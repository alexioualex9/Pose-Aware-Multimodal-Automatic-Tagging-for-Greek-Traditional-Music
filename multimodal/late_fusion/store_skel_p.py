import os
import json
import argparse
import numpy as np
from pathlib import Path
from tqdm import tqdm

import torch
import torch.nn as nn
from torch.utils.data import DataLoader

from skeletons.config import MODELS_CONFIG, MODELS_DIR, LABELS_SUBSET_DIR, OUT_DIR
from skeletons.skeleton_dataset import SkelAllClipsPerVideoCTv
from skeletons.build_model import return_model


def swap_np(x, i, j):
    x = x.copy()
    x[i], x[j] = x[j], x[i]
    return x

def swap_torch(x, i , j):
    x = x.clone()
    x[i], x[j] = x[j].clone(), x[i].clone()
    return x


def save_stgcn_probs(args):
    # ---- config ----
    config = MODELS_CONFIG[args.dataset].copy()
    config["dataset"] = args.dataset
    config["model_name"] = "STGCN"
    config["device"] = torch.device(args.device)

    # Skeleton embeddings path
    skel_emb_path = os.path.join(OUT_DIR, config["dataset"], args.time_window, "STGCN")

    # Get test index path
    index_path = os.path.join(skel_emb_path, args.split, "skeleton_index.json")
    if not os.path.exists(index_path):
        raise FileNotFoundError(f"Missing index: {index_path}")

    # Load labels
    labels = json.load(open(LABELS_SUBSET_DIR))["labels"]
    C_out = len(labels)

    # Dataset/Loader
    ds = SkelAllClipsPerVideoCTv(index_path, min_valid=config.get("min_valid", 0))
    loader = DataLoader(ds, batch_size=1, shuffle=False, num_workers=0)

    # Get input's dimension
    C_in = ds.C_in

    # Build model
    model = return_model(
        model_name="STGCN",
        C_out=C_out,
        C_in=C_in,
        config=config,
    )

    # Checkpoint's path
    ckpt_path = args.ckpt
    if ckpt_path is None:
        ckpt_path = os.path.join(MODELS_DIR, args.time_window, "STGCN", str(args.seed), "best.pt")
    if not os.path.exists(ckpt_path):
        raise FileNotFoundError(f"Missing checkpoint: {ckpt_path}")

    # Path in which probs are going to be saved
    out_dir = os.path.join(args.out_dir, args.time_window, str(args.seed), args.split)
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    # Define sigmoid
    sigmoid = nn.Sigmoid()

    # Load checkpoint
    state = torch.load(ckpt_path, map_location=config["device"])
    model.load_state_dict(state)
    model.eval()

    # ---- inference loop ----
    with torch.no_grad():
        for Xs, y, vid in tqdm(loader, desc=f"Save probs ({args.split})"):
            vid_str = str(vid[0]) if isinstance(vid, (list, tuple)) else str(vid)

            if isinstance(Xs, (list, tuple)):
                Xs = Xs[0]
            if isinstance(y, (list, tuple)):
                y = y[0]

            # squeeze batch dim if present
            if Xs.dim() == 5:
                # [1, Nv, C, T, V] -> [Nv, C, T, V]
                Xs_ = Xs.squeeze(0)
            elif Xs.dim() == 4:
                # [Nv, C, T, V]
                Xs_ = Xs
            else:
                raise RuntimeError(f"Unexpected Xs shape: {tuple(Xs.shape)}")

            if y.dim() == 2:
                y_ = y.squeeze(0)  # [C_out]
            else:
                y_ = y

            # Swap labels (Lyra, Oud)
            y_ = swap_torch(y_, 20, 21)

            Xs_ = Xs_.to(config["device"])
            y_ = y_.to(config["device"])

            # model outputs per-clip logits: [Nv, C_out]
            logits_clips = model(Xs_)  # STGCNModel should accept [Nv,C,T,V]
            if logits_clips.dim() != 2 or logits_clips.shape[1] != C_out:
                raise RuntimeError(f"Unexpected logits shape: {tuple(logits_clips.shape)}")

            # video-level aggregation
            aggregate = logits_clips.mean(dim=0)
            probs_video = sigmoid(aggregate)

            # Swap labels
            probs_video = swap_torch(probs_video, 20, 21)

            # move to numpy
            skel_probs = probs_video.detach().cpu().numpy().astype(np.float32).reshape(-1)
            labels_np = y_.detach().cpu().numpy().astype(np.int32).reshape(-1)

            # save
            out_path = out_dir / f"{vid_str}.npz"
            payload = dict(
                labels=labels_np,
                skel_probs=skel_probs,
                skel_id=np.asarray(vid_str),
                agg_skel=np.asarray("mean_probs_skeleton_clips"),
            )
            np.savez_compressed(out_path, **payload)

    print(f"[done] wrote .npz files to: {out_dir}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset", type=str, default="lyra")
    parser.add_argument("--device", type=str, default="cpu", choices=["cpu", "cuda", "mps"])
    parser.add_argument("--split", type=str, default="test", choices=["train", "val", "test"])
    parser.add_argument("--time_window", type=str, default="3.69", choices=["3.69", "8.00"])
    parser.add_argument("--seed", type=int, default=42, choices=[42, 123, 1337, 2024, 9999])
    parser.add_argument("--out_dir", type=str, default="skeleton_probs", help="Output folder where <vid>.npz will be saved")
    parser.add_argument("--ckpt", type=str, default=None, help="Path to STGCN checkpoint (defaults to MODELS_DIR/STGCN/best.pt)")
    args = parser.parse_args()

    save_stgcn_probs(args)
