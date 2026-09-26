import os
import sys
import torch
import torch.nn as nn
import tqdm
import argparse
import numpy as np
from pathlib import Path
from sklearn.metrics import classification_report, roc_auc_score, average_precision_score

from ccml.config import MODELS_CONFIG, EVALUATIONS_DIR, DATA_DIR, MODELS_DIR, ROOT_DIR
from ccml.training.datasets.utils import get_dataset_test_loader, split_spectrogram, get_dataset_train_val_loader, get_dataset_mean_std
from ccml.training.models.ast import ASTModel
from ccml.training.models.vgg_ish import ShortChunkCNN
from ccml.training.models.musicnn import Musicnn

import warnings

# ---- TF32 warning μόνο σε CUDA. Σε MPS/CPU το κρύβουμε.
if torch.cuda.is_available():
    torch.set_float32_matmul_precision("high")
else:
    warnings.filterwarnings("ignore", message=".*TF32.*", category=UserWarning)

from pathlib import Path


def evaluate(config, subset, out_dir):

    test_dataset, test_dataloader = get_dataset_test_loader(config, subset)

    if 'normalize_input' in config and config['normalize_input']:
        _, train_loader, _ = get_dataset_train_val_loader(config)
        config['norm_mean'], config['norm_std'] = get_dataset_mean_std(
            train_loader)

    # load model
    saved_models_dir = os.path.join(MODELS_DIR, config['dataset'])

    if subset:
       model_path = os.path.join(saved_models_dir, "subset", f'{config["model_name"]}.pth')
    else:
       model_path = os.path.join(saved_models_dir, f'{config["model_name"]}.pth')

    if 'musicnn' in config['model_name']:
        model = Musicnn(n_class=test_dataset.len_labels)
        model.load_state_dict(torch.load(
            model_path, map_location=torch.device(config['device'])))

    elif 'vgg_ish' in config['model_name']:
        model = ShortChunkCNN(n_class=test_dataset.len_labels)
        model.load_state_dict(torch.load(
            model_path, map_location=torch.device(config['device'])))

    elif 'ast' in config['model_name']:
        model = ASTModel(input_tdim=test_dataset.input_length,
                         label_dim=test_dataset.len_labels, model_size='base384')
        model.load_state_dict(torch.load(
            model_path, map_location=torch.device(config['device'])))
    else:
        raise NotImplementedError(
            'No model implementation found for the given config.')

    model = model.to(config['device'])
    model.eval()

    # get model prediction for each sample in the test set (store also its label)
    y = []
    y_ = []
    estimated = []
    sigmoid = torch.nn.Sigmoid()
    for i, single_sample_batch in enumerate(tqdm.tqdm(test_dataloader)):
        [mel_spectrogram], [label], [sample_id] = single_sample_batch
        vid = str(sample_id[0]) if isinstance(sample_id, (list, tuple)) else str(sample_id)

        splitted_spectrogram = split_spectrogram(
            mel_spectrogram, test_dataset.input_length)
        splits_scores = []
        for spectrogram in splitted_spectrogram:
            spectrogram = spectrogram[np.newaxis, :, :]

            if 'normalize_input' in config and config['normalize_input']:
                # normalize the input audio spectrogram so that the dataset mean
                # and standard deviation are 0 and 0.5 respectively (as in training)
                spectrogram = (
                    spectrogram - config['norm_mean']) / (config['norm_std'] * 2)

            out = model.forward(spectrogram.float().to(config['device']))
            splits_scores.append(out.detach().cpu().numpy())

        splits_scores = np.vstack(splits_scores)

        # average pooling on chunks scores
        tags_scores = np.mean(splits_scores, axis=0)


        if isinstance(config['loss_function'], nn.BCEWithLogitsLoss):
            # models with no sigmoid at their final layer when trained
            if isinstance(tags_scores, np.ndarray):
                tags_scores = torch.from_numpy(
                    tags_scores).float().to(config['device'])
            tags_scores = sigmoid(tags_scores)
            tags_scores = tags_scores.detach().cpu().numpy()

        # ----- μετά το tags_scores -----
        audio_probs = tags_scores.astype(np.float32).reshape(-1)   # probabilities (C,)
        labels = label.detach().cpu().numpy().astype(np.int32).reshape(-1)  # (C,)



        # -------- save --------
        out_path = out_dir / f"{vid}.npz"
        payload = dict(
            labels=labels,
            audio_probs= audio_probs,
            audio_id=np.asarray(vid),
            agg_audio=np.asarray("mean_probs_audioonly_chunks"),
        )
        np.savez_compressed(out_path, **payload)



if __name__ == '__main__':

    parser = argparse.ArgumentParser(
        formatter_class=argparse.ArgumentDefaultsHelpFormatter)

    parser.add_argument('--dataset', type=str, default='lyra', choices=[
                        'magnatagatune', 'fma', 'lyra', 'makam', 'hindustani', 'carnatic'])
    parser.add_argument('--data_dir', type=str,
                        default=os.path.join(DATA_DIR, 'lyra'))
    parser.add_argument('--model_name', type=str, default='vgg_ish', choices=['vgg_ish', 'ast'])
    parser.add_argument('--subset', type=bool, default=False)
    parser.add_argument('--device', type=str, default='cpu',
                        choices=['cpu', 'cuda:0', 'cuda:1', 'cuda:2', 'mps'])
    parser.add_argument('--out_dir', type=str, default=os.path.join(ROOT_DIR, 'save_audio_probs'))
    args = parser.parse_args()

    config = MODELS_CONFIG[args.dataset][args.model_name].copy()
    config['dataset'] = args.dataset
    config['data_dir'] = args.data_dir
    config['model_name'] = args.model_name
    config['device'] = args.device

    print(config)

    # Fix paths
    if args.subset:
       dataset_folder = "subset"
    else:
       dataset_folder = "whole_dataset"

    if args.model_name == "vgg_ish":
       time_folder = "3.69"
    else:
       time_folder = "8.00"

    set_folder = "test"
    temp_path = os.path.join(args.out_dir, time_folder, dataset_folder, set_folder)
    out_dir = Path(temp_path)
    out_dir.mkdir(parents=True, exist_ok=True)

    evaluate(config, args.subset, out_dir)
