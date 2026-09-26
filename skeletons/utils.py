import torch
import numpy as np
from sklearn.metrics import (
    classification_report,
    roc_auc_score,
    average_precision_score,
    f1_score,
)
from STGCN_model import STGCNModel
from helpers.mlp_head import MLPHead


# Save metrics in a .txt file
def save_metrics(metrics, out_dir, model_name, dataset, split):

    # TXT classification report
    with open(out_dir, "w", encoding="utf-8") as f:
        f.write(f'Evaluation of model "{model_name}" on "{dataset}" {split} set:\n')
        f.write(f'ROC-AUC score: {metrics["roc_macro"]}\n')
        f.write(f'PR-AUC score: {metrics["pr_macro"]}\n')
        f.write(f'F1 score: {metrics["f1_macro"]}\n')
       # f.write(metrics["report"])


def load_embed_stats(stats_path: str):
    z = np.load(stats_path)
    mu = z["mu"].astype(np.float32)
    std = z["std"].astype(np.float32)
    std = np.maximum(std, 1e-6)
    return mu, std



def collate_videos(batch):
    # batch: list of (Xs [Nv,D], y [Cout], vid)
    Xs_list, y_list, vids = zip(*batch)
    lengths = torch.tensor([x.size(0) for x in Xs_list], dtype=torch.long)

    D = Xs_list[0].size(1)
    Tmax = int(lengths.max().item())

    Xpad = torch.zeros((len(batch), Tmax, D), dtype=Xs_list[0].dtype)
    for i, x in enumerate(Xs_list):
        Xpad[i, :x.size(0)] = x

    Y = torch.stack(y_list, dim=0)  # [B, Cout]
    return Xpad, lengths, Y, list(vids)


def return_model(model_name: str, C_out: int, C_in: int, config: dict):
    """
    model_name:
      - "stgcn"  -> STGCNModel
    """
    return STGCNModel(
         num_class=C_out,
         in_channels=C_in,
         hidden_channels=config['hidden_channels'],
         num_layers=config['num_layers'],
         use_edge_importance=config['no_edge_importance'],
         multi_scale_tcn=config['multi_scale_tcn'],
         root=config['root'],
    ).to(config['device'])


def safe_auc(fn, Y, S, average: str):
    try:
        return fn(Y, S, average=average)
    except Exception:
        return float("nan")


def compute_global_metrics(Y, S, threshold, label_names):
    P = (S >= threshold).astype(np.int32)

    roc_micro = safe_auc(roc_auc_score, Y, S, average="micro")
    roc_macro = safe_auc(roc_auc_score, Y, S, average="macro")
    pr_micro  = safe_auc(average_precision_score, Y, S, average="micro")
    pr_macro  = safe_auc(average_precision_score, Y, S, average="macro")

    f1_micro = f1_score(Y, P, average="micro", zero_division=0)
    f1_macro = f1_score(Y, P, average="macro", zero_division=0)

    report = classification_report(Y, P, target_names=label_names, zero_division=0, digits=2)

    return dict(
        report=report,
        roc_micro=roc_micro, roc_macro=roc_macro,
        pr_micro=pr_micro, pr_macro=pr_macro,
        f1_micro=f1_micro, f1_macro=f1_macro,
    )


# Print test results
def test_results(m, model_name=None, dataset=None, split="test",
                 auc_kind="macro", decimals=2):
    """
    m: dict: report, roc_micro, roc_macro, pr_micro, pr_macro
    """

    if model_name is not None and dataset is not None:
        print(f'\nEvaluation of model "{model_name}" on "{dataset}" {split} set:')
    else:
        # fallback if names are not given
        print(f"\nEvaluation results ({split} set):")

    roc_key = f"roc_{auc_kind}"
    pr_key  = f"pr_{auc_kind}"

    if roc_key in m:
        print(f"ROC-AUC score: {m[roc_key]}")
    if pr_key in m:
        print(f"PR-AUC score: {m[pr_key]}")
    print()

    # Convert classification report into 2 decimals
    report = m.get("report", "")
    if isinstance(report, str) and report:
        import re
        def _fmt(match):
            return f"{float(match.group(0)):.{decimals}f}"
        report = re.sub(r"\d+\.\d{4,}", _fmt, report)

    print(report)
