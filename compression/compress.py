"""
Phase 5: Model compression and edge feasibility (Section 3.8).

Applies iterative structured channel pruning to the trained FaultClassifier's
encoder, following the general approach of Xu et al. (2023): rank channels
by their contribution (here, L1 norm of each output channel's filter
weights, the standard magnitude-pruning criterion), remove the least
important channels, and retrain briefly to recover any lost accuracy.

This prunes the three INTERMEDIATE convolutional layers of Conv1DEncoder
(the ones producing 32, 64, and 128 channels) and leaves the final layer's
output (embed_dim) untouched, since that dimension feeds directly into the
classification head — pruning it would require also resizing the head's
first Linear layer, which is a reasonable extension but adds complexity
this first pass avoids. Genuine structural pruning is performed: the
resulting Conv1d/BatchNorm1d layers are physically smaller (fewer stored
parameters), not just zeroed out, so the reported size and speed
differences are real, not simulated.

Usage:
    python compress.py --dataset paderborn --data_dir paderborn --fs 64000 \
        --segment_len 4096 --checkpoint trained_model_100pct.pt \
        --pruning_ratios 0.1 0.2 0.3 0.4 0.5 --out compression_results.csv

The --checkpoint file is produced by finetune.py's --save_model flag,
run once at the end of a normal fine-tuning sweep (typically the full,
100%-label-fraction condition, your best-performing model).
"""

import argparse
import copy
import csv
import time

import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from sklearn.metrics import f1_score, accuracy_score

from finetune import (
    build_labelled_records, file_level_split, build_crop_pool,
    subsample_pool_by_fraction, PoolDataset, set_all_seeds, compute_class_weights,
)
from model import Conv1DEncoder, ClassificationHead, FaultClassifier


# Indices within Conv1DEncoder.net of the three prunable Conv1d layers and
# their paired BatchNorm1d layers (see model.py's Sequential definition).
_PRUNABLE_CONV_IDX = [0, 3, 6]
_PRUNABLE_BN_IDX = [1, 4, 7]


def channel_importance(conv_layer):
    """L1 norm of each output channel's filter weights — the standard,
    simple magnitude-pruning criterion."""
    weight = conv_layer.weight.data  # (out_channels, in_channels, kernel)
    return weight.abs().sum(dim=(1, 2))


def build_pruned_encoder(encoder, pruning_ratio, in_channels=1, embed_dim=128):
    """
    Returns a NEW, smaller Conv1DEncoder with the least important channels
    removed from each of the three intermediate conv layers, weights
    copied over from the original (kept channels only). The final conv
    layer's output width (embed_dim) is unchanged, so the classification
    head needs no modification.
    """
    old_layers = list(encoder.net)
    original_out_channels = [old_layers[i].out_channels for i in _PRUNABLE_CONV_IDX]

    kept_indices = []  # kept_indices[k] = sorted tensor of channel indices kept at prunable layer k
    new_out_channels = []
    for k, conv_idx in enumerate(_PRUNABLE_CONV_IDX):
        conv = old_layers[conv_idx]
        importance = channel_importance(conv)
        n_keep = max(1, int(round(conv.out_channels * (1 - pruning_ratio))))
        keep_idx = torch.argsort(importance, descending=True)[:n_keep]
        keep_idx, _ = torch.sort(keep_idx)
        kept_indices.append(keep_idx)
        new_out_channels.append(n_keep)

    # Build the new encoder with reduced widths passed as construction args
    # is awkward since Conv1DEncoder's __init__ doesn't expose per-layer
    # widths — so we construct the Sequential directly here instead.
    c1_out, c2_out, c3_out = new_out_channels
    new_net = nn.Sequential(
        nn.Conv1d(in_channels, c1_out, kernel_size=15, stride=2, padding=7),
        nn.BatchNorm1d(c1_out), nn.GELU(),
        nn.Conv1d(c1_out, c2_out, kernel_size=9, stride=2, padding=4),
        nn.BatchNorm1d(c2_out), nn.GELU(),
        nn.Conv1d(c2_out, c3_out, kernel_size=5, stride=2, padding=2),
        nn.BatchNorm1d(c3_out), nn.GELU(),
        nn.Conv1d(c3_out, embed_dim, kernel_size=3, stride=1, padding=1),
        nn.BatchNorm1d(embed_dim), nn.GELU(),
    )

    # Copy weights: for each prunable conv, keep only the selected OUTPUT
    # channels; for the conv immediately after it, keep only the matching
    # INPUT channels (since that layer's input width shrank too).
    new_conv_idx = _PRUNABLE_CONV_IDX + [9]  # include the final, unpruned-output conv
    prev_keep = None
    for step, conv_idx in enumerate(new_conv_idx):
        old_conv = old_layers[conv_idx]
        new_conv = new_net[conv_idx]

        w = old_conv.weight.data
        b = old_conv.bias.data if old_conv.bias is not None else None

        if prev_keep is not None:
            w = w[:, prev_keep, :]  # narrow input channels to match previous layer's pruning

        if conv_idx in _PRUNABLE_CONV_IDX:
            keep_idx = kept_indices[_PRUNABLE_CONV_IDX.index(conv_idx)]
            w = w[keep_idx, :, :]
            if b is not None:
                b = b[keep_idx]
            new_conv.weight.data.copy_(w)
            if b is not None:
                new_conv.bias.data.copy_(b)

            # matching BatchNorm
            old_bn = old_layers[_PRUNABLE_BN_IDX[_PRUNABLE_CONV_IDX.index(conv_idx)]]
            new_bn = new_net[_PRUNABLE_BN_IDX[_PRUNABLE_CONV_IDX.index(conv_idx)]]
            new_bn.weight.data.copy_(old_bn.weight.data[keep_idx])
            new_bn.bias.data.copy_(old_bn.bias.data[keep_idx])
            new_bn.running_mean.copy_(old_bn.running_mean[keep_idx])
            new_bn.running_var.copy_(old_bn.running_var[keep_idx])

            prev_keep = keep_idx
        else:
            # final conv (index 9): output width unchanged, only input narrowed
            new_conv.weight.data.copy_(w)
            if b is not None:
                new_conv.bias.data.copy_(b)
            # final BatchNorm (index 10) is untouched — same width as before
            old_bn = old_layers[10]
            new_bn = new_net[10]
            new_bn.load_state_dict(old_bn.state_dict())

    new_encoder = Conv1DEncoder.__new__(Conv1DEncoder)
    nn.Module.__init__(new_encoder)
    new_encoder.net = new_net

    print(f"    Pruned widths: {original_out_channels} -> {new_out_channels} "
          f"(ratio={pruning_ratio:.0%})")
    return new_encoder


def count_parameters(module):
    return sum(p.numel() for p in module.parameters())


def measure_inference_latency(model, segment_len, n_runs=100, device="cpu"):
    """Average single-sample forward-pass latency in milliseconds, CPU."""
    model.eval()
    dummy = torch.randn(1, 1, segment_len, device=device)
    with torch.no_grad():
        for _ in range(10):  # warmup
            model(dummy)
        start = time.perf_counter()
        for _ in range(n_runs):
            model(dummy)
        elapsed = time.perf_counter() - start
    return (elapsed / n_runs) * 1000  # ms per sample


def retrain_briefly(model, loader, device, epochs, lr, class_weights=None):
    optim = torch.optim.AdamW(model.parameters(), lr=lr)
    criterion = nn.CrossEntropyLoss(weight=class_weights)
    model.to(device)
    model.train()
    for epoch in range(1, epochs + 1):
        total_loss, n = 0.0, 0
        for x, y in loader:
            x, y = x.to(device), y.to(device)
            logits = model(x)
            loss = criterion(logits, y)
            optim.zero_grad()
            loss.backward()
            optim.step()
            total_loss += loss.item()
            n += 1
        print(f"      retrain epoch {epoch:02d} | loss: {total_loss / max(n,1):.4f}")
    return model


def evaluate(model, loader, device):
    model.eval()
    all_true, all_pred = [], []
    with torch.no_grad():
        for x, y in loader:
            x = x.to(device)
            logits = model(x)
            pred = torch.softmax(logits, dim=-1).argmax(dim=-1).cpu().numpy()
            all_true.extend(y.numpy().tolist())
            all_pred.extend(pred.tolist())
    acc = accuracy_score(all_true, all_pred)
    macro_f1 = f1_score(all_true, all_pred, average="macro", zero_division=0)
    return acc, macro_f1


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset", choices=["cwru", "paderborn"], required=True)
    parser.add_argument("--data_dir", required=True)
    parser.add_argument("--fs", type=int, required=True)
    parser.add_argument("--segment_len", type=int, default=2048)
    parser.add_argument("--checkpoint", required=True,
                         help="Trained FaultClassifier state_dict (from finetune.py --save_model)")
    parser.add_argument("--pruning_ratios", type=float, nargs="+",
                         default=[0.1, 0.2, 0.3, 0.4, 0.5])
    parser.add_argument("--retrain_epochs", type=int, default=20)
    parser.add_argument("--lr", type=float, default=1e-4)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--out", type=str, default="compression_results.csv")
    parser.add_argument("--save_pruned_model", type=str, default=None,
                         help="Save the pruned+retrained model at --save_ratio to this path.")
    parser.add_argument("--save_ratio", type=float, default=None,
                         help="Which pruning ratio (must be in --pruning_ratios) to save via "
                              "--save_pruned_model. Required if --save_pruned_model is set.")
    args = parser.parse_args()

    set_all_seeds(args.seed)
    device = torch.device("cpu")  # edge feasibility proxy — measured on CPU
    print(f"Using device: {device} (CPU timing used as edge-hardware proxy)")

    records, classes = build_labelled_records(args.dataset, args.data_dir)
    train_records, test_records = file_level_split(records, classes, test_frac=0.3)
    train_pool = build_crop_pool(train_records, crops_per_file=20, segment_len=args.segment_len, seed=0)
    test_pool = build_crop_pool(test_records, crops_per_file=16, segment_len=args.segment_len, seed=1)
    test_ds = PoolDataset(test_pool, classes, epoch_multiplier=1)
    test_loader = DataLoader(test_ds, batch_size=min(32, len(test_ds)), shuffle=False)
    train_ds = PoolDataset(train_pool, classes, epoch_multiplier=1)
    train_loader = DataLoader(train_ds, batch_size=32, shuffle=True, drop_last=True)
    class_weights = compute_class_weights(train_pool, classes, device)

    base_model = FaultClassifier(in_channels=1, embed_dim=128, num_classes=len(classes))
    state = torch.load(args.checkpoint, map_location="cpu")
    base_model.load_state_dict(state)
    print(f"Loaded checkpoint from {args.checkpoint}")

    results = []

    # Baseline (unpruned) measurement first
    base_params = count_parameters(base_model)
    base_latency = measure_inference_latency(base_model, args.segment_len)
    base_acc, base_f1 = evaluate(base_model, test_loader, device)
    print(f"\nBaseline (unpruned): params={base_params}, latency={base_latency:.3f} ms, "
          f"accuracy={base_acc:.3f}, macro_f1={base_f1:.3f}")
    results.append({
        "pruning_ratio": 0.0, "parameters": base_params, "latency_ms": base_latency,
        "accuracy": base_acc, "macro_f1": base_f1,
        "size_reduction_pct": 0.0, "accuracy_retained_pct": 100.0,
    })

    for ratio in args.pruning_ratios:
        print(f"\n--- pruning ratio {ratio:.0%} ---")
        pruned_encoder = build_pruned_encoder(base_model.encoder, ratio)

        pruned_model = FaultClassifier(in_channels=1, embed_dim=128, num_classes=len(classes))
        pruned_model.encoder = pruned_encoder
        pruned_model.head = copy.deepcopy(base_model.head)  # head untouched, embed_dim unchanged

        pruned_model = retrain_briefly(pruned_model, train_loader, device,
                                        epochs=args.retrain_epochs, lr=args.lr,
                                        class_weights=class_weights)

        params = count_parameters(pruned_model)
        latency = measure_inference_latency(pruned_model, args.segment_len)
        acc, macro_f1 = evaluate(pruned_model, test_loader, device)

        size_reduction = (1 - params / base_params) * 100
        accuracy_retained = (acc / base_acc) * 100 if base_acc > 0 else float("nan")

        print(f"    -> params={params} ({size_reduction:.1f}% smaller), "
              f"latency={latency:.3f} ms, accuracy={acc:.3f} "
              f"({accuracy_retained:.1f}% of baseline), macro_f1={macro_f1:.3f}")

        results.append({
            "pruning_ratio": ratio, "parameters": params, "latency_ms": latency,
            "accuracy": acc, "macro_f1": macro_f1,
            "size_reduction_pct": size_reduction, "accuracy_retained_pct": accuracy_retained,
        })

        if args.save_pruned_model and args.save_ratio is not None and abs(ratio - args.save_ratio) < 1e-9:
            torch.save(pruned_model.state_dict(), args.save_pruned_model)
            # also save the exact pruned widths, since a loader needs these
            # to reconstruct the right-shaped model before loading weights
            widths_path = args.save_pruned_model.rsplit(".", 1)[0] + "_widths.txt"
            with open(widths_path, "w") as wf:
                pruned_widths = [pruned_model.encoder.net[i].out_channels for i in _PRUNABLE_CONV_IDX]
                wf.write(",".join(str(w) for w in pruned_widths))
            print(f"    Saved pruned model to {args.save_pruned_model} "
                  f"(widths saved to {widths_path} — needed to reload this model later)")

    with open(args.out, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=list(results[0].keys()))
        writer.writeheader()
        writer.writerows(results)
    print(f"\nSaved compression results to {args.out}")
    print("This table (pruning ratio vs. size reduction, latency, and retained "
          "accuracy) is your Phase 5 results table — the trade-off curve for "
          "choosing a deployment-ready compression level.")


if __name__ == "__main__":
    main()