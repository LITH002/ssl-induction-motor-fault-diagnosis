"""
Standalone validation: loads a saved pruned model and confirms it genuinely
works, real predictions on real held-out test recordings, per-class
accuracy breakdown, and a confusion matrix, not just one aggregate number.
Run this on your own machine BEFORE bothering with Raspberry Pi deployment.

Usage:
    python validate_pruned_model.py --dataset paderborn --data_dir paderborn \
        --fs 64000 --segment_len 4096 \
        --pruned_model pruned_30pct.pt --widths_file pruned_30pct_widths.txt \
        --n_examples 15
"""

import argparse

import numpy as np
import torch
import torch.nn as nn

from finetune import build_labelled_records, file_level_split, build_crop_pool, PoolDataset
from model import Conv1DEncoder, ClassificationHead, FaultClassifier


def build_pruned_shell(widths, embed_dim=128, in_channels=1, num_classes=3):
    """Reconstruct an empty model with the exact pruned channel widths,
    ready to have the saved state_dict loaded into it."""
    c1, c2, c3 = widths
    net = nn.Sequential(
        nn.Conv1d(in_channels, c1, kernel_size=15, stride=2, padding=7),
        nn.BatchNorm1d(c1), nn.GELU(),
        nn.Conv1d(c1, c2, kernel_size=9, stride=2, padding=4),
        nn.BatchNorm1d(c2), nn.GELU(),
        nn.Conv1d(c2, c3, kernel_size=5, stride=2, padding=2),
        nn.BatchNorm1d(c3), nn.GELU(),
        nn.Conv1d(c3, embed_dim, kernel_size=3, stride=1, padding=1),
        nn.BatchNorm1d(embed_dim), nn.GELU(),
    )
    encoder = Conv1DEncoder.__new__(Conv1DEncoder)
    nn.Module.__init__(encoder)
    encoder.net = net

    model = FaultClassifier.__new__(FaultClassifier)
    nn.Module.__init__(model)
    model.encoder = encoder
    model.head = ClassificationHead(embed_dim, num_classes)
    return model


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset", choices=["cwru", "paderborn"], required=True)
    parser.add_argument("--data_dir", required=True)
    parser.add_argument("--fs", type=int, required=True)
    parser.add_argument("--segment_len", type=int, default=2048)
    parser.add_argument("--pruned_model", required=True)
    parser.add_argument("--widths_file", required=True)
    parser.add_argument("--n_examples", type=int, default=15,
                         help="How many individual predictions to print for manual inspection")
    args = parser.parse_args()

    with open(args.widths_file) as f:
        widths = [int(x) for x in f.read().strip().split(",")]
    print(f"Loading pruned model with channel widths {widths} ...")

    records, classes = build_labelled_records(args.dataset, args.data_dir)
    _, test_records = file_level_split(records, classes, test_frac=0.3)
    test_pool = build_crop_pool(test_records, crops_per_file=16, segment_len=args.segment_len, seed=1)

    model = build_pruned_shell(widths, num_classes=len(classes))
    state = torch.load(args.pruned_model, map_location="cpu")
    model.load_state_dict(state)
    model.eval()
    print(f"Model loaded successfully. Parameters: {sum(p.numel() for p in model.parameters())}")

    # Full evaluation: confusion matrix + per-class accuracy
    confusion = np.zeros((len(classes), len(classes)), dtype=int)
    all_records_for_display = []

    with torch.no_grad():
        for item in test_pool:
            x = torch.from_numpy(item["segment"]).unsqueeze(0).unsqueeze(0)  # (1,1,seg_len)
            logits = model(x)
            probs = torch.softmax(logits, dim=-1).squeeze(0)
            pred_idx = probs.argmax().item()
            true_idx = classes.index(item["fault_label"])
            confusion[true_idx, pred_idx] += 1
            all_records_for_display.append((item, true_idx, pred_idx, probs))

    print(f"\n{'='*60}")
    print("CONFUSION MATRIX (rows = true class, columns = predicted class)")
    print(f"{'='*60}")
    header = "".join(f"{c[:10]:>12}" for c in classes)
    print(f"{'':14}{header}")
    for i, c in enumerate(classes):
        row = "".join(f"{confusion[i,j]:>12}" for j in range(len(classes)))
        print(f"{c[:12]:<14}{row}")

    print(f"\n{'='*60}")
    print("PER-CLASS ACCURACY")
    print(f"{'='*60}")
    for i, c in enumerate(classes):
        total = confusion[i].sum()
        correct = confusion[i, i]
        acc = correct / total if total > 0 else float("nan")
        print(f"  {c:<12} {correct}/{total} correct  ({acc:.1%})")

    overall_acc = np.trace(confusion) / confusion.sum()
    print(f"\n  Overall accuracy: {overall_acc:.1%}")

    print(f"\n{'='*60}")
    print(f"SAMPLE INDIVIDUAL PREDICTIONS (first {args.n_examples})")
    print(f"{'='*60}")
    rng = np.random.default_rng(0)
    sample_indices = rng.choice(len(all_records_for_display),
                                 size=min(args.n_examples, len(all_records_for_display)),
                                 replace=False)
    for idx in sample_indices:
        item, true_idx, pred_idx, probs = all_records_for_display[idx]
        status = "CORRECT" if true_idx == pred_idx else "WRONG  "
        prob_str = ", ".join(f"{classes[k]}={probs[k]:.2f}" for k in range(len(classes)))
        print(f"  [{status}] file={item['source_file']:<30} "
              f"true={classes[true_idx]:<10} pred={classes[pred_idx]:<10} ({prob_str})")

    print(f"\n{'='*60}")
    print("If the confusion matrix and per-class accuracy look sensible (no class "
          "collapsed entirely to 0%, no obviously degenerate always-predict-one-class "
          "pattern), this model is validated and ready to move to Raspberry Pi deployment. "
          "If something looks wrong here, it's better to catch it now than after setting "
          "up hardware.")


if __name__ == "__main__":
    main()