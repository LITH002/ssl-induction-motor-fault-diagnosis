"""
Week 2 driver script: fine-tune (or train from scratch) a fault classifier
at varying label fractions, comparing SSL-pretrained, MAML-pretrained, and
fully-supervised conditions.
"""

import argparse
import csv
import random
from collections import defaultdict

import numpy as np
import torch
from torch.utils.data import Dataset, DataLoader
from sklearn.metrics import f1_score, roc_auc_score, accuracy_score

from preprocessing import load_cwru_mat_dir, load_paderborn_mat_dir, rms_normalise, random_crop_segment
from model import FaultClassifier
from labels import (
    cwru_label_from_filename, CWRU_CLASSES,
    paderborn_label_from_filename, PADERBORN_CLASSES,
)


def set_all_seeds(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def build_labelled_records(dataset, data_dir):
    if dataset == "cwru":
        raw = load_cwru_mat_dir(data_dir)
        classes = CWRU_CLASSES
        for r in raw:
            parsed = cwru_label_from_filename(r["label"])
            r["fault_label"] = parsed[0] if parsed else None
    elif dataset == "paderborn":
        raw = load_paderborn_mat_dir(data_dir)
        classes = PADERBORN_CLASSES
        for r in raw:
            r["fault_label"] = paderborn_label_from_filename(r["label"])
    else:
        raise ValueError(dataset)

    kept = [r for r in raw if r["fault_label"] is not None]
    dropped = len(raw) - len(kept)
    if dropped:
        print(f"[info] dropped {dropped} record(s) with unresolved/excluded labels")

    by_class = defaultdict(list)
    for r in kept:
        by_class[r["fault_label"]].append(r)
    for c in classes:
        print(f"  class '{c}': {len(by_class.get(c, []))} recordings")

    return kept, classes


def file_level_split(records, classes, test_frac=0.3, seed=0):
    rng = random.Random(seed)
    by_class = defaultdict(list)
    for r in records:
        by_class[r["fault_label"]].append(r)

    train, test = [], []
    for c in classes:
        items = by_class.get(c, [])
        rng.shuffle(items)
        n_test = max(1, int(len(items) * test_frac)) if len(items) > 1 else 0
        test.extend(items[:n_test])
        train.extend(items[n_test:])
    return train, test


def build_crop_pool(train_records, crops_per_file=20, segment_len=2048, seed=0):
    rng = np.random.default_rng(seed)
    pool = []
    for r in train_records:
        signal = rms_normalise(r["signal"])
        for _ in range(crops_per_file):
            seg = random_crop_segment(signal, segment_len, rng=rng)
            pool.append({
                "segment": seg.astype(np.float32),
                "fault_label": r["fault_label"],
                "source_file": r["file"],
            })
    return pool


def subsample_pool_by_fraction(pool, classes, fraction, seed=0):
    rng = random.Random(seed)
    by_class = defaultdict(list)
    for item in pool:
        by_class[item["fault_label"]].append(item)

    subset = []
    for c in classes:
        items = by_class.get(c, [])
        if not items:
            continue
        rng.shuffle(items)
        n = max(1, round(len(items) * fraction))
        subset.extend(items[:n])
    return subset


class PoolDataset(Dataset):
    def __init__(self, pool_items, classes, epoch_multiplier=1):
        self.items = pool_items
        self.classes = classes
        self.class_to_idx = {c: i for i, c in enumerate(classes)}
        self.epoch_multiplier = max(1, epoch_multiplier)

    def __len__(self):
        return len(self.items) * self.epoch_multiplier

    def __getitem__(self, idx):
        item = self.items[idx % len(self.items)]
        x = torch.from_numpy(item["segment"]).unsqueeze(0)
        y = self.class_to_idx[item["fault_label"]]
        return x, y


def compute_class_weights(pool_items, classes, device):
    counts = defaultdict(int)
    for item in pool_items:
        counts[item["fault_label"]] += 1
    weights = []
    for c in classes:
        n = counts.get(c, 1)
        weights.append(1.0 / n)
    weights = torch.tensor(weights, dtype=torch.float32, device=device)
    weights = weights / weights.sum() * len(classes)
    return weights


def train_classifier(model, loader, device, epochs=15, lr=1e-3, class_weights=None):
    optim = torch.optim.AdamW(filter(lambda p: p.requires_grad, model.parameters()), lr=lr)
    criterion = torch.nn.CrossEntropyLoss(weight=class_weights)
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
        print(f"    epoch {epoch:03d} | train loss: {total_loss / max(n,1):.4f}")
    return model


def evaluate(model, loader, device, num_classes):
    model.eval()
    all_true, all_pred, all_probs = [], [], []
    with torch.no_grad():
        for x, y in loader:
            x = x.to(device)
            logits = model(x)
            probs = torch.softmax(logits, dim=-1).cpu().numpy()
            pred = probs.argmax(axis=-1)
            all_true.extend(y.numpy().tolist())
            all_pred.extend(pred.tolist())
            all_probs.extend(probs.tolist())

    acc = accuracy_score(all_true, all_pred)
    macro_f1 = f1_score(all_true, all_pred, average="macro", zero_division=0)
    try:
        auc = roc_auc_score(all_true, all_probs, multi_class="ovr", labels=list(range(num_classes)))
    except ValueError:
        auc = float("nan")
    return acc, macro_f1, auc


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset", choices=["cwru", "paderborn"], required=True)
    parser.add_argument("--data_dir", required=True)
    parser.add_argument("--fs", type=int, required=True)
    parser.add_argument("--segment_len", type=int, default=2048)
    parser.add_argument("--pretrained", type=str, default=None)
    parser.add_argument("--maml_checkpoint", type=str, default=None)
    parser.add_argument("--freeze_encoder", action="store_true")
    parser.add_argument("--fractions", type=float, nargs="+", default=[0.01, 0.05, 0.10, 1.0])
    parser.add_argument("--crops_per_file", type=int, default=20)
    parser.add_argument("--epochs", type=int, default=15)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--out", type=str, default="finetune_results.csv")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--save_model", type=str, default=None,
                         help="Path to save the trained classifier (encoder+head) after the "
                              "LAST fraction in --fractions (typically 100%%). This is the "
                              "checkpoint the compression phase (compress.py) prunes.")
    args = parser.parse_args()

    set_all_seeds(args.seed)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")
    if args.maml_checkpoint:
        print(f"Condition: MAML meta-trained ({args.maml_checkpoint})")
    elif args.pretrained:
        print(f"Condition: SSL pretrained ({args.pretrained})")
    else:
        print("Condition: Supervised baseline (no pretraining)")

    records, classes = build_labelled_records(args.dataset, args.data_dir)
    train_records, test_records = file_level_split(records, classes, test_frac=0.3)
    print(f"Train files: {len(train_records)} | Test files: {len(test_records)}")

    test_pool = build_crop_pool(test_records, crops_per_file=16, segment_len=args.segment_len, seed=1)
    test_ds = PoolDataset(test_pool, classes, epoch_multiplier=1)
    test_loader = DataLoader(test_ds, batch_size=min(32, len(test_ds)), shuffle=False)
    print(f"Test pool size: {len(test_pool)} crops from {len(test_records)} files")

    train_pool = build_crop_pool(train_records, crops_per_file=args.crops_per_file,
                                  segment_len=args.segment_len, seed=0)
    print(f"Train pool size: {len(train_pool)} crops from {len(train_records)} files "
          f"({args.crops_per_file} crops/file)")

    results = []
    for frac in args.fractions:
        subset = subsample_pool_by_fraction(train_pool, classes, frac)
        print(f"\n--- label fraction {frac:.0%} | {len(subset)} training crops "
              f"(of {len(train_pool)} available) ---")

        target_samples_per_epoch = 256
        multiplier = max(1, target_samples_per_epoch // max(1, len(subset)))
        train_ds = PoolDataset(subset, classes, epoch_multiplier=multiplier)
        batch_size = min(32, max(1, len(train_ds) // 2))
        train_loader = DataLoader(train_ds, batch_size=batch_size, shuffle=True, drop_last=True)

        model = FaultClassifier(
            in_channels=1, embed_dim=128, num_classes=len(classes),
            pretrained_encoder_path=args.pretrained, freeze_encoder=args.freeze_encoder,
        )
        if args.maml_checkpoint:
            state = torch.load(args.maml_checkpoint, map_location="cpu")
            model.load_state_dict(state)

        class_weights = compute_class_weights(subset, classes, device)
        model = train_classifier(model, train_loader, device, epochs=args.epochs, lr=args.lr,
                                  class_weights=class_weights)
        acc, macro_f1, auc = evaluate(model, test_loader, device, num_classes=len(classes))
        print(f"    -> accuracy={acc:.3f}  macro_f1={macro_f1:.3f}  auc={auc:.3f}")

        condition = "maml" if args.maml_checkpoint else ("ssl_pretrained" if args.pretrained else "supervised_scratch")
        results.append({
            "dataset": args.dataset, "condition": condition, "label_fraction": frac,
            "n_train_crops": len(subset), "accuracy": acc, "macro_f1": macro_f1, "auc": auc,
        })

        if args.save_model and frac == args.fractions[-1]:
            torch.save(model.state_dict(), args.save_model)
            print(f"    Saved trained model (this fraction) to {args.save_model}")

    with open(args.out, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=list(results[0].keys()))
        writer.writeheader()
        writer.writerows(results)
    print(f"\nSaved results to {args.out}")


if __name__ == "__main__":
    main()