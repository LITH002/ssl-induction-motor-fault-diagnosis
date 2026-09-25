"""
First-Order MAML (FOMAML) meta-training baseline (Section 3.7), following
the general approach of Pourghoraba et al. (2025) — same motor type, same
signal, same fault categories as your dissertation.

Uses First-Order MAML rather than full second-order MAML: it skips the
expensive Hessian-vector products full MAML needs, at a small accuracy
cost that's well-documented as negligible in practice, and makes CPU
training actually feasible within your timeline. This is the same
simplification most published few-shot fault-diagnosis papers make.

How it works, in plain terms: each "task" is a small K-shot episode (K
support examples + Q query examples per class, always the same 3 fault
classes, since your problem doesn't vary classes across episodes the way
classic few-shot image classification does — it varies which specific
examples/conditions make up each episode). For each task: clone the
model, take a few gradient steps on the support set (this is the "fast
adaptation" MAML is named for), then compute the loss on the query set
using the adapted clone. That query-set gradient is copied onto the
ORIGINAL model and used for the outer-loop update — this copy step is
exactly what makes it "first-order" (it skips backpropagating through the
adaptation step itself).

After meta-training, the saved checkpoint is a full FaultClassifier
(encoder + head) — this is what --maml_checkpoint in finetune.py loads
before running the SAME 1%/5%/10%/100% fine-tuning/evaluation sweep used
for the SSL and supervised conditions, for a fair three-way comparison.

Usage:
    python maml_pretrain.py --dataset paderborn --data_dir paderborn \
        --fs 64000 --segment_len 4096 --meta_iterations 300 \
        --out maml_paderborn.pt
"""

import argparse
import copy

import numpy as np
import torch
import torch.nn as nn

from finetune import (
    build_labelled_records, file_level_split, build_crop_pool,
    set_all_seeds,
)
from model import FaultClassifier


def sample_task(pool_by_class, classes, k_shot, q_query, rng):
    support_x, support_y, query_x, query_y = [], [], [], []
    for class_idx, c in enumerate(classes):
        items = pool_by_class[c]
        if len(items) < k_shot + q_query:
            chosen = [items[rng.integers(0, len(items))] for _ in range(k_shot + q_query)]
        else:
            idxs = rng.choice(len(items), size=k_shot + q_query, replace=False)
            chosen = [items[i] for i in idxs]
        for item in chosen[:k_shot]:
            support_x.append(item["segment"])
            support_y.append(class_idx)
        for item in chosen[k_shot:]:
            query_x.append(item["segment"])
            query_y.append(class_idx)

    support_x = torch.from_numpy(np.stack(support_x)).unsqueeze(1)
    support_y = torch.tensor(support_y, dtype=torch.long)
    query_x = torch.from_numpy(np.stack(query_x)).unsqueeze(1)
    query_y = torch.tensor(query_y, dtype=torch.long)
    return support_x, support_y, query_x, query_y


def fomaml_train(model, pool_by_class, classes, device,
                  meta_iterations=300, meta_batch_size=4, k_shot=5, q_query=10,
                  inner_lr=0.01, inner_steps=3, outer_lr=1e-3, seed=0):
    rng = np.random.default_rng(seed)
    criterion = nn.CrossEntropyLoss()
    meta_optim = torch.optim.Adam(model.parameters(), lr=outer_lr)
    model.to(device)

    for it in range(1, meta_iterations + 1):
        meta_optim.zero_grad()
        accumulated_grads = [torch.zeros_like(p) for p in model.parameters()]

        for _ in range(meta_batch_size):
            support_x, support_y, query_x, query_y = sample_task(
                pool_by_class, classes, k_shot, q_query, rng)
            support_x, support_y = support_x.to(device), support_y.to(device)
            query_x, query_y = query_x.to(device), query_y.to(device)

            clone = copy.deepcopy(model)
            clone_optim = torch.optim.SGD(clone.parameters(), lr=inner_lr)

            clone.train()
            for _ in range(inner_steps):
                logits = clone(support_x)
                loss = criterion(logits, support_y)
                clone_optim.zero_grad()
                loss.backward()
                clone_optim.step()

            query_logits = clone(query_x)
            query_loss = criterion(query_logits, query_y)
            clone.zero_grad()
            query_loss.backward()

            for acc_grad, clone_param in zip(accumulated_grads, clone.parameters()):
                if clone_param.grad is not None:
                    acc_grad += clone_param.grad / meta_batch_size

        for param, grad in zip(model.parameters(), accumulated_grads):
            param.grad = grad
        meta_optim.step()

        if it % 20 == 0 or it == 1:
            print(f"  meta-iter {it:04d}/{meta_iterations} | query loss (last task): {query_loss.item():.4f}")

    return model


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset", choices=["cwru", "paderborn"], required=True)
    parser.add_argument("--data_dir", required=True)
    parser.add_argument("--fs", type=int, required=True)
    parser.add_argument("--segment_len", type=int, default=2048)
    parser.add_argument("--crops_per_file", type=int, default=20)
    parser.add_argument("--meta_iterations", type=int, default=300)
    parser.add_argument("--meta_batch_size", type=int, default=4)
    parser.add_argument("--k_shot", type=int, default=5)
    parser.add_argument("--q_query", type=int, default=10)
    parser.add_argument("--inner_lr", type=float, default=0.01)
    parser.add_argument("--inner_steps", type=int, default=3)
    parser.add_argument("--outer_lr", type=float, default=1e-3)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--out", type=str, default="maml_checkpoint.pt")
    args = parser.parse_args()

    set_all_seeds(args.seed)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    records, classes = build_labelled_records(args.dataset, args.data_dir)
    train_records, _ = file_level_split(records, classes, test_frac=0.3)
    print(f"Meta-training pool: {len(train_records)} files")

    train_pool = build_crop_pool(train_records, crops_per_file=args.crops_per_file,
                                  segment_len=args.segment_len, seed=0)

    pool_by_class = {c: [item for item in train_pool if item["fault_label"] == c] for c in classes}
    for c in classes:
        print(f"  class '{c}': {len(pool_by_class[c])} crops available for episode sampling")

    model = FaultClassifier(in_channels=1, embed_dim=128, num_classes=len(classes))

    print(f"\nMeta-training: {args.meta_iterations} iterations, "
          f"{args.meta_batch_size} tasks/iter, {args.k_shot}-shot/{args.q_query}-query, "
          f"inner_lr={args.inner_lr}, inner_steps={args.inner_steps}, outer_lr={args.outer_lr}")

    model = fomaml_train(
        model, pool_by_class, classes, device,
        meta_iterations=args.meta_iterations, meta_batch_size=args.meta_batch_size,
        k_shot=args.k_shot, q_query=args.q_query,
        inner_lr=args.inner_lr, inner_steps=args.inner_steps, outer_lr=args.outer_lr,
        seed=args.seed,
    )

    torch.save(model.state_dict(), args.out)
    print(f"\nSaved MAML-meta-trained checkpoint (encoder + head) to {args.out}")
    print("Next: run finetune.py with --maml_checkpoint pointing at this file, "
          "using the same --fractions sweep as your SSL/supervised runs, for a "
          "fair three-way comparison.")


if __name__ == "__main__":
    main()