"""
Week 1 driver script: pretrain the masked-reconstruction encoder on the
pooled unlabelled CWRU + Paderborn signal segments.

Usage (after editing the paths below to point at your local downloads):

    python pretrain.py --cwru_dir /path/to/CWRU --paderborn_dir /path/to/Paderborn

This is deliberately single-pretext-task (masked reconstruction only) to get
something end-to-end and working this week. The contrastive design and the
head-to-head comparison between the two (Section 3.6) is Week 2 scope.
"""

import argparse
import numpy as np
import torch
from torch.utils.data import Dataset, DataLoader

from preprocessing import (
    load_cwru_mat_dir,
    load_paderborn_mat_dir,
    SSLPretrainDataset,
)
from model import MaskedReconstructionModel, apply_mask, weighted_reconstruction_loss


class TorchSegmentDataset(Dataset):
    """Thin torch Dataset wrapper around SSLPretrainDataset — draws a fresh
    random crop each __getitem__ call, which is what gives the shifting-
    window behaviour across epochs.

    `epoch_multiplier` makes each epoch draw that many random crops per
    underlying recording, rather than exactly one. This matters a lot for
    small file counts (e.g. 16 CWRU recordings) — without it, one epoch is
    only 16 samples total, which is both too little signal per epoch and,
    combined with a batch size larger than the dataset, can silently produce
    zero batches if the DataLoader drops incomplete batches.
    """

    def __init__(self, ssl_dataset, epoch_multiplier=1):
        self.ds = ssl_dataset
        self.epoch_multiplier = max(1, epoch_multiplier)

    def __len__(self):
        return len(self.ds) * self.epoch_multiplier

    def __getitem__(self, idx):
        real_idx = idx % len(self.ds)
        seg = self.ds.get_segment(real_idx, domain="time")  # (segment_len,)
        return torch.from_numpy(seg).float().unsqueeze(0)  # (1, segment_len)


def build_pooled_dataset(cwru_dir, paderborn_dir, fs=12000, segment_len=2048):
    records = []
    if cwru_dir:
        cwru_records = load_cwru_mat_dir(cwru_dir)
        print(f"Loaded {len(cwru_records)} CWRU recordings")
        records.extend(cwru_records)
    if paderborn_dir:
        pb_records = load_paderborn_mat_dir(paderborn_dir)
        print(f"Loaded {len(pb_records)} Paderborn recordings")
        records.extend(pb_records)
    if not records:
        raise ValueError("No data loaded — provide at least one of --cwru_dir / --paderborn_dir")
    print(f"Pooled unlabelled pretraining set: {len(records)} recordings "
          f"(shifting-window cropping means effectively many more samples per epoch)")
    return SSLPretrainDataset(records, fs=fs, segment_len=segment_len)


def train(model, loader, device, epochs=20, lr=1e-3, mask_rate=0.4, mask_span=32):
    optim = torch.optim.AdamW(model.parameters(), lr=lr)
    model.to(device)
    model.train()

    for epoch in range(1, epochs + 1):
        total_loss, n_batches = 0.0, 0
        for x in loader:
            x = x.to(device)
            x_masked, mask = apply_mask(x, mask_rate=mask_rate, mask_span=mask_span)
            x_hat = model(x_masked)
            loss = weighted_reconstruction_loss(x_hat, x, mask)

            optim.zero_grad()
            loss.backward()
            optim.step()

            total_loss += loss.item()
            n_batches += 1

        print(f"epoch {epoch:03d} | avg masked-reconstruction loss: {total_loss / max(n_batches,1):.6f}")

    return model


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--cwru_dir", type=str, default=None,
                         help="Path to extracted CWRU .mat files")
    parser.add_argument("--paderborn_dir", type=str, default=None,
                         help="Path to extracted Paderborn .mat files")
    parser.add_argument("--fs", type=int, default=12000,
                         help="Sampling rate of the loaded signals (CWRU DE data is commonly 12kHz)")
    parser.add_argument("--segment_len", type=int, default=2048)
    parser.add_argument("--batch_size", type=int, default=64)
    parser.add_argument("--epochs", type=int, default=20)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--mask_rate", type=float, default=0.4)
    parser.add_argument("--mask_span", type=int, default=32)
    parser.add_argument("--out", type=str, default="pretrained_encoder.pt")
    args = parser.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    ssl_dataset = build_pooled_dataset(args.cwru_dir, args.paderborn_dir, fs=args.fs, segment_len=args.segment_len)

    # Auto-pick an epoch multiplier so small file counts still get a
    # reasonable number of samples per epoch, and adapt batch size down if
    # the (possibly multiplied) dataset is still smaller than the requested
    # batch size, so drop_last never silently zeroes out every batch.
    target_samples_per_epoch = 512
    epoch_multiplier = max(1, target_samples_per_epoch // max(1, len(ssl_dataset)))
    torch_dataset = TorchSegmentDataset(ssl_dataset, epoch_multiplier=epoch_multiplier)

    effective_batch_size = min(args.batch_size, max(1, len(torch_dataset) // 2))
    if effective_batch_size != args.batch_size:
        print(f"[info] reducing batch size from {args.batch_size} to {effective_batch_size} "
              f"to fit the dataset size ({len(torch_dataset)} samples/epoch after "
              f"{epoch_multiplier}x epoch multiplier)")

    loader = DataLoader(torch_dataset, batch_size=effective_batch_size, shuffle=True,
                         drop_last=True, num_workers=2)
    print(f"Samples per epoch: {len(torch_dataset)} (epoch_multiplier={epoch_multiplier}), "
          f"batch size: {effective_batch_size}, batches/epoch: {len(loader)}")

    model = MaskedReconstructionModel(in_channels=1, embed_dim=128)
    model = train(model, loader, device, epochs=args.epochs, lr=args.lr,
                  mask_rate=args.mask_rate, mask_span=args.mask_span)

    torch.save(model.encoder.state_dict(), args.out)
    print(f"Saved pretrained encoder weights to {args.out}")
    print("Next (Week 2): load these encoder weights, attach a small classification "
          "head, and fine-tune on labelled CWRU/Paderborn subsets at 1%/5%/10%/full "
          "label fractions (Section 3.7).")


if __name__ == "__main__":
    main()