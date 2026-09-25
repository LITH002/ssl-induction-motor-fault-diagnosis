"""
Self-contained edge-hardware latency measurement — designed specifically
for a resource-constrained device (Raspberry Pi Zero 2 W, 512MB RAM).

No dataset, no scipy, no scikit-learn — only torch, and only the model
architecture itself, since that's all real inference timing needs. Copy
this file plus your saved checkpoint(s) to the Pi, nothing else.

Usage (baseline, unpruned model):
    python3 measure_on_pi.py --checkpoint trained_model_100pct.pt

Usage (pruned model — pass the widths saved alongside it):
    python3 measure_on_pi.py --checkpoint pruned_30pct.pt --widths 22 45 90

Reports latency as mean ± std across multiple SEPARATE timing sessions
(not just back-to-back runs within one session), which is more robust to
the kind of OS scheduling noise that affected the earlier CPU-proxy
measurements taken on the development laptop.
"""

import argparse
import time

import torch
import torch.nn as nn


class Conv1DEncoder(nn.Module):
    def __init__(self, widths, in_channels=1, embed_dim=128):
        super().__init__()
        c1, c2, c3 = widths
        self.net = nn.Sequential(
            nn.Conv1d(in_channels, c1, kernel_size=15, stride=2, padding=7),
            nn.BatchNorm1d(c1), nn.GELU(),
            nn.Conv1d(c1, c2, kernel_size=9, stride=2, padding=4),
            nn.BatchNorm1d(c2), nn.GELU(),
            nn.Conv1d(c2, c3, kernel_size=5, stride=2, padding=2),
            nn.BatchNorm1d(c3), nn.GELU(),
            nn.Conv1d(c3, embed_dim, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm1d(embed_dim), nn.GELU(),
        )

    def forward(self, x):
        return self.net(x)


class ClassificationHead(nn.Module):
    def __init__(self, embed_dim=128, num_classes=3):
        super().__init__()
        self.pool = nn.AdaptiveAvgPool1d(1)
        self.fc = nn.Linear(embed_dim, num_classes)

    def forward(self, z):
        return self.fc(self.pool(z).squeeze(-1))


class FaultClassifier(nn.Module):
    def __init__(self, widths, in_channels=1, embed_dim=128, num_classes=3):
        super().__init__()
        self.encoder = Conv1DEncoder(widths, in_channels, embed_dim)
        self.head = ClassificationHead(embed_dim, num_classes)

    def forward(self, x):
        return self.head(self.encoder(x))


def measure_session(model, segment_len, n_runs=50):
    dummy = torch.randn(1, 1, segment_len)
    model.eval()
    with torch.no_grad():
        for _ in range(5):  # warmup
            model(dummy)
        start = time.perf_counter()
        for _ in range(n_runs):
            model(dummy)
        elapsed = time.perf_counter() - start
    return (elapsed / n_runs) * 1000  # ms/sample


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--widths", type=int, nargs=3, default=[32, 64, 128],
                         help="Encoder channel widths — default is the unpruned baseline. "
                              "For a pruned model, pass the three numbers from its "
                              "*_widths.txt file, e.g. --widths 22 45 90")
    parser.add_argument("--segment_len", type=int, default=4096)
    parser.add_argument("--num_classes", type=int, default=3)
    parser.add_argument("--n_sessions", type=int, default=5)
    parser.add_argument("--n_runs_per_session", type=int, default=50)
    args = parser.parse_args()

    print(f"Building model with encoder widths {args.widths} ...")
    model = FaultClassifier(widths=args.widths, num_classes=args.num_classes)
    state = torch.load(args.checkpoint, map_location="cpu")
    model.load_state_dict(state)

    n_params = sum(p.numel() for p in model.parameters())
    print(f"Loaded checkpoint from {args.checkpoint}")
    print(f"Parameters: {n_params}")

    print(f"\nRunning {args.n_sessions} separate timing sessions "
          f"({args.n_runs_per_session} forward passes each) ...")
    latencies = []
    for s in range(1, args.n_sessions + 1):
        lat = measure_session(model, args.segment_len, args.n_runs_per_session)
        latencies.append(lat)
        print(f"  session {s}: {lat:.3f} ms/sample")

    mean_lat = sum(latencies) / len(latencies)
    std_lat = (sum((x - mean_lat) ** 2 for x in latencies) / len(latencies)) ** 0.5

    print(f"\n{'='*50}")
    print(f"RESULT — real edge hardware measurement")
    print(f"{'='*50}")
    print(f"Parameters:        {n_params}")
    print(f"Latency:            {mean_lat:.3f} +/- {std_lat:.3f} ms/sample")
    print(f"(across {args.n_sessions} independent sessions of {args.n_runs_per_session} runs each)")
    print(f"\nCopy this result back for Chapter 4 — this replaces the CPU-proxy "
          f"timing from the development laptop with a genuine edge-hardware number.")


if __name__ == "__main__":
    main()