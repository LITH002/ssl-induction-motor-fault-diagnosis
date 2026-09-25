"""
Encoder + decoder for masked-signal-reconstruction pretraining, plus the
Week 2 fine-tuning classifier built on the same encoder.
"""

import torch
import torch.nn as nn


class Conv1DEncoder(nn.Module):
    def __init__(self, in_channels=1, embed_dim=128):
        super().__init__()
        self.net = nn.Sequential(
            nn.Conv1d(in_channels, 32, kernel_size=15, stride=2, padding=7),
            nn.BatchNorm1d(32), nn.GELU(),
            nn.Conv1d(32, 64, kernel_size=9, stride=2, padding=4),
            nn.BatchNorm1d(64), nn.GELU(),
            nn.Conv1d(64, 128, kernel_size=5, stride=2, padding=2),
            nn.BatchNorm1d(128), nn.GELU(),
            nn.Conv1d(128, embed_dim, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm1d(embed_dim), nn.GELU(),
        )

    def forward(self, x):
        return self.net(x)


class Conv1DDecoder(nn.Module):
    def __init__(self, embed_dim=128, out_channels=1):
        super().__init__()
        self.net = nn.Sequential(
            nn.ConvTranspose1d(embed_dim, 128, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm1d(128), nn.GELU(),
            nn.ConvTranspose1d(128, 64, kernel_size=5, stride=2, padding=2, output_padding=1),
            nn.BatchNorm1d(64), nn.GELU(),
            nn.ConvTranspose1d(64, 32, kernel_size=9, stride=2, padding=4, output_padding=1),
            nn.BatchNorm1d(32), nn.GELU(),
            nn.ConvTranspose1d(32, out_channels, kernel_size=15, stride=2, padding=7, output_padding=1),
        )

    def forward(self, z):
        return self.net(z)


class MaskedReconstructionModel(nn.Module):
    def __init__(self, in_channels=1, embed_dim=128):
        super().__init__()
        self.encoder = Conv1DEncoder(in_channels, embed_dim)
        self.decoder = Conv1DDecoder(embed_dim, in_channels)

    def forward(self, x_masked):
        z = self.encoder(x_masked)
        x_hat = self.decoder(z)
        if x_hat.shape[-1] != x_masked.shape[-1]:
            min_len = min(x_hat.shape[-1], x_masked.shape[-1])
            x_hat = x_hat[..., :min_len]
        return x_hat


def apply_mask(x, mask_rate=0.4, mask_span=32):
    batch, channels, length = x.shape
    mask = torch.zeros(batch, 1, length, dtype=torch.bool, device=x.device)
    n_spans = max(1, int((length * mask_rate) / mask_span))
    for b in range(batch):
        for _ in range(n_spans):
            start = torch.randint(0, max(1, length - mask_span), (1,)).item()
            mask[b, 0, start:start + mask_span] = True
    x_masked = x.clone()
    x_masked = x_masked.masked_fill(mask, 0.0)
    return x_masked, mask


def weighted_reconstruction_loss(x_hat, x_true, mask, eps=1e-3):
    mask_f = mask.float()
    weights = 1.0 / (x_true.abs() + eps)
    weights = weights / (weights.sum(dim=-1, keepdim=True) + eps)
    sq_err = (x_hat - x_true) ** 2
    weighted_err = sq_err * weights * mask_f
    denom = mask_f.sum() + eps
    return weighted_err.sum() / denom


class ClassificationHead(nn.Module):
    def __init__(self, embed_dim=128, num_classes=3):
        super().__init__()
        self.pool = nn.AdaptiveAvgPool1d(1)
        self.fc = nn.Linear(embed_dim, num_classes)

    def forward(self, z):
        pooled = self.pool(z).squeeze(-1)
        return self.fc(pooled)


class FaultClassifier(nn.Module):
    def __init__(self, in_channels=1, embed_dim=128, num_classes=3,
                 pretrained_encoder_path=None, freeze_encoder=False):
        super().__init__()
        self.encoder = Conv1DEncoder(in_channels, embed_dim)
        if pretrained_encoder_path is not None:
            state = torch.load(pretrained_encoder_path, map_location="cpu")
            self.encoder.load_state_dict(state)
        if freeze_encoder:
            for p in self.encoder.parameters():
                p.requires_grad = False
        self.head = ClassificationHead(embed_dim, num_classes)

    def forward(self, x):
        z = self.encoder(x)
        return self.head(z)