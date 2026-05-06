"""PyTorch neural model definitions for paper-quality attacks.

Architectures:
  - ResNet1DTransformer (for A1 distinguishability)
  - UNet1D (for A3 neural reconstruction)
  - SiameseEncoder (for A4-CI role-stratified linkage)

All models import torch lazily so other code paths remain torch-free.
Use `pip install torch` (or `pip install -e .[heavy]`) to enable.
"""

from __future__ import annotations
import logging
from typing import Optional

logger = logging.getLogger(__name__)


def _try_import_torch():
    try:
        import torch
        import torch.nn as nn
        return torch, nn
    except ImportError:
        return None, None


# ============================================================================
# ResNet-1D + Transformer head (for A1 Distinguishability)
# ============================================================================


def make_resnet1d_transformer(
    n_channels: int = 5,
    n_classes: int = 2,
    base_filters: int = 32,
    n_resblocks: int = 4,
    transformer_d_model: int = 128,
    transformer_nhead: int = 4,
    transformer_layers: int = 2,
):
    """Create a ResNet-1D backbone + transformer head.

    Returns nn.Module or None if torch unavailable.
    """
    torch, nn = _try_import_torch()
    if torch is None:
        return None

    class ResBlock1D(nn.Module):
        def __init__(self, in_ch, out_ch, stride=1):
            super().__init__()
            self.conv1 = nn.Conv1d(in_ch, out_ch, 7, stride=stride, padding=3)
            self.bn1 = nn.BatchNorm1d(out_ch)
            self.conv2 = nn.Conv1d(out_ch, out_ch, 7, padding=3)
            self.bn2 = nn.BatchNorm1d(out_ch)
            self.shortcut = (nn.Conv1d(in_ch, out_ch, 1, stride=stride)
                             if stride != 1 or in_ch != out_ch else nn.Identity())
            self.relu = nn.ReLU(inplace=True)

        def forward(self, x):
            identity = self.shortcut(x)
            out = self.relu(self.bn1(self.conv1(x)))
            out = self.bn2(self.conv2(out))
            return self.relu(out + identity)

    class ResNet1DTransformer(nn.Module):
        def __init__(self):
            super().__init__()
            self.input_conv = nn.Sequential(
                nn.Conv1d(n_channels, base_filters, 15, padding=7),
                nn.BatchNorm1d(base_filters),
                nn.ReLU(inplace=True),
                nn.MaxPool1d(2),
            )
            blocks = []
            ch = base_filters
            for i in range(n_resblocks):
                out_ch = ch * 2 if i > 0 else ch
                blocks.append(ResBlock1D(ch, out_ch, stride=2 if i > 0 else 1))
                ch = out_ch
            self.resblocks = nn.Sequential(*blocks)
            # Project to transformer d_model
            self.proj = nn.Conv1d(ch, transformer_d_model, 1)
            encoder_layer = nn.TransformerEncoderLayer(
                d_model=transformer_d_model,
                nhead=transformer_nhead,
                batch_first=True,
                dim_feedforward=transformer_d_model * 2,
            )
            self.transformer = nn.TransformerEncoder(encoder_layer, num_layers=transformer_layers)
            self.head = nn.Linear(transformer_d_model, n_classes)

        def forward(self, x):  # x: (B, C, L)
            x = self.input_conv(x)
            x = self.resblocks(x)         # (B, ch, L')
            x = self.proj(x)              # (B, d, L')
            x = x.transpose(1, 2)         # (B, L', d)
            x = self.transformer(x)
            x = x.mean(dim=1)             # global avg
            return self.head(x)

    return ResNet1DTransformer()


# ============================================================================
# 1D-UNet (for A3 Neural Reconstruction)
# ============================================================================


def make_unet1d(
    n_channels: int = 5,
    base_filters: int = 32,
    depth: int = 4,
):
    """Simple 1D U-Net for signal reconstruction. Returns nn.Module or None."""
    torch, nn = _try_import_torch()
    if torch is None:
        return None

    class DoubleConv1D(nn.Module):
        def __init__(self, in_ch, out_ch):
            super().__init__()
            self.block = nn.Sequential(
                nn.Conv1d(in_ch, out_ch, 3, padding=1),
                nn.BatchNorm1d(out_ch),
                nn.ReLU(inplace=True),
                nn.Conv1d(out_ch, out_ch, 3, padding=1),
                nn.BatchNorm1d(out_ch),
                nn.ReLU(inplace=True),
            )
        def forward(self, x):
            return self.block(x)

    class UNet1D(nn.Module):
        def __init__(self):
            super().__init__()
            self.downs = nn.ModuleList()
            self.pools = nn.ModuleList()
            self.ups = nn.ModuleList()
            self.upconvs = nn.ModuleList()
            ch = n_channels
            channels = []
            for d in range(depth):
                out = base_filters * (2 ** d)
                self.downs.append(DoubleConv1D(ch, out))
                self.pools.append(nn.MaxPool1d(2))
                channels.append(out)
                ch = out
            self.bottleneck = DoubleConv1D(ch, ch * 2)
            ch = ch * 2
            for d in range(depth):
                target = channels[-(d + 1)]
                self.upconvs.append(nn.ConvTranspose1d(ch, target, 2, stride=2))
                self.ups.append(DoubleConv1D(ch, target))
                ch = target
            self.out_conv = nn.Conv1d(ch, n_channels, 1)

        def forward(self, x):
            skips = []
            for down, pool in zip(self.downs, self.pools):
                x = down(x)
                skips.append(x)
                x = pool(x)
            x = self.bottleneck(x)
            for up, upconv, skip in zip(self.ups, self.upconvs, reversed(skips)):
                x = upconv(x)
                # match length if odd
                if x.size(-1) != skip.size(-1):
                    x = nn.functional.interpolate(x, size=skip.size(-1), mode="linear")
                x = torch.cat([x, skip], dim=1)
                x = up(x)
            return self.out_conv(x)

    return UNet1D()


# ============================================================================
# Siamese Encoder (for A4-CI Role-stratified Linkage)
# ============================================================================


def make_siamese_encoder(
    n_channels: int = 5,
    embed_dim: int = 64,
    base_filters: int = 32,
    n_blocks: int = 3,
):
    """Twin CNN encoder mapping signals to embedding for contrastive learning."""
    torch, nn = _try_import_torch()
    if torch is None:
        return None

    class SiameseEncoder(nn.Module):
        def __init__(self):
            super().__init__()
            layers = []
            ch = n_channels
            for i in range(n_blocks):
                out = base_filters * (2 ** i)
                layers.append(nn.Conv1d(ch, out, 7, padding=3, stride=2))
                layers.append(nn.BatchNorm1d(out))
                layers.append(nn.ReLU(inplace=True))
                ch = out
            self.encoder = nn.Sequential(*layers)
            self.pool = nn.AdaptiveAvgPool1d(1)
            self.proj = nn.Linear(ch, embed_dim)

        def forward(self, x):  # (B, C, L) -> (B, embed_dim)
            x = self.encoder(x)
            x = self.pool(x).squeeze(-1)
            x = self.proj(x)
            return nn.functional.normalize(x, dim=-1)

    return SiameseEncoder()


# ============================================================================
# Training utilities (works for any of the above)
# ============================================================================


def _best_device():
    """Return best available device string. Prefers CUDA if available."""
    try:
        import torch
        if torch.cuda.is_available():
            return "cuda"
    except ImportError:
        pass
    return "cpu"


def _wrap_multi_gpu(model, device):
    """Wrap model with DataParallel if multiple GPUs are available."""
    try:
        import torch
        if device == "cuda" and torch.cuda.device_count() > 1:
            return torch.nn.DataParallel(model)
    except ImportError:
        pass
    return model


def train_classifier(model, train_X, train_y, val_X=None, val_y=None,
                     epochs=20, batch_size=32, lr=1e-3, device=None,
                     verbose=False):
    """Train a classifier with simple SGD loop. Returns trained model + history."""
    torch, nn = _try_import_torch()
    if torch is None:
        raise RuntimeError("PyTorch required. Install with: pip install torch")
    device = device or _best_device()
    model = _wrap_multi_gpu(model.to(device), device)
    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)

    X = torch.from_numpy(train_X).float().to(device)
    y = torch.from_numpy(train_y).long().to(device)
    n = len(X)
    history = {"train_loss": [], "val_acc": []}

    for epoch in range(epochs):
        model.train()
        perm = torch.randperm(n, device=device)
        epoch_loss = 0.0
        for i in range(0, n, batch_size):
            idx = perm[i:i + batch_size]
            optimizer.zero_grad()
            out = model(X[idx])
            loss = criterion(out, y[idx])
            loss.backward()
            optimizer.step()
            epoch_loss += loss.item() * len(idx)
        history["train_loss"].append(epoch_loss / n)
        if val_X is not None and val_y is not None:
            model.eval()
            with torch.no_grad():
                vX = torch.from_numpy(val_X).float().to(device)
                vy = torch.from_numpy(val_y).long().to(device)
                pred = model(vX).argmax(dim=1)
                val_acc = (pred == vy).float().mean().item()
            history["val_acc"].append(val_acc)
            if verbose:
                logger.info(f"epoch {epoch+1}/{epochs}: train_loss={history['train_loss'][-1]:.4f}, val_acc={val_acc:.4f}")
    return model, history


def predict_classifier(model, X, device=None):
    """Return predicted class indices."""
    torch, _nn = _try_import_torch()
    device = device or _best_device()
    model.eval()
    with torch.no_grad():
        X_t = torch.from_numpy(X).float().to(device)
        out = model(X_t)
        return out.argmax(dim=1).cpu().numpy(), out.softmax(dim=1).cpu().numpy()


def train_reconstruction(model, train_TX, train_X, val_TX=None, val_X=None,
                          epochs=30, batch_size=16, lr=1e-3, device=None,
                          verbose=False):
    """Train UNet for signal reconstruction (transformed -> original)."""
    torch, nn = _try_import_torch()
    if torch is None:
        raise RuntimeError("PyTorch required")
    device = device or _best_device()
    model = _wrap_multi_gpu(model.to(device), device)
    criterion = nn.MSELoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)

    TX = torch.from_numpy(train_TX).float().to(device)
    X = torch.from_numpy(train_X).float().to(device)
    n = len(X)

    for epoch in range(epochs):
        model.train()
        perm = torch.randperm(n, device=device)
        epoch_loss = 0.0
        for i in range(0, n, batch_size):
            idx = perm[i:i + batch_size]
            optimizer.zero_grad()
            out = model(TX[idx])
            loss = criterion(out, X[idx])
            loss.backward()
            optimizer.step()
            epoch_loss += loss.item() * len(idx)
        if verbose and (epoch + 1) % 5 == 0:
            logger.info(f"recon epoch {epoch+1}/{epochs}: train_loss={epoch_loss/n:.6f}")
    return model


def train_siamese(model, segments_with_ids, n_epochs=20, batch_size=32,
                   lr=1e-3, temperature=0.1, device=None, verbose=False):
    """Train siamese encoder with InfoNCE-style contrastive loss.

    segments_with_ids: list of (signal_array, persona_id_int).
    """
    torch, nn = _try_import_torch()
    if torch is None:
        raise RuntimeError("PyTorch required")
    device = device or _best_device()
    model = _wrap_multi_gpu(model.to(device), device)
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)

    signals = np.array([s for s, _ in segments_with_ids])
    pids = np.array([p for _, p in segments_with_ids])
    n = len(signals)

    X = torch.from_numpy(signals).float().to(device)
    pids_t = torch.from_numpy(pids).long().to(device)

    for epoch in range(n_epochs):
        model.train()
        perm = torch.randperm(n, device=device)
        epoch_loss = 0.0
        for i in range(0, n, batch_size):
            idx = perm[i:i + batch_size]
            if len(idx) < 4:
                continue
            optimizer.zero_grad()
            embeds = model(X[idx])  # (B, D)
            # Compute similarity matrix
            sim = torch.matmul(embeds, embeds.t()) / temperature
            # Mask: same persona = positive
            ids = pids_t[idx]
            mask_pos = (ids[:, None] == ids[None, :]).float()
            mask_pos.fill_diagonal_(0)
            # Skip if no positives in batch
            if mask_pos.sum() == 0:
                continue
            # InfoNCE-style
            log_softmax = nn.functional.log_softmax(sim, dim=1)
            # Mean log-likelihood of positive pairs
            pos_sum = (log_softmax * mask_pos).sum(dim=1)
            pos_count = mask_pos.sum(dim=1).clamp(min=1)
            loss = -(pos_sum / pos_count).mean()
            loss.backward()
            optimizer.step()
            epoch_loss += loss.item()
        if verbose and (epoch + 1) % 5 == 0:
            logger.info(f"siamese epoch {epoch+1}/{n_epochs}: loss={epoch_loss:.4f}")
    return model


# Make numpy import explicit (used in train_siamese)
import numpy as np
