"""Encoder stacks. RawPixelEncoder is the pure numpy/PIL floor; TinyCNNAE and
TinyCNNPred are small from-scratch torch stacks (reconstructive vs
predictive objective) that stand in for blocked V-JEPA-2/DINO encoders."""
import hashlib
import io

import numpy as np
import torch
import torch.nn as nn
from PIL import Image

from physics_auditor.generator.clip import Clip

RAW_PIXEL_SIZE = 16
LATENT_DIM = 128


class RawPixelEncoder:
    """Raw-pixel control: greyscale, downsampled to 16x16, flattened."""

    name = "raw-pixel"
    cache_key = "raw16-v1"

    def encode(self, clip: Clip) -> np.ndarray:
        gray = clip.frames.mean(axis=-1) / 255.0  # (T, H, W) float
        out = np.empty((gray.shape[0], RAW_PIXEL_SIZE * RAW_PIXEL_SIZE), dtype=np.float32)
        for t in range(gray.shape[0]):
            frame_u8 = (gray[t] * 255.0).astype(np.uint8)
            small = Image.fromarray(frame_u8).resize((RAW_PIXEL_SIZE, RAW_PIXEL_SIZE), Image.BILINEAR)
            out[t] = (np.asarray(small, dtype=np.float32) / 255.0).reshape(-1)
        return out


def _state_dict_hash(module: nn.Module) -> str:
    buf = io.BytesIO()
    torch.save(module.state_dict(), buf)
    return hashlib.sha1(buf.getvalue()).hexdigest()[:8]


def _frames_to_tensor(clip: Clip) -> torch.Tensor:
    frames = clip.frames.astype(np.float32) / 255.0  # (T, 64, 64, 3)
    return torch.from_numpy(frames).permute(0, 3, 1, 2).contiguous()  # (T, 3, 64, 64)


class _ConvEncoder(nn.Module):
    """Conv(3,16,k4,s2,p1)-ReLU-Conv(16,32,k4,s2,p1)-ReLU-Conv(32,64,k4,s2,p1)
    -ReLU-Flatten-Linear(4096,128). 64x64 -> 32x32 -> 16x16 -> 8x8x64=4096."""

    def __init__(self):
        super().__init__()
        self.net = nn.Sequential(
            nn.Conv2d(3, 16, 4, 2, 1), nn.ReLU(),
            nn.Conv2d(16, 32, 4, 2, 1), nn.ReLU(),
            nn.Conv2d(32, 64, 4, 2, 1), nn.ReLU(),
            nn.Flatten(),
            nn.Linear(4096, LATENT_DIM),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x)


class _ConvDecoder(nn.Module):
    """Mirror of _ConvEncoder: Linear(128,4096) -> reshape(64,8,8) -> three
    ConvTranspose2d(k4,s2,p1) back up to 64x64x3, sigmoid-squashed to [0,1]."""

    def __init__(self):
        super().__init__()
        self.fc = nn.Linear(LATENT_DIM, 4096)
        self.deconv = nn.Sequential(
            nn.ReLU(),
            nn.ConvTranspose2d(64, 32, 4, 2, 1), nn.ReLU(),
            nn.ConvTranspose2d(32, 16, 4, 2, 1), nn.ReLU(),
            nn.ConvTranspose2d(16, 3, 4, 2, 1),
            nn.Sigmoid(),
        )

    def forward(self, z: torch.Tensor) -> torch.Tensor:
        x = self.fc(z).view(-1, 64, 8, 8)
        return self.deconv(x)


class _AEModel(nn.Module):
    def __init__(self):
        super().__init__()
        self.encoder = _ConvEncoder()
        self.decoder = _ConvDecoder()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.decoder(self.encoder(x))


class _PredModel(nn.Module):
    def __init__(self):
        super().__init__()
        self.encoder = _ConvEncoder()
        self.head = nn.Sequential(
            nn.Linear(4 * LATENT_DIM, 512), nn.ReLU(), nn.Linear(512, LATENT_DIM)
        )
        self.decoder = _ConvDecoder()

    def forward(self, context: torch.Tensor) -> torch.Tensor:
        # context: (B, 4, 3, 64, 64) -- 4 consecutive frames
        b = context.shape[0]
        zs = [self.encoder(context[:, i]) for i in range(4)]
        z_cat = torch.cat(zs, dim=1)  # (B, 512)
        z_next = self.head(z_cat)  # (B, 128)
        return self.decoder(z_next)


class TinyCNNAE:
    """Reconstructive torch stack: conv autoencoder trained with pixel MSE.
    The frozen encoder half is the stack's latent representation."""

    name = "tiny-cnn-ae"

    def __init__(self):
        self.model = _AEModel()
        self.model.eval()

    @property
    def cache_key(self) -> str:
        return f"{self.name}-{_state_dict_hash(self.model.encoder)}"

    def encode(self, clip: Clip) -> np.ndarray:
        self.model.eval()
        x = _frames_to_tensor(clip)
        with torch.no_grad():
            z = self.model.encoder(x)
        return z.numpy().astype(np.float32)

    def fit(self, frames: np.ndarray, epochs: int = 3, batch_size: int = 128, lr: float = 1e-3, seed: int = 0) -> list[float]:
        """frames: (N, 64, 64, 3) float32 in [0,1]. Reinitialises weights
        deterministically from `seed` then trains as an autoencoder (MSE)."""
        torch.manual_seed(seed)
        self.model = _AEModel()
        self.model.train()
        x = torch.from_numpy(frames.astype(np.float32)).permute(0, 3, 1, 2).contiguous()
        n = x.shape[0]
        opt = torch.optim.Adam(self.model.parameters(), lr=lr)
        loss_fn = nn.MSELoss()
        losses = []
        for _epoch in range(epochs):
            perm = torch.randperm(n)
            total = 0.0
            n_batches = 0
            for start in range(0, n, batch_size):
                idx = perm[start:start + batch_size]
                batch = x[idx]
                opt.zero_grad()
                recon = self.model(batch)
                loss = loss_fn(recon, batch)
                loss.backward()
                opt.step()
                total += loss.item()
                n_batches += 1
            losses.append(total / max(n_batches, 1))
        self.model.eval()
        return losses


class TinyCNNPred:
    """Predictive torch stack: same conv encoder, trained via a next-frame
    prediction head on 4-frame context windows. Only the frozen encoder is
    used as the stack's representation at encode time."""

    name = "tiny-cnn-pred"

    def __init__(self):
        self.model = _PredModel()
        self.model.eval()

    @property
    def cache_key(self) -> str:
        return f"{self.name}-{_state_dict_hash(self.model.encoder)}"

    def encode(self, clip: Clip) -> np.ndarray:
        self.model.eval()
        x = _frames_to_tensor(clip)
        with torch.no_grad():
            z = self.model.encoder(x)
        return z.numpy().astype(np.float32)

    def fit(self, frames: np.ndarray, epochs: int = 3, batch_size: int = 128, lr: float = 1e-3, seed: int = 0) -> list[float]:
        """frames: (N, 64, 64, 3) float32 in [0,1]. Reinitialises weights
        deterministically from `seed`, trains on 5-frame windows: 4-frame
        context -> predicted next frame's pixels (MSE)."""
        torch.manual_seed(seed)
        self.model = _PredModel()
        self.model.train()
        x_all = torch.from_numpy(frames.astype(np.float32)).permute(0, 3, 1, 2).contiguous()
        n_windows = x_all.shape[0] - 4
        if n_windows <= 0:
            raise ValueError("need at least 5 frames to form one context+target window")
        contexts = torch.stack([x_all[i:i + 4] for i in range(n_windows)])  # (W,4,3,64,64)
        targets = torch.stack([x_all[i + 4] for i in range(n_windows)])  # (W,3,64,64)

        opt = torch.optim.Adam(self.model.parameters(), lr=lr)
        loss_fn = nn.MSELoss()
        losses = []
        for _epoch in range(epochs):
            perm = torch.randperm(n_windows)
            total = 0.0
            n_batches = 0
            for start in range(0, n_windows, batch_size):
                idx = perm[start:start + batch_size]
                ctx_batch = contexts[idx]
                tgt_batch = targets[idx]
                opt.zero_grad()
                pred = self.model(ctx_batch)
                loss = loss_fn(pred, tgt_batch)
                loss.backward()
                opt.step()
                total += loss.item()
                n_batches += 1
            losses.append(total / max(n_batches, 1))
        self.model.eval()
        return losses
