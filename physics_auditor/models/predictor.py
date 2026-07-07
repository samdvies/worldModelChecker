"""Per-stack latent dynamics predictor: MLP(4*D -> 512 -> 512 -> D) trained
with teacher-forced next-latent MSE on cached latents. Frozen encoder +
this predictor form a LatentStackAdapter (see models/stacks.py); closed-loop
rollout() feeds predictions back in as context for VoE surprise scoring.

The MLP predicts a RESIDUAL on top of the last-seen latent (delta), and its
final layer is zero-initialised. This means an untrained predictor is
*exactly* the predict-last-latent baseline (delta=0) -- training can only
improve on that floor, never silently regress below it (see reviewer
finding: predictors previously underperformed the trivial baseline by
15x-130x because a plain from-scratch MLP had to relearn the near-identity
mapping from randomly initialised weights within very few epochs)."""
import copy
import json
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn

K_DEFAULT = 4
HIDDEN_DEFAULT = 512


class PredictorVersionMismatch(Exception):
    """Raised when a saved predictor's recorded encoder cache_key does not
    match (or is missing for) the encoder currently in use -- i.e. the
    predictor was trained against a different (or unknown) encoder's latent
    space and must not be silently loaded."""


class _PredictorMLP(nn.Module):
    def __init__(self, dim: int, k: int = K_DEFAULT, hidden: int = HIDDEN_DEFAULT):
        super().__init__()
        self.dim = dim
        self.net = nn.Sequential(
            nn.Linear(k * dim, hidden), nn.ReLU(),
            nn.Linear(hidden, hidden), nn.ReLU(),
            nn.Linear(hidden, dim),
        )
        # Zero-init the final layer: at init, delta == 0, so forward()
        # reduces exactly to the last-latent baseline.
        nn.init.zeros_(self.net[-1].weight)
        nn.init.zeros_(self.net[-1].bias)

    def forward(self, context_flat: torch.Tensor) -> torch.Tensor:
        last = context_flat[:, -self.dim:]
        delta = self.net(context_flat)
        return last + delta


def _windows(latents: list[np.ndarray], k: int) -> tuple[torch.Tensor, torch.Tensor]:
    """Build teacher-forced (context, target) windows within each clip's
    latents (no cross-clip windows). context: (N, k*D), target: (N, D)."""
    contexts, targets = [], []
    for z in latents:
        z = z.astype(np.float32)
        t = z.shape[0]
        for i in range(t - k):
            contexts.append(z[i:i + k].reshape(-1))
            targets.append(z[i + k])
    if not contexts:
        raise ValueError("need at least k+1 frames in some clip to form one window")
    return (
        torch.from_numpy(np.stack(contexts)),
        torch.from_numpy(np.stack(targets)),
    )


class LatentPredictor:
    """A per-stack next-latent predictor. context is k consecutive latents
    (flattened); target is the latent that follows."""

    def __init__(self, dim: int, k: int = K_DEFAULT, hidden: int = HIDDEN_DEFAULT):
        self.dim = dim
        self.k = k
        self.hidden = hidden
        self.model = _PredictorMLP(dim, k, hidden)
        self.model.eval()

    def fit(
        self,
        train_latents: list[np.ndarray],
        val_latents: list[np.ndarray] | None = None,
        epochs: int = 15,
        batch_size: int = 256,
        lr: float = 1e-3,
        seed: int = 0,
    ) -> dict:
        """train_latents/val_latents: lists of (T, D) float32 arrays, one per
        clip. Reinitialises weights deterministically from `seed`. Returns
        {'train_losses': [...], 'val_losses': [...] or None}."""
        torch.manual_seed(seed)
        self.model = _PredictorMLP(self.dim, self.k, self.hidden)
        self.model.train()

        ctx, tgt = _windows(train_latents, self.k)
        n = ctx.shape[0]

        val_ctx = val_tgt = None
        if val_latents is not None:
            val_ctx, val_tgt = _windows(val_latents, self.k)

        opt = torch.optim.Adam(self.model.parameters(), lr=lr)
        loss_fn = nn.MSELoss()
        train_losses = []
        val_losses = [] if val_latents is not None else None

        # Best-val-loss checkpointing. The untrained (zero-init-delta) model
        # is included as the FIRST candidate: since forward() at init is
        # exactly the last-latent baseline, this guarantees fit() can never
        # return a model whose val MSE is worse than that baseline, even if
        # a few epochs of Adam noise nudge weights the wrong way on latents
        # that already barely change frame-to-frame.
        best_val = None
        best_state = None
        if val_latents is not None:
            self.model.eval()
            with torch.no_grad():
                best_val = loss_fn(self.model(val_ctx), val_tgt).item()
            best_state = copy.deepcopy(self.model.state_dict())
            self.model.train()

        for _epoch in range(epochs):
            perm = torch.randperm(n)
            total = 0.0
            n_batches = 0
            for start in range(0, n, batch_size):
                idx = perm[start:start + batch_size]
                c_batch, t_batch = ctx[idx], tgt[idx]
                opt.zero_grad()
                pred = self.model(c_batch)
                loss = loss_fn(pred, t_batch)
                loss.backward()
                opt.step()
                total += loss.item()
                n_batches += 1
            train_losses.append(total / max(n_batches, 1))

            if val_latents is not None:
                self.model.eval()
                with torch.no_grad():
                    val_loss = loss_fn(self.model(val_ctx), val_tgt).item()
                val_losses.append(val_loss)
                if val_loss < best_val:
                    best_val = val_loss
                    best_state = copy.deepcopy(self.model.state_dict())
                self.model.train()

        if best_state is not None:
            self.model.load_state_dict(best_state)
        self.model.eval()
        return {"train_losses": train_losses, "val_losses": val_losses}

    def rollout(self, context: np.ndarray, n: int) -> np.ndarray:
        """context: (k, D) float32 array of recent latents. Autoregressively
        predicts n future latents in closed loop (each prediction is fed
        back in as context for the next step). Returns (n, D) float32."""
        self.model.eval()
        ctx = context.astype(np.float32).copy()
        outs = []
        with torch.no_grad():
            for _ in range(n):
                flat = torch.from_numpy(ctx.reshape(1, -1))
                pred = self.model(flat).numpy()[0].astype(np.float32)
                outs.append(pred)
                ctx = np.vstack([ctx[1:], pred[None, :]])
        return np.stack(outs).astype(np.float32)


def _meta_path(weights_path: Path) -> Path:
    return weights_path.with_suffix(".meta.json")


def save_predictor(
    pred: LatentPredictor,
    weights_path: Path,
    encoder_cache_key: str,
    baseline_val_mse: float | None = None,
    final_val_mse: float | None = None,
) -> None:
    """Saves predictor weights plus a sidecar .meta.json recording the
    encoder cache_key it was trained against (so a later load can detect a
    stale predictor/encoder pairing -- see PredictorVersionMismatch) and,
    optionally, the predict-last-latent baseline val MSE it was trained
    against alongside its own final val MSE, so callers that only load
    (never retrain) can warn if a predictor doesn't beat that floor without
    needing to recompute anything."""
    weights_path = Path(weights_path)
    weights_path.parent.mkdir(parents=True, exist_ok=True)
    torch.save(pred.model.state_dict(), weights_path)
    meta = {"encoder_cache_key": encoder_cache_key}
    if baseline_val_mse is not None:
        meta["baseline_val_mse"] = baseline_val_mse
    if final_val_mse is not None:
        meta["final_val_mse"] = final_val_mse
    _meta_path(weights_path).write_text(json.dumps(meta), encoding="utf-8")


def load_predictor_meta(weights_path: Path) -> dict:
    """Reads the sidecar .meta.json for a saved predictor without loading
    the (possibly large) weight tensors. Raises PredictorVersionMismatch if
    the meta file is missing."""
    meta_path = _meta_path(Path(weights_path))
    if not meta_path.exists():
        raise PredictorVersionMismatch(f"no meta sidecar at {meta_path}")
    return json.loads(meta_path.read_text(encoding="utf-8"))


def load_predictor(
    weights_path: Path,
    dim: int,
    encoder_cache_key: str,
    k: int = K_DEFAULT,
    hidden: int = HIDDEN_DEFAULT,
) -> LatentPredictor:
    """Loads a predictor only if its sidecar meta records the SAME encoder
    cache_key as `encoder_cache_key`. Raises PredictorVersionMismatch if the
    meta file is missing or records a different cache_key -- this is the
    guard against silently pairing a predictor with a retrained encoder."""
    weights_path = Path(weights_path)
    meta_path = _meta_path(weights_path)
    if not meta_path.exists():
        raise PredictorVersionMismatch(
            f"no meta sidecar at {meta_path}; cannot verify which encoder "
            f"{weights_path} was trained against"
        )
    meta = json.loads(meta_path.read_text(encoding="utf-8"))
    saved_key = meta.get("encoder_cache_key")
    if saved_key != encoder_cache_key:
        raise PredictorVersionMismatch(
            f"predictor at {weights_path} was trained against encoder "
            f"cache_key {saved_key!r} but the current encoder's cache_key is "
            f"{encoder_cache_key!r} -- retrain the predictor (--force)"
        )
    pred = LatentPredictor(dim=dim, k=k, hidden=hidden)
    pred.model.load_state_dict(torch.load(weights_path, map_location="cpu"))
    pred.model.eval()
    return pred
