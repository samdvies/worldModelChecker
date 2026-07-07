"""End-to-end training + latent-cache population for the FLOOR encoder stacks.

Part 1 (this script, always run): train TinyCNNAE + TinyCNNPred on the
lawful train_clips(), save weights to artifacts/weights/, then populate the
on-disk latent cache for train/val/eval clips across all 3 stacks
(raw-pixel, tiny-cnn-ae, tiny-cnn-pred).

Part 2 (--predictors): train a LatentPredictor per stack (raw-pixel,
tiny-cnn-ae, tiny-cnn-pred) on CACHED train/val latents, teacher-forced
next-latent MSE, 15 epochs. Reports train/val loss per stack and the
predict-last-latent baseline (val MSE of just repeating z[t] as z[t+1]).

Usage: uv run python scripts/train_stacks.py [--force] [--predictors]
"""
import argparse
import time
from pathlib import Path

import numpy as np
import torch

from physics_auditor.generator.clip import Clip
from physics_auditor.generator.dataset import eval_pairs, train_clips, val_clips
from physics_auditor.laws import ALL_LAWS
from physics_auditor.models.cache import encode_clip_cached
from physics_auditor.models.encoders import RawPixelEncoder, TinyCNNAE, TinyCNNPred
from physics_auditor.models.predictor import (
    LatentPredictor,
    PredictorVersionMismatch,
    _windows,
    load_predictor,
    save_predictor,
)

ROOT = Path(__file__).resolve().parent.parent
WEIGHTS_DIR = ROOT / "artifacts" / "weights"
CACHE_DIR = ROOT / "cache"

AE_WEIGHTS = WEIGHTS_DIR / "tiny-cnn-ae.pt"
PRED_WEIGHTS = WEIGHTS_DIR / "tiny-cnn-pred.pt"
PREDICTOR_WEIGHTS = {
    "raw-pixel": WEIGHTS_DIR / "predictor_raw-pixel.pt",
    "tiny-cnn-ae": WEIGHTS_DIR / "predictor_tiny-cnn-ae.pt",
    "tiny-cnn-pred": WEIGHTS_DIR / "predictor_tiny-cnn-pred.pt",
}


def _all_frames(clips: list[Clip]) -> np.ndarray:
    """Concatenate every clip's frames into one (N, 64, 64, 3) float32 [0,1] array."""
    return np.concatenate([c.frames for c in clips], axis=0).astype(np.float32) / 255.0


def load_or_train_ae(clips: list[Clip], force: bool = False) -> tuple[TinyCNNAE, list[float] | None, float]:
    enc = TinyCNNAE()
    if AE_WEIGHTS.exists() and not force:
        enc.model.load_state_dict(torch.load(AE_WEIGHTS, map_location="cpu"))
        enc.model.eval()
        return enc, None, 0.0

    frames = _all_frames(clips)
    t0 = time.time()
    losses = enc.fit(frames, epochs=3, batch_size=128, lr=1e-3, seed=0)
    elapsed = time.time() - t0

    WEIGHTS_DIR.mkdir(parents=True, exist_ok=True)
    torch.save(enc.model.state_dict(), AE_WEIGHTS)
    return enc, losses, elapsed


def load_or_train_pred(clips: list[Clip], force: bool = False) -> tuple[TinyCNNPred, list[float] | None, float]:
    enc = TinyCNNPred()
    if PRED_WEIGHTS.exists() and not force:
        enc.model.load_state_dict(torch.load(PRED_WEIGHTS, map_location="cpu"))
        enc.model.eval()
        return enc, None, 0.0

    # Concatenate per-clip frames; the tiny fraction of 4-frame windows that
    # straddle a clip boundary is negligible noise at this scale (127 seams
    # out of ~6000 frames) -- documented simplification, not a correctness bug.
    frames = _all_frames(clips)
    t0 = time.time()
    losses = enc.fit(frames, epochs=3, batch_size=128, lr=1e-3, seed=0)
    elapsed = time.time() - t0

    WEIGHTS_DIR.mkdir(parents=True, exist_ok=True)
    torch.save(enc.model.state_dict(), PRED_WEIGHTS)
    return enc, losses, elapsed


def populate_caches(stacks: list, force: bool = False) -> dict[str, int]:
    """Populate the on-disk latent cache for train/val/eval clips across all
    given stacks. Returns {cache_key: n_files_written_or_present}."""
    counts: dict[str, int] = {}

    def _do(clip: Clip) -> None:
        for enc in stacks:
            encode_clip_cached(enc, clip, cache_dir=str(CACHE_DIR))

    for clip in train_clips():
        _do(clip)
    for clip in val_clips():
        _do(clip)
    for law_name in ALL_LAWS:
        for pair in eval_pairs(law_name):
            _do(pair.obey)
            _do(pair.violate)

    for enc in stacks:
        key_dir = CACHE_DIR / enc.cache_key
        counts[enc.cache_key] = len(list(key_dir.glob("*.npy"))) if key_dir.exists() else 0
    return counts


def _last_latent_baseline(latents: list[np.ndarray], k: int = 4) -> float:
    """MSE of predicting z[t+k] as z[t+k-1] (predict-last-latent), matching
    the same k+1 windowing LatentPredictor.fit uses -- the baseline a
    trained predictor must beat to be worth anything."""
    errs = []
    for z in latents:
        t = z.shape[0]
        for i in range(t - k):
            pred = z[i + k - 1]
            target = z[i + k]
            errs.append(np.mean((pred - target) ** 2))
    return float(np.mean(errs)) if errs else float("nan")


def train_predictors(stacks: list, force: bool = False) -> dict[str, dict]:
    """Train (or load) a LatentPredictor per given stack on cached train/val
    latents. Returns {stack_name: {'train_losses', 'val_losses',
    'baseline_val_mse', 'elapsed', 'trained': bool}}."""
    train = list(train_clips())
    val = list(val_clips())
    results: dict[str, dict] = {}

    for enc in stacks:
        train_latents = [encode_clip_cached(enc, c, cache_dir=str(CACHE_DIR)) for c in train]
        val_latents = [encode_clip_cached(enc, c, cache_dir=str(CACHE_DIR)) for c in val]
        dim = train_latents[0].shape[1]
        baseline = _last_latent_baseline(val_latents)

        weights_path = PREDICTOR_WEIGHTS[enc.name]

        if weights_path.exists() and not force:
            try:
                pred = load_predictor(weights_path, dim=dim, encoder_cache_key=enc.cache_key)
                val_ctx, val_tgt = _windows(val_latents, pred.k)
                with torch.no_grad():
                    final_val_mse = torch.nn.functional.mse_loss(pred.model(val_ctx), val_tgt).item()
                results[enc.name] = {
                    "train_losses": None, "val_losses": None,
                    "baseline_val_mse": baseline, "final_val_mse": final_val_mse,
                    "elapsed": 0.0, "trained": False,
                }
                continue
            except PredictorVersionMismatch as exc:
                # Stale predictor (trained against a since-retrained encoder,
                # or a pre-versioning weights file with no meta sidecar) --
                # fall through and retrain rather than silently pairing it
                # with the current encoder.
                print(f"  {enc.name}: {exc} -- retraining")

        pred = LatentPredictor(dim=dim)
        t0 = time.time()
        out = pred.fit(train_latents, val_latents=val_latents, epochs=15, batch_size=256, lr=1e-3, seed=0)
        elapsed = time.time() - t0

        # fit() checkpoints on best val loss (see predictor.py), so the
        # returned model may not be the LAST epoch in val_losses -- recompute
        # the actual final model's val MSE for an honest beats-baseline check.
        val_ctx, val_tgt = _windows(val_latents, pred.k)
        with torch.no_grad():
            final_val_mse = torch.nn.functional.mse_loss(pred.model(val_ctx), val_tgt).item()
        save_predictor(
            pred, weights_path, encoder_cache_key=enc.cache_key,
            baseline_val_mse=baseline, final_val_mse=final_val_mse,
        )
        results[enc.name] = {
            "train_losses": out["train_losses"], "val_losses": out["val_losses"],
            "baseline_val_mse": baseline, "final_val_mse": final_val_mse,
            "elapsed": elapsed, "trained": True,
        }
    return results


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--force", action="store_true", help="retrain even if weights already exist")
    parser.add_argument("--predictors", action="store_true", help="also train a LatentPredictor per stack")
    args = parser.parse_args()

    print("Building lawful train clips (seeds 100-132, all 4 laws)...")
    train = list(train_clips())
    print(f"  {len(train)} clips, {sum(c.frames.shape[0] for c in train)} frames")

    print("\nTraining TinyCNNAE (reconstructive)...")
    ae, ae_losses, ae_time = load_or_train_ae(train, force=args.force)
    if ae_losses is None:
        print(f"  weights already present at {AE_WEIGHTS}, skipped (use --force to retrain)")
    else:
        print(f"  epoch losses: {[f'{l:.5f}' for l in ae_losses]}  ({ae_time:.1f}s)")
    print(f"  cache_key: {ae.cache_key}")

    print("\nTraining TinyCNNPred (predictive)...")
    pred, pred_losses, pred_time = load_or_train_pred(train, force=args.force)
    if pred_losses is None:
        print(f"  weights already present at {PRED_WEIGHTS}, skipped (use --force to retrain)")
    else:
        print(f"  epoch losses: {[f'{l:.5f}' for l in pred_losses]}  ({pred_time:.1f}s)")
    print(f"  cache_key: {pred.cache_key}")

    stacks = [RawPixelEncoder(), ae, pred]

    print("\nPopulating latent caches (train + val + eval clips x 3 stacks)...")
    t0 = time.time()
    counts = populate_caches(stacks)
    print(f"  done in {time.time() - t0:.1f}s")
    for key, n in counts.items():
        print(f"  {key}: {n} cached files")

    if args.predictors:
        print("\nTraining LatentPredictors (raw-pixel, tiny-cnn-ae, tiny-cnn-pred)...")
        results = train_predictors(stacks, force=args.force)
        for name, r in results.items():
            # final_val_mse is the ACTUAL returned model's val MSE (fit()
            # checkpoints on best val loss, so this may differ from
            # val_losses[-1]); that's what must beat the baseline.
            beats = r["final_val_mse"] <= r["baseline_val_mse"] * (1 + 1e-6)
            if not r["trained"]:
                print(f"  {name}: weights already present, skipped (use --force to retrain)")
                print(f"    final val MSE: {r['final_val_mse']:.6f}  vs "
                      f"predict-last-latent baseline {r['baseline_val_mse']:.6f}  "
                      f"({'BEATS/TIES' if beats else 'DOES NOT BEAT'} baseline)")
                continue
            print(f"  {name}: ({r['elapsed']:.1f}s)")
            print(f"    train losses: {[f'{l:.5f}' for l in r['train_losses']]}")
            print(f"    val losses:   {[f'{l:.5f}' for l in r['val_losses']]}")
            print(f"    final (best-checkpoint) val MSE: {r['final_val_mse']:.6f}  vs "
                  f"predict-last-latent baseline {r['baseline_val_mse']:.6f}  "
                  f"({'BEATS/TIES' if beats else 'DOES NOT BEAT'} baseline)")
            if not beats:
                print(f"    WARNING: {name} predictor does not beat the trivial baseline -- "
                      f"investigate before trusting its rollout-based surprise scores.")


if __name__ == "__main__":
    main()
