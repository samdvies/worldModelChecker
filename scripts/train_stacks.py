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
import sys
import time
from pathlib import Path

import numpy as np
import torch

from physics_auditor.generator.clip import Clip
from physics_auditor.generator.dataset import eval_pairs, train_clips, val_clips
from physics_auditor.laws import ALL_LAWS
from physics_auditor.models.cache import encode_clip_cached, encode_clips_cached
from physics_auditor.models.encoders import RawPixelEncoder, TinyCNNAE, TinyCNNPred
from physics_auditor.models.predictor import (
    LatentPredictor,
    PredictorVersionMismatch,
    _windows,
    load_predictor,
    save_predictor,
)
from physics_auditor.models.pretrained import DINOv2Encoder, VJEPA2Encoder
from physics_auditor.probes.mechanistic.data import probe_pairs

ROOT = Path(__file__).resolve().parent.parent
WEIGHTS_DIR = ROOT / "artifacts" / "weights"
CACHE_DIR = ROOT / "cache"

AE_WEIGHTS = WEIGHTS_DIR / "tiny-cnn-ae.pt"
PRED_WEIGHTS = WEIGHTS_DIR / "tiny-cnn-pred.pt"
PREDICTOR_WEIGHTS = {
    "raw-pixel": WEIGHTS_DIR / "predictor_raw-pixel.pt",
    "tiny-cnn-ae": WEIGHTS_DIR / "predictor_tiny-cnn-ae.pt",
    "tiny-cnn-pred": WEIGHTS_DIR / "predictor_tiny-cnn-pred.pt",
    "dinov2-s14": WEIGHTS_DIR / "predictor_dinov2-s14.pt",
    "vjepa2-vitl": WEIGHTS_DIR / "predictor_vjepa2-vitl.pt",
}
# The 3 from-scratch stacks are trained/populated by default (unchanged
# behaviour); the 2 pretrained stacks are opt-in via --stacks since their
# real weights only load on the GPU box (see models/pretrained.py).
DEFAULT_STACKS = ["raw-pixel", "tiny-cnn-ae", "tiny-cnn-pred"]
PRETRAINED_STACKS = ["dinov2-s14", "vjepa2-vitl"]
ALL_STACK_NAMES = DEFAULT_STACKS + PRETRAINED_STACKS


def plan_stacks(requested: list[str]) -> list[str]:
    """Pure, side-effect-free cache-warm plan: given the requested stack
    names (already validated against ALL_STACK_NAMES), returns exactly the
    stack names that will be built/encoded, in build order (local stacks
    first, then pretrained). This is the single source of truth `main()`
    consults for stack selection -- MUST exactly match `requested`, with no
    extra local-stack work slipping in when only pretrained stacks are
    requested (see docs/failure-sweeps.md class G: a prior GPU run invoked
    with `--stacks dinov2-s14,vjepa2-vitl` wasted ~2h re-encoding all 3
    local stacks anyway)."""
    plan = [name for name in DEFAULT_STACKS if name in requested]
    plan += [name for name in PRETRAINED_STACKS if name in requested]
    return plan


def _load_pretrained_encoder(name: str):
    """Loads a pretrained stack's REAL weights via its lazy .load() -- only
    safe to call on a machine with network access (GPU box)."""
    if name == "dinov2-s14":
        return DINOv2Encoder().load()
    if name == "vjepa2-vitl":
        return VJEPA2Encoder().load()
    raise ValueError(f"unknown pretrained stack {name}")


def probe_clips() -> list[Clip]:
    """All Clips backing the mechanistic-probe latent caches: for every law
    in ALL_LAWS, for both splits ("train": seeds 300..331, "test": seeds
    400..415), every pair's obey + violate clip. Pure fan-out, no encoding
    -- this is the single source of truth for what --probe-caches warms.
    Expected count: 4 laws x (32 + 16) seeds x 2 clips = 384 (see
    docs/failure-sweeps.md class I: this replaces run_report_card.py/
    run_mechanistic.py as the way remote_job.sh warms these caches, since
    both scripts crash on the GPU box with FileNotFoundError on predictor
    weights that are never trained there)."""
    clips: list[Clip] = []
    for law_name in ALL_LAWS:
        for split in ("train", "test"):
            for pair in probe_pairs(law_name, split):
                clips.append(pair.obey)
                clips.append(pair.violate)
    return clips


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


def populate_caches(stacks: list, force: bool = False, include_probe_caches: bool = False) -> dict[str, int]:
    """Populate the on-disk latent cache for train/val/eval clips (and, when
    include_probe_caches is set, the mechanistic-probe clips too -- see
    probe_clips()) across all given stacks. Returns {cache_key:
    n_files_written_or_present}. Default (include_probe_caches=False)
    behaviour is byte-identical to before this flag existed."""
    counts: dict[str, int] = {}

    all_clips: list[Clip] = []
    all_clips.extend(train_clips())
    all_clips.extend(val_clips())
    for law_name in ALL_LAWS:
        for pair in eval_pairs(law_name):
            all_clips.append(pair.obey)
            all_clips.append(pair.violate)
    if include_probe_caches:
        all_clips.extend(probe_clips())

    for enc in stacks:
        encode_clips_cached(enc, all_clips, cache_dir=str(CACHE_DIR))

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
    parser.add_argument(
        "--stacks", type=str, default=None,
        help=f"comma list of stacks to build, from {ALL_STACK_NAMES}. "
             f"Default: {DEFAULT_STACKS} (unchanged legacy behaviour). "
             f"Pass e.g. 'dinov2-s14,vjepa2-vitl' to add the pretrained "
             f"stacks WITHOUT retraining/recomputing the existing three.",
    )
    parser.add_argument(
        "--probe-caches", action="store_true",
        help="also warm the mechanistic-probe latent caches (probe_clips(): "
             "4 laws x train(300-331)+test(400-415) seeds x 2 clips = 384) "
             "through the same encode_clips_cached batched path, per "
             "selected stack. Absent: byte-identical legacy behaviour.",
    )
    args = parser.parse_args(sys.argv[1:])
    requested = [s.strip() for s in args.stacks.split(",")] if args.stacks else DEFAULT_STACKS
    unknown = set(requested) - set(ALL_STACK_NAMES)
    if unknown:
        raise ValueError(f"unknown stack(s) {sorted(unknown)}, choose from {ALL_STACK_NAMES}")

    print("Building lawful train clips (seeds 100-132, all 4 laws)...")
    train = list(train_clips())
    print(f"  {len(train)} clips, {sum(c.frames.shape[0] for c in train)} frames")

    plan = plan_stacks(requested)
    stacks = []

    if "raw-pixel" in plan:
        stacks.append(RawPixelEncoder())

    if "tiny-cnn-ae" in plan:
        print("\nTraining TinyCNNAE (reconstructive)...")
        ae, ae_losses, ae_time = load_or_train_ae(train, force=args.force)
        if ae_losses is None:
            print(f"  weights already present at {AE_WEIGHTS}, skipped (use --force to retrain)")
        else:
            print(f"  epoch losses: {[f'{l:.5f}' for l in ae_losses]}  ({ae_time:.1f}s)")
        print(f"  cache_key: {ae.cache_key}")
        stacks.append(ae)

    if "tiny-cnn-pred" in plan:
        print("\nTraining TinyCNNPred (predictive)...")
        pred, pred_losses, pred_time = load_or_train_pred(train, force=args.force)
        if pred_losses is None:
            print(f"  weights already present at {PRED_WEIGHTS}, skipped (use --force to retrain)")
        else:
            print(f"  epoch losses: {[f'{l:.5f}' for l in pred_losses]}  ({pred_time:.1f}s)")
        print(f"  cache_key: {pred.cache_key}")
        stacks.append(pred)

    for name in PRETRAINED_STACKS:
        if name in plan:
            print(f"\nLoading pretrained {name} (real weights, GPU box only)...")
            enc = _load_pretrained_encoder(name)
            print(f"  cache_key: {enc.cache_key}")
            stacks.append(enc)

    probe_suffix = " + probe clips" if args.probe_caches else ""
    print(f"\nPopulating latent caches (train + val + eval clips{probe_suffix} x {len(stacks)} stack(s))...")
    t0 = time.time()
    counts = populate_caches(stacks, include_probe_caches=args.probe_caches)
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
