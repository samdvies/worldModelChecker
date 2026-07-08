"""Static minimal-pair gallery data export (ladder step 2, docs/gallery-design.md
phase 1). For ONE showcase pair per law (seed=0, outside every sacred probe
seed range), exports:

  - obey + violate frames as base64 PNGs, upscaled 3x nearest-neighbour
    (64x64 -> 192x192) so pixel blocks stay visible.
  - per-frame one-step surprise traces for oracle, pixel-static,
    pixel-linear, and the 3 trained local stacks (raw-pixel, tiny-cnn-ae,
    tiny-cnn-pred), each normalised by its own obey-clip max.
  - per-stack monitor fire_frame on the showcase obey/violate clips, using
    the SAME calibration recipe as scripts/run_monitor_eval.py (concept
    direction + threshold from lawful probe-TRAIN clips, seeds 300..331,
    lambda_s=0.0 pinned variant) -- the showcase pair itself (seed 0) is
    never used for calibration.
  - metadata: critical_frame, per-law behavioural AUROC + pixel floor +
    n_pairs + distinct_scores (from artifacts/voe_scores.json) and the
    genuine/shortcut-suspect/shortcut/not-sensitive verdict per stack (from
    artifacts/mechanistic_scores.json) -- the SAME honesty-constraint
    caveats the report card carries, not re-derived numbers.

Output format: docs/gallery/gallery_data.js, a single `const GALLERY_DATA =
{...};` statement (NOT gallery_data.json). This is deliberate: the phase-2
HTML gallery must be a single self-contained file openable via file:// with
no server -- a <script src="gallery_data.js"> same-directory include works
under file://, while `fetch('gallery_data.json')` is blocked by Chrome/most
browsers' CORS policy for local files (fetch of file:// XHRs is disallowed
cross-"origin" even same-directory). A .js const sidesteps that entirely.

Usage: uv run python scripts/build_gallery_data.py
"""
import json
from pathlib import Path

from physics_auditor.gallery.traces import (
    clip_frames_base64,
    latent_stack_one_step_trace,
    normalize_trace_pair,
    oracle_one_step_trace,
    pixel_linear_one_step_trace,
    pixel_static_one_step_trace,
)
from physics_auditor.laws import ALL_LAWS
from physics_auditor.models.cache import encode_clip_cached
from physics_auditor.monitor.detector import LawMonitor
from physics_auditor.probes.mechanistic.data import probe_pairs
from physics_auditor.probes.mechanistic.intervention import concept_direction

ROOT = Path(__file__).resolve().parent.parent
ARTIFACTS_DIR = ROOT / "artifacts"
CACHE_DIR = ROOT / "cache"
OUT_DIR = ROOT / "docs" / "gallery"
OUT_PATH = OUT_DIR / "gallery_data.js"

SHOWCASE_SEED = 0  # outside every sacred probe-train (300..331) / probe-test (400..415) range
STACK_NAMES = ["raw-pixel", "tiny-cnn-ae", "tiny-cnn-pred", "dinov2-s14", "vjepa2-vitl"]
FRAME_SCALE = 3  # 64x64 -> 192x192


def _load_stack_encoder_and_predictor(name: str):
    # local import: scripts/run_monitor_eval.py's loaders are the single
    # source of truth for how each stack's encoder + predictor are wired
    # together (weights paths, cache_key checks); reuse verbatim rather
    # than re-deriving.
    from scripts.run_monitor_eval import _load_encoder, _load_predictor_for

    encoder = _load_encoder(name)
    predictor = _load_predictor_for(name, encoder)
    return encoder, predictor


def _trace_entry(obey_trace, violate_trace) -> dict:
    norm_obey, norm_violate, normalizer = normalize_trace_pair(obey_trace, violate_trace)
    return {
        "obey": [round(float(x), 6) for x in norm_obey],
        "violate": [round(float(x), 6) for x in norm_violate],
        "normalizer": normalizer,
    }


def build_law_entry(law_name: str, voe_scores: dict, mechanistic_scores: dict) -> dict:
    law_cls = ALL_LAWS[law_name]
    pair = law_cls().generate_pair(seed=SHOWCASE_SEED)

    entry: dict = {
        "law": law_name,
        "critical_frame": pair.critical_frame,
        "n_frames": pair.obey.frames.shape[0],
        "obey_frames": clip_frames_base64(pair.obey, scale=FRAME_SCALE),
        "violate_frames": clip_frames_base64(pair.violate, scale=FRAME_SCALE),
        "traces": {},
        "monitor": {},
    }

    # oracle + pixel baselines
    entry["traces"]["oracle"] = _trace_entry(
        oracle_one_step_trace(pair.obey), oracle_one_step_trace(pair.violate)
    )
    entry["traces"]["pixel-static"] = _trace_entry(
        pixel_static_one_step_trace(pair.obey), pixel_static_one_step_trace(pair.violate)
    )
    entry["traces"]["pixel-linear"] = _trace_entry(
        pixel_linear_one_step_trace(pair.obey), pixel_linear_one_step_trace(pair.violate)
    )

    # local stacks: per-frame teacher-forced trace + monitor fire_frame
    train_pairs = probe_pairs(law_name, "train")
    for stack_name in STACK_NAMES:
        encoder, predictor = _load_stack_encoder_and_predictor(stack_name)

        z_obey = encode_clip_cached(encoder, pair.obey, cache_dir=str(CACHE_DIR))
        z_violate = encode_clip_cached(encoder, pair.violate, cache_dir=str(CACHE_DIR))
        entry["traces"][stack_name] = _trace_entry(
            latent_stack_one_step_trace(z_obey, predictor, k=4),
            latent_stack_one_step_trace(z_violate, predictor, k=4),
        )

        direction, _alpha = concept_direction(train_pairs, encoder, cache_dir=str(CACHE_DIR))
        monitor = LawMonitor(
            encoder=encoder, predictor=predictor, direction=direction,
            law_name=law_name, lambda_s=0.0,
        )
        monitor.calibrate([p.obey for p in train_pairs], cache_dir=str(CACHE_DIR))
        entry["monitor"][stack_name] = {
            "fire_frame_obey": monitor.fire_frame(pair.obey, cache_dir=str(CACHE_DIR)),
            "fire_frame_violate": monitor.fire_frame(pair.violate, cache_dir=str(CACHE_DIR)),
            "threshold": monitor.threshold,
        }

    # metadata + caveats copied verbatim from artifacts JSONs (honesty
    # constraint, docs/gallery-design.md) -- never re-derived here.
    law_voe = voe_scores[law_name]
    entry["metadata"] = {
        "pixel_floor": law_voe["pixel_floor"],
        "auroc": {
            model: {
                "auroc": law_voe[model]["auroc"],
                "n_pairs": len(law_voe[model]["obey_scores"]),
                "distinct_scores": law_voe[model]["distinct_scores"],
                "degenerate": law_voe[model]["distinct_scores"] < len(law_voe[model]["obey_scores"]),
            }
            for model in ("oracle", "pixel-static", "pixel-linear", *STACK_NAMES)
        },
        "verdict": {
            row["stack"]: row["verdict"]
            for row in mechanistic_scores["verdicts"]
            if row["law"] == law_name
        },
        "degenerate_cells": [
            cell for cell in mechanistic_scores["degenerate_cells"] if f"/{law_name}:" in cell
        ],
    }
    return entry


def main() -> None:
    voe_scores = json.loads((ARTIFACTS_DIR / "voe_scores.json").read_text(encoding="utf-8"))
    mechanistic_scores = json.loads((ARTIFACTS_DIR / "mechanistic_scores.json").read_text(encoding="utf-8"))

    data = {
        "showcase_seed": SHOWCASE_SEED,
        "frame_scale": FRAME_SCALE,
        "stack_names": STACK_NAMES,
        "laws": {
            law_name: build_law_entry(law_name, voe_scores, mechanistic_scores)
            for law_name in ALL_LAWS
        },
        "caveats": [
            "Same honesty constraint as artifacts/report_card_v2.md: these are "
            "n=16-eval-pair AUROCs (n_pairs shown per cell); degenerate cells "
            "(distinct_scores < n_pairs) are flagged, not hidden.",
            "The showcase pair per law (seed=0) is illustrative only -- it is "
            "outside the sacred probe-train (300..331) / probe-test (400..415) "
            "seed ranges and contributes to no reported metric.",
            "permanence's monitor projection term carries no signal for the "
            "per-frame stacks (concept-direction alpha collapses to ~0, "
            "report_card_v2.md rung-3 finding) -- an honest null, not a bug. "
            "vjepa2-vitl is the exception: as a clip-level video encoder it is "
            "the only stack genuinely sensitive to permanence.",
            "vjepa2-vitl traces are non-causal: V-JEPA-2 attends over the whole "
            "clip, so post-violation frames perturb latents of earlier frames "
            "too (the obey/violate inputs are byte-identical before the critical "
            "frame). Its surprise may rise BEFORE the marker line -- future-frame "
            "leakage, not early detection. Per-frame stacks' traces are causal.",
        ],
    }

    OUT_DIR.mkdir(parents=True, exist_ok=True)
    js = "const GALLERY_DATA = " + json.dumps(data, separators=(",", ":")) + ";\n"
    OUT_PATH.write_text(js, encoding="utf-8")
    size_mb = OUT_PATH.stat().st_size / (1024 * 1024)
    print(f"wrote {OUT_PATH} ({size_mb:.2f} MB)")


if __name__ == "__main__":
    main()
