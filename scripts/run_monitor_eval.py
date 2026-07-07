"""Thin-slice runtime monitor evaluation (STRETCH): for every (stack, law),
builds a LawMonitor from the stack's frozen encoder + trained predictor +
concept direction (reused verbatim from probes.mechanistic.intervention),
calibrates its threshold on LAWFUL (obey) probe-TRAIN clips (seeds 300..331),
then evaluates on probe-TEST pairs (seeds 400..415):

  - clip-level detection AUROC: violate vs obey, score = max post-k frame score
  - false-positive rate on obey clips at the calibrated threshold
  - median detection latency = fire_frame - critical_frame, over violate
    clips that fire

Runs BOTH pinned lambda_s variants (0.0 = pure concept projection, 1.0 =
projection + residual-magnitude term) and reports both. Does not tune the
threshold percentile or lambda_s beyond these two pinned values.

Usage: uv run python scripts/run_monitor_eval.py
"""
import json
import statistics
import time
from pathlib import Path

import torch

from physics_auditor.laws import ALL_LAWS
from physics_auditor.models.encoders import RawPixelEncoder, TinyCNNAE, TinyCNNPred
from physics_auditor.models.predictor import load_predictor
from physics_auditor.monitor.detector import LawMonitor
from physics_auditor.probes.behavioural.metrics import auroc
from physics_auditor.probes.mechanistic.data import probe_pairs
from physics_auditor.probes.mechanistic.intervention import concept_direction

ROOT = Path(__file__).resolve().parent.parent
WEIGHTS_DIR = ROOT / "artifacts" / "weights"
CACHE_DIR = ROOT / "cache"
ARTIFACTS_DIR = ROOT / "artifacts"

AE_WEIGHTS = WEIGHTS_DIR / "tiny-cnn-ae.pt"
PRED_WEIGHTS = WEIGHTS_DIR / "tiny-cnn-pred.pt"
PREDICTOR_WEIGHTS = {
    "raw-pixel": WEIGHTS_DIR / "predictor_raw-pixel.pt",
    "tiny-cnn-ae": WEIGHTS_DIR / "predictor_tiny-cnn-ae.pt",
    "tiny-cnn-pred": WEIGHTS_DIR / "predictor_tiny-cnn-pred.pt",
}
STACK_NAMES = ["raw-pixel", "tiny-cnn-ae", "tiny-cnn-pred"]
LAMBDA_VARIANTS = [0.0, 1.0]


def _load_encoder(name: str):
    if name == "raw-pixel":
        return RawPixelEncoder()
    if name == "tiny-cnn-ae":
        enc = TinyCNNAE()
        enc.model.load_state_dict(torch.load(AE_WEIGHTS, map_location="cpu"))
        enc.model.eval()
        return enc
    if name == "tiny-cnn-pred":
        enc = TinyCNNPred()
        enc.model.load_state_dict(torch.load(PRED_WEIGHTS, map_location="cpu"))
        enc.model.eval()
        return enc
    raise ValueError(f"unknown stack {name}")


def _load_predictor_for(name: str, encoder):
    state = torch.load(PREDICTOR_WEIGHTS[name], map_location="cpu")
    dim = state["net.0.weight"].shape[1] // 4
    return load_predictor(PREDICTOR_WEIGHTS[name], dim=dim, encoder_cache_key=encoder.cache_key)


def evaluate_monitor(monitor: LawMonitor, train_pairs, test_pairs, cache_dir: str) -> dict:
    monitor.calibrate([p.obey for p in train_pairs], cache_dir=cache_dir)

    obey_scores = [monitor.clip_score_max(p.obey, cache_dir=cache_dir) for p in test_pairs]
    violate_scores = [monitor.clip_score_max(p.violate, cache_dir=cache_dir) for p in test_pairs]
    clip_auroc = auroc(pos=violate_scores, neg=obey_scores)

    obey_fires = [monitor.fire_frame(p.obey, cache_dir=cache_dir) is not None for p in test_pairs]
    fpr = sum(obey_fires) / len(obey_fires)

    latencies = []
    for p in test_pairs:
        fired = monitor.fire_frame(p.violate, cache_dir=cache_dir)
        if fired is not None:
            latencies.append(fired - p.critical_frame)
    median_latency = statistics.median(latencies) if latencies else None
    n_violate_fired = len(latencies)

    return {
        "threshold": monitor.threshold,
        "clip_auroc": clip_auroc,
        "obey_fpr": fpr,
        "n_violate_fired": n_violate_fired,
        "n_test_pairs": len(test_pairs),
        "median_latency": median_latency,
    }


def main() -> None:
    results: dict[str, dict] = {}
    timings: dict[str, float] = {}

    for stack_name in STACK_NAMES:
        encoder = _load_encoder(stack_name)
        predictor = _load_predictor_for(stack_name, encoder)

        for law_name in ALL_LAWS:
            train_pairs = probe_pairs(law_name, "train")
            test_pairs = probe_pairs(law_name, "test")

            direction, alpha = concept_direction(train_pairs, encoder, cache_dir=str(CACHE_DIR))

            for lambda_s in LAMBDA_VARIANTS:
                key = f"{stack_name}/{law_name}/lambda_s={lambda_s}"
                t0 = time.time()
                monitor = LawMonitor(
                    encoder=encoder, predictor=predictor, direction=direction,
                    law_name=law_name, lambda_s=lambda_s,
                )
                r = evaluate_monitor(monitor, train_pairs, test_pairs, cache_dir=str(CACHE_DIR))
                r["alpha"] = alpha
                elapsed = time.time() - t0
                r["elapsed_s"] = elapsed
                timings[key] = elapsed
                results[key] = r
                print(f"  {key}: AUROC={r['clip_auroc']:.3f} FPR={r['obey_fpr']:.3f} "
                      f"latency={r['median_latency']} ({elapsed:.1f}s)")

    # ---- markdown assembly -------------------------------------------------
    md_lines: list[str] = []
    md_lines.append("# Runtime Monitor Evaluation (thin slice, STRETCH)")
    md_lines.append("")
    md_lines.append(
        "LawMonitor: residual r_t = z_t - predictor.rollout(z[t-4:t], 1)[0]; "
        "score_t = |proj r_t onto unit concept direction d| + lambda_s * ||r_t||_2. "
        "Threshold = 99th percentile of pooled frame scores over LAWFUL (obey) "
        "probe-TRAIN clips (seeds 300..331); evaluated on probe-TEST pairs "
        "(seeds 400..415). Two pinned lambda_s variants reported, no tuning "
        "beyond these."
    )
    md_lines.append("")
    md_lines.append(
        "| stack | law | lambda_s | clip AUROC | obey FPR | n violate fired | "
        "median latency (frames) | alpha | threshold |"
    )
    md_lines.append("| --- | --- | --- | --- | --- | --- | --- | --- | --- |")
    for stack_name in STACK_NAMES:
        for law_name in ALL_LAWS:
            for lambda_s in LAMBDA_VARIANTS:
                key = f"{stack_name}/{law_name}/lambda_s={lambda_s}"
                r = results[key]
                lat = f"{r['median_latency']:.1f}" if r["median_latency"] is not None else "n/a"
                md_lines.append(
                    f"| {stack_name} | {law_name} | {lambda_s} | {r['clip_auroc']:.3f} | "
                    f"{r['obey_fpr']:.3f} | {r['n_violate_fired']}/{r['n_test_pairs']} | {lat} | "
                    f"{r['alpha']:.4f} | {r['threshold']:.4f} |"
                )
    md_lines.append("")
    md_lines.append("## Caveats")
    md_lines.append("")
    md_lines.append(
        "- `permanence` is expected to be at-or-near chance / structurally blind: "
        "its concept direction alpha collapses to ~0 in the intervention probe "
        "(rung-3 finding, report_card_v2.md), so the projection term of the "
        "monitor score carries no signal for this law regardless of stack -- this "
        "is reported as the honest negative it is, not tuned around."
    )
    md_lines.append(
        "- Threshold is calibrated ONLY on probe-TRAIN (300..331) obey clips; "
        "AUROC/FPR/latency are reported ONLY on probe-TEST (400..415) pairs, per "
        "the sacred seed-range split."
    )
    md_lines.append(
        "- lambda_s=0.0 (pure concept projection) and lambda_s=1.0 (projection + "
        "residual-magnitude term) are the two pinned variants; neither the "
        "percentile nor lambda_s was tuned beyond these fixed values."
    )
    markdown = "\n".join(md_lines)

    ARTIFACTS_DIR.mkdir(parents=True, exist_ok=True)
    (ARTIFACTS_DIR / "monitor_eval.md").write_text(markdown, encoding="utf-8")
    (ARTIFACTS_DIR / "monitor_scores.json").write_text(
        json.dumps(results, indent=2), encoding="utf-8"
    )

    print(markdown)
    print("\nTimings (s):")
    for key, t in timings.items():
        print(f"  {key}: {t:.2f}s")


if __name__ == "__main__":
    main()
