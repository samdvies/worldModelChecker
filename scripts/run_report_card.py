"""Builds the report card v1: for every law x {oracle, pixel-static,
pixel-linear, raw-pixel, tiny-cnn-ae, tiny-cnn-pred}, runs behavioural VoE
on the 16 eval pairs and writes artifacts/report_card.md +
artifacts/voe_scores.json (per-pair obey/violate scores).

Also adds a delta_vs_pixel_floor column for the 3 learned encoder stacks,
where pixel_floor(law) = max(auroc(pixel-static), auroc(pixel-linear)).

Usage: uv run python scripts/run_report_card.py
"""
import argparse
import json
import time
from pathlib import Path

import torch

from physics_auditor.generator.dataset import eval_pairs
from physics_auditor.laws import ALL_LAWS
from physics_auditor.models.cache import encode_clip_cached
from physics_auditor.models.encoders import RawPixelEncoder, TinyCNNAE, TinyCNNPred
from physics_auditor.models.oracle import OracleAdapter
from physics_auditor.models.predictor import load_predictor, load_predictor_meta
from physics_auditor.models.pretrained import resolve_cache_only
from physics_auditor.models.stacks import LatentStackAdapter
from physics_auditor.probes.behavioural.pixel_baseline import PixelLinearAdapter, PixelStaticAdapter
from physics_auditor.probes.behavioural.voe import run_voe
from physics_auditor.report.card import ReportCard

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
DEFAULT_STACKS = ["raw-pixel", "tiny-cnn-ae", "tiny-cnn-pred"]
PRETRAINED_STACKS = ["dinov2-s14", "vjepa2-vitl"]
ALL_STACK_NAMES = DEFAULT_STACKS + PRETRAINED_STACKS
LATENT_STACKS = DEFAULT_STACKS  # default kept for backward-compat callers


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
    if name == "dinov2-s14":
        return resolve_cache_only(name)
    if name == "vjepa2-vitl":
        return resolve_cache_only(name)
    raise ValueError(f"unknown stack {name}")


def build_adapters(stacks: list[str] = DEFAULT_STACKS) -> list:
    """Builds oracle + 2 pixel baselines + one LatentStackAdapter (frozen
    encoder + trained predictor loaded from disk) per requested stack name."""
    adapters: list = [OracleAdapter(), PixelStaticAdapter(), PixelLinearAdapter()]

    for name in stacks:
        encoder = _load_encoder(name)
        # Predictor input is k*dim wide (net.0.weight); dividing out k=4 avoids
        # an extra (expensive) encode() call just to learn the latent dim.
        state = torch.load(PREDICTOR_WEIGHTS[name], map_location="cpu")
        dim = state["net.0.weight"].shape[1] // 4
        # load_predictor asserts the on-disk predictor's recorded encoder
        # cache_key matches this encoder's current cache_key -- refuses to
        # silently pair a predictor with a since-retrained encoder.
        predictor = load_predictor(PREDICTOR_WEIGHTS[name], dim=dim, encoder_cache_key=encoder.cache_key)

        meta = load_predictor_meta(PREDICTOR_WEIGHTS[name])
        baseline_mse = meta.get("baseline_val_mse")
        final_mse = meta.get("final_val_mse")
        if baseline_mse is not None and final_mse is not None and final_mse > baseline_mse * (1 + 1e-6):
            print(f"WARNING: predictor '{name}' does not beat the trivial predict-last-latent "
                  f"baseline (final val MSE {final_mse:.6g} > baseline {baseline_mse:.6g}) -- "
                  f"its rollout-based surprise scores may not be trustworthy.")

        adapters.append(LatentStackAdapter(name, encoder, predictor))

    return adapters


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--stacks", type=str, default=None,
        help=f"comma list of latent stacks to score, from {ALL_STACK_NAMES}. "
             f"Default: {DEFAULT_STACKS} (unchanged legacy behaviour).",
    )
    args = parser.parse_args()
    stacks = [s.strip() for s in args.stacks.split(",")] if args.stacks else DEFAULT_STACKS
    unknown = set(stacks) - set(ALL_STACK_NAMES)
    if unknown:
        raise ValueError(f"unknown stack(s) {sorted(unknown)}, choose from {ALL_STACK_NAMES}")

    adapters = build_adapters(stacks)
    card = ReportCard()
    voe_scores: dict = {}
    timings: dict[str, float] = {}

    for law_name in ALL_LAWS:
        pairs = eval_pairs(law_name)
        law_scores: dict[str, dict] = {}
        auroc_by_model: dict[str, float] = {}

        for adapter in adapters:
            t0 = time.time()
            result = run_voe(adapter, pairs, law_name)
            elapsed = time.time() - t0
            timings[f"{law_name}/{adapter.name}"] = elapsed

            card.add_row(adapter.name, law_name, "behavioural", "auroc", result.auroc, result.n_pairs)
            auroc_by_model[adapter.name] = result.auroc
            law_scores[adapter.name] = {
                "auroc": result.auroc,
                "obey_scores": result.obey_scores,
                "violate_scores": result.violate_scores,
                "distinct_scores": result.distinct_scores,
            }
            # A collapsed score distribution (e.g. a pixel-MSE baseline that
            # is translation-invariant along the only randomised dimension)
            # means the reported n_pairs massively overstates the effective
            # sample size behind this AUROC -- flag it loudly rather than
            # let a "n=16" cell imply more statistical weight than it has.
            if result.distinct_scores < result.n_pairs:
                print(f"CAVEAT: {law_name}/{adapter.name} AUROC is based on only "
                      f"{result.distinct_scores} distinct score value(s) across "
                      f"{result.n_pairs} eval pairs -- effective sample size is much "
                      f"smaller than n_pairs suggests.")

        pixel_floor = max(auroc_by_model["pixel-static"], auroc_by_model["pixel-linear"])
        for name in stacks:
            law_scores[name]["delta_vs_pixel_floor"] = auroc_by_model[name] - pixel_floor
        law_scores["pixel_floor"] = pixel_floor

        voe_scores[law_name] = law_scores

        if auroc_by_model["oracle"] < 0.95:
            print(f"WARNING: oracle AUROC {auroc_by_model['oracle']:.3f} < 0.95 on law '{law_name}' "
                  f"-- validity gate failed, investigate upstream.")

    md_lines = [card.to_markdown(), "", "## delta_vs_pixel_floor (learned stacks only)", ""]
    md_lines.append("| law | model | auroc | pixel_floor | delta_vs_pixel_floor |")
    md_lines.append("| --- | --- | --- | --- | --- |")
    for law_name in ALL_LAWS:
        pf = voe_scores[law_name]["pixel_floor"]
        for name in stacks:
            entry = voe_scores[law_name][name]
            md_lines.append(
                f"| {law_name} | {name} | {entry['auroc']:.3f} | {pf:.3f} | {entry['delta_vs_pixel_floor']:+.3f} |"
            )

    all_model_names = [a.name for a in adapters]
    caveat_rows = [
        (law_name, name, voe_scores[law_name][name]["distinct_scores"])
        for law_name in ALL_LAWS
        for name in all_model_names
        if voe_scores[law_name][name]["distinct_scores"] < 16
    ]
    md_lines += ["", "## Caveats: degenerate score distributions", "",
                 "Cells below have fewer distinct score values than eval pairs (n=16) -- "
                 "their effective sample size is smaller than n_pairs implies; treat their "
                 "AUROC as a much weaker signal than a naive n=16 read would suggest.", ""]
    if caveat_rows:
        md_lines.append("| law | model | distinct_scores (of 32 obey+violate) |")
        md_lines.append("| --- | --- | --- |")
        for law_name, name, distinct in caveat_rows:
            md_lines.append(f"| {law_name} | {name} | {distinct} |")
    else:
        md_lines.append("(none)")
    markdown = "\n".join(md_lines)

    artifacts_dir = ROOT / "artifacts"
    artifacts_dir.mkdir(parents=True, exist_ok=True)
    (artifacts_dir / "report_card.md").write_text(markdown, encoding="utf-8")
    (artifacts_dir / "voe_scores.json").write_text(json.dumps(voe_scores, indent=2), encoding="utf-8")

    print(markdown)
    print("\nTimings (s):")
    for key, t in timings.items():
        print(f"  {key}: {t:.2f}s")


if __name__ == "__main__":
    main()
