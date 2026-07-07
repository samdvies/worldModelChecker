"""Builds the mechanistic rung of the report card (spec 5.2/5.3): for every
stack x law, runs decodability (ridge read-out probes) and intervention
(concept-patch vs placebo, paired bootstrap) on the sacred probe-train
(300..331) / probe-test (400..415) seed ranges, derives a genuine /
shortcut-suspect / shortcut / not-sensitive verdict from the behavioural v1
data (artifacts/voe_scores.json) plus the two mechanistic probes, and merges
everything into artifacts/report_card_v2.md + artifacts/mechanistic_scores.json.

Usage: uv run python scripts/run_mechanistic.py
"""
import argparse
import json
import time
from pathlib import Path

import torch

from physics_auditor.laws import ALL_LAWS
from physics_auditor.models.encoders import RawPixelEncoder, TinyCNNAE, TinyCNNPred
from physics_auditor.models.predictor import load_predictor
from physics_auditor.models.pretrained import DINOv2Encoder, VJEPA2Encoder
from physics_auditor.probes.mechanistic.decodability import PRIMARY_VARIABLE, decode_law_stack
from physics_auditor.probes.mechanistic.intervention import run_intervention_law_stack
from physics_auditor.report.verdict import is_decodable, is_voe_sensitive, verdict

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
    "dinov2-s14": WEIGHTS_DIR / "predictor_dinov2-s14.pt",
    "vjepa2-vitl": WEIGHTS_DIR / "predictor_vjepa2-vitl.pt",
}
DEFAULT_STACKS = ["raw-pixel", "tiny-cnn-ae", "tiny-cnn-pred"]
PRETRAINED_STACKS = ["dinov2-s14", "vjepa2-vitl"]
ALL_STACK_NAMES = DEFAULT_STACKS + PRETRAINED_STACKS
STACK_NAMES = DEFAULT_STACKS  # default kept for backward-compat callers


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
        return DINOv2Encoder().load()
    if name == "vjepa2-vitl":
        return VJEPA2Encoder().load()
    raise ValueError(f"unknown stack {name}")


def _load_predictor_for(name: str, encoder):
    state = torch.load(PREDICTOR_WEIGHTS[name], map_location="cpu")
    dim = state["net.0.weight"].shape[1] // 4
    return load_predictor(PREDICTOR_WEIGHTS[name], dim=dim, encoder_cache_key=encoder.cache_key)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--stacks", type=str, default=None,
        help=f"comma list of latent stacks to run mechanistic probes on, from "
             f"{ALL_STACK_NAMES}. Default: {DEFAULT_STACKS} (unchanged legacy "
             f"behaviour). voe_scores.json (from run_report_card.py) must "
             f"already contain an entry for every requested stack.",
    )
    args = parser.parse_args()
    stack_names = [s.strip() for s in args.stacks.split(",")] if args.stacks else DEFAULT_STACKS
    unknown = set(stack_names) - set(ALL_STACK_NAMES)
    if unknown:
        raise ValueError(f"unknown stack(s) {sorted(unknown)}, choose from {ALL_STACK_NAMES}")

    voe_scores = json.loads((ARTIFACTS_DIR / "voe_scores.json").read_text(encoding="utf-8"))

    decode_results: dict[str, dict] = {}
    intervention_results: dict[str, dict] = {}
    verdict_rows: list[tuple[str, str, bool, bool, bool, str]] = []
    timings: dict[str, float] = {}
    degenerate_cells: list[str] = []  # "stack/law: decode|intervention" for the caveats section

    for stack_name in stack_names:
        encoder = _load_encoder(stack_name)
        predictor = _load_predictor_for(stack_name, encoder)

        for law_name in ALL_LAWS:
            key = f"{stack_name}/{law_name}"

            t0 = time.time()
            decode_result = decode_law_stack(law_name, encoder, cache_dir=str(CACHE_DIR))
            decode_elapsed = time.time() - t0
            decode_results[key] = {
                "metrics": decode_result.metrics,
                "degenerate": decode_result.degenerate,
                "elapsed_s": decode_elapsed,
            }

            t0 = time.time()
            intervention_result = run_intervention_law_stack(
                law_name, encoder, predictor, seed=0, cache_dir=str(CACHE_DIR)
            )
            intervention_elapsed = time.time() - t0
            intervention_results[key] = {
                "effect_concept_mean": float(sum(intervention_result.effect_concept) / len(intervention_result.effect_concept)),
                "effect_placebo_mean": float(sum(intervention_result.effect_placebo) / len(intervention_result.effect_placebo)),
                "delta_mean": intervention_result.delta_mean,
                "ci_lo": intervention_result.ci_lo,
                "ci_hi": intervention_result.ci_hi,
                "n_positive": intervention_result.n_positive,
                "n_pairs": len(intervention_result.effect_concept),
                "causal_positive": intervention_result.causal_positive,
                "alpha": intervention_result.alpha,
                "degenerate": intervention_result.degenerate,
                "elapsed_s": intervention_elapsed,
            }
            timings[f"decode/{key}"] = decode_elapsed
            timings[f"intervention/{key}"] = intervention_elapsed
            print(f"  {key}: decode {decode_elapsed:.1f}s, intervention {intervention_elapsed:.1f}s")

            degenerate_parts = []
            if decode_result.degenerate:
                degenerate_parts.append("decode")
            if intervention_result.degenerate:
                degenerate_parts.append("intervention")
            if degenerate_parts:
                degenerate_cells.append(f"{key}: {'+'.join(degenerate_parts)}")

            primary_var = PRIMARY_VARIABLE[law_name]
            primary_metric = decode_result.metrics[primary_var]
            decodable = is_decodable(primary_metric)

            delta_vs_pixel_floor = voe_scores[law_name][stack_name]["delta_vs_pixel_floor"]
            voe_sensitive = is_voe_sensitive(delta_vs_pixel_floor)

            causally_used = intervention_result.causal_positive

            v = verdict(voe_sensitive, decodable, causally_used)
            verdict_rows.append((stack_name, law_name, voe_sensitive, decodable, causally_used, v))

    # ---- markdown assembly -------------------------------------------------
    md_lines: list[str] = []

    behavioural_md = (ROOT / "artifacts" / "report_card.md").read_text(encoding="utf-8")
    md_lines.append("# Report Card v2 (behavioural + mechanistic)")
    md_lines.append("")
    md_lines.append("## Behavioural (v1, carried over verbatim)")
    md_lines.append("")
    md_lines.append(behavioural_md)
    md_lines.append("")

    md_lines.append("## Decodability (ridge read-out, probe-train 300..331 -> probe-test 400..415)")
    md_lines.append("")
    md_lines.append(
        "Rows marked VACUOUS: the fixed decode window (cf-4..cf+7) is pixel-identical "
        "between obey and violate for every probe-test pair of this law, so the "
        "reported value is a majority-class/constant-target artifact, not a genuine "
        "read of (non-)decodability -- see Caveats."
    )
    md_lines.append("")
    md_lines.append("| stack | law | variable | kind | value | window |")
    md_lines.append("| --- | --- | --- | --- | --- | --- |")
    for stack_name in stack_names:
        for law_name in ALL_LAWS:
            key = f"{stack_name}/{law_name}"
            window_flag = "**VACUOUS**" if decode_results[key]["degenerate"] else "ok"
            for var_name, m in decode_results[key]["metrics"].items():
                is_primary = " (primary)" if var_name == PRIMARY_VARIABLE[law_name] else ""
                md_lines.append(
                    f"| {stack_name} | {law_name} | {var_name}{is_primary} | {m['kind']} | {m['value']:.3f} | {window_flag} |"
                )
    md_lines.append("")

    md_lines.append("## Intervention: concept-patch vs random-direction placebo (16 probe-test pairs, 10k-resample bootstrap CI)")
    md_lines.append("")
    md_lines.append(
        "Rows marked DEGENERATE: alpha (the concept-direction magnitude) collapsed to "
        "~0, or the obey/violate observed gap was ~0 for every probe-test pair inside "
        "the fixed HORIZON window -- the concept and placebo patches are then "
        "numerically indistinguishable and the tight [0,0]-looking CI is a vacuous "
        "'test did not fire', not an informative null. See Caveats."
    )
    md_lines.append("")
    md_lines.append("| stack | law | effect_concept | effect_placebo | delta | 95% CI | n_positive/16 | causal_positive | alpha | window |")
    md_lines.append("| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |")
    for stack_name in stack_names:
        for law_name in ALL_LAWS:
            key = f"{stack_name}/{law_name}"
            r = intervention_results[key]
            window_flag = "**DEGENERATE**" if r["degenerate"] else "ok"
            md_lines.append(
                f"| {stack_name} | {law_name} | {r['effect_concept_mean']:.3f} | {r['effect_placebo_mean']:.3f} | "
                f"{r['delta_mean']:+.3f} | [{r['ci_lo']:+.3f}, {r['ci_hi']:+.3f}] | {r['n_positive']}/{r['n_pairs']} | "
                f"{r['causal_positive']} | {r['alpha']:.4f} | {window_flag} |"
            )
    md_lines.append("")

    md_lines.append("## Verdict trichotomy (genuine / shortcut-suspect / shortcut / not-sensitive)")
    md_lines.append("")
    md_lines.append("| stack | law | voe_sensitive | decodable | causally_used | verdict |")
    md_lines.append("| --- | --- | --- | --- | --- | --- |")
    for stack_name, law_name, vs, dec, ca, v in verdict_rows:
        md_lines.append(f"| {stack_name} | {law_name} | {vs} | {dec} | {ca} | **{v}** |")
    md_lines.append("")

    md_lines.append("## Caveats")
    md_lines.append("")
    md_lines.append(
        "- Decodability and intervention probes use scene-disjoint seed ranges "
        "(probe-train 300..331, probe-test 400..415), strictly disjoint from the "
        "behavioural eval pairs (0..15) and predictor train/val pairs (100..131 / 200..207)."
    )
    md_lines.append(
        "- `causal_positive` requires the 95% bootstrap CI on (effect_concept - "
        "effect_placebo) to exclude zero AND the mean to be positive; a stack can be "
        "decodable (a linear probe recovers the variable) without being causally-used "
        "(the stack's own predictor doesn't act on that direction) -- this is by design "
        "an honest negative, not a bug, when it occurs."
    )
    md_lines.append(
        "- 'not-sensitive' always wins regardless of decodability/causal-use: a stack "
        "that isn't behaviourally surprised by the violation isn't worth mechanistically "
        "auditing further for that law."
    )
    md_lines.append(
        "- See report_card.md's own caveats section (carried over above) for degenerate "
        "behavioural score distributions on some law/model cells."
    )
    if degenerate_cells:
        md_lines.append(
            "- **Structurally vacuous mechanistic probes (test did not fire, not an "
            "informative null):** " + "; ".join(degenerate_cells) + ". This is expected "
            "for `permanence`: its occluder is deliberately wide enough that the "
            "deleted ball stays pixel-invisible for several frames past "
            "`critical_frame` (see `laws/permanence.py`), so the fixed decode window "
            "(cf-4..cf+7) and intervention horizon (cf..cf+7) never reach the frame "
            "where obey/violate actually diverge (empirically ~cf+9 for the raw-pixel "
            "stack). Read these cells' numbers as 'this test never had a chance to "
            "detect anything' rather than a well-powered negative; they are excluded "
            "from changing any verdict logic or thresholds -- fixing this properly "
            "would require a law-specific longer window/horizon, which is out of scope "
            "here so as not to tune the probe until a result appears."
        )
    markdown = "\n".join(md_lines)

    ARTIFACTS_DIR.mkdir(parents=True, exist_ok=True)
    (ARTIFACTS_DIR / "report_card_v2.md").write_text(markdown, encoding="utf-8")

    mechanistic_scores = {
        "decodability": decode_results,
        "intervention": intervention_results,
        "verdicts": [
            {
                "stack": stack_name, "law": law_name,
                "voe_sensitive": vs, "decodable": dec, "causally_used": ca,
                "verdict": v,
            }
            for stack_name, law_name, vs, dec, ca, v in verdict_rows
        ],
        "degenerate_cells": degenerate_cells,
    }
    (ARTIFACTS_DIR / "mechanistic_scores.json").write_text(
        json.dumps(mechanistic_scores, indent=2), encoding="utf-8"
    )

    print(markdown)
    print("\nTimings (s):")
    for key, t in timings.items():
        print(f"  {key}: {t:.2f}s")


if __name__ == "__main__":
    main()
