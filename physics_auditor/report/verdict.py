"""Per-(stack, law) understand-vs-shortcut verdict (spec 5.3): combines
three independent booleans -- behavioural VoE-sensitivity, decodability
(linear read-out of the ground-truth variable), and causal-use (positive,
CI-excludes-zero intervention effect over a random-direction placebo) --
into a trichotomy label.

Gate thresholds:
  voe_sensitive:  auroc - pixel_floor >= 0.2
  decodable:      acc >= 0.9 (bool primary variable) or R2 >= 0.5 (regression)
  causally_used:  InterventionResult.causal_positive (CI excludes 0 and mean > 0)
"""

VOE_DELTA_THRESHOLD = 0.2
ACC_THRESHOLD = 0.9
R2_THRESHOLD = 0.5


def is_voe_sensitive(delta_vs_pixel_floor: float) -> bool:
    return delta_vs_pixel_floor >= VOE_DELTA_THRESHOLD


def is_decodable(primary_metric: dict) -> bool:
    """primary_metric: {"kind": "acc"|"r2", "value": float} for the law's
    PRIMARY_VARIABLE (see probes/mechanistic/decodability.py)."""
    kind = primary_metric["kind"]
    value = primary_metric["value"]
    if kind == "acc":
        return value >= ACC_THRESHOLD
    if kind == "r2":
        return value >= R2_THRESHOLD
    raise ValueError(f"unknown metric kind {kind!r}")


def verdict(voe_sensitive: bool, decodable: bool, causally_used: bool) -> str:
    """Trichotomy (spec 5.3):
      all three                       -> 'genuine'
      VoE + decodable, no causal use  -> 'shortcut-suspect'
      VoE, not decodable              -> 'shortcut'
      no VoE                          -> 'not-sensitive' (regardless of the rest)
    """
    if not voe_sensitive:
        return "not-sensitive"
    if not decodable:
        return "shortcut"
    if not causally_used:
        return "shortcut-suspect"
    return "genuine"
