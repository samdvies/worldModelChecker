"""Closed-form ridge decodability probes (spec 5.2): fit a linear read-out
from a stack's frozen latents to the true underlying physical variable a law
should expose, on probe-TRAIN pairs (seeds 300..331); evaluate on
probe-TEST pairs (seeds 400..415). numpy only -- no sklearn/scipy. Ridge is
solved in closed form; z-scoring statistics come from TRAIN only and are
applied verbatim to test (never recomputed on test)."""
from dataclasses import dataclass, field

import numpy as np

from physics_auditor.models.base import Encoder
from physics_auditor.probes.mechanistic import labels as L
from physics_auditor.probes.mechanistic.data import latent_frames, probe_pairs

RIDGE_LAMBDA = 1e-2
# Frame offsets relative to the critical frame, inclusive: cf-4 .. cf+7.
WINDOW_OFFSETS = range(-4, 8)


@dataclass
class DecodeResult:
    law: str
    stack: str
    metrics: dict = field(default_factory=dict)  # var_name -> {"kind": "acc"|"r2", "value": float}
    degenerate: bool = False


BLINDNESS_EPS = 1e-9


def _max_gap_in_window(obey_z: np.ndarray, violate_z: np.ndarray, cf: int, offsets) -> float:
    """Max L2 distance between obey/violate latents over the given frame
    offsets from `cf`; 0.0 if the window is empty for this pair."""
    best = 0.0
    for off in offsets:
        idx = cf + off
        if idx < 0 or idx >= obey_z.shape[0] or idx >= violate_z.shape[0]:
            continue
        best = max(best, float(np.linalg.norm(obey_z[idx] - violate_z[idx])))
    return best


def window_is_structurally_blind(
    pairs, encoder: Encoder, cache_dir: str = "cache", offsets=WINDOW_OFFSETS, eps: float = BLINDNESS_EPS
) -> bool:
    """True iff, for EVERY pair, obey and violate latents are ~identical at
    every frame inside the fixed decode window -- i.e. the probe window can
    never observe any obey/violate divergence for this law, regardless of
    what the ridge readout does with it (e.g. permanence's occluder hides
    the ball's deletion for longer than WINDOW_OFFSETS reaches). A decode
    accuracy/R2 computed under this condition is a majority-class /
    constant-target artifact, not evidence of (non-)decodability."""
    for pair in pairs:
        obey_z, violate_z = latent_frames(pair, encoder, cache_dir=cache_dir)
        if _max_gap_in_window(obey_z, violate_z, pair.critical_frame, offsets) > eps:
            return False
    return True


def zscore_fit(X: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """Per-feature mean/std computed from X (intended: TRAIN features only).
    Zero-variance features get std=1 so they z-score to 0, not NaN/inf."""
    mean = X.mean(axis=0)
    std = X.std(axis=0)
    std = np.where(std < 1e-8, 1.0, std)
    return mean, std


def zscore_apply(X: np.ndarray, mean: np.ndarray, std: np.ndarray) -> np.ndarray:
    """Applies previously-fit (train) mean/std to X. Never recomputes stats
    from X itself -- this is what keeps test evaluation honest."""
    return (X - mean) / std


def ridge_fit(X: np.ndarray, y: np.ndarray, lam: float = RIDGE_LAMBDA) -> np.ndarray:
    """Closed-form ridge regression with an unregularised bias term.

    X: (N, D) z-scored features (no bias column -- added internally).
    y: (N,) targets.
    Returns w: (D+1,) with w[-1] the bias.
    """
    if X.shape[0] != y.shape[0]:
        raise ValueError(f"X has {X.shape[0]} rows but y has {y.shape[0]}")
    n, d = X.shape
    Xb = np.hstack([X, np.ones((n, 1), dtype=X.dtype)])
    reg = np.eye(d + 1, dtype=X.dtype)
    reg[-1, -1] = 0.0  # bias unregularised
    w = np.linalg.solve(Xb.T @ Xb + lam * reg, Xb.T @ y)
    return w


def ridge_predict(X: np.ndarray, w: np.ndarray) -> np.ndarray:
    n = X.shape[0]
    Xb = np.hstack([X, np.ones((n, 1), dtype=X.dtype)])
    return Xb @ w


def r2_score(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    ss_res = np.sum((y_true - y_pred) ** 2)
    ss_tot = np.sum((y_true - y_true.mean()) ** 2)
    if ss_tot < 1e-12:
        return 0.0
    return float(1.0 - ss_res / ss_tot)


def accuracy_from_scores(y_true_bool: np.ndarray, y_pred_continuous: np.ndarray) -> float:
    pred_bool = y_pred_continuous >= 0.5
    return float(np.mean(np.asarray(y_true_bool, dtype=bool) == pred_bool))


# --- per-law variable registry -------------------------------------------

# (variable_name, kind, label_fn); kind in {"acc", "r2"}.
LAW_VARIABLES: dict[str, list[tuple[str, str, object]]] = {
    "support": [("support", "acc", L.support_label)],
    "permanence": [("permanence", "acc", L.permanence_label)],
    "solidity": [("ball_y", "r2", L.solidity_y), ("above_shelf", "acc", L.solidity_above_shelf)],
    "gravity": [("vy", "r2", L.gravity_vy)],
}

# The "primary" variable per law, used by report/verdict.py's decodable gate.
PRIMARY_VARIABLE: dict[str, str] = {
    "support": "support",
    "permanence": "permanence",
    "solidity": "ball_y",
    "gravity": "vy",
}


def _collect(pairs, encoder: Encoder, label_fn, cache_dir: str) -> tuple[np.ndarray, np.ndarray]:
    """Latents + labels for frames cf-4..cf+7 of both obey and violate clips,
    across all pairs in this split."""
    xs: list[np.ndarray] = []
    ys: list[float] = []
    for pair in pairs:
        obey_z, violate_z = latent_frames(pair, encoder, cache_dir=cache_dir)
        cf = pair.critical_frame
        for z_arr, clip in ((obey_z, pair.obey), (violate_z, pair.violate)):
            t = z_arr.shape[0]
            for offset in WINDOW_OFFSETS:
                idx = cf + offset
                if idx < 0 or idx >= t:
                    continue
                xs.append(z_arr[idx])
                ys.append(label_fn(clip.states[idx]))
    return np.asarray(xs, dtype=np.float64), np.asarray(ys)


def decode_law_stack(law_name: str, stack: Encoder, cache_dir: str = "cache") -> DecodeResult:
    """Fits + evaluates ridge read-outs for every GT variable of `law_name`
    on this stack's frozen latents. Train = probe-train pairs (300s), test =
    probe-test pairs (400s); z-scoring stats come from train only."""
    train_pairs = probe_pairs(law_name, "train")
    test_pairs = probe_pairs(law_name, "test")

    metrics: dict[str, dict] = {}
    for var_name, kind, label_fn in LAW_VARIABLES[law_name]:
        x_train, y_train = _collect(train_pairs, stack, label_fn, cache_dir)
        x_test, y_test = _collect(test_pairs, stack, label_fn, cache_dir)

        mean, std = zscore_fit(x_train)
        x_train_z = zscore_apply(x_train, mean, std)
        x_test_z = zscore_apply(x_test, mean, std)

        w = ridge_fit(x_train_z, y_train.astype(np.float64))
        pred_test = ridge_predict(x_test_z, w)

        if kind == "acc":
            value = accuracy_from_scores(y_test, pred_test)
        else:
            value = r2_score(y_test.astype(np.float64), pred_test)
        metrics[var_name] = {"kind": kind, "value": value}

    degenerate = window_is_structurally_blind(test_pairs, stack, cache_dir=cache_dir)
    return DecodeResult(law=law_name, stack=stack.name, metrics=metrics, degenerate=degenerate)
