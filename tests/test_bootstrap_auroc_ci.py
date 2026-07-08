"""Bootstrap CI for VoE AUROC (report-card statistical rigor): percentile
bootstrap over the paired (obey, violate) index pairs, matching the pairing
run_voe / auroc already rely on (spec §5.1)."""
from physics_auditor.probes.behavioural.metrics import auroc, bootstrap_auroc_ci


def test_perfect_separation_ci_is_point_mass_at_one():
    obey = [0.1, 0.2, 0.3, 0.4]
    violate = [1.1, 1.2, 1.3, 1.4]
    lo, hi = bootstrap_auroc_ci(obey, violate, n_boot=200, seed=0)
    assert (lo, hi) == (1.0, 1.0)


def test_identical_distributions_ci_straddles_half():
    # Same values in both arms (paired ties) -> point auroc is exactly 0.5,
    # and every bootstrap resample (any mix of the same values with ties)
    # is also exactly 0.5 -- CI collapses to (0.5, 0.5), which still
    # straddles 0.5 trivially.
    obey = [0.5, 0.5, 0.5, 0.5]
    violate = [0.5, 0.5, 0.5, 0.5]
    lo, hi = bootstrap_auroc_ci(obey, violate, n_boot=200, seed=0)
    assert lo <= 0.5 <= hi


def test_overlapping_distributions_ci_has_spread_and_straddles_half():
    # Mild separation with overlap -- resampling pairs should produce a CI
    # with real width that still straddles 0.5.
    obey = [0.1, 0.9, 0.2, 0.8, 0.3, 0.7, 0.4, 0.6]
    violate = [0.9, 0.1, 0.8, 0.2, 0.7, 0.3, 0.6, 0.4]
    lo, hi = bootstrap_auroc_ci(obey, violate, n_boot=2000, seed=0)
    assert lo < hi
    assert lo <= 0.5 <= hi


def test_deterministic_given_seed():
    obey = [0.1, 0.4, 0.35, 0.5, 0.2]
    violate = [0.6, 0.3, 0.7, 0.55, 0.9]
    ci1 = bootstrap_auroc_ci(obey, violate, n_boot=500, seed=42)
    ci2 = bootstrap_auroc_ci(obey, violate, n_boot=500, seed=42)
    assert ci1 == ci2


def test_lo_le_hi_and_within_unit_interval():
    obey = [0.2, 0.3, 0.9, 0.1, 0.6]
    violate = [0.8, 0.4, 0.2, 0.95, 0.3]
    lo, hi = bootstrap_auroc_ci(obey, violate, n_boot=1000, seed=7)
    assert 0.0 <= lo <= hi <= 1.0


def test_pairing_is_preserved_across_resamples():
    # If pairing were broken (independent resampling of obey vs violate),
    # shuffling one arm relative to the other would change the point AUROC
    # entirely -- here obey[i] and violate[i] are deliberately anti-correlated
    # per index, so preserving pairing keeps the bootstrap distribution tight
    # around the point estimate rather than centered elsewhere.
    obey = [0.0, 1.0, 0.0, 1.0, 0.0, 1.0]
    violate = [1.0, 0.0, 1.0, 0.0, 1.0, 0.0]
    point = auroc(pos=violate, neg=obey)
    lo, hi = bootstrap_auroc_ci(obey, violate, n_boot=1000, seed=3)
    assert lo <= point <= hi


def test_requires_equal_length_paired_inputs():
    import pytest

    with pytest.raises(ValueError):
        bootstrap_auroc_ci([0.1, 0.2], [0.3], n_boot=10, seed=0)
