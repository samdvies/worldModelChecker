"""AUROC is the core VoE metric (spec §5.1): surprise separating violate-vs-obey."""
import pytest

from physics_auditor.probes.behavioural.metrics import auroc, distinct_score_count


def test_perfect_separation_is_1():
    # violate scores all above obey scores
    assert auroc(pos=[2.0, 3.0, 4.0], neg=[0.1, 0.2, 0.3]) == 1.0


def test_no_separation_is_half():
    assert auroc(pos=[1.0, 2.0], neg=[1.0, 2.0]) == 0.5


def test_inverted_separation_is_0():
    assert auroc(pos=[0.1, 0.2], neg=[5.0, 6.0]) == 0.0


def test_ties_count_half():
    # pos=[1], neg=[1] -> single tied comparison -> 0.5
    assert auroc(pos=[1.0], neg=[1.0]) == 0.5


def test_partial_overlap():
    # comparisons: (2>1)=1, (2>3)=0, (4>1)=1, (4>3)=1 -> 3/4
    assert auroc(pos=[2.0, 4.0], neg=[1.0, 3.0]) == 0.75


def test_empty_raises():
    with pytest.raises(ValueError):
        auroc(pos=[], neg=[1.0])


def test_distinct_score_count_all_unique():
    # regression test (reviewer finding: solidity/permanence AUROC cells
    # collapse to 1-4 distinct score values across 16 "different" seeds,
    # e.g. PixelStaticAdapter returning the literal same float for solidity
    # regardless of seed -- n=16 in that case massively overstates the
    # effective sample size).
    assert distinct_score_count(obey=[1.0, 2.0, 3.0], violate=[4.0, 5.0, 6.0]) == 6


def test_distinct_score_count_collapsed_to_one_value():
    assert distinct_score_count(obey=[0.5284832790493965] * 16, violate=[0.5284832790493965] * 16) == 1


def test_distinct_score_count_partial_collapse():
    assert distinct_score_count(obey=[0.0, 0.0, 1.0], violate=[0.0, 2.0, 2.0]) == 3
