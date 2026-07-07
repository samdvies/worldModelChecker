"""TDD: report/verdict.py -- trichotomy verdict from three booleans
(voe_sensitive, decodable, causally_used), per spec 5.3."""
import pytest

from physics_auditor.report.verdict import verdict


@pytest.mark.parametrize(
    "voe_sensitive, decodable, causally_used, expected",
    [
        (True, True, True, "genuine"),
        (True, True, False, "shortcut-suspect"),
        (True, False, True, "shortcut"),  # causal flag irrelevant if not decodable
        (True, False, False, "shortcut"),
        (False, True, True, "not-sensitive"),  # no VoE always wins regardless
        (False, True, False, "not-sensitive"),
        (False, False, True, "not-sensitive"),
        (False, False, False, "not-sensitive"),
    ],
)
def test_verdict_trichotomy_table(voe_sensitive, decodable, causally_used, expected):
    assert verdict(voe_sensitive, decodable, causally_used) == expected


def test_verdict_gates_decodable_from_metrics():
    from physics_auditor.report.verdict import is_decodable

    assert is_decodable({"kind": "acc", "value": 0.9}) is True
    assert is_decodable({"kind": "acc", "value": 0.89999}) is False
    assert is_decodable({"kind": "r2", "value": 0.5}) is True
    assert is_decodable({"kind": "r2", "value": 0.4999}) is False


def test_verdict_gates_voe_sensitive_from_delta():
    from physics_auditor.report.verdict import is_voe_sensitive

    assert is_voe_sensitive(0.2) is True
    assert is_voe_sensitive(0.1999) is False
    assert is_voe_sensitive(-0.5) is False
