"""Characterization tests for ScenarioConfig.scenario_id."""
from physics_auditor.generator.config import ScenarioConfig


def _make_config(violate: bool = False) -> ScenarioConfig:
    return ScenarioConfig(
        seed=0,
        violate=violate,
        widths=(1.0, 2.0),
        heights=(1.0, 2.0),
        x_centres=(0.0, 5.0),
    )


def test_same_config_built_twice_has_identical_id():
    a = _make_config()
    b = _make_config()
    assert a.scenario_id == b.scenario_id


def test_flipping_violate_changes_id():
    obey = _make_config(violate=False)
    violate = _make_config(violate=True)
    assert obey.scenario_id != violate.scenario_id


def test_id_is_40_char_lowercase_sha1_hex():
    config = _make_config()
    scenario_id = config.scenario_id
    assert len(scenario_id) == 40
    assert scenario_id == scenario_id.lower()
    assert all(c in "0123456789abcdef" for c in scenario_id)
