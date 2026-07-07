"""Guard for failure class E: a test file that reads the real repo's
artifacts/ or cache/ directories (both gitignored, populated only by a real
GPU-box run) without a skipif/existence guard is green locally and red on
any fresh clone -- which can abort a GPU dispatch (test_report_card_e2e.py
hit this once; see docs/failure-sweeps.md).

This is a static check: any test file that references a literal "artifacts/"
or "cache/" path (not going through pytest's tmp_path/tmp_path_factory
fixtures) must also carry a `pytest.mark.skipif` (module-level or per-test)
guarding on that same path's existence.
"""
import re
from pathlib import Path

TESTS_DIR = Path(__file__).resolve().parent
REPO_ROOT = TESTS_DIR.parent

_UNTRACKED_PATH_RE = re.compile(r'["\'](artifacts|cache)/')
_SKIPIF_RE = re.compile(r"skipif")


def _test_files() -> list[Path]:
    return sorted(p for p in TESTS_DIR.glob("test_*.py") if p.name != "test_guard_untracked_state.py")


def test_every_untracked_state_reader_has_a_skip_guard():
    offenders = []
    for path in _test_files():
        text = path.read_text(encoding="utf-8")
        if _UNTRACKED_PATH_RE.search(text) and not _SKIPIF_RE.search(text):
            offenders.append(path.name)
    assert not offenders, (
        "these test files reference a literal artifacts/ or cache/ path "
        "without a pytest.mark.skipif existence guard, so they will fail "
        "(not skip) on a fresh clone with no untracked state: "
        f"{offenders} -- see tests/test_report_card_e2e.py for the pattern"
    )


def test_report_card_e2e_guard_matches_the_directory_it_reads():
    """Specifically pin the known offender-turned-fix: the skipif condition
    must check the SAME path (artifacts/weights) the test body reads."""
    path = TESTS_DIR / "test_report_card_e2e.py"
    text = path.read_text(encoding="utf-8")
    assert "artifacts/weights" in text
    assert re.search(r"skipif\(\s*not Path\(.artifacts/weights.\)\.is_dir\(\)", text)
