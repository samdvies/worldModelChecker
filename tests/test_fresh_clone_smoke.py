"""Guard for the fresh-clone gate's own blind spot (class E follow-up):
`git archive HEAD` silently omits anything uncommitted, so
scripts/fresh_clone_smoke.py must detect a missing-from-HEAD smoke file
BEFORE running the (then-meaningless) archive+pytest dance, rather than
surfacing a confusing FileNotFoundError from deep inside pytest.
"""
import subprocess
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "scripts"))
import fresh_clone_smoke  # noqa: E402


def _init_repo(tmp_path: Path) -> Path:
    repo = tmp_path / "repo"
    repo.mkdir()
    subprocess.run(["git", "init", "-q"], cwd=repo, check=True)
    subprocess.run(["git", "config", "user.email", "t@example.com"], cwd=repo, check=True)
    subprocess.run(["git", "config", "user.name", "T"], cwd=repo, check=True)
    return repo


def test_missing_tracked_smoke_files_empty_when_all_committed(tmp_path):
    repo = _init_repo(tmp_path)
    (repo / "tests").mkdir()
    (repo / "tests" / "test_smoke.py").write_text("# smoke\n")
    (repo / "tests" / "test_smoke_torch.py").write_text("# smoke torch\n")
    subprocess.run(["git", "add", "-A"], cwd=repo, check=True)
    subprocess.run(["git", "commit", "-q", "-m", "init"], cwd=repo, check=True)

    missing = fresh_clone_smoke._missing_tracked_smoke_files(
        repo, ["tests/test_smoke.py", "tests/test_smoke_torch.py"]
    )
    assert missing == []


def test_missing_tracked_smoke_files_flags_untracked_file(tmp_path):
    repo = _init_repo(tmp_path)
    (repo / "tests").mkdir()
    (repo / "tests" / "test_smoke.py").write_text("# smoke\n")
    subprocess.run(["git", "add", "-A"], cwd=repo, check=True)
    subprocess.run(["git", "commit", "-q", "-m", "init"], cwd=repo, check=True)
    # test_smoke_torch.py exists on disk but is never git-added -- exactly
    # this session's real situation.
    (repo / "tests" / "test_smoke_torch.py").write_text("# smoke torch\n")

    missing = fresh_clone_smoke._missing_tracked_smoke_files(
        repo, ["tests/test_smoke.py", "tests/test_smoke_torch.py"]
    )
    assert missing == ["tests/test_smoke_torch.py"]


def test_missing_tracked_smoke_files_flags_file_absent_entirely(tmp_path):
    repo = _init_repo(tmp_path)
    (repo / "tests").mkdir()
    (repo / "tests" / "test_smoke.py").write_text("# smoke\n")
    subprocess.run(["git", "add", "-A"], cwd=repo, check=True)
    subprocess.run(["git", "commit", "-q", "-m", "init"], cwd=repo, check=True)

    missing = fresh_clone_smoke._missing_tracked_smoke_files(
        repo, ["tests/test_smoke.py", "tests/test_does_not_exist.py"]
    )
    assert missing == ["tests/test_does_not_exist.py"]


def test_tracked_files_at_head_returns_none_outside_git_repo(tmp_path):
    not_a_repo = tmp_path / "plain_dir"
    not_a_repo.mkdir()
    assert fresh_clone_smoke._tracked_files_at_head(not_a_repo) is None

