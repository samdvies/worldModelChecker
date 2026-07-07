"""Class C guard, broadened to every shell script in the repo: `bash -n`
(syntax-only, no execution) on every .sh file, plus the specific unguarded-
env-var-expansion-under-set -u rule (the exact mechanism that bypassed the
ERR trap in remote_job.sh -- see docs/failure-sweeps.md and
tests/test_remote_job_ordering.py for the remote_job.sh-specific pin).
"""
import re
import shutil
import subprocess
from pathlib import Path

import pytest

REPO_ROOT = Path(__file__).resolve().parent.parent
ALL_SH = sorted(REPO_ROOT.glob("scripts/**/*.sh"))

# Scripts that execute in an environment where HOME/USER-style login
# environment variables are NOT guaranteed to be set (remote/user-data
# contexts) -- these are the ones the unguarded-expansion rule applies to.
REMOTELY_EXECUTED = {"remote_job.sh"}


def test_at_least_one_shell_script_found():
    assert ALL_SH, "expected at least one scripts/**/*.sh file"


@pytest.mark.parametrize("sh_path", ALL_SH, ids=[p.name for p in ALL_SH])
def test_bash_syntax_check(sh_path):
    bash = shutil.which("bash")
    if bash is None:
        pytest.skip("bash not available in this environment")
    result = subprocess.run([bash, "-n", str(sh_path)], capture_output=True, text=True)
    assert result.returncode == 0, f"bash -n failed for {sh_path.name}:\n{result.stderr}"


@pytest.mark.parametrize(
    "sh_path", [p for p in ALL_SH if p.name in REMOTELY_EXECUTED], ids=lambda p: p.name if hasattr(p, "name") else str(p)
)
def test_home_var_is_guarded_under_set_u(sh_path):
    """A script that runs under `set -u` in an environment with no login
    shell (EC2 user-data / remote_job.sh) must never bare-expand $HOME
    BEFORE it has been guarded: the FIRST occurrence of $HOME in the script
    must be a guarded form (`${HOME:-...}`), which then makes every later
    plain `$HOME` use safe (it is guaranteed set by that point)."""
    text = sh_path.read_text(encoding="utf-8")
    assert "set -u" in text or "set -euo pipefail" in text, (
        f"{sh_path.name} is remotely executed but does not set -u -- add it "
        f"so unguarded var bugs fail loudly instead of silently using an "
        f"empty string"
    )
    first_use = re.search(r"\$\{?HOME\b", text)
    assert first_use, f"{sh_path.name}: expected at least one $HOME reference"
    tail = text[first_use.end():first_use.end() + 2]
    assert tail.startswith(":-") or tail.startswith(":?"), (
        f"{sh_path.name}: first use of $HOME is not guarded with "
        f"${{HOME:-default}} -- an unguarded expansion aborts under set -u "
        f"WITHOUT firing the ERR trap (class C)"
    )
