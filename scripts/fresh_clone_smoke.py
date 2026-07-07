"""Pre-GPU-launch gate (class E): simulates a fresh clone (no .venv, no
artifacts/, no cache/) by exporting the current HEAD via `git archive` into
a scratch directory and running the fast smoke tier there.

This is read-only against the working repo (git archive HEAD does not touch
the working tree or history) and takes ~seconds, so it belongs immediately
before every scripts/aws/launch_gpu.sh dispatch -- see scripts/aws/runbook.md.

IMPORTANT: `git archive HEAD` only ever includes COMMITTED content -- any
smoke-tier file that exists on disk but isn't committed will silently be
missing from the export, and the failure that surfaces (pytest can't find
the test file) gives no hint that the real cause is "go commit your guard
files". `_missing_tracked_smoke_files()` below checks that up front and
fails loudly with an actionable message instead.

Usage: uv run python scripts/fresh_clone_smoke.py
Exit code: 0 = smoke tier green in the fresh export AND within the <=10s
wall-time budget; non-zero otherwise.
"""
import shutil
import subprocess
import sys
import tempfile
import time
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parent.parent

# Every file the `smoke` pytest marker's collection needs. If any of these
# is untracked/uncommitted, `git archive HEAD` will silently omit it and
# the fresh-clone simulation is not simulating anything real.
REQUIRED_TRACKED_FILES = [
    "tests/test_smoke.py",
    "tests/test_smoke_torch.py",
]

SMOKE_WALL_BUDGET_S = 10.0


def _tracked_files_at_head(repo_root: Path) -> set[str] | None:
    """Returns the set of paths git considers tracked at HEAD, or None if
    git isn't usable here (e.g. not a repo / no commits yet)."""
    proc = subprocess.run(
        ["git", "ls-tree", "-r", "--name-only", "HEAD"],
        cwd=repo_root,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        text=True,
        check=False,
    )
    if proc.returncode != 0:
        return None
    return set(proc.stdout.splitlines())


def _missing_tracked_smoke_files(repo_root: Path, required: list[str]) -> list[str]:
    """Subset of `required` that is NOT tracked at HEAD -- i.e. would be
    silently missing from `git archive HEAD`. Returns [] if the tracked-set
    lookup itself failed (nothing to flag confidently)."""
    tracked = _tracked_files_at_head(repo_root)
    if tracked is None:
        return []
    return [f for f in required if f not in tracked]


def main() -> int:
    missing = _missing_tracked_smoke_files(REPO_ROOT, REQUIRED_TRACKED_FILES)
    if missing:
        print(
            "FAILED: the following smoke-tier files exist locally but are "
            "NOT committed to git, so `git archive HEAD` (this fresh-clone "
            "simulation) will omit them and the gate below is meaningless:\n"
            + "\n".join(f"  - {f}" for f in missing)
            + "\nCommit these files (or ask Sam to) before relying on this "
            "gate or dispatching to the GPU box.",
            file=sys.stderr,
        )
        return 3

    with tempfile.TemporaryDirectory(prefix="physics-auditor-freshclone-") as tmp:
        scratch = Path(tmp)
        print(f"== exporting HEAD via git archive into {scratch} ==")
        archive_proc = subprocess.run(
            ["git", "archive", "--format=tar", "HEAD"],
            cwd=REPO_ROOT,
            stdout=subprocess.PIPE,
            check=False,
        )
        if archive_proc.returncode != 0 or not archive_proc.stdout:
            print(
                "FAILED: `git archive HEAD` produced no output -- is this a "
                "git repo with at least one commit? Skipping fresh-clone gate.",
                file=sys.stderr,
            )
            return archive_proc.returncode or 1

        tar_path = scratch / "archive.tar"
        tar_path.write_bytes(archive_proc.stdout)
        extract_dir = scratch / "export"
        extract_dir.mkdir()
        subprocess.run(["tar", "-xf", str(tar_path), "-C", str(extract_dir)], check=True)
        tar_path.unlink()

        for stale in ("artifacts", "cache", ".venv"):
            stale_path = extract_dir / stale
            if stale_path.exists():
                shutil.rmtree(stale_path)

        print(f"== running 'uv run pytest -m smoke' in the fresh export =={extract_dir}")
        # Warm the environment UNTIMED: in a fresh export `uv run` first
        # resolves and installs the whole venv (torch ~2 min), which is
        # environment-setup cost, not smoke-tier cost. Timing it against the
        # 10s budget made the gate fail unconditionally on every fresh clone.
        print("== warming fresh venv (untimed): uv sync ==")
        sync = subprocess.run(["uv", "sync"], cwd=extract_dir)
        if sync.returncode != 0:
            print("== fresh-clone smoke gate: FAIL -- uv sync failed ==", file=sys.stderr)
            return sync.returncode

        t0 = time.monotonic()
        result = subprocess.run(
            ["uv", "run", "pytest", "-m", "smoke", "-q", "tests/test_smoke.py"],
            cwd=extract_dir,
        )
        elapsed = time.monotonic() - t0
        print(f"== smoke tier wall time: {elapsed:.1f}s (budget {SMOKE_WALL_BUDGET_S:.0f}s) ==")

        if result.returncode == 0 and elapsed > SMOKE_WALL_BUDGET_S:
            print(
                f"== fresh-clone smoke gate: FAIL -- smoke tier passed but took "
                f"{elapsed:.1f}s, over the {SMOKE_WALL_BUDGET_S:.0f}s hard-rule "
                "budget (see docs/failure-sweeps.md) ==",
                file=sys.stderr,
            )
            return 4

        if result.returncode == 0:
            print("== fresh-clone smoke gate: PASS -- safe to launch_gpu.sh --preflight next ==")
        else:
            print("== fresh-clone smoke gate: FAIL -- do NOT launch the GPU box ==", file=sys.stderr)
        return result.returncode


if __name__ == "__main__":
    raise SystemExit(main())
