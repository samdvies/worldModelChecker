"""Guard for failure class C (sweep finding, scripts/aws/remote_job.sh):
`${BUCKET:?msg}`-style parameter-expansion failures abort a `set -u` script
WITHOUT firing the ERR trap (confirmed empirically -- see docs/failure-sweeps.md).
So the ERR trap and the cost-backstop shutdown must be registered/scheduled
BEFORE any command that can fail this way, or a bad env var leaves the box
running with no diagnostics uploaded and no backstop.
"""
import re
from pathlib import Path

REMOTE_JOB = Path(__file__).resolve().parent.parent / "scripts" / "aws" / "remote_job.sh"


def _line_of(pattern: str, text: str) -> int:
    for i, line in enumerate(text.splitlines(), start=1):
        if re.search(pattern, line):
            return i
    raise AssertionError(f"pattern not found in {REMOTE_JOB}: {pattern!r}")


def _code_lines(text: str) -> list[tuple[int, str]]:
    """(lineno, line) pairs for non-comment, non-blank lines."""
    out = []
    for i, line in enumerate(text.splitlines(), start=1):
        stripped = line.strip()
        if not stripped or stripped.startswith("#"):
            continue
        out.append((i, line))
    return out


def test_no_bare_parameter_expansion_failure_checks_in_code():
    """`${VAR:?msg}` aborts a `set -u` script WITHOUT firing the ERR trap
    (class C). remote_job.sh must not use this form in actual code (comments
    documenting the hazard are fine) -- failing checks must be normal
    commands (`if ...; then ...; fi`) that the ERR trap can see."""
    text = REMOTE_JOB.read_text(encoding="utf-8")
    for lineno, line in _code_lines(text):
        assert not re.search(r"\$\{\w+:\?", line), (
            f"line {lineno} uses ${{VAR:?...}} in code -- this bypasses the "
            f"ERR trap under set -u (class C); use an explicit `if` check instead"
        )


def test_trap_and_backstop_precede_the_bucket_check():
    text = REMOTE_JOB.read_text(encoding="utf-8")

    trap_line = _line_of(r"trap\s+on_error\s+ERR", text)
    shutdown_line = _line_of(r"shutdown -h \+240", text)
    bucket_check_line = _line_of(r'-z\s+"\$\{BUCKET:-', text)

    assert trap_line < bucket_check_line, (
        f"ERR trap registered at line {trap_line} but the BUCKET failure "
        f"check is at line {bucket_check_line} -- trap must come first"
    )
    assert shutdown_line < bucket_check_line, (
        f"cost-backstop shutdown scheduled at line {shutdown_line} but the "
        f"BUCKET failure check is at line {bucket_check_line} -- backstop "
        f"must come first"
    )


def test_partial_sync_loop_starts_after_err_trap_and_bucket_check():
    """Class G (see docs/failure-sweeps.md): the background partial-sync
    loop must not start before BUCKET is validated (it needs BUCKET to sync
    anywhere useful) but must still exist to protect any failure from that
    point on -- both the ERR trap and the BUCKET check must precede it."""
    text = REMOTE_JOB.read_text(encoding="utf-8")
    trap_line = _line_of(r"trap\s+on_error\s+ERR", text)
    bucket_check_line = _line_of(r'-z\s+"\$\{BUCKET:-', text)
    loop_start_line = _line_of(r"start_partial_sync_loop\s*$", text)
    assert trap_line < loop_start_line
    assert bucket_check_line < loop_start_line


def test_partial_sync_loop_precedes_repo_download():
    """The loop must be armed before the (slow, failure-prone) repo
    download/untar step, so a download failure still gets a partial-cache
    sync attempt via on_error."""
    text = REMOTE_JOB.read_text(encoding="utf-8")
    loop_start_line = _line_of(r"start_partial_sync_loop\s*$", text)
    download_line = _line_of(r"download \+ untar repo", text)
    assert loop_start_line < download_line


def test_cache_preseed_precedes_any_encoding_step():
    """Pre-seeding cache/ from cache-partial must happen after repo untar
    but before any of the (expensive) encoding steps -- otherwise the
    resumability the pre-seed exists for never kicks in."""
    text = REMOTE_JOB.read_text(encoding="utf-8")
    untar_line = _line_of(r"tar -xzf .*physics-auditor\.tar\.gz", text)
    preseed_line = _line_of(r"aws s3 sync .*cache-partial.*REPO_DIR/cache", text)
    train_stacks_line = _line_of(r"train_stacks\.py --stacks", text)
    assert untar_line < preseed_line < train_stacks_line


def test_stop_partial_sync_loop_called_in_on_error_and_success_path():
    """Both the on_error trap handler and the normal success path must stop
    the loop (kill + final sync) -- a job that succeeds without pushing its
    last few minutes of cache work, or one that fails and leaves the loop
    running into the shutdown, both defeat the point of this guard."""
    text = REMOTE_JOB.read_text(encoding="utf-8")
    on_error_start = _line_of(r"^on_error\(\)\s*\{", text)
    trap_line = _line_of(r"trap\s+on_error\s+ERR", text)
    on_error_body = "\n".join(text.splitlines()[on_error_start - 1: trap_line - 1])
    assert "stop_partial_sync_loop" in on_error_body, (
        "on_error() must call stop_partial_sync_loop before shutdown"
    )

    shutdown_now_line = _line_of(r"shutdown -h now$", text)
    lines = text.splitlines()
    # the LAST "stop_partial_sync_loop" call in the file, outside on_error,
    # must occur before the final (success-path) shutdown -h now.
    success_path_calls = [
        i for i, line in enumerate(lines, start=1)
        if re.search(r"^stop_partial_sync_loop\s*$", line)
    ]
    assert success_path_calls, "expected a top-level stop_partial_sync_loop call in the success path"
    assert success_path_calls[-1] < shutdown_now_line


def test_cuda_gate_after_uv_sync_and_before_train_stacks():
    """Class I (see docs/failure-sweeps.md): pyproject.toml used to pin the
    CPU torch wheel index globally, so `uv sync --group gpu` on the GPU box
    silently installed torch==X+cpu and encoding fell back to CPU (~20x
    slower). remote_job.sh must fail fast (seconds, not hours) if torch
    resolved a CPU wheel, right after `uv sync --group gpu` and before any
    encoding work starts."""
    text = REMOTE_JOB.read_text(encoding="utf-8")
    uv_sync_line = _line_of(r"uv sync --group gpu", text)
    cuda_gate_line = _line_of(r"torch\.cuda\.is_available\(\)", text)
    train_stacks_line = _line_of(r"train_stacks\.py --stacks", text)
    assert uv_sync_line < cuda_gate_line < train_stacks_line


def test_train_stacks_line_uses_probe_caches_flag():
    """Class I: run_report_card.py/run_mechanistic.py (previously used to
    warm the mechanistic-probe latent caches) crash on the GPU box with
    FileNotFoundError -- predictor weights are never trained there. The
    probe caches are now warmed via train_stacks.py --probe-caches
    instead."""
    text = REMOTE_JOB.read_text(encoding="utf-8")
    train_stacks_line_text = next(
        line for line in text.splitlines() if re.search(r"train_stacks\.py --stacks", line)
    )
    assert "--probe-caches" in train_stacks_line_text


def test_report_card_and_mechanistic_scripts_no_longer_invoked():
    """Class I: both scripts crash on the GPU box (FileNotFoundError on
    predictor weights never trained there) -- they must no longer be
    invoked from remote_job.sh."""
    text = REMOTE_JOB.read_text(encoding="utf-8")
    assert "run_report_card.py" not in text
    assert "run_mechanistic.py" not in text


def test_artifacts_tar_no_longer_produced_or_uploaded():
    """Class I: nothing on the box produces artifacts/ any more (predictor
    training/report-card/mechanistic steps were removed), so the
    artifacts.tar.gz tar+upload is dead weight -- must be removed."""
    text = REMOTE_JOB.read_text(encoding="utf-8")
    assert "artifacts.tar.gz" not in text


def test_no_unguarded_var_expansion_under_set_u():
    """Every shell script that may run remotely (under `set -u`) must guard
    variable expansions that are not parameters/locals with a default
    (`${VAR:-...}`) or an explicit `:?` check -- a bare `$VAR`/`${VAR}` for
    an environment-sourced variable aborts silently under `set -u` (class C).
    This only scans variables we know are environment-sourced in this repo's
    remote scripts, to avoid false positives on ordinary local variables.
    """
    text = REMOTE_JOB.read_text(encoding="utf-8")
    assert "set -euo pipefail" in text or "set -u" in text

    env_sourced_vars = ["HOME", "BUCKET", "REGION"]
    for var in env_sourced_vars:
        # every appearance of $VAR or ${VAR} (not ${VAR:-...} / ${VAR:?...}
        # / ${VAR//...} assignment sites) must be guarded somewhere before
        # first bare use -- concretely: the FIRST occurrence of the var name
        # in the script must be a guarded form.
        first_bare = re.search(rf"\$\{{?{var}\b(?!:[-?])", text)
        guarded = re.search(rf"\$\{{{var}:[-?]", text) or re.search(rf'{var}="\$\{{{var}:', text)
        assert guarded, f"{var} is never guarded with ${{{var}:-...}} or ${{{var}:?...}} in remote_job.sh"
