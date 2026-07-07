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
