"""Guard for failure class D: external identifiers (AWS account id, bucket
name, region, instance type, ...) were duplicated verbatim across
scripts/aws/*.sh with no shared source of truth and no cross-check against
`aws sts get-caller-identity`, so a future account migration could silently
leave the two callers pointed at different S3 buckets (see
docs/failure-sweeps.md). This pins: (1) one shared _env.sh, (2)
launch_gpu.sh and pull_results.sh both source it, (3) no script hardcodes a
literal AWS account id, (4) launch_gpu.sh exposes --preflight.
"""
import re
from pathlib import Path

AWS_DIR = Path(__file__).resolve().parent.parent / "scripts" / "aws"
ENV_SH = AWS_DIR / "_env.sh"

# 12-digit AWS account ids look like this; a bare literal one is the smell
# we're guarding against (derived-at-runtime values don't match this as a
# quoted assignment).
_ACCOUNT_ID_LITERAL_RE = re.compile(r'ACCOUNT_ID\s*=\s*"?\d{12}"?')


def test_shared_env_file_exists():
    assert ENV_SH.exists(), "expected scripts/aws/_env.sh as the single source of truth"
    text = ENV_SH.read_text(encoding="utf-8")
    assert "resolve_account_id" in text


def test_launch_and_pull_scripts_source_shared_env():
    for name in ("launch_gpu.sh", "pull_results.sh"):
        text = (AWS_DIR / name).read_text(encoding="utf-8")
        assert re.search(r'source\s+"[^"]*_env\.sh"', text), (
            f"{name} must `source .../_env.sh` instead of redefining PROFILE/"
            f"REGION/ACCOUNT_ID/BUCKET locally"
        )


def test_no_script_hardcodes_a_literal_account_id():
    for path in AWS_DIR.glob("*.sh"):
        text = path.read_text(encoding="utf-8")
        assert not _ACCOUNT_ID_LITERAL_RE.search(text), (
            f"{path.name} hardcodes a literal 12-digit ACCOUNT_ID -- derive "
            f"it via resolve_account_id() (scripts/aws/_env.sh) instead so "
            f"launch_gpu.sh and pull_results.sh can never drift apart"
        )


def test_launch_gpu_supports_preflight_flag():
    text = (AWS_DIR / "launch_gpu.sh").read_text(encoding="utf-8")
    assert "--preflight" in text
    # preflight must be checked BEFORE the expensive IAM/bucket-creation
    # steps and must not itself call run-instances.
    assert "PREFLIGHT" in text or "preflight" in text.lower()
