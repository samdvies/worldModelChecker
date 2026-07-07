# Shared config for scripts/aws/*.sh -- single source of truth for external
# identifiers (class D guard: these were previously duplicated verbatim in
# launch_gpu.sh and pull_results.sh with no cross-check against reality, see
# docs/failure-sweeps.md). Source this, don't copy these values elsewhere.
#
# Usage: `source "$(dirname "${BASH_SOURCE[0]}")/_env.sh"` from a script
# that has already done `set -euo pipefail`.

PROFILE="${PROFILE:-claude-admin}"
REGION="${REGION:-eu-west-1}"
INSTANCE_TYPE="${INSTANCE_TYPE:-g4dn.xlarge}"
PROJECT_TAG="${PROJECT_TAG:-physics-auditor}"
IAM_ROLE_NAME="${IAM_ROLE_NAME:-physics-auditor-gpu-role}"
IAM_PROFILE_NAME="${IAM_PROFILE_NAME:-physics-auditor-gpu-profile}"
SG_NAME="${SG_NAME:-physics-auditor-gpu-sg}"
ROOT_VOLUME_GB="${ROOT_VOLUME_GB:-100}"

# ACCOUNT_ID is derived live from the caller's own credentials rather than
# hardcoded, so a profile/account migration can never leave the two callers
# (launch_gpu.sh, pull_results.sh) silently pointed at different buckets --
# each resolves it itself, from the same `aws sts get-caller-identity` call
# each already needs for a credential sanity check.
resolve_account_id() {
    aws sts get-caller-identity --profile "$PROFILE" --region "$REGION" \
        --query "Account" --output text
}

# BUCKET is only fully resolved once ACCOUNT_ID is known (see
# resolve_account_id above) -- callers must set:
#   ACCOUNT_ID="$(resolve_account_id)"
#   BUCKET="physics-auditor-${ACCOUNT_ID}"
