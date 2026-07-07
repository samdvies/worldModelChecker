#!/usr/bin/env bash
# Pulls the populated latent caches + logs down from S3 after a GPU job
# finishes, and untars them into the local repo. Run FROM the laptop.
#
# Usage: scripts/aws/pull_results.sh
set -euo pipefail

PROFILE="claude-admin"
REGION="eu-west-1"
ACCOUNT_ID="652742769396"
BUCKET="physics-auditor-${ACCOUNT_ID}"

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "$SCRIPT_DIR/../.." && pwd)"
DEST_DIR="$(mktemp -d)"

echo "== syncing results from s3://${BUCKET}/results/ =="
aws s3 sync "s3://${BUCKET}/results/" "$DEST_DIR" \
    --profile "$PROFILE" --region "$REGION"

if [ -f "$DEST_DIR/cache.tar.gz" ]; then
    echo "== untarring cache.tar.gz into $REPO_ROOT =="
    tar -xzf "$DEST_DIR/cache.tar.gz" -C "$REPO_ROOT"
else
    echo "!! cache.tar.gz not found in results -- job may have failed early. Check remote_job.log below."
fi

if [ -f "$DEST_DIR/artifacts.tar.gz" ]; then
    echo "== untarring artifacts.tar.gz into $REPO_ROOT =="
    tar -xzf "$DEST_DIR/artifacts.tar.gz" -C "$REPO_ROOT"
fi

if [ -f "$DEST_DIR/remote_job.log" ]; then
    echo "== last 40 lines of remote_job.log =="
    tail -n 40 "$DEST_DIR/remote_job.log"
    cp "$DEST_DIR/remote_job.log" "$REPO_ROOT/remote_job.log"
fi

echo ""
echo "== next steps =="
echo "1. Retrain predictors for the new stacks:"
echo "   uv run python scripts/train_stacks.py --predictors --stacks raw-pixel,tiny-cnn-ae,tiny-cnn-pred,dinov2-s14,vjepa2-vitl"
echo "2. Rerun the report cards including the new stacks:"
echo "   uv run python scripts/run_report_card.py --stacks raw-pixel,tiny-cnn-ae,tiny-cnn-pred,dinov2-s14,vjepa2-vitl"
echo "   uv run python scripts/run_mechanistic.py --stacks raw-pixel,tiny-cnn-ae,tiny-cnn-pred,dinov2-s14,vjepa2-vitl"
