#!/usr/bin/env bash
# Runs ON the GPU box (invoked by EC2 user-data). Defensive: any failure
# uploads the log to S3 before shutdown, and a hard 240-minute shutdown
# backstop is scheduled at the very start regardless of outcome, so a wedged
# job can never burn spot-instance money indefinitely.
set -euo pipefail

# No --profile on the box itself: it authenticates via the instance profile
# attached by launch_gpu.sh, not a named CLI profile. Region is still
# explicit on every call.
REGION="${REGION:-eu-west-1}"
BUCKET="${BUCKET:?BUCKET env var must be set (passed by launch_gpu.sh user-data)}"

WORKDIR="/opt/physics-auditor"
REPO_DIR="$WORKDIR/repo"
LOG_FILE="$WORKDIR/remote_job.log"
mkdir -p "$WORKDIR" "$REPO_DIR"
exec > >(tee -a "$LOG_FILE") 2>&1

echo "== cost backstop: hard shutdown scheduled at +240 min regardless of outcome =="
sudo shutdown -h +240 || true

upload_log() {
    aws s3 cp "$LOG_FILE" "s3://${BUCKET}/results/remote_job.log" --region "$REGION" || true
}

on_error() {
    local exit_code=$?
    echo "== FAILED (exit $exit_code) at $(date -u +%FT%TZ) -- uploading log before shutdown =="
    upload_log
    sudo shutdown -h now || true
    exit "$exit_code"
}
trap on_error ERR

t0=$(date +%s)
log_elapsed() { echo "[+$(( $(date +%s) - t0 ))s] $*"; }

log_elapsed "== download + untar repo =="
aws s3 cp "s3://${BUCKET}/repo/physics-auditor.tar.gz" "$WORKDIR/physics-auditor.tar.gz" --region "$REGION"
tar -xzf "$WORKDIR/physics-auditor.tar.gz" -C "$REPO_DIR"
cd "$REPO_DIR"

log_elapsed "== install uv =="
curl -LsSf https://astral.sh/uv/install.sh | sh
export PATH="$HOME/.local/bin:$PATH"

log_elapsed "== uv sync --group gpu =="
uv sync --group gpu

log_elapsed "== pytest -q (fail-fast: red tests abort the job) =="
uv run pytest -q

log_elapsed "== smoke test: DINOv2 (one clip end-to-end) =="
uv run python -c "
from physics_auditor.generator.dataset import train_clips
from physics_auditor.models.pretrained import DINOv2Encoder
clip = next(iter(train_clips()))
enc = DINOv2Encoder().load()
z = enc.encode(clip)
print('dinov2-s14 smoke:', z.shape, z.dtype)
"

log_elapsed "== smoke test: V-JEPA-2 (one clip end-to-end) =="
uv run python -c "
from physics_auditor.generator.dataset import train_clips
from physics_auditor.models.pretrained import VJEPA2Encoder
clip = next(iter(train_clips()))
enc = VJEPA2Encoder().load()
z = enc.encode(clip)
print('vjepa2-vitl smoke:', z.shape, z.dtype)
"

log_elapsed "== populate latent caches: train(100..131)/val(200..207)/eval(0..15, all 4 laws) =="
uv run python scripts/train_stacks.py --stacks dinov2-s14,vjepa2-vitl

log_elapsed "== populate probe latent caches: probe-train(300..331)/probe-test(400..415) via mechanistic run =="
uv run python scripts/run_report_card.py --stacks dinov2-s14,vjepa2-vitl
uv run python scripts/run_mechanistic.py --stacks dinov2-s14,vjepa2-vitl

log_elapsed "== tar cache/ + logs, upload to S3 =="
tar -czf "$WORKDIR/cache.tar.gz" -C "$REPO_DIR" cache
tar -czf "$WORKDIR/artifacts.tar.gz" -C "$REPO_DIR" artifacts
aws s3 cp "$WORKDIR/cache.tar.gz" "s3://${BUCKET}/results/cache.tar.gz" --region "$REGION"
aws s3 cp "$WORKDIR/artifacts.tar.gz" "s3://${BUCKET}/results/artifacts.tar.gz" --region "$REGION"
upload_log

log_elapsed "== done -- shutting down (spot terminates via instance-initiated-shutdown-behavior=terminate) =="
sudo shutdown -h now
