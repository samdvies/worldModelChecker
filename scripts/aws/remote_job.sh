#!/usr/bin/env bash
# Runs ON the GPU box (invoked by EC2 user-data). Defensive: any failure
# uploads the log to S3 before shutdown, and a hard 240-minute shutdown
# backstop is scheduled at the very start regardless of outcome, so a wedged
# job can never burn spot-instance money indefinitely.
set -euo pipefail

# user-data runs as root with no login environment: HOME may be unset, and
# under `set -u` an unbound variable aborts WITHOUT firing the ERR trap.
export HOME="${HOME:-/root}"

# No --profile on the box itself: it authenticates via the instance profile
# attached by launch_gpu.sh, not a named CLI profile. Region is still
# explicit on every call.
REGION="${REGION:-eu-west-1}"

WORKDIR="/opt/physics-auditor"
REPO_DIR="$WORKDIR/repo"
LOG_FILE="$WORKDIR/remote_job.log"
mkdir -p "$WORKDIR" "$REPO_DIR"
exec > >(tee -a "$LOG_FILE") 2>&1

# IMPORTANT (class C, see docs/failure-sweeps.md): the ERR trap and the cost
# backstop MUST be set up before ANY command that can fail -- including
# `${BUCKET:?...}` below. Parameter-expansion failures like `:?` abort a
# `set -u` script WITHOUT firing the ERR trap, so if BUCKET is unset/empty
# and this check ran before the trap/backstop existed, the instance would
# keep running with no automatic shutdown and no log ever reaching S3.
echo "== cost backstop: hard shutdown scheduled at +240 min regardless of outcome =="
sudo shutdown -h +240 || true

upload_log() {
    aws s3 cp "$LOG_FILE" "s3://${BUCKET:-unknown-bucket}/results/remote_job.log" --region "$REGION" || true
}

# Class G (see docs/failure-sweeps.md): a prior run re-encoded ~2,290 clips'
# worth of latents that were then LOST when the job was killed, because
# nothing synced partial progress to S3 until the very end. SYNC_PID tracks
# the background partial-sync loop (started below, once BUCKET is known) so
# both the success path and on_error can cleanly stop it and push one final
# sync -- initialised here (before the loop exists) so on_error can safely
# reference it even if a failure happens before the loop is started.
SYNC_PID=""

stop_partial_sync_loop() {
    if [ -n "$SYNC_PID" ]; then
        kill "$SYNC_PID" >/dev/null 2>&1 || true
        wait "$SYNC_PID" 2>/dev/null || true
        SYNC_PID=""
    fi
    # Final sync regardless of whether the loop had even started -- cheap
    # and idempotent (aws s3 sync only pushes new/changed files), and it's
    # the last chance to persist any cache/ work before the box terminates.
    aws s3 sync "$REPO_DIR/cache" "s3://${BUCKET:-unknown-bucket}/results/cache-partial" --region "$REGION" || true
}

on_error() {
    local exit_code=$?
    echo "== FAILED (exit $exit_code) at $(date -u +%FT%TZ) -- syncing partial cache + uploading log before shutdown =="
    stop_partial_sync_loop
    upload_log
    sudo shutdown -h now || true
    exit "$exit_code"
}
trap on_error ERR

# BUCKET has no safe default (it names the S3 bucket everything is
# read/written from), so an unset/empty value is a hard failure -- but as a
# NORMAL command (`if`/`echo`/`exit`), not a `${VAR:?}` parameter-expansion
# abort, so it triggers the ERR trap above like any other failure instead of
# bypassing it.
if [ -z "${BUCKET:-}" ]; then
    echo "== FAILED: BUCKET env var must be set (passed by launch_gpu.sh user-data) =="
    false
fi

# Class G (see docs/failure-sweeps.md): resumability + redundant recompute.
# Started right after the ERR trap/backstop are armed and BUCKET is known
# to be valid, so ANY subsequent failure (including one before the repo is
# even downloaded) still gets a partial-cache push via stop_partial_sync_loop
# in on_error above. `$REPO_DIR/cache` may not exist/be empty yet at this
# point -- `mkdir -p` + an empty-source `aws s3 sync` are both no-ops, so
# starting this early is safe.
SYNC_INTERVAL_S="${SYNC_INTERVAL_S:-600}"
mkdir -p "$REPO_DIR/cache"

start_partial_sync_loop() {
    (
        while true; do
            sleep "$SYNC_INTERVAL_S"
            aws s3 sync "$REPO_DIR/cache" "s3://${BUCKET}/results/cache-partial" --region "$REGION" || true
        done
    ) &
    SYNC_PID=$!
    echo "== partial-sync loop started (pid $SYNC_PID, every ${SYNC_INTERVAL_S}s) =="
}
start_partial_sync_loop

t0=$(date +%s)
log_elapsed() { echo "[+$(( $(date +%s) - t0 ))s] $*"; }

log_elapsed "== download + untar repo =="
aws s3 cp "s3://${BUCKET}/repo/physics-auditor.tar.gz" "$WORKDIR/physics-auditor.tar.gz" --region "$REGION"
tar -xzf "$WORKDIR/physics-auditor.tar.gz" -C "$REPO_DIR"
cd "$REPO_DIR"

log_elapsed "== pre-seed cache/ from any previously-synced partial results (resumability, class G) =="
aws s3 sync "s3://${BUCKET}/results/cache-partial" "$REPO_DIR/cache" --region "$REGION" || true

log_elapsed "== install uv =="
curl -LsSf https://astral.sh/uv/install.sh | sh
export PATH="$HOME/.local/bin:$PATH"

log_elapsed "== uv sync --group gpu =="
uv sync --group gpu

log_elapsed "== cuda gate (class I): refuse to run on CPU wheels =="
uv run python -c "
import torch
assert torch.cuda.is_available(), f'CUDA unavailable -- torch {torch.__version__} is a CPU wheel; see docs/failure-sweeps.md class I'
print('cuda OK:', torch.__version__, torch.cuda.get_device_name(0))
"

log_elapsed "== pytest -m smoke (fast torch-free pre-flight -- fail fast in seconds, not minutes) =="
uv run pytest -m smoke -q tests/test_smoke.py

log_elapsed "== pytest -m smoke_torch (torch-backed pre-flight, mock-injected models only) =="
uv run pytest -m smoke_torch -q tests/test_smoke_torch.py

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

log_elapsed "== populate latent caches: train(100..131)/val(200..207)/eval(0..15, all 4 laws) + probe caches (train 300..331/test 400..415) =="
uv run python scripts/train_stacks.py --stacks dinov2-s14,vjepa2-vitl --probe-caches

log_elapsed "== stop partial-sync loop + push final cache-partial sync =="
stop_partial_sync_loop

log_elapsed "== tar cache/ + logs, upload to S3 =="
tar -czf "$WORKDIR/cache.tar.gz" -C "$REPO_DIR" cache
aws s3 cp "$WORKDIR/cache.tar.gz" "s3://${BUCKET}/results/cache.tar.gz" --region "$REGION"
upload_log

log_elapsed "== done -- shutting down (spot terminates via instance-initiated-shutdown-behavior=terminate) =="
sudo shutdown -h now
