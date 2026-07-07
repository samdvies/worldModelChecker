# GPU runbook: populating the pretrained-encoder latent caches

Populates the `dinov2-s14` and `vjepa2-vitl` latent caches on a spot
`g4dn.xlarge` in `eu-west-1` (account `652742769396`), then terminates
itself. Run every step below **from the laptop** unless marked otherwise.

## Cost / time expectations

- Spot `g4dn.xlarge` in `eu-west-1`: roughly **$0.16-0.53/hr** depending on
  spot pricing at request time.
- **V-JEPA-2 throughput (class G, see docs/failure-sweeps.md):** measured
  ~40s/clip fp32 unbatched on a T4 -- at that rate, 672 clips x 2 encoders
  needs 10+ hours, which is why the fp16 + batching work in
  `physics_auditor/models/pretrained.py` exists. With `.load()`'s fp16 cast
  and `VJEPA2Encoder.encode_batch` (default batch_size=4 on cuda) the
  expected per-clip cost drops substantially (fp16 typically ~2x, batching
  of 4 typically another ~2-3x on a T4 for a ViT-L-sized model) -- treat
  this as a rough **~5-8x** combined speedup estimate, not a measured
  number, until confirmed against a real T4 run. DINOv2 is comparatively
  cheap (ViT-S, already frame-batched) and was not the bottleneck.
- Expected total job time: **1-3 hours** on a *resumed* run picking up from
  a prior `cache-partial/` sync (see Resume semantics below); a cold run
  with no partial cache and the old fp32/unbatched V-JEPA-2 path could take
  10+ hours -- always check whether `s3://physics-auditor-<account>/results/cache-partial/`
  already has content before assuming a fresh multi-hour budget.
- Expected cost per run: **roughly $0.20-1.60** for a resumed/fp16-batched
  run; budget more for a cold run against the old (fixed) per-clip cost.
- Hard backstop: `remote_job.sh` schedules `sudo shutdown -h +240` (4h) at
  the very start, so a wedged job cannot run away with cost.

## Resume semantics + cache-partial layout (class G)

`remote_job.sh` is now resumable by construction:

1. **Pre-seed:** immediately after repo untar (before any encoding step),
   it runs `aws s3 sync s3://$BUCKET/results/cache-partial ./cache`. Since
   `models/cache.py`'s `encode_clip_cached` already skips any `.npy` that
   already exists on disk, every clip already cached from a prior run (or
   from the committed local stacks) is free -- nothing is re-downloaded or
   re-encoded.
2. **Background partial-sync loop:** started right after the ERR trap and
   cost backstop are armed and `BUCKET` is validated (before the repo is
   even downloaded), it pushes `./cache -> s3://$BUCKET/results/cache-partial`
   every `SYNC_INTERVAL_S` seconds (default 600 = 10 min). This used to be
   a manual `aws s3 sync` loop bolted on ad hoc via SSM mid-run -- it is now
   part of the script itself.
3. **Stop + final sync:** both the normal success path and `on_error` call
   a shared `stop_partial_sync_loop` (kill the background loop, then push
   one last sync) before shutdown, so at most `SYNC_INTERVAL_S` seconds of
   work (rather than the whole job) can ever be lost to a kill or crash.

**Layout:** `cache-partial/<encoder_cache_key>/<scenario_id>.npy`, e.g.
`cache-partial/dinov2-s14-74d24787/deadbeef00.npy`. `<encoder_cache_key>` is
`{name}-{sha1(state_dict bytes)[:8]}` (see `models/pretrained.py`), so a
retrained/re-fingerprinted encoder gets its own subdirectory rather than
silently mixing latents from two different weight versions. After a full
successful run, `cache-partial/` should mirror `cache/` inside
`results/cache.tar.gz` for the requested stacks.

If a run is killed or fails partway through, the NEXT dispatch (same or
different `--stacks`) will pick up automatically from whatever
`cache-partial/` holds -- no manual intervention needed beyond re-running
`launch_gpu.sh`.

## Mandatory pre-launch sequence (do all three, in order, every time)

These are cheap (seconds to a few minutes) and exist specifically to avoid
paying for a ~10-minute box boot only to hit something a local check could
have caught (see `docs/failure-sweeps.md` for the incidents that motivated
each one):

1. **Full local suite green:**
   ```
   uv run pytest -q
   ```
2. **Fresh-clone smoke gate** (class E: catches tests that quietly depend on
   untracked local state like `artifacts/`/`cache/` -- simulates a clean
   checkout via `git archive HEAD` into a scratch dir and runs the smoke
   tier there; commit any new test files first, since `git archive` only
   sees committed content):
   ```
   uv run python scripts/fresh_clone_smoke.py
   ```
3. **launch_gpu.sh preflight** (class D: cheaply verifies AMI resolution,
   instance-type region-offering, and bucket reachability WITHOUT creating
   any resources):
   ```
   scripts/aws/launch_gpu.sh --preflight
   ```

Only proceed to the steps below once all three pass.

## Order of operations

1. **Dry run first, always.** Confirms credentials and prints every AWS CLI
   call without executing anything:
   ```
   scripts/aws/launch_gpu.sh --dry-run
   ```
   Review the printed commands (AMI resolution, bucket/IAM/instance
   creation, the run-instances call) before proceeding.

2. **Launch for real:**
   ```
   scripts/aws/launch_gpu.sh
   ```
   Add `--on-demand` if spot capacity is unavailable (`InsufficientInstanceCapacity`).
   This tars the repo (`git archive HEAD`), uploads it + `remote_job.sh` to
   `s3://physics-auditor-652742769396/`, then requests the instance. No
   inbound ports are opened -- the box is reachable only via SSM Session
   Manager.

3. **Check the job is running (from the laptop, via SSM):**
   ```
   aws ssm start-session --target <instance-id> --profile claude-admin --region eu-west-1
   ```
   Then, on the box:
   ```
   tail -f /opt/physics-auditor/remote_job.log
   ```
   The log should show, **in this order**: repo download, `uv sync --group
   gpu`, `pytest -m smoke` (fast fail in seconds if the wiring broke),
   `pytest -q` (job aborts here if red), the DINOv2 smoke test, the
   V-JEPA-2 smoke test, then cache population for train/val/eval/probe clip
   sets. Every step is timestamped with elapsed seconds so you can sanity
   check the cost as it runs.

4. **Kill switch** (job wedged, wrong instance, or you just want to stop
   paying): from the laptop,
   ```
   aws ec2 terminate-instances --instance-ids <instance-id> --profile claude-admin --region eu-west-1
   ```
   This is safe at any point -- the instance never holds anything that
   isn't already uploaded to S3 as it goes.

5. **Pull results down** once the box has terminated (job succeeded or
   failed -- the log is always uploaded before shutdown):
   ```
   scripts/aws/pull_results.sh
   ```
   This syncs `cache.tar.gz`, `artifacts.tar.gz`, and `remote_job.log` from
   `s3://physics-auditor-652742769396/results/`, untars the cache and
   artifacts into the repo, and prints the last 40 lines of the log. If
   `cache.tar.gz` is missing, the job failed before completing -- read the
   log for the `FAILED (exit ...)` line.

6. **After pulling results,** retrain the predictors for the new stacks and
   rerun the report cards (printed by `pull_results.sh`, repeated here):
   ```
   uv run python scripts/train_stacks.py --predictors --stacks raw-pixel,tiny-cnn-ae,tiny-cnn-pred,dinov2-s14,vjepa2-vitl
   uv run python scripts/run_report_card.py --stacks raw-pixel,tiny-cnn-ae,tiny-cnn-pred,dinov2-s14,vjepa2-vitl
   uv run python scripts/run_mechanistic.py --stacks raw-pixel,tiny-cnn-ae,tiny-cnn-pred,dinov2-s14,vjepa2-vitl
   ```

## Notes

- `launch_gpu.sh` and `pull_results.sh` always pass `--profile claude-admin
  --region eu-west-1` explicitly on every AWS CLI call.
- `remote_job.sh` runs ON the box and authenticates via its attached
  instance profile (SSM Core + scoped S3 access to
  `physics-auditor-652742769396` only) -- it never needs a named CLI
  profile or any embedded credentials.
- No weights are ever downloaded on the laptop. `DINOv2Encoder.load()` and
  `VJEPA2Encoder.load()` only run on the GPU box, inside `remote_job.sh`.
- IAM role/instance-profile/S3-bucket creation in `launch_gpu.sh` is
  idempotent -- safe to rerun `launch_gpu.sh` for subsequent jobs.
- `PROFILE`/`REGION`/`INSTANCE_TYPE`/IAM/SG names live in one place,
  `scripts/aws/_env.sh`, sourced by both `launch_gpu.sh` and
  `pull_results.sh`. `ACCOUNT_ID` is never hardcoded -- both scripts derive
  it live via `aws sts get-caller-identity` so they can never silently
  drift onto different buckets (class D, see `docs/failure-sweeps.md`).
- **`docs/gallery/index.html` changes need a live browser pass, not just a
  static check.** Ordering/hoisting bugs in the page's JS (class F) do not
  reliably surface via `node --check`, a text read, or even repeated static
  review -- only live execution caught the one instance of this hit so far,
  and it needed multiple page loads (cold + warm cache) to reproduce. After
  any edit to `docs/gallery/index.html`, load it in Chrome (e.g. via the
  claude-in-chrome tooling) a few times and check the console for errors
  before treating the change as done. `tests/test_gallery_js_ordering.py`
  pins the specific known-bad ordering pattern but is not a substitute for
  this.
