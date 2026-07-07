# GPU runbook: populating the pretrained-encoder latent caches

Populates the `dinov2-s14` and `vjepa2-vitl` latent caches on a spot
`g4dn.xlarge` in `eu-west-1` (account `652742769396`), then terminates
itself. Run every step below **from the laptop** unless marked otherwise.

## Cost / time expectations

- Spot `g4dn.xlarge` in `eu-west-1`: roughly **$0.16-0.53/hr** depending on
  spot pricing at request time.
- Expected total job time: **1-3 hours** (pytest + both smoke tests + cache
  population across train/val/eval/probe clip sets for both encoders).
- Expected cost per run: **roughly $0.20-1.60**.
- Hard backstop: `remote_job.sh` schedules `sudo shutdown -h +240` (4h) at
  the very start, so a wedged job cannot run away with cost.

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
