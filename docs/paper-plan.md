# Getting past review: objection-driven roadmap

Source: adversarial review of the current bench (2026-07-08). Each workstream names the
objection it kills, the concrete deliverable, an acceptance gate decided BEFORE running,
and rough cost. Ordered by leverage, not difficulty. P0 is already in flight.

The standing honesty rule applies throughout: a gate that fails is a reported finding,
not a reason to tune until it passes.

---

## P0 — statistical + structural hardening (IN FLIGHT)

**Kills:** "half the grid can't discriminate; n=16; ceiling effects" (objection 4),
part of "headline is confounded" (objection 2).

Already being built: eval n=16 -> 64 with paired-bootstrap 95% CIs; `permanence-ext`
(64-frame clips, re-emergence ~frame 30, >=25 frames of visible evidence) so per-frame
stacks get a fair shot at permanence; `support-hard` (pixel-decoy tower collapsing in
both clips) so the support row stops being pixel-trivial; the
`vjepa2-vitl-causal-w16`/`-w32` memory-horizon sweep (sliding-window encode,
causally valid per-frame latents: w16 = window < occlusion, an intentional
negative control; w32 = window >= occlusion, the fair test).

**Gates:**
- causal V-JEPA-2 permanence VoE >= 0.8 -> headline survives ("permanence sensitivity is
  not future-frame leakage"). < 0.6 -> reframe headline as "whole-clip change detection";
  both outcomes are reportable, only one is a capability claim.
- support-hard pixel floor <= 0.85 while oracle = 1.0, else iterate geometry.
- CI half-widths shrink enough that at least the headline cells separate from their
  pixel floor with non-overlapping intervals.

**Cost:** 3 subagent lanes (running) + one GPU box run (~1-2 h, ~$1).

---

## P1 — native VoE: audit V-JEPA-2's OWN predictor

**Kills:** "you didn't audit a world model" (objection 1) — the fatal one — at its
cheapest point of attack.

The HF checkpoint we already run (`facebook/vjepa2-vitl-fpc64-256`) contains the
V-JEPA-2 predictor: model outputs expose `predictor_output.last_hidden_state` next to
the encoder's `last_hidden_state`. Meta trained that predictor; we did not. Its masked-
prediction error on our minimal pairs is a real deployed model's own surprise.

**Deliverable:** a `native-voe` adapter: context = tokens up to the critical frame,
predict masked post-critical tubelets, surprise_t = prediction error at t (in the
model's own latent space). New report-card column `vjepa2-native`. Direct comparison
row: native surprise vs our trained-probe surprise, same pairs, same AUROC/CI harness.

**Gate (pre-registered):** if native VoE ranks laws in the same order as probe VoE, the
probe methodology is validated ("our cheap probes predict what the real predictor
knows"). If it diverges, that IS the paper's most interesting table either way.

**Design notes:**
- masking scheme: mask everything after critical_frame; context tubelets unmasked.
  Check how HF exposes predictor masking (the fine-tuning notebook on the model card
  shows the API); if the public head only supports its pretraining mask pattern,
  fall back to per-tubelet leave-future-out masking and document.
- runs on the existing GPU pipeline (encoder+predictor forward fits a T4 at fp16).
- Also unlocks a NATIVE monitor row (model's own error as the runtime signal).

**Cost:** ~1 subagent lane + 1 GPU run. Days, not weeks. Do this before anything below.

## P2 — nuisance-factor render axes

**Kills:** "one renderer, zero visual diversity; shortcuts everywhere" (objection 6),
half of "2D toy says nothing" (objection 3).

**Deliverable:** renderer variants as controlled axes, same physics/seeds: (a) textured
sprites, (b) non-uniform backgrounds, (c) 128px + anti-aliasing, (d) small camera
jitter (same jitter in obey/violate — nuisance, not signal). Verdict grid recomputed
per render domain. Headline claim becomes: "verdicts are (in)stable across visual
domains" — a robustness statement no single-domain benchmark can make.

**Gate:** a verdict cell keeps its label across >= 3 of 4 domains -> "stable"; flips ->
reported as domain-fragile (and the shortcut machinery should say why).

**Extra credit (verdict-validity experiment):** plant a deliberate shortcut (e.g. a
pixel-count-correlated artifact in violate clips of one domain) and show the
trichotomy correctly downgrades genuine -> shortcut. A benchmark that demonstrates its
own failure-detection works is much harder to dismiss.

**Cost:** renderer work is local/CPU; re-encodes for pretrained stacks = 1 GPU run per
domain (batch them into one box session). ~1 week of lanes.

## P3 — audit one actual generative rollout model

**Kills:** the rest of objection 1; makes the title honest at the flagship level.

**Deliverable:** one open generative world model rolled out from the pre-critical
prefix; surprise = divergence between its rollout distribution and the observed
continuation (obey vs violate). Candidates, in order of preference:
1. NVIDIA Cosmos (world-foundation-model framing matches ours; open weights),
2. LTX-Video / CogVideoX (fast open video gen, conditioning-friendly),
3. SVD (image-conditioned only — weakest fit).
Selection criterion: must accept >= 8-frame video conditioning (prefix-conditioned
rollout is the whole point). Scoring: per-frame distance between rollout and ground
truth continuation, aggregated like every other stack -> it becomes just another
report-card row with the same CIs and verdict machinery (mechanistic layer optional
for this stack; behavioural row alone already answers objection 1).

**Cost:** the real one. Needs a bigger box (g5.xlarge A10G 24GB or g6e, ~$1-2/h,
hours). New adapter lane + prompt-conditioning plumbing. Schedule after P1/P2; if time
runs out before the FYP deadline, P1 alone already blunts objection 1.

## P4 — probe-robustness bundle

**Kills:** "standard probing with known illusions" (objection 5).

**Deliverable:** three additions, all cheap on frozen caches:
1. nonlinear probe agreement: rerun decodability with a 2-layer MLP probe; report
   linear-vs-nonlinear verdict agreement matrix.
2. latent-patching intervention: instead of nudging along a diff-of-means direction,
   transplant the concept subspace of a violate latent into its paired obey latent
   (activation-patching style) and require predictor surprise to follow the patch.
   Stronger causal evidence than direction-nudging; keep the norm-matched placebo.
3. seed variance: 5 probe seeds x 3 predictor seeds per cell -> error bars on the
   verdict itself (report verdict flip-rate, not just score variance).

**Gate:** verdicts stable across probe class and seeds -> "genuine" survives scrutiny.

**Cost:** local CPU lanes only. No GPU. Good parallel filler while GPU runs elsewhere.

## P5 — external anchor (sim-to-real bridge)

**Kills:** the remainder of objection 3.

**Deliverable:** run the SAME VoE harness (native + probe) on a public IntPhys-style
dev block (real or photoreal videos, O1 permanence tasks map directly). One scatter
plot: our synthetic per-law scores vs external benchmark scores, per stack. Any
positive rank correlation is a bridge; even a null is a finding about synthetic
benchmarks generally (bigger claim than ours alone).

**Cost:** dataset download + adapter lane + 1 GPU encode run. After P1 (reuses its
native-VoE machinery).

## P6 — grid breadth (stretch)

Two more laws with the existing recipe (continuity: teleport behind occluder;
inertia: velocity flip without cause). Each is now ~1 lane + rides along on whatever
GPU run is next. Do opportunistically; stop adding rows once every column has at
least one discriminative cell per objection-4 gate.

---

## Sequencing

```
now      P0 lanes land -> integrate -> GPU run #1 (n=64 + new laws + causal stack)
next     P1 native-VoE lane        -> GPU run #2 (can merge into #1 if lanes are quick)
then     P2 render axes (local)    -> GPU run #3 (all domains batched)
         P4 probe bundle (local, parallel with any of the above)
later    P3 generative rollout (bigger box), P5 external anchor, P6 extra laws
```

Positioning note for the write-up (objection 5's novelty half): the delta vs
arXiv 2602.07050 after P1 is exactly: validity-first scaffold + verdict trichotomy +
NATIVE-VoE-validates-probe-VoE. Lead with that trio; the probing mechanics are
commodity and should be framed as such.
