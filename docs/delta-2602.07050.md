# Delta vs arXiv 2602.07050

**Correct citation:** Joseph, Garrido, Balestriero, Kowal, Fel, Bakhtiari, Richards, Rabbat.
"Interpreting Physics in Video World Models." arXiv:2602.07050 (Feb 2026). ID confirmed correct —
no retitling found.

## 1. What the paper actually does

The paper is an **interpretability study of two frozen, pretrained video encoders**
(V-JEPA-2 at L/H/giant scale, VideoMAE-v2-G) — not a benchmark, not a new model, and not an
audit tool. Using linear and attentive-MLP probes over mean-pooled space-time patches, they
find a sharp mid-depth "Physics Emergence Zone" where motion direction becomes linearly
decodable via a circular (population-code) geometry, while scalar speed/acceleration are
available from early layers. They validate on IntPhys (real matched possible/impossible pairs),
a self-generated synthetic toy-ball set (Kubric, controlled direction/speed/acceleration), and
CLEVRER for generalization, with ImageNet/SSv2/shuffled-video as non-physics controls. Causally,
they do targeted attention-head ablation at the emergence zone (physics tasks degrade, ImageNet
does not) and iterative orthogonal-probe "steering" (patch a stack of probe directions to rotate
the represented motion angle; full-subspace steering hits <0.5° error vs >80° for single-direction
steering). Headline claim: physical variables are encoded as **distributed, task-specific
population codes**, not factorized physics-engine-style state, and causal control of direction
requires coordinated multi-direction intervention, not a single vector.

## 2. Capability comparison vs Physics Auditor

| Capability | 2602.07050 | Physics Auditor (our spec) |
|---|---|---|
| Controllable generator, minimal pairs identical-until-critical-frame | Partial — synthetic toy-ball set varies motion params, but no obey/violate pair sharing a critical frame; IntPhys pairs exist but are theirs, fixed, not ours to extend | Yes — engine-agnostic generator, our own primitive |
| Oracle/pixel validity sandwich | No — no oracle upper bound, no pixel-baseline lower bound reported | Yes — non-negotiable in spec §6 |
| Decodability (linear probe → GT variable) | Yes, extensively (this is their core method) | Yes, but treated as stage 1 of 2, not the finding itself |
| Causal intervention scored vs GT counterfactual | No — ablation shows degradation, steering hits a target angle; neither is scored against a generator's ground-truth counterfactual world | Yes — effect ratio vs GT counterfactual is the core metric |
| Placebo/random-direction control on intervention | No explicit placebo direction (ImageNet used as task-control, not a representation-space placebo) | Yes — non-negotiable, isolates "real handle" from "any poke perturbs" |
| Verdict trichotomy (genuine / shortcut / not-sensitive) | No — five representational hypotheses compared narratively, no per-law per-model verdict grid | Yes — is the falsifiable headline |
| Behavioural VoE with pixel-floor control | Uses IntPhys possible/impossible classification accuracy; no explicit pixel/flow floor subtraction | Yes — model-AUROC minus pixel-baseline-AUROC |
| Runtime monitor (mid-rollout law-break detection) | Not addressed | Stretch goal, our startup seed |
| Models covered | 2 (V-JEPA-2, VideoMAE-v2), pretrained only | 3 CPU stacks now; V-JEPA-2/DINO adapters planned |
| Law coverage | Motion (direction/speed/accel) + IntPhys's permanence/continuity/shape-constancy categories | Object permanence, support, solidity, gravity-continuity |

## 3. The delta statement

They already nail the mechanistic-decodability half of our stage 1 — layerwise probing of
physical variables in real pretrained video encoders is their whole paper, done more thoroughly
than our current CPU-stack decodability grid (population-code geometry, attention-head ablation,
orthogonal-probe steering are all more sophisticated readout tools than our ridge probes). What
they structurally cannot do, by design of their method, is close the causal-fidelity loop: they
never score an intervention against a generator's own ground-truth counterfactual world, never
run a placebo-controlled effect-ratio, never build obey/violate minimal pairs sharing a critical
frame, and never emit a per-(model, law) verdict distinguishing "genuinely used" from "decodable
but shortcut." Their steering result (rotate a direction to a target angle) is impressive but is
an existence proof of controllability, not a scored comparison against what the *physics itself*
would produce — there is no generator underneath IntPhys or the toy-ball set that they built or
own, so no minimal-pair/counterfactual protocol is possible without one. This is closer to 40%
overlap than 80%: they own the decodability/geometry insight for real pretrained encoders; we own
the controllable-generator, counterfactual-scored causal test and the verdict trichotomy — the
part of the spec that turns "physics is decodable" into "physics is used, provably, with a
control for false positives." Per spec §9's risk register, this is *not* the "they already nail
80%" scenario that would force a full pivot to the monitor axis — the causal/counterfactual/
verdict machinery is untouched territory. Two real risks worth weight-shifting for: (a) they
share an author (Garrido) with the SSL-intuitive-physics claim we cite, so expect their group to
extend toward causal work next — move faster on the intervention+verdict layer, not the
decodability layer, since that's where we're exposed; (b) their attention-ablation method is a
cheaper, real-pretrained-model alternative to our trained-predictor-on-synthetic-worlds core —
worth a stretch comparison once V-JEPA-2 access exists, but does not change the floor plan.

## 4. Sources

- [arXiv:2602.07050 abstract](https://arxiv.org/abs/2602.07050)
- [arXiv:2602.07050 HTML (full text)](https://arxiv.org/html/2602.07050v1)
- [arXiv:2602.07050 PDF](https://arxiv.org/pdf/2602.07050)
- [Sonia Joseph, author page/blog post on the paper](https://www.soniajoseph.ai/interpreting-ph/)
- [Paper note issue, AkihikoWatanabe/paper_notes #5740](https://github.com/AkihikoWatanabe/paper_notes/issues/5740)
