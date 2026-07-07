# Physics Auditor

A controllable-generative causal test-bench for video world models: it
generates minimal pairs of physics clips (one obeying a physical law, one
violating it at a known critical frame) and measures whether a model's
predictions are surprised by the violation. This is the feasibility slice —
one law (support/stability), 2D PyMunk physics, and a perfect oracle model
closing the loop end-to-end.

## Repo layout

- `physics_auditor/generator/` — PyMunk-backed world simulation: state types,
  scenario config, the `Engine` (steps physics, extracts support edges from
  real contacts, can ghost a block), rendering to 64x64 RGB frames, and
  `Clip`/`simulate` to produce a full frame+state sequence.
- `physics_auditor/laws/` — `Law` protocol and `MinimalPair`; `SupportLaw`
  builds a 3-block tower and generates an obey/violate pair (violate ghosts
  the middle block at the critical frame).
- `physics_auditor/models/` — `ModelAdapter` protocol and `OracleAdapter`, a
  perfect-physics model that rolls the true engine forward from the
  pre-critical-frame state and scores surprise as accumulated position error.
- `physics_auditor/probes/behavioural/` — `auroc` metric and `run_voe`, the
  violation-of-expectation probe that scores obey vs. violate clips and
  reports separability.
- `physics_auditor/probes/mechanistic/` — `data.py`/`labels.py` build
  probe-train/probe-test latent datasets and physical-variable labels;
  `decodability.py` runs ridge read-out probes; `intervention.py` runs
  concept-patch vs. random-direction placebo interventions with bootstrap
  CIs. Feeds `artifacts/mechanistic_scores.json` and the v2 report card.
- `physics_auditor/report/` — `ReportCard`, a tabular results accumulator
  with markdown rendering.
- `physics_auditor/monitor/` — live-monitoring logic (see its own module
  docs for current scope).
- `artifacts/findings.md` — the findings writeup: validity scaffold, per-law
  narrative, mechanistic verdicts, limitations, fallback log, and next
  steps, grounded in `artifacts/{report_card.md,report_card_v2.md,
  voe_scores.json,mechanistic_scores.json}`.

## Running

Tests:

```
uv run pytest
```

Feasibility demo (prints the report card and per-pair surprise scores, and
saves `artifacts/obey.png` / `artifacts/violate.png` filmstrips of the
seed-0 minimal pair):

```
uv run python scripts/run_feasibility.py
```

Other scripts: `scripts/train_stacks.py` trains the three latent stacks
(raw-pixel, tiny-cnn-ae, tiny-cnn-pred) and populates the latent cache;
`scripts/run_report_card.py` runs the behavioural VoE probes and writes
`artifacts/{report_card.md,voe_scores.json}`; `scripts/run_mechanistic.py`
runs the decodability + intervention probes and writes
`artifacts/{report_card_v2.md,mechanistic_scores.json,
mechanistic_intervention_scores.json}`.
