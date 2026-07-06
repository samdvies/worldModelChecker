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
- `physics_auditor/probes/mechanistic/` — reserved for future internal-state
  probes (empty in this slice).
- `physics_auditor/report/` — `ReportCard`, a tabular results accumulator
  with markdown rendering.
- `physics_auditor/monitor/` — reserved for future live-monitoring logic
  (empty in this slice).

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
