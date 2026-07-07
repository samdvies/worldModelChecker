# Minimal-Pair Gallery — design note (benched)

*Parked 2026-07-07. Build after the pretrained-encoder (V-JEPA-2 / DINOv2) results exist.
Everything below generates from existing artifacts (`artifacts/*.json`, clip frames) — no
pipeline restructuring needed.*

## Pitch

Show, don't tell, the project's core move: two videos pixel-identical until a chosen frame,
then one quietly breaks a law of physics. Synced side-by-side playback with a critical-frame
marker delivers the "they were the same until *just now*" beat that no static figure can.

## Core layout (per experiment)

- **Two synced video panes** (obey | violate), shared scrubber, vertical marker at the
  critical frame. Play/pause/step controls.
- **Surprise traces** under the panes, drawn in sync with playback: one line per stack
  (oracle, pixel baselines, learned stacks) showing prediction error per frame.
  - Oracle spikes exactly at the critical frame (validity, visible).
  - Solidity punchline: the pixel baseline spikes on the *lawful bounce* and stays flat
    through the violation — the AUROC-0.000 inversion becomes a visible moment instead of a
    table cell.
- **Monitor flag**: a "smoke alarm" indicator that fires when the runtime monitor's
  calibrated threshold is crossed (shows detection latency visually).

## Per-law showcase notes

- **Permanence** is the interactive hook: ask the viewer to spot the violation frame — they
  can't (occluder hides it); reveal the oracle detected it instantly from state. One
  interaction that communicates "invisible to pixels, obvious to understanding".
- **Support/solidity/gravity**: straight side-by-side with traces.

## Build ladder (increasing effort)

1. **Animated GIFs** of pairs + surprise-trace strips — nearly free; embeddable in README,
   proposal email, interim slides.
2. **Static HTML gallery** (this doc's subject): synced player + scrubber + traces; no
   server; GitHub Pages. The supervisor screen-share demo. ~1 day.
3. **Interactive playground** (post-interim / OSS end-game): sliders for physics variables
   regenerating pairs live; "audit your model" upload; public leaderboard.

## Honesty constraint

The gallery must carry the same caveats as the report card (n=16 pairs, degenerate-score
cells, pinned thresholds) — it is a scientific artifact, not a sales page. Numbers shown on
screen come from the same JSONs the writeup cites.

## Side benefit

Scrubbing pairs with traces is the fastest generator-bug detector (e.g. the from_state
friction drift found in review would be instantly visible as a creeping tower).
