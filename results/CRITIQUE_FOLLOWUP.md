# Follow-up on an external critique of the 1.08x/1.09x E7 result

A detailed critique (via another model) argued: (1) beating MWPM isn't the paper's
actual claim and isn't necessary, (2) the E2 gate failing worse than E7 succeeds is
diagnostic of a real problem, not just "surgery is harder," (3) receptive field vs.
graph diameter might explain it, (4) shot count might be a bottleneck (rare-event
tail), (5) the 1.08x gap might not be statistically separable from 1.0, (6) E6 at
0.30 is the actually-concerning number for H3. Checked each empirically. Session
ended (laptop shutdown) before all follow-ups completed — see
`LR_INSTABILITY_DIAGNOSTIC.md` for the most important unfinished thread.

## 1. Receptive field vs. graph diameter — checked, ruled out

Computed actual shortest-path diameters (networkx) on real sampled graphs at
p=0.002 (sub-threshold, sparse syndromes):

- Memory (E2) graphs: mean diameter 1.08, max 3, over 285 measured components.
- Surgery (k=1) graphs: mean diameter 1.42, max 8, over 1518 components; only
  0.2% exceed diameter 6.

The model has 6 message-passing layers (6-hop receptive field). At this
sub-threshold operating point, sparse syndromes produce small connected
components — diameter is not the bottleneck. (Might matter near/above threshold
where syndromes are denser; not the reported operating regime.)

## 2. Shot count for E2 — real effect, doesn't close the gap

10x shots (300,000 train/test vs. the usual 30,000), same hyperparameters
(hidden_dim=128, lr=1e-3, edge_dim=5), 3 seeds:

| shots | GNN | MWPM | ratio |
|---|---|---|---|
| 30,000 | 0.0013 | 0.0007 | 1.86x |
| 300,000 | 0.001409 | 0.000858 | **1.64x** |

Real, consistent movement in the predicted direction. Also corrected a factual
point in the critique: 30,000 shots at p=0.002 on the actual E2 circuit gives
**672** true-positive training examples (measured directly), not ~21 — that
number conflated MWPM's own residual error count with the number of positive
training labels available to the GNN. Still a real effect; just not as extreme
as "learning from 21 examples" would imply. Not pushed further (e.g. 100x) —
would need real time budget to test whether it eventually closes the gap or
plateaus.

## 3. Is the E7 1.08x gap separable from 1.0? — yes, confirmed

Paired bootstrap (same 30,000 test shots decoded by both GNN and MWPM, 10,000
resamples) on E7 spacelike, fresh model trained at current best config:

```
GNN error rate: 0.05387   MWPM error rate: 0.04897
95% CI on (GNN - MWPM):  [0.0029, 0.0069]   (entirely above zero)
P(diff <= 0) across resamples: 0.0
```

The gap is real and statistically significant, not noise — narrower than the
original ~2.3-2.8x gap, but genuinely present. The unpaired mean+-std comparisons
used elsewhere in this repo (e.g. `results/main_table.md`) are less powerful than
this paired test; worth redoing for other headline comparisons if this repo is
picked up again.

## 4. E6/H3 — real methodological bug found, partial fix found, NOT fully resolved

See `LR_INSTABILITY_DIAGNOSTIC.md` for the full story. Short version: E6 with
only 2 swept configs always trains on exactly 1 config (no real domain
randomization happens at all), and separately, `lr=1e-3` (validated only on k=1
data) causes real training instability/failure on k=2/k=3-only training data
(near-chance train-set performance, not a generalization issue). A lower lr
(3e-4) helps but an 8-seed confirmation run was interrupted at n=5 with mixed
results (not a clean fix). **This also raises a real, unresolved question about
whether the H4 threshold sweep's "GNN shows no distance-scaling advantage"
finding is itself confounded by this same k>=2 training instability** — the
GNN's k=2 curve in `results/timelike_threshold.png` shows the same "jumps to
chance and flatlines" signature as the confirmed training failures here. This
was NOT rechecked before the session ended. Treat the current H4 "negative"
finding in `LIMITATIONS.md`/`paper/main.tex` as unconfirmed pending this recheck,
not as settled.

## 5. Boundary-weight edge feature — tested, reverted (see `EDGE_FEATURE_CHECK.md` era commits)

A second architecture idea (DEM "boundary"/weight-1 error mechanisms as a node
feature, since 246 of them carry real probability mass and were previously
completely invisible to the GNN) was implemented, tested at n=8 matched to
main-table scale, came in worse (0.0550 +/- 0.0029 vs. baseline 0.0523 +/- 0.0016),
and was reverted via `git checkout` before committing anything broken. Not
included in the shipped model. Worth retrying with a different encoding
(e.g. as an additional self-loop edge rather than a raw node feature) if
revisited, since the theoretical motivation (this probability mass is real and
currently discarded) is still sound even though this specific encoding didn't help.

## Priority if this repo is picked up again

1. Finish the lr instability investigation (item 4) — this is the one that could
   change conclusions already written into `LIMITATIONS.md`/the paper, not just
   add a new result.
2. Recheck the H4 threshold sweep with a k-appropriate learning rate.
3. Only then decide whether E6/H3 needs a genuinely redesigned metric (report
   per-excluded-configuration rather than pooled, as already flagged in
   `LIMITATIONS.md`) on top of the optimization fix.
