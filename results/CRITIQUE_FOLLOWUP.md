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

## 4. E6/H3 — real methodological bug found, partial fix confirmed, NOT fully resolved

See `LR_INSTABILITY_DIAGNOSTIC.md` for the full story, now updated with a
completed n=8 confirmation (the earlier n=5 was from a session interruption,
since resolved). Short version: E6 with only 2 swept configs always trains on
exactly 1 config (no real domain randomization happens at all), and
separately, `lr=1e-3` (validated only on k=1 data) causes real training
degradation on k=2/k=3-only training data. **Confirmed with the full n=8**:
lowering lr to 3e-4 cuts the k=1-excluded training failure roughly in half
(0.478 -> 0.342 mean train-set error) — a real, substantial, but partial fix.
Training on k=2/k=3 alone still converges meaningfully worse than training
with k=1 included (0.342 vs 0.136), so lr was not the whole story.

## 4b. H4 threshold-sweep recheck — lr was NOT the explanation there; budget is the leading suspect

Reran the k=2 GNN curve (`experiments/e7_threshold_sweep.py`'s own p-grid,
spacelike + timelike) at `lr=3e-4` instead of the original `1e-3`, first at
the threshold sweep's own (smaller) budget — 8000 train shots, 15 epochs:

| p | spacelike (lr=3e-4) | spacelike (original, lr=1e-3) | timelike (lr=3e-4) | timelike (original, lr=1e-3) |
|---|---|---|---|---|
| 0.0012 | 0.090 | 0.065 | 0.486 | 0.476 |
| 0.0025 | 0.490 | 0.482 | 0.495 | 0.502 |
| 0.005  | 0.501 | 0.488 | 0.499 | 0.504 |
| 0.01   | 0.504 | 0.499 | 0.500 | 0.508 |
| 0.02   | 0.494 | 0.495 | 0.485 | 0.509 |

**Essentially unchanged.** Lowering lr alone, at the threshold sweep's own
smaller shot/epoch budget, did NOT fix the "jumps to chance at p=0.0025 and
stays flat" pattern.

**Disambiguating follow-up**: reran k=2 spacelike at the *larger*, E6-matching
budget (15000 train/test shots, 20 epochs) + lr=3e-4 -- the same combination
that gave E6 its real (if partial) improvement:

| p | rate | for reference: MWPM at same (k=2, p) |
|---|---|---|
| 0.0012 | 0.0565 | 0.0068 |
| 0.0025 | 0.4875 | 0.0569 |
| 0.005  | 0.5011 | 0.2679 |
| 0.01   | 0.4995 | 0.4919 |
| 0.02   | 0.4987 | 0.4968 |

**Still essentially unchanged.** Neither lr nor shot/epoch budget, alone or
together, fixes k=2's near-chance performance at p=0.0025 or p=0.005.

**The critical read, checking against MWPM's own numbers**: at p=0.01 and
p=0.02, MWPM itself is already near chance (0.49-0.50) -- the underlying
decoding problem is too hard for *any* decoder there (well above k=2's
threshold), so the GNN being near chance at those points is not a meaningful
comparison and was never real evidence either way. But **at p=0.0025 and
p=0.005, MWPM clearly does non-trivial, real decoding (5.7% and 26.8% error)
while the GNN sits at chance (~49-50%) across all three configurations tested
here** (original lr=1e-3; lr=3e-4 at the small sweep budget; lr=3e-4 at the
larger E6-matching budget). This survived two different, real attempts at a
fix. **This is now the most defensible read of H4**: the GNN has a genuine,
specific difficulty learning k=2 spacelike decoding in the regime where the
problem is provably still solvable (MWPM proves it) -- not simply an
optimization/lr artifact that a hyperparameter tweak resolves, but also not
exhaustively investigated (more epochs at even lower lr, a from-scratch check
of the k=2 data pipeline for a subtle correctness bug, or architecture
changes specific to larger graphs were not tried). The original "no real
distance-scaling advantage" H4 finding stands, now on firmer ground than
before this recheck, but should be described as "not explained by the two
most likely optimization causes" rather than as a fully understood
architectural limit.

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

## Status: both threads now resolved to a stable conclusion

1. ~~Finish the lr instability investigation~~ — done: real, partial fix
   (lr=3e-4 roughly halves the k=1-excluded training failure) but not complete.
2. ~~Recheck the H4 threshold sweep~~ — done: neither lr nor shot/epoch budget
   fixes it at the p points that actually matter (0.0025, 0.005, where MWPM
   proves the problem is solvable). H4's negative finding is now on firmer
   ground, described precisely (not fully architecturally explained, but
   survived two real fix attempts) rather than left as "might just be a bug."
3. Folded into `LIMITATIONS.md` and `paper/main.tex`'s H4 sections.

## If this repo is picked up again

- E6/H3 could still use a genuinely redesigned metric (report
  per-excluded-configuration rather than pooled, as flagged in
  `LIMITATIONS.md`) — the lr fix alone won't make the pooled number
  meaningful, since it's dominated by which config gets excluded.
- The k=2 training failure at p=0.0025/0.005 (item above) was not
  exhaustively investigated — a data-pipeline correctness check specific to
  k=2, or an even lower lr with a longer epoch budget, would be the next
  things to try, not assumed to be a closed question.
