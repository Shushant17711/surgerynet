# lr=1e-3 instability on k>=2 training

**Status: n=8 confirmation completed (resumed after a session interruption at n=5;
the original interrupted n=5 table is kept below for the record). Real, partial
fix found — not a complete one. H4 threshold-sweep recheck is the next step,
see `CRITIQUE_FOLLOWUP.md`.**

## n=8 result (completed)

| condition | mean train-set error rate | n |
|---|---|---|
| lr=1e-3 (original), k=1 excluded from training | 0.459-0.478 (near chance) | 9 (5 from `E6_EXTENDED.md` + 4 from main E6 table) |
| **lr=3e-4, k=1 excluded from training** | **0.3416** | 4 |
| lr=3e-4, k=1 included in training | 0.1356 | 4 |

Lowering lr from 1e-3 to 3e-4 cuts the k=1-excluded training failure roughly in
half (0.48 -> 0.34) -- a real, substantial improvement, confirming lr is a real
part of the problem. **But it does not fully close it**: even at lr=3e-4,
training on k=2/k=3 alone (no k=1 "easy" anchor) still converges meaningfully
worse (0.34) than training with k=1 included (0.14). This means there is a
second, still-unexplained factor beyond learning rate -- plausibly that k=2/k=3
graphs (bigger, more rounds) need more epochs or shots than the current
per-config budget (15,000 shots, 20 epochs) provides, or that k=1 data provides
a genuinely useful easier-curriculum signal that a same-size k>=2-only dataset
doesn't replicate. Not isolated further here.

Full per-seed data (distances=(3,5,7), lr=3e-4, hidden_dim=128, edge_dim=5):

| seed | held_out_d | held_out_rate | train_rate | gap |
|---|---|---|---|---|
| 0 | [5] | 0.1346 | 0.1756 | -0.0410 |
| 1 | [3] | 0.1930 | 0.2446 | -0.0516 |
| 2 | [3] | 0.1091 | 0.3830 | -0.2739 |
| 3 | [3] | 0.1539 | 0.2610 | -0.1071 |
| 4 | [3] | 0.2795 | 0.4777 | -0.1982 |
| 5 | [7] | 0.3755 | 0.0947 | 0.2808 |
| 6 | [7] | 0.3229 | 0.0871 | 0.2358 |
| 7 | [5] | 0.1465 | 0.1850 | -0.0385 |

**Pooled held-out rate: 0.2144 ± 0.0987** (mean ± std, n=8) -- barely different
from the lr=1e-3 pooled number (0.2518 ± 0.0763) or the original 2-config
baseline (0.2224 ± 0.1014). **The lr fix does not show up in the pooled metric**
because that metric is still dominated by which configuration gets held out
(a `k=7`-held-out seed still gets evaluated on the hardest config regardless of
how well training converged), not by training quality. This reinforces the
earlier finding that E6's single pooled number is not a good metric for this
question — the real signal is in the train_rate breakdown above, not the
held-out rate. A genuinely informative version of E6 would need to report
per-configuration numbers from the start, not average across an
uncontrolled mix of "which config got excluded."

---

## Original interrupted n=5 partial run (kept for the record)

**Status: INTERRUPTED — session ended before completion. This is n=5 of a planned
n=8 confirmation run, saved so the finding and partial evidence aren't lost.**

## The finding

`results/E6_EXTENDED.md`'s full 8-seed rerun (distances=(3,5,7), current tuned
config incl. `lr=1e-3`) showed a clear bimodal pattern in `train_configuration_error_rate`:

- Whenever k=1 (d=3) was part of the training mix: train rate 0.08-0.28 (converges).
- Whenever training was k=2+k=3 ONLY (no k=1 at all — seeds where `held_out_d=[3]`):
  train rate 0.46-0.48 — **near chance, training essentially fails**.

This is NOT explained by the physics being harder at larger k: the threshold sweep
(`results/THRESHOLD_SWEEP.md`) shows MWPM decodes k=2 at p=0.002 just fine
(~2-5% error rate), so a k=2-only training set is a solvable problem in principle.

Isolated single-config diagnostic (`/tmp` scratch, not saved as a script — reproduce
via `experiments.e5_surgery_trained.run` on a k=2 circuit, train_shots=15000):

| condition | result |
|---|---|
| lr=1e-3, 20 epochs (current default) | 0.462 (near chance — reproduces the E6 failure) |
| lr=3e-4 (the pre-retune value), 20 epochs | 0.174 |
| lr=1e-3, 60 epochs (3x budget) | 0.115 |

**Conclusion so far: `lr=1e-3` (adopted project-wide after the edge-feature retune,
which was validated ONLY on k=1 data — E2 memory, E5/E7 at k=1) is too aggressive
for the bigger k=2/k=3 surgery graphs**, causing real optimization instability, not
a capability limit. Lower lr or a longer epoch budget both mitigate it in isolated
single-config tests.

## Partial re-run of E6 (distances=(3,5,7)) at lr=3e-4, interrupted at n=5/8

| seed | held_out_d | held_out_rate | train_rate | gap |
|---|---|---|---|---|
| 0 | [5] | 0.1346 | 0.1756 | -0.0410 |
| 1 | [3] | 0.1930 | 0.2446 | -0.0516 |
| 2 | [3] | 0.1091 | 0.3830 | -0.2739 |
| 3 | [3] | 0.1539 | 0.2610 | -0.1071 |
| 4 | [3] | 0.2795 | 0.4777 | -0.1982 |

**This is not a clean confirmation.** Seeds 1-3 (train configs = {5,7}, no k=1) show
much better train rates (0.24-0.38) than the lr=1e-3 run's 0.46-0.48 — real
improvement. But seed 4 (also train configs = {5,7}) still shows 0.4777, барely
different from the lr=1e-3 failure case. **lr=3e-4 helps but does not reliably fix
k=2/k=3-only training** — there may be a seed-dependent optimization instability
(bad initialization basin) beyond just the learning rate, or 3 more seeds would
have clarified whether 0.4777 is an outlier or a real recurring failure mode at
this lr too.

## What to do next

1. ~~Finish this run to n=8~~ — **done above.** Confirmed: lr=3e-4 is a real,
   partial fix (0.48 -> 0.34 on k=1-excluded training) but not a complete one.
2. The residual gap (0.34 vs. 0.14 when k=1 is included) is unexplained —
   try more epochs specifically for k>=2-only training, or an even lower lr
   (1e-4), before concluding this is an irreducible data/capacity limit.
3. **Reconsider whether `results/timelike_threshold.png`'s GNN k=2 curve is
   confounded by this same instability.** The GNN k=2 spacelike curve in that
   sweep jumps to ~0.48-0.50 immediately at p=0.0025 and stays flat — the same
   signature as the training failures documented here — while MWPM degrades
   smoothly. `e7_threshold_sweep.py`'s `gnn_sweep` used `lr=1e-3` (the current
   default) for all k, including k=2. **The H4 "no real distance-scaling
   advantage" conclusion in `LIMITATIONS.md`/`paper/main.tex` may be partly or
   wholly a training-instability artifact, not a genuine capability limit, and
   should be rechecked with a k-appropriate (lower) learning rate before being
   treated as a settled finding.** This is flagged but NOT corrected in
   `LIMITATIONS.md`/the paper yet — do that alongside the rerun.
4. Once resolved, update `experiments/e6_config_randomization.py` and
   `experiments/e7_threshold_sweep.py` to use a k-appropriate learning rate
   (or make it adaptive) rather than the flat k=1-tuned `lr=1e-3` default that
   `experiments/hparam_search.py`/`experiments/edge_feature_check.py` never
   actually tested beyond k=1.
