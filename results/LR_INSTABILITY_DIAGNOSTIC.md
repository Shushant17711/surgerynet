# lr=1e-3 instability on k>=2 training (partial/interrupted investigation)

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

## What to do next (not done — session ended here)

1. Finish this run to n=8 (3 more seeds) to get a real mean ± std.
2. If seed 4-style failures recur at lr=3e-4, try an even lower lr (1e-4) or a
   warmup schedule specifically for k>=2 training, rather than assuming lr=3e-4
   fully fixes it.
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
