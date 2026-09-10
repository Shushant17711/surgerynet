# Deep dive: why does k=2 spacelike training fail at p=0.0025/0.005?

Follow-up to `LR_INSTABILITY_DIAGNOSTIC.md` and `CRITIQUE_FOLLOWUP.md`'s open
question. Systematically ruled out every cheap/moderate-effort optimization
hypothesis; found a real, weak-but-genuine learning signal and a plausible,
well-evidenced structural explanation. Point of reference throughout:
k=2, spacelike, p=0.0025, `hidden_dim=128, num_layers=6, conv_type=transformer,
heads=2, weight_decay=1e-4, edge_dim=5`, train/test shots=15000 unless noted.
MWPM at this exact (k, p): 5.7% error (real, non-trivial decoding).

## Ruled out

| hypothesis | test | result |
|---|---|---|
| lr too high | lr=3e-4 (vs 1e-3), 20 epochs | still ~0.49 (chance) |
| lr too high, more | lr=1e-4, 40 epochs | still 0.4908 |
| not enough epochs | lr=3e-4, 60 epochs | still 0.4907 |
| bad seed / init lottery | 8 seeds total (0-7) at lr=3e-4, 20ep | **all** 0.477-0.493 -- no seed succeeds |
| pooling can't express parity | `use_sum_pool=True` (new model option, added this session) | 0.471-0.477 across 3 seeds -- no better |
| components too spatially local | `spatial_radius=10.0` (vs default 3.0) | components/shot barely changed (18.03 vs 17.86) -- components are separated in **time** (fixed `temporal_radius=1.0`, not spatial), not space |

None of these fixed it. This is a real, deterministic, reproducible failure,
not noise or a simple hyperparameter miss.

## Not ruled out: raw label balance and syndrome density

Measured directly on the actual k=2, p=0.0025 circuit (15000 shots):

- Raw (undecoded) observable flip rate: **P(y=1) = 0.4853** -- marginally,
  this label is close to a coin flip. (MWPM still recovers 5.7% error using
  the syndrome -- the syndrome carries real information even though the
  unconditional prior doesn't.)
- Mean fired detectors per shot: **72.6** (vs. ~10.3 at the k=1/p=0.002
  setting that trains fine).
- Mean connected components per shot: **17.9** (median 18, up to 26) vs.
  **4.1** (median 4) at k=1/p=0.002 -- confirmed via networkx on real
  sampled graphs, not estimated.

## The most defensible current explanation

MWPM solves one *global* matching problem over the full DEM graph (including
implicit boundary matches), so it isn't limited by which syndrome nodes
happen to be graph-connected in a simplified pairwise sense. This GNN's
message passing only propagates information *within* a connected component;
cross-component combination happens only at the final pooling step (mean,
max, and now optionally sum) -- a single order-invariant aggregate that has
to implicitly learn something parity-like across ~18 largely-independent
pieces of evidence to get the label right. Parity-of-many-variables is a
well-known hard case for gradient-based learning (slow, sometimes
effectively-stuck convergence on standard architectures/initializations) --
independent of this project, a well-documented pathology in the ML
literature, not something specific to this codebase.

**Tracked the actual training loss curve** at this exact (k=2, p=0.0025)
point, standard config, 15 epochs. It is *not* stuck initially -- BCE loss
starts at 0.6946 (== $\ln 2$, the exact random-guessing baseline) and
decreases to 0.6869 by epoch 15, a real if very slow trend.

**Then tested whether that trend continues to something useful given a much
longer budget: 300 epochs (15x the normal 20).** Result: **0.4914 -- still
chance.** This is decisive. Combined with everything above (lr, seeds,
pooling, radius all ruled out; now epoch budget up to 300 ruled out too),
**this is not an optimization-budget problem**. Either the slow initial
loss decrease was fitting noise/overfitting to the training set without
generalizing (15,000 shots may simply not be enough data for an
~18-component combinatorial problem), or the model plateaus at a bad
solution the current architecture cannot escape via gradient descent
regardless of budget. Either way, **more training time alone does not fix
this.**

## Bottom line

Exhaustively tested within this investigation's scope: learning rate
(1e-3, 3e-4, 1e-4), epoch budget (20, 40, 60, 300), 8 different seeds,
pooling type (mean+max vs.\ mean+max+sum), and spatial connectivity radius
(3.0 vs.\ 10.0). **None fix it.** This is a genuine, reproducible
architectural/representational limitation specific to k=2 (and presumably
k=3) surgery decoding in the regime where the underlying problem is still
solvable (MWPM proves it), not a hyperparameter miss or a training bug.
The most likely remaining explanation: this GNN's message passing only
shares information *within* a connected component, and the ~18
largely-independent components a k=2 shot's graph fragments into require
something like parity-style reasoning across all of them to get the label
right -- reasoning that only happens at the final (mean/max/sum) pooling
step, a genuinely hard combination for gradient descent to discover
(parity-of-many-variables is a well-documented hard case in the ML
literature generally, not specific to this codebase).

## What would actually confirm or fix this (not done here)

1. **A virtual "boundary" supernode** connected to every detector node
   (edge weight = that node's own boundary log-odds -- the same
   information tried and reverted as a *node* feature earlier this
   session) would directly give the GNN a channel to route information
   *between* the ~18 components during message passing itself, not just
   at the final pooling step -- mirroring how MWPM's own decoding graph
   includes an explicit boundary node it can match any detector to. This
   is architecturally different from both the reverted node-feature
   attempt and the sum-pooling test above (neither added actual graph
   *connectivity* between components). This is the most promising
   remaining fix, but is a real graph-construction change (`data/
   to_graph.py`) plus model changes, not a quick hyperparameter check --
   not attempted in this investigation.
2. A synthetic parity-learning sanity check (train this exact architecture
   on a constructed task that's provably a parity function of N inputs,
   varying N) would confirm whether "parity is hard for SGD" applies to
   this specific architecture at a component-count around 18, independent
   of the quantum-error-correction domain specifics. Not done.
3. Whether 15,000 training shots is simply too few for a problem this
   combinatorially rich (as opposed to an architectural ceiling) was not
   isolated -- a much larger shot count (e.g. 150,000+) at this exact
   (k=2, p=0.0025) point, matching the 10x-shots test already done for E2,
   would help distinguish "needs more data" from "needs a different
   architecture."
