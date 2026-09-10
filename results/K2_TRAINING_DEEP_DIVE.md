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

**Support for this reading, not yet a proof**: tracked the actual training
loss curve at this exact (k=2, p=0.0025) point, standard config, 15 epochs.
It is *not* stuck -- BCE loss starts at 0.6946 (== $\ln 2$, the exact
random-guessing baseline) and decreases steadily to 0.6869 by epoch 15, a
real if very slow trend. This is consistent with (though doesn't prove) a
parity-like problem that SGD is slowly making progress on rather than one
that is architecturally unreachable. A 300-epoch run was launched to check
whether this slow trend continues to a meaningfully better error rate given
enough budget -- **result pending, not yet in this file; check
`/tmp/long_train.log` or a later revision of this file / `LIMITATIONS.md`
for the outcome** (this note exists so the interim state is captured even
if that run doesn't finish this session).

## What would actually confirm or refute this

Not done here (time-bounded investigation):

1. Finish the 300-epoch run above. If error rate drops substantially
   (e.g. below ~30%, matching order-of-magnitude with MWPM's 5.7%), this
   is a slow-convergence/capacity issue, not a hard architectural wall --
   worth knowing precisely where the diminishing-returns point is.
2. A virtual "boundary" supernode connected to every detector node (edge
   weight = that node's own boundary log-odds, the same information tried
   and reverted as a *node* feature earlier this session) would directly
   give the GNN a channel to route information *between* the ~18
   components, mirroring how MWPM's own decoding graph includes an
   explicit boundary node. This is architecturally different from both the
   reverted node-feature attempt and the sum-pooling test above (neither
   added actual graph *connectivity* between components) and is the most
   promising untried fix, but is a real graph-construction change (`data/
   to_graph.py`), not a quick hyperparameter check -- not attempted here.
3. A synthetic parity-learning sanity check (train this exact architecture
   on a constructed task that's provably a parity function of N inputs,
   varying N) would confirm whether the "parity is hard for SGD" hypothesis
   applies to this specific architecture at a component-count around 18,
   independent of the quantum-error-correction domain specifics. Not done.
