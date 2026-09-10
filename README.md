# SurgeryNet: A Graph Neural Decoder for Lattice Surgery

> Among *learned* decoders — which now outperform MWPM on memory experiments —
> architecture determines whether the decoder can operate on lattice surgery at
> all. We show that fixed-input architectures cannot transfer, that a GNN can,
> and we report the first learned-decoder accuracy numbers on merge/split
> operations, including timelike error rates.

Full research framing, hypotheses (H1-H4), and the 12-week plan this repo
implements: [`11-qec-lattice-surgery-gnn.md`](11-qec-lattice-surgery-gnn.md).
Implementation status, task-by-task: [`tasks.md`](tasks.md). Honest limitations,
written from actually running this pipeline (not filled in after the fact):
[`LIMITATIONS.md`](LIMITATIONS.md) — **read that file before citing any number
from this repo.**

## What's here

- `circuits/` — merge/split circuit construction. The actual backend is TQEC
  (`tqec_backend.py`); a hand-rolled Stim builder (`lattice_surgery.py`) is kept
  as a documented, non-working reference attempt (see its module docstring).
  `validate.py` is the week-5 MWPM validation gate — run it first, always.
- `data/` — detection-event sampling (`generate.py`) and graph construction
  (`to_graph.py`, region/phase label recovery in `spacetime_labels.py`).
- `models/` — the GNN decoder (`gnn_decoder.py`, two heads: spacelike/timelike),
  MLP/CNN comparison decoders that raise `ShapeInfeasibleError` on
  surgery-shaped input (the H1 "architecturally infeasible" evidence), and the
  training loop (`train.py`).
- `baselines/` — MWPM (`mwpm.py`) and union-find (`union_find.py`).
- `experiments/` — E1-E9 from the design doc, plus post-hoc additions:
  `hparam_search.py` (real hyperparameter search against the E2 gate — see
  below), `e7_threshold_sweep.py` (real spacelike/timelike threshold curves
  for H4), `e3_isolation_check.py` (isolates whether the E3 zero-shot
  regression below is caused by the tuned architecture or by training
  scale), `e6_extended.py` (a less thin H3 configuration-generalization
  test than `run_all.py`'s own 2-config default), and
  `edge_feature_check.py` (A/B evidence for giving the GNN access to the
  DEM's own per-edge error probability — see below).
- `results/` — every generated table, plot, and raw JSON-lines result file.
- `paper/main.tex` — a paper draft built from the real numbers in `results/`.

## Environment setup

Two separate virtualenvs, because TQEC does not support Python 3.14 yet:

```bash
# main env: stim, pymatching, sinter, torch, torch_geometric
python3.14 -m venv .venv
.venv/bin/pip install -r requirements.txt -r requirements-dev.txt

# TQEC env: circuit generation only, talks to the main env via .stim files on disk
yay -S python312   # or however you get a Python 3.12 interpreter
python3.12 -m venv .venv-tqec
.venv-tqec/bin/pip install tqec
```

The two environments never import each other — `experiments/common.py`'s
`generate_surgery_circuit` shells out to `.venv-tqec/bin/python
circuits/tqec_backend.py` and reads back a portable `.stim` file.

GPU: `models/train.py`/`experiments/common.py::train_gnn` auto-detect CUDA
(`torch.cuda.is_available()`) and use it by default; pass `--device cpu` to any
experiment CLI to force CPU. Every number in `results/` after
`results/HPARAM_SEARCH.md` was produced on an RTX 4060 (8GB) — training that
used to take minutes per run on CPU takes single-digit seconds to ~2 minutes on
GPU depending on scale, which is what actually made a real hyperparameter
search practical.

## Reproducing the results

```bash
.venv/bin/python -m pytest                    # 84 passed, 3 xfailed (documented, see LIMITATIONS.md)
scripts/reproduce.sh --check                   # verify both envs + all scripts exist, no run
.venv/bin/python circuits/validate.py          # THE week-5 gate — must pass before anything else is trustworthy
scripts/reproduce.sh                           # full E1-E9 pipeline -> results/main_table.md, results/ABLATIONS.md
.venv/bin/python -m experiments.hparam_search  # rerun the hyperparameter search (results/HPARAM_SEARCH.md)
.venv/bin/python -m experiments.e7_threshold_sweep  # H4 threshold curves -> results/timelike_threshold.png
```

`scripts/reproduce.sh` and `experiments/run_all.py` now default to the tuned
hyperparameters from `results/HPARAM_SEARCH.md` + `results/EDGE_FEATURE_CHECK.md`
(`hidden_dim=128, num_layers=6, conv_type="transformer", heads=2, lr=1e-3,
weight_decay=1e-4`, plus DEM-log-odds edge features — see below) rather than
the untuned design-doc defaults — pass
`--hidden-dim`/`--num-layers`/`--conv-type`/`--heads`/`--lr`/`--weight-decay`/`--edge-dim`
to override. Pass `--shots`/`--epochs`/`--num-seeds` to control scale; see
`results/main_table.md`'s own header row for what scale actually produced the
current numbers.

## Results

$p=0.002$ (validated sub-threshold, cross-checked by `results/THRESHOLD_SWEEP.md`), 8 seeds, 30,000 shots, 20 epochs, tuned hyperparameters + DEM edge-weight features (`hidden_dim=128, num_layers=6, conv_type=transformer, heads=2, lr=1e-3, weight_decay=1e-4, edge_dim=5`). Full table with E8/E9 rows: [`results/main_table.md`](results/main_table.md).

| Exp. | What it measures | Decoder | Logical error rate | vs. MWPM |
|---|---|---|---|---|
| E2 | Plain memory (the design doc's own gate) | GNN | 0.0013 ± 0.0002 | 1.86x |
| E2 | Plain memory | MWPM | 0.0007 ± 0.0002 | — |
| E3 | Zero-shot transfer to surgery | GNN | 0.4825 ± 0.0324 | — |
| E3 | Zero-shot transfer to surgery | MLP / CNN | **N/A — architecturally infeasible** | — |
| E5 | Surgery-trained | GNN | 0.0523 ± 0.0016 | — |
| E6 | Config. generalization (held-out $k$) | GNN | 0.3023 ± 0.0521 | — |
| E7 | Surgery, spacelike | GNN | **0.0530 ± 0.0011** | **1.08x** |
| E7 | Surgery, spacelike | MWPM | 0.0491 ± 0.0012 | — |
| E7 | Surgery, timelike | GNN | **0.1082 ± 0.0046** | **1.09x** |
| E7 | Surgery, timelike | MWPM | 0.0990 ± 0.0014 | — |
| E9 | Ablation baseline (surgery-trained) | GNN | 0.0538 ± 0.0024 | — |

**Ablations** ([`results/ABLATIONS.md`](results/ABLATIONS.md), leave-one-out, same 8-seed/30000-shot scale):

| Variant | Mean rate | Std | Δ vs. baseline |
|---|---|---|---|
| Baseline (all components on) | 0.0531 | 0.0021 | — |
| Region/phase features removed | 0.0534 | 0.0018 | +0.0003 |
| DEM edges removed (radius-only) | 0.0614 | 0.0021 | **+0.0084** |
| Radius edges removed (DEM-only) | 0.0536 | 0.0014 | +0.0006 |
| Shallower (4 layers vs. 6) | 0.0537 | 0.0023 | +0.0007 |
| Normalization removed | 0.0518 | 0.0013 | -0.0012 |
| **Edge weights removed (topology only)** | 0.0613 | 0.0016 | **+0.0082** |

The new edge-weight feature is now tied for the single most load-bearing
component in the model (removing it costs almost exactly as much as
removing DEM edges outright) — strong confirmation it's doing real work,
not a marginal tweak.

**H4 threshold sweep** ([`results/THRESHOLD_SWEEP.md`](results/THRESHOLD_SWEEP.md), [`results/timelike_threshold.png`](results/timelike_threshold.png)), rerun at the edge-feature config:

| Observable | Decoder | Pseudo-threshold $p$ |
|---|---|---|
| Spacelike | MWPM | ≈ 0.0033 |
| Spacelike | GNN | ≈ 0.0149 — **read before citing**: this is a linear-interpolation artifact between two chance-floor (≈0.50) points, not a real crossing; see below |
| Timelike | MWPM | ≈ 0.0030 |
| Timelike | GNN | not found in swept range (0.0008–0.02) |

**Don't just cite the 0.0149 number.** At $p=0.005$ (still clearly
sub-threshold, MWPM decodes it easily) the GNN's $k=2$ rate (0.488) is
still far worse than its $k=1$ rate (0.295) — the same qualitative failure
as before edge features. The reported "crossing" falls between $p=0.005$
and $p=0.01$, where *both* $k=1$ (0.504) and $k=2$ (0.499) rates have
already collapsed to chance. The estimator has no floor-awareness and will
report a number even when both bracketing points are noise. **Honest
reading**: in the regime where the GNN is actually decoding non-trivially,
more code distance still doesn't help it the way it helps MWPM — edge
features improved absolute accuracy a lot without fixing this.

**Follow-up: is this a training bug, not a real finding?** Checked directly
— see `results/LR_INSTABILITY_DIAGNOSTIC.md`, `results/K2_TRAINING_DEEP_DIVE.md`,
and `results/CRITIQUE_FOLLOWUP.md`.
A real optimization problem was found (training on k≥2 data alone degrades
badly at the current tuned `lr=1e-3`, partially fixed by `lr=3e-4`), so the
k=2 curve here was rechecked at the lower lr — twice, at two different
shot/epoch budgets. **Neither fixed it.** Cross-checking against MWPM's own
numbers: at p=0.01/0.02 MWPM itself is already at chance, so the GNN
failing there was never real evidence; but at **p=0.0025 and p=0.005,
MWPM clearly decodes non-trivially (5.7%, 26.8% error) while the GNN sits
at chance across all three configurations tested.** This survived two real
fix attempts.

A further, exhaustive pass then ruled out every remaining optimization
explanation at the exact failing point (k=2, spacelike, p=0.0025): **8
different seeds** all fail identically (0.477–0.493, not a seed lottery);
`lr=1e-4` with 40 epochs still fails (0.4908); 60 epochs at `lr=3e-4` still
fails (0.4907); a new sum-pooling model option (testing whether mean+max
pooling was discarding parity-relevant information across the ~18
largely-disconnected components a k=2 shot's graph typically fragments
into, vs. ~4 at the working k=1 setting) doesn't help (0.471–0.477); and a
much larger spatial connectivity radius barely changes the component count
(18.03 vs. 17.86) — the components are separated in *time*, not space. Most
tellingly, the training loss curve isn't stuck — it starts at exactly ln(2)
and decreases steadily over 15 epochs, a real if slow trend — but a
**300-epoch run (15x the normal budget) testing whether that trend
continues to something useful still lands at 0.4914, essentially chance.**
More optimization budget, in every form tried, does not fix this — the
question this pass set out to answer (lr, epochs, seeds, pooling, data
locality) is now settled. The most likely remaining explanation: this
GNN's message passing only shares information *within* a connected
component, so combining ~18 largely-independent components into one label
happens only at the final pooling step — a genuinely hard, parity-like
combination for gradient descent, unlike MWPM's single global matching
problem. The most promising untried fix is a virtual "boundary" supernode
connected to every detector node, giving message passing itself (not just
pooling) a cross-component channel, mirroring MWPM's own boundary node —
a real graph-construction change, not attempted here.

## The headline result so far

The design doc's own gate (`tasks.md`'s header): "if you cannot beat MWPM on
memory [E2], the problem is your GNN, not lattice surgery." Before any tuning,
at design-doc-default hyperparameters, the GNN lost to MWPM by **~15x** on plain
memory. A real hyperparameter search (`experiments/hparam_search.py`, 58
trials, GPU) narrowed that to **~1.8x-1.9x** and it has stayed there since —
**the GNN still does not beat MWPM on plain memory**, and that has not
changed. That's reported plainly, not softened.

But E2 (plain memory) was never really the point of this project — the
actual research question is lattice surgery (E3/E5/E6/E7), and there the
picture moved a lot. The GNN originally saw graph *topology* only — never the
DEM's own per-edge error probability, which is exactly the number MWPM's
matching weight comes from. Adding that as an edge feature
(`experiments/edge_feature_check.py`, `LIMITATIONS.md`'s "fourth pass") did
essentially nothing on E2's uniform-error-rate memory circuit, but produced a
real, consistent improvement on surgery data (where error rates vary
spatially, so the weight is actually informative) — and a small follow-up
hyperparameter re-check on top of it (`hidden_dim=128`, `lr=1e-3`) pushed
further. Net result, full 8-seed/30000-shot rerun:

- **E7 (spacelike): gap to MWPM closed from ~2.3x to ~1.08x.**
- **E7 (timelike): gap to MWPM closed from ~2.8x to ~1.09x.**
- E5 (surgery-trained): 0.1156 (untuned) → 0.0622 (tuned hyperparameters) →
  **0.0523 (+ edge features and a further re-tune)** — three successive real
  improvements, not one (see `LIMITATIONS.md` for the full multi-pass numbers).
- E3 (zero-shot) and E6 (config. generalization) both got *worse*, continuing
  a pattern already isolated in an earlier pass
  (`experiments/e3_isolation_check.py`): more memory-training capacity/budget
  doesn't help — and can hurt — the zero-shot number, even as it helps
  everything that trains directly on surgery data.

See `LIMITATIONS.md`'s "fourth pass" section for the full A/B evidence, the
complete before/after table, and the honest accounting of what got worse
alongside what got much better.

## What's genuinely unresolved

Short version — full version in `LIMITATIONS.md`:

- GNN still trails MWPM everywhere in this repo, including on the
  now-much-closer E7 surgery numbers.
- E3 (zero-shot) and E6 (configuration generalization) got worse as the
  model gained capacity and edge features — a real, reproduced pattern, not
  fully explained (isolated to training scale/capacity, not architecture
  choice, but the underlying mechanism is still a hypothesis).
- The edge-feature hyperparameter re-check was a handful of hand-picked
  configs, not a systematic search — a full `hparam_search.py`-style search
  at this new baseline might find something better still.
- The GNN still shows no genuine distance-scaling advantage (H4) in the
  regime where it's actually decoding non-trivially — the H4 threshold
  sweep's naive linear-interpolation estimator reports a numeric spacelike
  "crossing" now, but it's an artifact of two chance-floor points, not a
  real signal (see the threshold table above).
- Unequal-distance patches and independent routing-width are unimplemented —
  checked at the TQEC API level, not just observed; see `LIMITATIONS.md`'s
  Task-2.3-adjacent entry for the exact API citation.
- No comparison to any published neural decoder (e.g. AlphaQubit), no real
  hardware noise model.
- The hand-rolled Stim circuit builder (`circuits/lattice_surgery.py`) does not
  work; every reported result comes from the TQEC backend.
