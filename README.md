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
  scale), and `e6_extended.py` (a less thin H3 configuration-generalization
  test than `run_all.py`'s own 2-config default).
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
hyperparameters from `results/HPARAM_SEARCH.md`
(`hidden_dim=64, num_layers=6, conv_type="transformer", heads=2, lr=3e-4,
weight_decay=1e-4`) rather than the untuned design-doc defaults — pass
`--hidden-dim`/`--num-layers`/`--conv-type`/`--heads`/`--lr`/`--weight-decay` to
override. Pass `--shots`/`--epochs`/`--num-seeds` to control scale; see
`results/main_table.md`'s own header row for what scale actually produced the
current numbers.

## The headline result so far

The design doc's own gate (`tasks.md`'s header): "if you cannot beat MWPM on
memory [E2], the problem is your GNN, not lattice surgery." Before any tuning,
at design-doc-default hyperparameters, the GNN lost to MWPM by **~15x** on plain
memory. A real hyperparameter search (`experiments/hparam_search.py`, 58
trials, GPU) — varying hidden dim, depth, conv type, learning rate, batch size,
weight decay — closed that to **~1.8x** (confirmed not to be a training-budget
artifact: 3x the epochs and 2x the test shots on the winning config changed
the ratio from 1.79x to 1.88x, i.e. not at all). The GNN still does not beat
MWPM on memory. That is reported plainly, not softened — see
`results/HPARAM_SEARCH.md` for the full search and
`results/main_table.md`/`LIMITATIONS.md` for what this means for every
downstream (lattice-surgery) number.

At the tuned hyperparameters and a much larger scaled run (8 seeds, 30000
shots, 20 epochs — see `results/main_table.md`), every metric that trains
*directly* on surgery data improved substantially (E5: 0.116 → 0.062; E7
spacelike gap to MWPM: 2.3x → 1.26x; E7 timelike: 2.8x → 1.25x) — but E3's
zero-shot transfer got measurably *worse* (32% → 42% error). A follow-up
(`experiments/e3_isolation_check.py`, `results/E3_ISOLATION_CHECK.md`)
isolated why: it's **training scale**, not the tuned architecture, that
drives this — training the *old*, untuned architecture at the *new*, larger
scale reproduces the regression (and then some: 51.7% ± 15.3%), which ruled
out our first guess (a deeper/tuned model "specializing" to memory
statistics) before it made it into anything final. See `LIMITATIONS.md`'s
"third pass" section for the full corrected story.

## What's genuinely unresolved

Short version — full version in `LIMITATIONS.md`:

- GNN still trails MWPM everywhere, including after real tuning.
- Unequal-distance patches and independent routing-width are unimplemented —
  checked at the TQEC API level, not just observed; see `LIMITATIONS.md`'s
  Task-2.3-adjacent entry for the exact API citation.
- No comparison to any published neural decoder (e.g. AlphaQubit), no real
  hardware noise model.
- The hand-rolled Stim circuit builder (`circuits/lattice_surgery.py`) does not
  work; every reported result comes from the TQEC backend.
