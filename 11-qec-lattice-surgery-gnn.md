# SurgeryNet: A Graph Neural Decoder for Lattice Surgery

> Neural QEC decoders are trained and evaluated almost entirely on quantum *memory* —
> one static patch, held still. Real fault-tolerant computation is lattice surgery:
> patches merge and split, the decoding graph changes shape mid-computation, and the
> code distance varies with time. Fixed-input neural decoders cannot even be
> *evaluated* in that setting. A graph neural network can, because a changing graph is
> its native input.

---

## 1. The one-paragraph pitch

Every neural decoder result you have read — AlphaQubit, the GNN decoders, the
transformer decoders — is a **memory experiment**: prepare a logical state, run `r`
rounds of stabilizer measurement, see if it survived. Useful, but it is the quantum
equivalent of benchmarking a CPU on idle. Actual computation requires logical *gates*,
and on surface codes those are done by lattice surgery: two patches are merged by
measuring joint stabilizers across a routing region, held merged for `d` rounds, then
split.

During that window the decoding problem is a different problem. New detector vertices
appear where the patches join. The boundaries move. The effective code distance is
time-varying. And the merged patch can, in the worst case, span the entire device.

**A neural decoder with a fixed-size input cannot be run on it at all** — not "runs
badly," but has no valid input shape. A GNN over detection events has no such
constraint. That structural mismatch is the whole thesis of this project, and it is a
better argument than "our model scores 3% higher."

## 2. The research gap

A 2026 paper on maximum-likelihood decoding states the situation directly: to date,
the most advanced research has focused on memory experiments, where the goal is to
preserve a logical state over time, whereas universal fault-tolerant computation
requires executing logical gates via lattice surgery, braiding, or transversal
operations. It goes on to say that these operations create geometrically complex,
irregular spacetime decoding graphs that differ fundamentally from the regular lattice
structures of memory experiments, and that lattice surgery specifically introduces new
types of boundaries, time-varying code distances, and correlated measurement errors
that traditional decoders struggle to handle optimally.

The real-time decoding review adds a second, independent gap: **high thresholds for
space-like as well as time-like errors when performing lattice surgery still need to be
demonstrated.** Timelike errors are the ones that corrupt the *parity measurement
outcome* rather than the stored state, and they have no analogue in a memory
experiment. Almost nobody reports them.

And the August 2026 ML decoding survey confirms the framing from the learning side:
its benchmark review focuses primarily on quantum memory experiments, which remain
the most common and systematically studied setting for evaluating decoder performance.

> Sources: arXiv:2605.17230 (MLD, §4.3 "Decoding beyond memory experiments");
> arXiv:2303.00054 (real-time decoding review, §E "Lattice surgery");
> arXiv:2608.15760 (ML decoding survey, §4).

### What already exists (be honest about this in your paper)

| Work | What it does | Why it does not close the gap |
|---|---|---|
| LATTE (arXiv, 2025) | FPGA/CPU hybrid, spatially parallel block decoding for lattice surgery, with a light neural *local* decoding unit | A systems/latency contribution. The neural component is a small local unit, not a learned decoder over the merged spacetime graph. |
| Spatially parallel decoding (arXiv:2403.01353) | Divides physical qubits into overlapping groups, assigns a decoder module to each | Matching-based, addresses throughput not learned accuracy. |
| Network-integrated decoding (arXiv:2504.11805) | Decoder system architecture supporting lattice surgery | Systems architecture, not a learned decoder. |
| MWPM / union-find | Work correctly on lattice surgery **given** a correct detector error model and carefully specified logical representatives | These are your baselines, and they are strong. See §4. |

**Nobody appears to have asked whether a learned decoder transfers from memory to
lattice surgery, or trained one to handle the merge/split structure directly.** That
is your paper.

## 3. What you must NOT claim

Get this wrong and a reviewer dismisses the whole thing in one sentence.

**Do not claim MWPM cannot decode lattice surgery.** It can. The real-time review is
explicit that graph-based decoders including union-find work on lattice surgery as
long as logical representatives are correctly specified through the full syndrome
history. Your contribution is *not* "we made the impossible possible."

The correct claim is narrower and defensible:

> Among *learned* decoders — which now outperform MWPM on memory experiments —
> architecture determines whether the decoder can operate on lattice surgery at all.
> We show that fixed-input architectures cannot transfer, that a GNN can, and we
> report the first learned-decoder accuracy numbers on merge/split operations,
> including timelike error rates.

Write that sentence into your abstract early and keep checking your claims against it.

## 4. Hypotheses

> **H1 (architectural).** A GNN decoder trained *only on memory experiments* can be
> evaluated zero-shot on merge/split rounds with no architectural change, while MLP
> and CNN decoders trained on the same data cannot be evaluated at all. Its accuracy
> will degrade relative to memory, and quantifying that degradation is the result.

> **H2 (localization).** The accuracy loss is not uniform across the operation. It
> concentrates in the **merge window** and specifically near the **routing region
> boundary**, where the graph structure is most unlike anything in the training
> distribution. A per-round, per-region error-rate map is the paper's key figure.

> **H3 (the fix).** Training on a *distribution* of surgery configurations — varying
> patch sizes, merge durations, and routing-region geometries — yields a single model
> that handles unseen configurations, recovering most of the H1 gap. Same domain-
> randomization logic that works in sim-to-real robotics.

> **H4 (timelike).** Spacelike and timelike logical error rates behave differently
> under a learned decoder, and a decoder tuned for one is not optimal for the other.
> Since timelike thresholds for lattice surgery remain undemonstrated, even a careful
> measurement here is a contribution.

H2 and H4 are the ones that make this more than a benchmark exercise.

## 5. Method

### 5.1 Generating the circuits — the hard part, budget accordingly

`stim.Circuit.generated()` only produces built-in memory and repetition-code circuits.
There is no `generated("lattice_surgery")`. You must construct the syndrome extraction
circuit yourself.

Three routes, in order of what I would try:

1. **TQEC** (open-source, github.com/tqec/tqec) builds Stim circuits from spacetime
   block / ZX-diagram descriptions of lattice surgery. If it works for your case this
   saves you a month. **Verify its current state before relying on it** — check the
   repo activity and whether the circuits it emits give sensible MWPM thresholds.
2. **Crumble** (Stim's browser-based circuit editor) to hand-build a small merge/split
   circuit, then export. Slow but you will *understand* the circuit, which matters more
   than speed at week 3.
3. **Build it manually in Python.** Two rotated surface code patches, a routing region
   between them, and a schedule: measure individual stabilizers → prepare routing
   qubits → measure joint stabilizers for `d` rounds → cease joint measurement → resume
   individual. This is the most work and the most reliable understanding.

**The target operation:** a logical `Z⊗Z` (or `X⊗X`) parity measurement between two
distance-`d` patches via merge and split. Start at `d=3`. Do not start bigger.

**Validation checkpoint, non-negotiable:** run PyMatching on your circuit's detector
error model and confirm you get a sensible threshold and the expected `Λ` scaling. If
your MWPM numbers look wrong, your circuit is wrong, and every neural result after that
is meaningless. Do not proceed past this.

### 5.2 Graph construction

Nodes = detection events that fired. Features:

- Spatial coordinates `(x, y)`, **normalized by patch dimension**
- Round index, normalized by total rounds
- Stabilizer type (X or Z)
- **Region label** — one-hot: `patch_A`, `patch_B`, `routing_region`. This is the
  feature that carries lattice-surgery structure and does not exist in memory
  experiments.
- **Phase label** — one-hot: `pre_merge`, `merged`, `post_split`
- Local detection-event density

Edges: pairs of detection events within a space-time radius, **plus** the edges implied
by the detector error model (`circuit.detector_error_model()` gives you the true
correlation structure — use it, do not guess a radius).

**Invariance discipline.** No absolute coordinates, no fixed patch size baked into a
dense layer, no assumption about how many rounds the merge lasts. Every one of those
silently breaks the transfer experiment and costs you two weeks to find.

For H1 (zero-shot from memory), the region and phase features must be *absent or
zeroed* at memory-training time and only supplied at surgery-evaluation time — or
omitted entirely. Design this carefully; it determines whether H1 is a fair test.

### 5.3 Model

- 4–6 message-passing layers (`GATConv` or `TransformerConv` from `torch_geometric`)
- Hidden dim 64–128
- Global mean + max pooling
- **Two output heads**: spacelike logical observable flip, and the **parity measurement
  outcome** (the timelike quantity). Two heads, not one — this is what enables H4.

### 5.4 Comparison models

| Model | Runs on lattice surgery? | Role |
|---|---|---|
| MLP on flattened syndrome | **No** — input dim is fixed by patch size | H1 evidence |
| CNN on syndrome grid | **No** — grid shape changes at merge | H1 evidence |
| GNN (yours) | Yes | The hypothesis |
| MWPM / PyMatching | Yes, given a correct DEM | The baseline that matters |
| Union-find | Yes | Fast classical baseline |

The fact that two of these simply *cannot be evaluated* is a result, not a gap in your
experiments. Report it as a row of "N/A (architecturally infeasible)" and explain why.

## 6. Experiments

**E1 — Validation.** MWPM on your surgery circuit. Correct threshold, correct scaling.
Week 5 gate.

**E2 — Memory baseline.** Train the GNN on plain `d=3` memory. Beat MWPM there, as
prior work does. If you cannot, the problem is your GNN, not lattice surgery.

**E3 — Zero-shot transfer (H1).** Evaluate the memory-trained GNN on merge/split
rounds. Report the degradation. Report MLP/CNN as architecturally infeasible.

**E4 — Localization (H2).** Per-round, per-region logical error rate. Heatmap over the
spacetime volume showing *where* the decoder fails. This is the figure people will
screenshot.

**E5 — Surgery-trained model.** Train directly on merge/split data. How much of the
E3 gap closes?

**E6 — Configuration randomization (H3).** Train across varied patch sizes, merge
durations, routing geometries. Evaluate on held-out configurations.

**E7 — Timelike errors (H4).** Spacelike vs timelike logical error rate, separately,
across physical error rates. Report both thresholds. Almost nobody does this.

**E8 — Latency.** Merged-patch decoding time vs patch size, CPU and GPU. GNN inference
parallelizes; matching is more sequential. If there is an advantage here it is a
practically important one, and if there is not, say so.

**E9 — Ablations.** Region/phase features on/off; DEM-derived edges vs radius edges;
message-passing depth; normalization on/off.

## 7. Metrics

- **Spacelike logical error rate** per operation
- **Timelike logical error rate** — probability the parity measurement outcome is wrong
- **Both thresholds**, separately
- **Per-round error rate** through the merge window
- **Decoding latency and throughput** vs merged-patch size
- **Configuration generalization gap** — held-out vs training configurations

## 8. Repository structure

```
surgerynet/
├── README.md
├── LIMITATIONS.md
├── requirements.txt              # stim, pymatching, sinter, torch, torch_geometric
├── circuits/
│   ├── lattice_surgery.py        # merge/split circuit construction
│   ├── configurations.py         # patch sizes, merge durations, geometries
│   └── validate.py               # THE week-5 gate: MWPM sanity check
├── data/
│   ├── generate.py
│   └── to_graph.py               # detection events + region/phase labels → PyG Data
├── models/
│   ├── gnn_decoder.py            # two heads: spacelike + timelike
│   ├── mlp_decoder.py            # for the "cannot be evaluated" demonstration
│   └── cnn_decoder.py
├── baselines/
│   ├── mwpm.py
│   └── union_find.py
├── experiments/
│   ├── e3_zeroshot.py
│   ├── e4_localization.py
│   ├── e7_timelike.py
│   └── e8_latency.py
├── scripts/reproduce.sh
├── results/
│   ├── spacetime_heatmap.png     # THE figure
│   ├── timelike_threshold.png
│   └── main_table.md
└── paper/main.tex
```

## 9. Timeline (12 weeks)

| Week | Work |
|---|---|
| 1 | Surface code fundamentals. Stim tutorials. Reproduce a memory threshold with PyMatching. |
| 2 | Read the lattice surgery material properly — the review's §E, plus *Lattice Surgery for Dummies*. Understand merge/split before writing code. |
| 3 | Evaluate TQEC. If viable, use it; if not, start manual circuit construction. |
| 4 | Build the `d=3` merge/split circuit. Visualize the detector graph before and during merge — you should *see* the extra vertices appear. |
| **5** | **Gate: MWPM validation.** Correct threshold and scaling, or stop and debug. |
| 6 | Memory-trained GNN beating MWPM (E2). |
| 7 | Zero-shot transfer (E3). First real result. |
| 8 | Localization heatmap (E4). |
| 9 | Surgery-trained model + configuration randomization (E5, E6). |
| 10 | Timelike errors (E7). Expect this to take longer than planned. |
| 11 | Latency (E8) + ablations (E9). |
| 12 | Write-up. |

## 10. Compute

Laptop CPU carries weeks 1–8: Stim is fast, and `d=3` merge/split circuits are small.
Colab free tier for the threshold sweeps (many shots at low error rates) and GNN
training at larger `d`. Use `sinter` for parallel Monte Carlo — it handles the
statistics and the parallelism.

## 11. Failure modes

- **Circuit construction eats the semester.** The single biggest risk. Hard gate at
  week 5; if you have not validated by then, fall back to a *simpler* logical
  operation (patch deformation, or a single patch that grows and shrinks) which
  preserves the changing-graph thesis at much lower construction cost. Decide this in
  advance, not in a panic at week 8.
- **Zero-shot transfer is catastrophic, not graceful.** Entirely possible. It is still
  a result — "learned decoders do not transfer from memory to logical operations" is a
  useful thing for the field to know, and E5/E6 give you the constructive half.
- **TQEC does not do what you need.** Check early. Do not discover this at week 6.
- **Timelike errors are confusing.** They are. Budget extra time, and sanity-check
  against MWPM's timelike rate before trusting your GNN's.
- **Sample-limited floors.** At low `p` you cannot resolve small logical error rates.
  Report the measurement floor set by your shot count rather than claiming zero.

## 12. Why this one

The gap is stated in three separate 2026 sources. The competing work is systems and
FPGA engineering, not learned decoding. The architectural argument is clean enough to
state in one sentence. It runs on your laptop. And every baseline from the memory
literature transfers directly, so you are not building a benchmark from nothing.

The risk is concentrated entirely in circuit construction — which is why week 5 is a
gate and not a milestone.

## 13. References

- arXiv:2605.17230 — *Maximum Likelihood Decoding of QEC Codes*, §4.3 "Decoding beyond
  memory experiments" — your primary gap citation
- arXiv:2303.00054 — *Real-Time Decoding for Fault-Tolerant Quantum Computing*, §E —
  lattice surgery decoding requirements, and the timelike threshold gap
- arXiv:2608.15760 — *ML Approaches to Decoding Topological Quantum Codes* (Aug 2026)
- arXiv:2403.01353 — *Spatially parallel decoding for multi-qubit lattice surgery* —
  closest systems-side work; cite and distinguish
- arXiv:2504.11805 — *Network-Integrated Decoding System for Real-Time QEC with Lattice
  Surgery*
- Horsman et al., *Surface code quantum computing by lattice surgery* (2012) — the
  original; read it first
- *Lattice Surgery for Dummies* (PMC11946007) — the readable introduction
- Gidney, *Stim* (Quantum, 2021); Higgott, *PyMatching*; TQEC repo
- Bausch et al., *AlphaQubit* (Nature 2024) — the memory-experiment state of the art
  you are extending past
