# E3 isolation check: architecture (tuning) vs. scale

Three points, same p=0.002, same 8 seeds where applicable, same surgery circuit:

| condition | E2 (memory) rate | E3 (zero-shot) rate |
|---|---|---|
| old hyperparams, OLD scale (3000 shots/8 epochs/3 seeds, archived) | 0.0091 ± 0.0042 | 0.3206 ± 0.0356 |
| old hyperparams, NEW scale (30000 shots/20 epochs/8 seeds, this script) | 0.0017 ± 0.0002 | 0.5171 ± 0.1532 |
| new (tuned) hyperparams, NEW scale (30000 shots/20 epochs/8 seeds, main run) | 0.0013 ± 0.0002 | 0.4207 ± 0.0305 |

**Verdict**: old-hp/new-scale E3 rate (0.5171) is closer to the NEW-hyperparameter result -> training SCALE itself (not the tuned hyperparameters) is implicated in the zero-shot regression -- the 'specializes harder to memory statistics' story in LIMITATIONS.md needs revising.
