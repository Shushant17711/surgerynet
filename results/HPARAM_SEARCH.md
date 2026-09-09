# Hyperparameter search results

Search target: beat MWPM on plain memory (E2) at p=0.002. 58 total trials logged to `results/hparam_search.jsonl` (stage 1 ranking + stage 2 multi-seed confirmation).

## Stage 2 (confirmation) — mean +/- std over seeds

| config | mean GNN rate | std | n seeds | mean MWPM rate | beats MWPM |
|---|---|---|---|---|---|
| hd64_L6_transformer_h2_lr0.0003_bs128_wd0.0001 **<- winner** | 0.0014 | 0.0006 | 3 | 0.0008 | no |
| hd128_L5_transformer_h2_lr0.003_bs64_wd1e-05 | 0.0015 | 0.0008 | 3 | 0.0008 | no |
| hd128_L4_transformer_h4_lr0.003_bs64_wd0.0001 | 0.0015 | 0.0007 | 3 | 0.0008 | no |
| hd64_L4_transformer_h2_lr0.001_bs64_wd0.0001 | 0.0016 | 0.0006 | 3 | 0.0008 | no |
| hd64_L6_transformer_h4_lr0.0003_bs64_wd0.0001 | 0.0017 | 0.0006 | 3 | 0.0008 | no |
| hd128_L6_transformer_h2_lr0.0003_bs256_wd0.0001 | 0.0017 | 0.0006 | 3 | 0.0008 | no |

## Winning configuration

```
HparamConfig(hidden_dim=64, num_layers=6, conv_type='transformer', heads=2, lr=0.0003, batch_size=128, weight_decay=0.0001)
```

Mean GNN logical error rate: 0.0014, mean MWPM: 0.0008 (1.79x MWPM's rate, MWPM still wins).

This is the honest outcome of the search, stated plainly — a search is run to find out the answer, not to manufacture a specific one.

## Follow-up: is the winner undertrained?

The winner's learning rate (3e-4) is the lowest in the search space, which could mean
it just hadn't converged in 25 epochs. Ran it again with 3x the epochs (60) and 2x
the test shots (20000, to reduce Monte Carlo noise on a rate this low) across the
same 3 seeds:

| | mean logical error rate |
|---|---|
| GNN (60 epochs, 20000 test shots) | 0.00135 |
| MWPM (same test shots) | 0.00072 |

Ratio ~1.88x — essentially unchanged from the 25-epoch confirmation pass (1.79x).
**Not undertraining** — this is a converged, stable gap at this architecture/data
scale, not a training-budget artifact. Adopted as the new default hyperparameters
everywhere in this repo (`hidden_dim=64, num_layers=6, conv_type="transformer",
heads=2, lr=3e-4, batch_size=128, weight_decay=1e-4`), replacing the untuned
design-doc defaults (`hidden_dim=64, num_layers=4, conv_type="gat", heads=4,
lr=1e-3`) used everywhere before this search.
