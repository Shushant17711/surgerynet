# E6 extended: configuration generalization with 3 configs (k=1,2,3)

Supplementary to `results/main_table.md`'s E6 row, which uses only distances=(3,5) (1 train config, 1 held-out config -- as thin as this metric can get). This run uses distances=(3,5,7): 2 train configs, 1 held-out, still not many, but a real improvement.

| metric | mean ± std (n=8) |
|---|---|
| held-out logical error rate | 0.2518 ± 0.0763 |
| generalization gap (held-out minus train) | -0.0501 ± 0.1918 |

For comparison, `results/main_table.md`'s E6 row (distances=(3,5), n=8): held-out rate 0.2224 ± 0.1014.
