# Ablation report

What breaks when each component is removed, relative to baseline.

| variant | mean logical error rate | std | n seeds | delta vs baseline |
|---|---|---|---|---|
| baseline (all components on) | 0.0531 | 0.0021 | 8 | +0.0000 |
| region/phase features removed | 0.0534 | 0.0018 | 8 | +0.0003 |
| DEM edges removed (radius-only) | 0.0614 | 0.0021 | 8 | +0.0084 |
| radius edges removed (DEM-only) | 0.0536 | 0.0014 | 8 | +0.0006 |
| shallower (4 layers vs 6) | 0.0537 | 0.0023 | 8 | +0.0007 |
| normalization removed | 0.0518 | 0.0013 | 8 | -0.0012 |
| edge weights removed (topology only) | 0.0613 | 0.0016 | 8 | +0.0082 |

Positive delta = worse (higher logical error rate) than baseline when that component is removed.
