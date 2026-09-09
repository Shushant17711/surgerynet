# Ablation report

What breaks when each component is removed, relative to baseline.

| variant | mean logical error rate | std | n seeds | delta vs baseline |
|---|---|---|---|---|
| baseline (all components on) | 0.1120 | 0.0032 | 3 | +0.0000 |
| region/phase features removed | 0.1079 | 0.0068 | 3 | -0.0041 |
| DEM edges removed (radius-only) | 0.1113 | 0.0093 | 3 | -0.0007 |
| radius edges removed (DEM-only) | 0.1013 | 0.0029 | 3 | -0.0107 |
| deeper (6 layers vs 4) | 0.1171 | 0.0047 | 3 | +0.0051 |
| normalization removed | 0.1179 | 0.0053 | 3 | +0.0059 |

Positive delta = worse (higher logical error rate) than baseline when that component is removed.
