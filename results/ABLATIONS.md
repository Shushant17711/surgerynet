# Ablation report

What breaks when each component is removed, relative to baseline.

| variant | mean logical error rate | std | n seeds | delta vs baseline |
|---|---|---|---|---|
| baseline (all components on) | 0.0622 | 0.0021 | 8 | +0.0000 |
| region/phase features removed | 0.0620 | 0.0035 | 8 | -0.0002 |
| DEM edges removed (radius-only) | 0.0640 | 0.0021 | 8 | +0.0019 |
| radius edges removed (DEM-only) | 0.0563 | 0.0016 | 8 | -0.0059 |
| shallower (4 layers vs 6) | 0.0633 | 0.0021 | 8 | +0.0012 |
| normalization removed | 0.0645 | 0.0042 | 8 | +0.0024 |

Positive delta = worse (higher logical error rate) than baseline when that component is removed.
