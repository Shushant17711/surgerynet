# Edge-feature addition: A/B check + small re-tune

| variant | E2 (memory) mean rate | E5 (surgery) mean rate |
|---|---|---|
| previous_tuned_no_edge_feat | 0.0013 (n=3) | 0.0660 (n=3) |
| previous_tuned_with_edge_feat | 0.0013 (n=3) | 0.0625 (n=3) |
| retuned_with_edge_feat | 0.0011 (n=3) | 0.0573 (n=3) |

`previous_tuned_*` uses `results/HPARAM_SEARCH.md`'s winner ({'hidden_dim': 64, 'num_layers': 6, 'conv_type': 'transformer', 'heads': 2, 'lr': 0.0003, 'weight_decay': 0.0001}); `retuned_with_edge_feat` is a small follow-up check around it once edge features are on ({'hidden_dim': 128, 'num_layers': 6, 'conv_type': 'transformer', 'heads': 2, 'lr': 0.001, 'weight_decay': 0.0001}), not a full re-run of `hparam_search.py`'s 58-trial search.
