# Progress

**Run finished at 06:36 on 2026-11-03. No experiments are running. The final results are in `results/FINAL_RESULTS.md`.** The remaining 39 minutes were too short for another full training run, and the test clips have been used.

Updated 2026-11-03 05:20:48. New experiments stop in -1.1 hours. Disk used by this run: 0.91 GB of 10 GB.

Baseline (final_v2 (deployed)): validation top-1 79.38%, macro F1 78.87%.

Best so far: **exp_022 full_tcn_fair**, top-1 84.32%, macro F1 83.94%.

## Running now
- nothing

## Waiting (10)
- full_tcn_fair+seed1 (priority 7)
- full_tcn_fair+dropout (priority 6)
- full_tcn_fair+moretime (priority 6)
- full_tcn_fair+smooth (priority 6)
- tcn_always_s1+dropout (priority 6)
- tcn_always_s1+moretime (priority 6)
- tcn_always_s1+smooth (priority 6)
- tcn_always_s2+dropout (priority 6)
- tcn_always_s2+moretime (priority 6)
- tcn_always_s2+smooth (priority 6)

## Failed
- full_mixup_300: {"epoch": 175, "of": 300, "val_top1": 0.7991, "s": 7148}
- full_mixup_stack2: Stopped by hand at epoch 28: 102 s per epoch would need 5 hours, and two recurrent layers would double browser time
- full_mixup_velocity: {"epoch": 6, "of": 180, "val_top1": 0.14, "s": 177}
- full_cnn_attn_fair: Stopped by hand at epoch 86: best 68.7%, while the TCN was at 80.8% by epoch 72
- tcn_always_u96: {"epoch": 17, "of": 130, "val_top1": 0.6686, "s": 1876}

## Finished, best first

| id | name | features | top-1 | top-5 | macro F1 | gap | params | partial data |
|---|---|---|---:|---:|---:|---:|---:|---|
| exp_022 | full_tcn_fair | base | 84.32% | 96.63% | 83.94% | 15.6% | 1,322,344 |  |
| exp_028 | tcn_always_s1 | base | 83.84% | 96.89% | 83.41% | 15.8% | 1,322,344 |  |
| exp_029 | tcn_always_s2 | base | 83.60% | 96.86% | 83.09% | 16.2% | 1,322,344 |  |
| exp_025 | full_mixup_always+moretime | base | 83.15% | 96.92% | 82.49% | 13.1% | 396,392 |  |
| exp_017 | full_mixup_always | base | 82.12% | 96.52% | 81.44% | 12.4% | 396,392 |  |
| exp_024 | full_mixup_always+seed1 | base | 82.04% | 96.78% | 81.52% | 12.6% | 396,392 |  |
| exp_019 | full_mixup_u192 | base | 81.34% | 96.31% | 80.56% | 17.8% | 716,968 |  |
| exp_026 | always_velocity | base | 81.26% | 96.44% | 80.63% | 13.1% | 518,792 |  |
| exp_027 | full_mixup_always+smooth | base | 81.19% | 96.47% | 80.59% | 11.5% | 396,392 |  |
| exp_016 | full_mixup_smooth | base | 80.79% | 95.96% | 80.36% | 14.1% | 396,392 |  |
| exp_015 | full_mixup | base | 80.49% | 95.30% | 79.86% | 15.9% | 396,392 |  |
| exp_020 | full_mixup_body | base | 79.67% | 95.48% | 79.25% | 15.2% | 396,392 |  |
| exp_014 | gate_body_mix | g_base | 60.88% | 84.96% | 58.50% | 38.2% | 396,392 | 5000 |
| exp_009 | gate_mixup | g_base | 59.95% | 83.66% | 57.63% | 39.4% | 396,392 | 5000 |
| exp_011 | gate_smooth | g_base | 57.48% | 83.71% | 55.06% | 41.5% | 396,392 | 5000 |
| exp_006 | gate_velocity | g_base | 56.76% | 81.19% | 54.57% | 42.9% | 518,792 | 5000 |
| exp_003 | gate_interp | g_interp | 56.23% | 80.89% | 53.94% | 43.4% | 396,392 | 5000 |
| exp_005 | gate_body_aug | g_base | 55.81% | 81.50% | 53.60% | 43.1% | 396,392 | 5000 |
| exp_010 | gate_linear | g_linear | 55.62% | 81.19% | 53.40% | 44.1% | 396,392 | 5000 |
| exp_001 | gate_base | g_base | 55.51% | 80.89% | 53.51% | 44.0% | 396,392 | 5000 |
| exp_012 | gate_t45 | g_t45 | 52.86% | 79.09% | 50.52% | 46.5% | 396,392 | 5000 |
| exp_002 | gate_face | g_face | 48.74% | 76.27% | 46.24% | 48.3% | 452,712 | 5000 |
| exp_013 | gate_face_interp | g_face_interp | 46.13% | 74.70% | 43.59% | 51.5% | 452,712 | 5000 |
| exp_004 | gate_embed | g_base | 43.37% | 70.48% | 40.99% | 56.4% | 647,784 | 5000 |
| exp_008 | gate_cnn_attn | g_base | 9.81% | 29.76% | 6.77% | 8.9% | 720,040 | 5000 |
| exp_007 | gate_tcn | g_base | 0.08% | 0.43% | 0.00% | 0.2% | 1,082,024 | 5000 |
