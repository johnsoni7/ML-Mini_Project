# Research log

Written automatically by `python/auto_optimize.py` as each run finishes. Read these notes first:

- The first three quick tests (`gate_base`, `gate_face`, `gate_interp`) compare with `f_long_s0` because their reference was not yet set. That comparison is unfair, since they used 5,000 clips. Compare them with `gate_base` instead (55.51%).
- `gate_tcn` and `gate_cnn_attn` failed from a setup fault, not because of the architecture. See `research/DECISIONS.md`.
- Runs stopped by hand are listed in `research/DECISIONS.md` and `experiments/queue.json`, not here.

## exp_001: gate_base

- Started 2026-11-02 22:49:40, finished 2026-11-02 23:05:15, took 15 minutes
- Hypothesis: Control for the 5,000-clip gates, the same data as the 30 x 100 pipeline
- Change: `{"arch": "bilstm", "batch": 64, "drop": true, "dropout": 0.4, "epochs": 200, "features": "g_base", "geometry": true, "lr": 0.002, "max_minutes": 240, "mirror": true, "seed": 0, "threads": 3, "train_limit": 5000, "units": 128, "warp": true}`
- Queued by: diagnosis, quick gate on 5,000 training clips
- Validation: top-1 55.51%, top-5 80.89%, macro F1 53.51%, on 3763 clips. Clean training accuracy 99.53%, gap 44.0%. 396,392 parameters
- Against f_long_s0 (single BiLSTM, 180 epochs): -22.51% (95% interval -24.13% to -20.89%, 3763 shared clips). Real loss.
- Next, from this result: nothing new queued

## exp_003: gate_interp

- Started 2026-11-02 22:49:40, finished 2026-11-02 23:05:46, took 16 minutes
- Hypothesis: Bridging hand gaps may help, since accuracy falls as tracking gaps grow
- Change: `{"arch": "bilstm", "batch": 64, "drop": true, "dropout": 0.4, "epochs": 200, "features": "g_interp", "geometry": true, "lr": 0.002, "max_minutes": 240, "mirror": true, "seed": 0, "threads": 3, "train_limit": 5000, "units": 128, "warp": true}`
- Queued by: diagnosis, quick gate on 5,000 training clips
- Validation: top-1 56.23%, top-5 80.89%, macro F1 53.94%, on 3763 clips. Clean training accuracy 99.60%, gap 43.4%. 396,392 parameters
- Against f_long_s0 (single BiLSTM, 180 epochs): -21.79% (95% interval -23.44% to -20.06%, 3763 shared clips). Real loss.
- Next, from this result: nothing new queued

## exp_002: gate_face

- Started 2026-11-02 22:49:40, finished 2026-11-02 23:06:16, took 16 minutes
- Hypothesis: Face points may separate signs that differ by location on the face
- Change: `{"arch": "bilstm", "batch": 64, "drop": true, "dropout": 0.4, "epochs": 200, "features": "g_face", "geometry": true, "lr": 0.002, "max_minutes": 240, "mirror": true, "seed": 0, "threads": 3, "train_limit": 5000, "units": 128, "warp": true}`
- Queued by: diagnosis, quick gate on 5,000 training clips
- Validation: top-1 48.74%, top-5 76.27%, macro F1 46.24%, on 3763 clips. Clean training accuracy 97.03%, gap 48.3%. 452,712 parameters
- Against f_long_s0 (single BiLSTM, 180 epochs): -29.29% (95% interval -30.99% to -27.61%, 3763 shared clips). Real loss.
- Next, from this result: nothing new queued

## exp_005: gate_body_aug

- Started 2026-11-02 23:07:36, finished 2026-11-02 23:36:19, took 28 minutes
- Hypothesis: Data is the limit, so imitate new signers: per-clip arm length, hand size and hand turn
- Change: `{"arch": "bilstm", "batch": 64, "body_aug": 1.0, "drop": true, "dropout": 0.4, "epochs": 200, "features": "g_base", "geometry": true, "lr": 0.002, "max_minutes": 240, "mirror": true, "seed": 0, "threads": 2, "train_limit": 5000, "units": 128, "warp": true}`
- Queued by: gate results: data is the main limit
- Validation: top-1 55.81%, top-5 81.50%, macro F1 53.60%, on 3763 clips. Clean training accuracy 98.90%, gap 43.1%. 396,392 parameters
- Against gate_base: +0.29% (95% interval -1.12% to +1.62%, 3763 shared clips). Within noise.
- Next, from this result: nothing new queued

## exp_009: gate_mixup

- Started 2026-11-02 23:07:36, finished 2026-11-02 23:36:19, took 28 minutes
- Hypothesis: Blending clips reduces memorising: 44-point gap on 5,000 clips
- Change: `{"arch": "bilstm", "batch": 64, "drop": true, "dropout": 0.4, "epochs": 200, "features": "g_base", "geometry": true, "lr": 0.002, "max_minutes": 240, "mirror": true, "mixup": 0.5, "seed": 0, "threads": 2, "train_limit": 5000, "units": 128, "warp": true}`
- Queued by: gate results: data is the main limit
- Validation: top-1 59.95%, top-5 83.66%, macro F1 57.63%, on 3763 clips. Clean training accuracy 99.40%, gap 39.4%. 396,392 parameters
- Against gate_base: +4.44% (95% interval +3.08% to +5.82%, 3763 shared clips). Real gain.
- Next, from this result: nothing new queued

## exp_006: gate_velocity

- Started 2026-11-02 23:07:36, finished 2026-11-02 23:43:22, took 36 minutes
- Hypothesis: Motion as direct input, used by the top Kaggle solutions
- Change: `{"arch": "bilstm", "batch": 64, "drop": true, "dropout": 0.4, "epochs": 200, "features": "g_base", "geometry": true, "lr": 0.002, "max_minutes": 240, "mirror": true, "seed": 0, "threads": 2, "train_limit": 5000, "units": 128, "velocity": true, "warp": true}`
- Queued by: diagnosis, quick gate on 5,000 training clips
- Validation: top-1 56.76%, top-5 81.19%, macro F1 54.57%, on 3763 clips. Clean training accuracy 99.63%, gap 42.9%. 518,792 parameters
- Against gate_base: +1.25% (95% interval -0.11% to +2.60%, 3763 shared clips). Within noise.
- Next, from this result: nothing new queued

## exp_004: gate_embed

- Started 2026-11-02 23:07:36, finished 2026-11-02 23:59:29, took 52 minutes
- Hypothesis: Frame embedding: +2.78 points in the PoseFormer ablation
- Change: `{"arch": "bilstm", "batch": 64, "drop": true, "dropout": 0.4, "embed": [256, 256], "epochs": 200, "features": "g_base", "geometry": true, "lr": 0.002, "max_minutes": 240, "mirror": true, "seed": 0, "threads": 2, "train_limit": 5000, "units": 128, "warp": true}`
- Queued by: diagnosis, quick gate on 5,000 training clips
- Validation: top-1 43.37%, top-5 70.48%, macro F1 40.99%, on 3763 clips. Clean training accuracy 99.77%, gap 56.4%. 647,784 parameters
- Against gate_base: -12.14% (95% interval -13.66% to -10.63%, 3763 shared clips). Real loss.
- Next, from this result: nothing new queued

## exp_008: gate_cnn_attn

- Started 2026-11-02 23:07:36, finished 2026-11-03 00:02:31, took 54 minutes
- Hypothesis: Convolution plus attention, as in the Kaggle winner and PoseFormer
- Change: `{"arch": "cnn_attn", "batch": 64, "drop": true, "dropout": 0.4, "epochs": 200, "features": "g_base", "geometry": true, "lr": 0.001, "max_minutes": 240, "mirror": true, "seed": 0, "threads": 2, "train_limit": 5000, "units": 64, "warmup": 5, "warp": true}`
- Queued by: diagnosis, quick gate on 5,000 training clips
- Validation: top-1 9.81%, top-5 29.76%, macro F1 6.77%, on 3763 clips. Clean training accuracy 18.67%, gap 8.9%. 720,040 parameters
- Against gate_base: -45.71% (95% interval -47.41% to -44.14%, 3763 shared clips). Real loss.
- Next, from this result: nothing new queued

## exp_010: gate_linear

- Started 2026-11-02 23:36:19, finished 2026-11-03 00:05:32, took 29 minutes
- Hypothesis: Blending frames instead of nearest-frame sampling may keep more movement
- Change: `{"arch": "bilstm", "batch": 64, "drop": true, "dropout": 0.4, "epochs": 200, "features": "g_linear", "geometry": true, "lr": 0.002, "max_minutes": 240, "mirror": true, "seed": 0, "threads": 2, "train_limit": 5000, "units": 128, "warp": true}`
- Queued by: diagnosis, quick gate on 5,000 training clips
- Validation: top-1 55.62%, top-5 81.19%, macro F1 53.40%, on 3763 clips. Clean training accuracy 99.70%, gap 44.1%. 396,392 parameters
- Against gate_base: +0.11% (95% interval -1.25% to +1.38%, 3763 shared clips). Within noise.
- Next, from this result: nothing new queued

## exp_011: gate_smooth

- Started 2026-11-02 23:36:19, finished 2026-11-03 00:06:33, took 30 minutes
- Hypothesis: Label smoothing, as 0.1 helped earlier and 2nd place Kaggle used far more
- Change: `{"arch": "bilstm", "batch": 64, "drop": true, "dropout": 0.4, "epochs": 200, "features": "g_base", "geometry": true, "lr": 0.002, "max_minutes": 240, "mirror": true, "seed": 0, "smoothing": 0.2, "threads": 2, "train_limit": 5000, "units": 128, "warp": true}`
- Queued by: gate results: data is the main limit
- Validation: top-1 57.48%, top-5 83.71%, macro F1 55.06%, on 3763 clips. Clean training accuracy 98.93%, gap 41.5%. 396,392 parameters
- Against gate_base: +1.97% (95% interval +0.56% to +3.43%, 3763 shared clips). Real gain.
- Next, from this result: nothing new queued

## exp_007: gate_tcn

- Started 2026-11-02 23:07:36, finished 2026-11-03 00:17:08, took 69 minutes
- Hypothesis: Temporal convolution, with a fair training length
- Change: `{"arch": "tcn", "batch": 64, "drop": true, "dropout": 0.4, "epochs": 200, "features": "g_base", "geometry": true, "lr": 0.001, "max_minutes": 240, "mirror": true, "seed": 0, "threads": 2, "train_limit": 5000, "units": 64, "warmup": 5, "warp": true}`
- Queued by: diagnosis, quick gate on 5,000 training clips
- Validation: top-1 0.08%, top-5 0.43%, macro F1 0.00%, on 3763 clips. Clean training accuracy 0.23%, gap 0.2%. 1,082,024 parameters
- Against gate_base: -55.43% (95% interval -56.98% to -53.89%, 3763 shared clips). Real loss.
- Next, from this result: nothing new queued

## exp_012: gate_t45

- Started 2026-11-02 23:43:22, finished 2026-11-03 00:27:12, took 43 minutes
- Hypothesis: 45 frames. The diagnosis predicts little gain, so this checks it
- Change: `{"arch": "bilstm", "batch": 64, "drop": true, "dropout": 0.4, "epochs": 200, "features": "g_t45", "geometry": true, "lr": 0.002, "max_minutes": 240, "mirror": true, "seed": 0, "threads": 2, "train_limit": 5000, "units": 128, "warp": true}`
- Queued by: diagnosis, quick gate on 5,000 training clips
- Validation: top-1 52.86%, top-5 79.09%, macro F1 50.52%, on 3763 clips. Clean training accuracy 99.37%, gap 46.5%. 396,392 parameters
- Against gate_base: -2.66% (95% interval -4.07% to -1.20%, 3763 shared clips). Real loss.
- Next, from this result: nothing new queued

## exp_013: gate_face_interp

- Started 2026-11-02 23:59:30, finished 2026-11-03 00:39:48, took 40 minutes
- Hypothesis: Both feature changes together
- Change: `{"arch": "bilstm", "batch": 64, "drop": true, "dropout": 0.4, "epochs": 200, "features": "g_face_interp", "geometry": true, "lr": 0.002, "max_minutes": 240, "mirror": true, "seed": 0, "threads": 2, "train_limit": 5000, "units": 128, "warp": true}`
- Queued by: diagnosis, quick gate on 5,000 training clips
- Validation: top-1 46.13%, top-5 74.70%, macro F1 43.59%, on 3763 clips. Clean training accuracy 97.63%, gap 51.5%. 452,712 parameters
- Against gate_base: -9.38% (95% interval -10.92% to -7.89%, 3763 shared clips). Real loss.
- Next, from this result: nothing new queued

## exp_014: gate_body_mix

- Started 2026-11-03 00:02:31, finished 2026-11-03 00:41:49, took 39 minutes
- Hypothesis: Both together
- Change: `{"arch": "bilstm", "batch": 64, "body_aug": 1.0, "drop": true, "dropout": 0.4, "epochs": 200, "features": "g_base", "geometry": true, "lr": 0.002, "max_minutes": 240, "mirror": true, "mixup": 0.5, "seed": 0, "threads": 2, "train_limit": 5000, "units": 128, "warp": true}`
- Queued by: gate results: data is the main limit
- Validation: top-1 60.88%, top-5 84.96%, macro F1 58.50%, on 3763 clips. Clean training accuracy 99.03%, gap 38.2%. 396,392 parameters
- Against gate_base: +5.37% (95% interval +3.96% to +6.80%, 3763 shared clips). Real gain.
- Next, from this result: nothing new queued

## exp_015: full_mixup

- Started 2026-11-03 00:17:08, finished 2026-11-03 02:23:31, took 126 minutes
- Hypothesis: Mixup gained 4.4 points on 5,000 clips (interval +3.1 to +5.8). Does it hold on all training clips?
- Change: `{"arch": "bilstm", "batch": 64, "drop": true, "dropout": 0.4, "epochs": 180, "features": "base", "geometry": true, "lr": 0.002, "max_minutes": 240, "mirror": true, "mixup": 0.5, "seed": 0, "threads": 2, "units": 128, "warp": true}`
- Queued by: gate results
- Validation: top-1 80.49%, top-5 95.30%, macro F1 79.86%, on 3763 clips. Clean training accuracy 96.40%, gap 15.9%. 396,392 parameters
- Against f_long_s0 (single BiLSTM, 180 epochs): +2.47% (95% interval +1.41% to +3.61%, 3763 shared clips). Real gain.
- Next, from this result: full_mixup+dropout: gap still 15.9% with mixup, so raise dropout; full_mixup+moretime: best epoch 178 of 180, so it had not finished learning; full_mixup+smooth: top-5 is 14.8% above top-1, so soften targets to separate close signs; full_mixup+seed1: repeat with a new seed to see whether this result is stable

## exp_017: full_mixup_always

- Started 2026-11-03 00:17:08, finished 2026-11-03 02:24:32, took 127 minutes
- Hypothesis: Stronger blending, every batch instead of half
- Change: `{"arch": "bilstm", "batch": 64, "drop": true, "dropout": 0.4, "epochs": 180, "features": "base", "geometry": true, "lr": 0.002, "max_minutes": 240, "mirror": true, "mixup": 1.0, "seed": 0, "threads": 2, "units": 128, "warp": true}`
- Queued by: gate results
- Validation: top-1 82.12%, top-5 96.52%, macro F1 81.44%, on 3763 clips. Clean training accuracy 94.53%, gap 12.4%. 396,392 parameters
- Against f_long_s0 (single BiLSTM, 180 epochs): +4.09% (95% interval +2.98% to +5.26%, 3763 shared clips). Real gain.
- Next, from this result: full_mixup_always+moretime: best epoch 179 of 180, so it had not finished learning; full_mixup_always+smooth: top-5 is 14.4% above top-1, so soften targets to separate close signs; full_mixup_always+seed1: repeat with a new seed to see whether this result is stable

## exp_016: full_mixup_smooth

- Started 2026-11-03 00:17:08, finished 2026-11-03 02:26:02, took 128 minutes
- Hypothesis: Label smoothing was +2.0 on the small set, so test it on top of mixup
- Change: `{"arch": "bilstm", "batch": 64, "drop": true, "dropout": 0.4, "epochs": 180, "features": "base", "geometry": true, "lr": 0.002, "max_minutes": 240, "mirror": true, "mixup": 0.5, "seed": 0, "smoothing": 0.1, "threads": 2, "units": 128, "warp": true}`
- Queued by: gate results
- Validation: top-1 80.79%, top-5 95.96%, macro F1 80.36%, on 3763 clips. Clean training accuracy 94.90%, gap 14.1%. 396,392 parameters
- Against f_long_s0 (single BiLSTM, 180 epochs): +2.76% (95% interval +1.65% to +3.88%, 3763 shared clips). Real gain.
- Next, from this result: nothing new queued

## exp_020: full_mixup_body

- Started 2026-11-03 01:30:39, finished 2026-11-03 03:05:43, took 95 minutes
- Hypothesis: On 5,000 clips mixup plus hand and arm variation scored 60.9%, against 59.95% for mixup alone. Check on all clips
- Change: `{"arch": "bilstm", "batch": 64, "body_aug": 1.0, "drop": true, "dropout": 0.4, "epochs": 180, "features": "base", "geometry": true, "lr": 0.002, "max_minutes": 194.34211463133494, "mirror": true, "mixup": 0.5, "seed": 0, "threads": 2, "units": 128, "warp": true}`
- Queued by: gate results
- Validation: top-1 79.67%, top-5 95.48%, macro F1 79.25%, on 3763 clips. Clean training accuracy 94.90%, gap 15.2%. 396,392 parameters
- Against f_long_s0 (single BiLSTM, 180 epochs): +1.65% (95% interval +0.56% to +2.82%, 3763 shared clips). Real gain.
- Next, from this result: nothing new queued

## exp_019: full_mixup_u192

- Started 2026-11-03 00:39:48, finished 2026-11-03 03:22:17, took 162 minutes
- Hypothesis: With less memorising the model may use more capacity
- Change: `{"arch": "bilstm", "batch": 64, "drop": true, "dropout": 0.4, "epochs": 180, "features": "base", "geometry": true, "lr": 0.002, "max_minutes": 240, "mirror": true, "mixup": 0.5, "seed": 0, "threads": 2, "units": 192, "warp": true}`
- Queued by: gate results
- Validation: top-1 81.34%, top-5 96.31%, macro F1 80.56%, on 3763 clips. Clean training accuracy 99.10%, gap 17.8%. 716,968 parameters
- Against f_long_s0 (single BiLSTM, 180 epochs): +3.32% (95% interval +2.21% to +4.49%, 3763 shared clips). Real gain.
- Next, from this result: full_mixup_u192+dropout: gap still 17.8% with mixup, so raise dropout; full_mixup_u192+moretime: best epoch 174 of 180, so it had not finished learning; full_mixup_u192+smooth: top-5 is 15.0% above top-1, so soften targets to separate close signs

## exp_024: full_mixup_always+seed1

- Started 2026-11-03 02:27:03, finished 2026-11-03 03:33:19, took 66 minutes
- Hypothesis: repeat with a new seed to see whether this result is stable
- Change: `{"arch": "bilstm", "batch": 64, "drop": true, "dropout": 0.4, "epochs": 180, "features": "base", "geometry": true, "lr": 0.002, "max_minutes": 137.94129966497422, "mirror": true, "mixup": 1.0, "seed": 1, "threads": 2, "units": 128, "warp": true}`
- Queued by: controller
- Validation: top-1 82.04%, top-5 96.78%, macro F1 81.52%, on 3763 clips. Clean training accuracy 94.63%, gap 12.6%. 396,392 parameters
- Against full_mixup_always: -0.08% (95% interval -1.06% to +0.96%, 3763 shared clips). Within noise.
- Next, from this result: full_mixup_always+seed1+moretime: best epoch 171 of 180, so it had not finished learning; full_mixup_always+seed1+smooth: top-5 is 14.7% above top-1, so soften targets to separate close signs

## exp_025: full_mixup_always+moretime

- Started 2026-11-03 02:27:03, finished 2026-11-03 03:56:55, took 89 minutes
- Hypothesis: best epoch 179 of 180, so it had not finished learning
- Change: `{"arch": "bilstm", "batch": 64, "drop": true, "dropout": 0.4, "epochs": 240, "features": "base", "geometry": true, "lr": 0.002, "max_minutes": 137.9411970337232, "mirror": true, "mixup": 1.0, "seed": 0, "threads": 2, "units": 128, "warp": true}`
- Queued by: controller
- Validation: top-1 83.15%, top-5 96.92%, macro F1 82.49%, on 3763 clips. Clean training accuracy 96.27%, gap 13.1%. 396,392 parameters
- Against full_mixup_always: +1.04% (95% interval -0.03% to +2.07%, 3763 shared clips). Within noise.
- Next, from this result: full_mixup_always+moretime+smooth: top-5 is 13.8% above top-1, so soften targets to separate close signs; full_mixup_always+moretime+seed1: repeat with a new seed to see whether this result is stable

## exp_026: always_velocity

- Started 2026-11-03 03:05:43, finished 2026-11-03 04:39:06, took 93 minutes
- Hypothesis: Motion input on top of the best setting
- Change: `{"arch": "bilstm", "batch": 64, "drop": true, "dropout": 0.4, "epochs": 180, "features": "base", "geometry": true, "lr": 0.002, "max_minutes": 99.28121918439865, "mirror": true, "mixup": 1.0, "seed": 0, "threads": 2, "units": 128, "velocity": true, "warp": true}`
- Queued by: after exp_017
- Validation: top-1 81.26%, top-5 96.44%, macro F1 80.63%, on 3763 clips. Clean training accuracy 94.40%, gap 13.1%. 518,792 parameters
- Against full_mixup_always: -0.85% (95% interval -1.91% to +0.24%, 3763 shared clips). Within noise.
- Next, from this result: nothing new queued

## exp_027: full_mixup_always+smooth

- Started 2026-11-03 03:22:17, finished 2026-11-03 04:44:38, took 82 minutes
- Hypothesis: top-5 is 14.4% above top-1, so soften targets to separate close signs
- Change: `{"arch": "bilstm", "batch": 64, "drop": true, "dropout": 0.4, "epochs": 180, "features": "base", "geometry": true, "lr": 0.002, "max_minutes": 82.70993640025456, "mirror": true, "mixup": 1.0, "seed": 0, "smoothing": 0.2, "threads": 2, "units": 128, "warp": true}`
- Queued by: controller
- Validation: top-1 81.19%, top-5 96.47%, macro F1 80.59%, on 3763 clips. Clean training accuracy 92.73%, gap 11.5%. 396,392 parameters
- Against full_mixup_always: -0.93% (95% interval -1.94% to +0.05%, 3763 shared clips). Within noise.
- Next, from this result: nothing new queued

## exp_022: full_tcn_fair

- Started 2026-11-03 02:24:32, finished 2026-11-03 04:46:08, took 141 minutes
- Hypothesis: Earlier TCN was stuck at chance from a setup fault (no normalisation before convolution, narrow head with heavy dropout). Fair test with those fixed
- Change: `{"arch": "tcn", "batch": 64, "drop": true, "dropout": 0.2, "epochs": 180, "features": "base", "geometry": true, "head": 256, "lr": 0.001, "max_minutes": 140.45845609903336, "mirror": true, "mixup": 0.5, "prenorm": true, "seed": 0, "threads": 2, "units": 64, "warmup": 5, "warp": true}`
- Queued by: debugging the failed gates
- Validation: top-1 84.32%, top-5 96.63%, macro F1 83.94%, on 3763 clips. Clean training accuracy 99.97%, gap 15.6%. 1,322,344 parameters
- Against f_long_s0 (single BiLSTM, 180 epochs): +6.30% (95% interval +5.13% to +7.52%, 3763 shared clips). Real gain.
- Next, from this result: full_tcn_fair+dropout: gap still 15.6% with mixup, so raise dropout; full_tcn_fair+moretime: best epoch 148 of 158, so it had not finished learning; full_tcn_fair+smooth: top-5 is 12.3% above top-1, so soften targets to separate close signs; full_tcn_fair+seed1: repeat with a new seed to see whether this result is stable

## exp_028: tcn_always_s1

- Started 2026-11-03 03:29:48, finished 2026-11-03 05:18:47, took 109 minutes
- Hypothesis: Second seed: is the TCN stable, and does a pair improve it? The browser has room for several
- Change: `{"arch": "tcn", "batch": 64, "drop": true, "dropout": 0.2, "epochs": 130, "features": "base", "geometry": true, "head": 256, "lr": 0.001, "max_minutes": 110.19247504870097, "mirror": true, "mixup": 1.0, "prenorm": true, "seed": 1, "threads": 2, "units": 64, "warmup": 5, "warp": true}`
- Queued by: TCN results and browser timing
- Validation: top-1 83.84%, top-5 96.89%, macro F1 83.41%, on 3763 clips. Clean training accuracy 99.63%, gap 15.8%. 1,322,344 parameters
- Against full_tcn_fair: -0.48% (95% interval -1.44% to +0.45%, 3763 shared clips). Within noise.
- Next, from this result: tcn_always_s1+dropout: gap still 15.8% with mixup, so raise dropout; tcn_always_s1+moretime: best epoch 119 of 130, so it had not finished learning; tcn_always_s1+smooth: top-5 is 13.0% above top-1, so soften targets to separate close signs

## exp_029: tcn_always_s2

- Started 2026-11-03 03:33:19, finished 2026-11-03 05:20:48, took 107 minutes
- Hypothesis: Third seed for stability and combining
- Change: `{"arch": "tcn", "batch": 64, "drop": true, "dropout": 0.2, "epochs": 130, "features": "base", "geometry": true, "head": 256, "lr": 0.001, "max_minutes": 106.67844443321228, "mirror": true, "mixup": 1.0, "prenorm": true, "seed": 2, "threads": 2, "units": 64, "warmup": 5, "warp": true}`
- Queued by: TCN results and browser timing
- Validation: top-1 83.60%, top-5 96.86%, macro F1 83.09%, on 3763 clips. Clean training accuracy 99.77%, gap 16.2%. 1,322,344 parameters
- Against full_tcn_fair: -0.72% (95% interval -1.67% to +0.24%, 3763 shared clips). Within noise.
- Next, from this result: tcn_always_s2+dropout: gap still 16.2% with mixup, so raise dropout; tcn_always_s2+moretime: best epoch 126 of 127, so it had not finished learning; tcn_always_s2+smooth: top-5 is 13.3% above top-1, so soften targets to separate close signs
