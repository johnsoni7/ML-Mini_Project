# Literature review

Notes on what each source says and what it means for Signo. **Published numbers are not comparable with ours unless stated.** Most use a different vocabulary size, split or metric.

## Sources that were read

| Source | Problem | Model and input | Reported result | Use for Signo | Doable in 10 hours? |
|---|---|---|---|---:|---|
| Desai et al., ASL Citizen, NeurIPS 2023 ([arXiv](https://arxiv.org/abs/2304.05934)) | Signer-independent recognition for dictionary lookup, 2,731 signs, 52 signers | I3D on 64 RGB frames. ST-GCN on 27 MediaPipe keypoints, shoulder-normalised, up to 128 frames, shear and rotation augmentation | I3D Rec@1 63.10%, Rec@5 86.09%. ST-GCN Rec@1 59.52%, Rec@5 82.68%. On a 1,075-sign subset, 74.16% top-1 | Confirms the protocol: signers held out, best validation checkpoint used for test. Rec@1 equals top-1 | Already followed |
| [Microsoft ASL-citizen-code](https://github.com/microsoft/ASL-citizen-code) | Official baselines | `pose.py` stores x and y only for 543 points. `asl_citizen_dataset_pose.py` centres on shoulders and scales by shoulder distance, using means over the clip, 128 frames | Example checkpoint name shows validation 73.64%, which is only an example | Uses up to 128 frames against our 30, so frame count is worth testing | Yes |
| Vandendriessche et al., [One-shot ISLR](https://arxiv.org/abs/2502.20171) | Pose-based sign embeddings across languages | PoseFormer: input convolutions, frame embedding, intermediate convolutions, self-attention, on MediaPipe keypoints | Pretraining on ASL Citizen: Rec@1 about 0.75, which is 0.1199 above I3D. Removing the frame embedding costs 0.0278 Rec@1, removing intermediate convolutions costs 0.018, input convolutions have limited impact | **Strong lead.** Our BiLSTM reads coordinates directly and has no frame embedding. It also shows pose can beat RGB here | Yes, frame embedding and conv plus attention |
| Hoyeol Sohn, [Kaggle ISLR 1st place notebook](https://github.com/hoyso48/Google---Isolated-Sign-Language-Recognition-1st-place-solution) | 250 signs from MediaPipe landmarks | 1D conv blocks (kernel 17) plus transformer blocks. Input: lips, both hands, nose, both eyes, no body points. x and y plus two motion channels (change over 1 and 2 frames). Up to 384 frames. Resampling 0.5x to 1.5x, flip, affine, temporal and spatial masking | Only the code was read, so no score is quoted here | **Strong lead.** Face points and motion are direct inputs, and Signo has only the nose from the face | Yes, but needs new features in the browser |
| [Habr write-up, 2nd place of the same competition](https://habr.com/en/articles/795109) | Same | CNN on 80 points, including lips and hands, and a transformer on 61 points. Per-point motion, pairwise hand and lip distances, finger angles. Affine, time interpolation, finger rotation, masking, mixup, cutmix, label smoothing 0.50 to 0.65, 100 to 210 epochs | Leaderboard 0.80 image model, 0.78 transformer | Supports lips, motion and heavy augmentation, and long training | Partly |
| dos Santos et al., [Landmark subsets for LIBRAS](https://arxiv.org/html/2510.24887v3) | Which landmarks to use, leave-one-person-out | Skeleton images into ResNet-18. Best subset: lips, hands and pose (80 points). Spline filling of missing points | F1 0.94 (subset) against 0.66 (all points) on MINDS-Libras. Filling gaps adds 4 to 6 points | Supports a chosen face subset over all points, and filling hand gaps | Yes |
| Kvanchiani et al., [Training strategies for ISLR](https://arxiv.org/html/2412.11553v1) | RGB only | MViTv2 and I3D, speed changes, random frame drop and add, boundary shift, mixup and cutmix | WLASL 62.89, Slovo 78.21%. Removing image augmentation and mixing costs 4.06 points | Different input. Supports speed augmentation and mixing | Mixing only |
| Bai et al., [TCN vs recurrent](https://arxiv.org/abs/1803.01271) (abstract only) | General sequence tasks | Dilated convolution | Simple convolution beats canonical LSTMs across many tasks | Worth testing, and earlier Signo tests were cut short | Yes |

## Sources that could not be read

These are recorded so nobody assumes they were used.

| Source | What happened |
|---|---|
| Varanasi et al., CVPRW 2026 | 403 Forbidden. The listing's full title says Indian Sign Language, so it is not an ASL Citizen paper |
| Li et al., AAAI 2021 (`ojs.aaai.org/.../16247`) | Blocked by a bot check |
| PoET (HAL, 2026) | Blocked by a bot check. A search summary says it evaluates a pose encoder on ASL Citizen, but no number was seen |
| Kaggle discussion page of the 1st-place solution | Only the title loaded. The notebook was read instead |

## What this changes

1. **Face points.** Every strong landmark solution here includes lips or face points. Signo has only the nose. Many ASL signs differ by where the hand touches the face, so this is the first feature to test.
2. **Frame embedding.** A measured +2.78 points in the PoseFormer ablation, and cheap to add.
3. **Motion input.** Used by both top Kaggle solutions.
4. **Filling hand gaps.** Measured gain in the LIBRAS paper, and our audit shows gaps are common.
5. **Conv plus attention.** Needs a fair test, as earlier Signo conv runs stopped at 34 to 72 epochs on a time budget.

Not chosen: graph networks (the earlier attempt reached 3 epochs in 17 minutes, and ST-GCN scored below I3D in the ASL Citizen paper) and RGB models (no browser path at this size).
