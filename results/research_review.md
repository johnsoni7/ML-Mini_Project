# Research review: improving pose-based sign recognition

This review separates what papers report from what we measured. **Published numbers use different datasets, signers, vocabularies and splits, so none of them can be compared directly with Signo's scores.**

## What we could and couldn't read

| Source | Access | What it says that matters here |
|---|---|---|
| Desai et al., ASL Citizen, NeurIPS 2023 ([arXiv](https://arxiv.org/abs/2304.05934)) | Summary of the results | Baselines on all 2,731 signs with unseen signers: I3D (video) about 63.1% top-1, ST-GCN (pose) about 59.5% |
| Dafnis et al., LREC 2022 ([paper](https://aclanthology.org/2022.lrec-1.797/)) | Full text | Graph model on a corrected version of WLASL: 77.43% top-1, 94.54% top-5. Augments with random sampling, **mirroring**, rotation, scaling and shifting. Corrected gloss labels so each sign has one label |
| Tunga et al., WACV Workshops 2021 ([arXiv](https://arxiv.org/abs/2012.00781)) | Abstract only (the PDF refused access) | A GCN for spatial relations and BERT for time, combined by late fusion. Reports an improvement of "up to 5%" on WLASL over earlier pose methods |
| SignBart, 2025 ([arXiv](https://arxiv.org/abs/2506.21592)) | Full text | Reports 81.45% on "ASL-Citizen-1000", labelled validation. It doesn't say how the subset or split was made, so this can't be compared with ours. Normalising body and hands separately gave its largest ablation gain |
| 2nd place, Google ISLR Kaggle ([repository](https://github.com/ffs333/2nd_place_GISLR)) | Write-up | MediaPipe landmarks. Flips x-coordinates, rotates, and varies timing. Ensembling gave about 0.01 |
| SignGraph, IEEE Access 2023 ([DOI](https://doi.org/10.1109/ACCESS.2023.3247761)) | **Not read** | The DOI resolves, but the publisher page returned no content |
| Hand-Aware GCN, 2024 ([DOI](https://doi.org/10.1016/j.jiixd.2024.08.001)) | **Not read** | The DOI resolves, but the publisher page returned no content |
| MSE-GCN, IEEE TETCI ([DOI](https://doi.org/10.1109/TETCI.2024.3509500)) | **Not read** | The DOI resolves, but the publisher page returned no content |

We make no claims about the three unread papers.

## Techniques considered

| Technique | Why it might help | Feasibility here | What we measured |
|---|---|---|---|
| Fresh augmentation every epoch | Stops the model memorising fixed copies | Easy | Largest gain: 69.5% to 77.3% with a larger model and mirroring |
| Mirroring | Left-dominant signers look like flipped right-dominant signers | Easy. Used in published ASL work | Large gain for left-dominant signers |
| Time warping | Signers differ in speed | Easy | About 2 points |
| Label smoothing | Reduces overconfidence | Easy | About 1 point, within noise, so not adopted |
| AdamW weight decay | Limits large weights | Easy | No gain, so rejected |
| Stacked BiLSTM and CNN + BiLSTM | More capacity | Easy, and it loads in TensorFlow.js | Far behind within the time budget, because it trains slowly on CPU |
| GRU, temporal convolution only | Lighter alternatives | Easy, but not run | Not tested, because the evidence pointed elsewhere first |
| Lightweight ST-GCN | Uses the real skeleton | Built from layers TensorFlow.js supports, and fast in the browser | **Inconclusive.** About 5.5 minutes per epoch on this CPU, so it never really trained |
| Transformer | Long-range timing | Would need a substantial rewrite for TensorFlow.js | Not tested |
| Wrist-relative hands | Hand shape separated from position | Works inside the model, with the 30 x 100 input unchanged | No gain within the budget |
| Class weights, focal loss, balanced sampling | Help when classes are imbalanced | Easy | Not run: imbalance is only 11 to 17 clips per sign (1.5 to 1), and errors don't follow clip counts closely |
| Ensembles | Models make different mistakes | Possible as one TensorFlow.js model | Two models deployed (+1.4 points). Three rejected at about 291 ms |
| More data | More signers and examples | WLASL is the realistic source. Academic-use licence (C-UDA), and about 12,000 videos over 2,000 words | Not downloaded. See below |

## Additional data

Training on half the clips scored about 6.5 points lower, so more real clips should help. WLASL ([repository](https://github.com/dxli94/wlasl), [Hugging Face copy](https://huggingface.co/datasets/Voxel51/WLASL)) averages about 6 videos per word, and fewer would match our list. Extraction would take about an hour on the CPU used for training, and labels and possible signer overlap would need checking. A rough estimate of the gain is 1 to 3 points, so we left it as the top future step.
