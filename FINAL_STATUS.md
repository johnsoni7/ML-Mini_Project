# Signo: final status

**Ready to demo, with one gap: the 1,000-sign model hasn't been tried on a real person's webcam.** Do that first, and allow 10 minutes for it.

## Start it

```bash
cd ~/Desktop/signo
.venv/bin/python -m http.server 8000 --bind 127.0.0.1
```

If it says the address is already in use, the server is already running, so carry on. Then open http://localhost:8000 in Chrome and press **Cmd+Shift+R** once to load the newest page. You need internet, because TensorFlow.js and MediaPipe load from a CDN.

## Demonstrate it

1. Check that the top bar says **Large ASL model (1000 signs)**. If it doesn't, pick it from the dropdown.
2. Click **Wake up Signo!** and allow the camera.
3. Keep your head, shoulders and both hands in frame.
4. Sign one word at normal speed, then lower your hands. Try **family, home, book, water** first.
5. Read the caption on the video and the top-3 bars.

The full five-minute talk and 20 viva answers are in `DEMO_SCRIPT.md`.

**If the large model does badly on you:** switch the dropdown to **Small ASL model (50 signs)** and try **more, friend, know, home, big**.

## The model

**Selected: `w1000_u48_acc`, a BiLSTM with 48 units, 1,000 signs and 110,872 parameters.** It had the best validation top-1 and top-5 of the six runs, and it's the smallest. Larger models fitted the training clips better without doing better on new people. The test set played no part in the choice.

| Measure | Validation | Test | Random guessing |
|---|---:|---:|---:|
| Top-1 | 69.5% | **61.4%** | 0.1% |
| Top-5 | 90.2% | **84.4%** | 0.5% |
| Clips | 3,763 | 3,983 | |
| Signers | 5 | 11 | |

Mean per-sign precision is 65.0% and recall is 61.5%. There are 14,626 training clips from 35 signers, and no signer appears in more than one split. Quote the test numbers in your report.

The 50-sign fallback scored 76.7% top-1 and 95.9% top-5 on its own test set. The two sets differ, so this isn't a direct comparison.

## What was checked

### Completed

| Task | Result |
|---|---|
| Training and selection | 6 runs compared on validation only |
| Test evaluation | Run once, logged in `results_1000/test_log.json` |
| TensorFlow.js export | Succeeded: input `[null, 30, 100]`, 1,000 outputs |
| Python vs TF.js | Same top-1 and top-3 order on 1,000 of 1,000 test sequences |
| Browser integration | Loads automatically as the default, with labels and output size checked |
| Fallback | Missing or corrupt large model: the app opens on the 50-sign model |
| Model switching | Labels and output size stay matched |
| Held-out videos in the browser | 60 clips: 65% top-1, 85% top-5, agrees with Python on 53 |
| Tests | All 21 pass |
| Git | Committed locally, not pushed, no videos or datasets included |

### Partly completed

| Task | Status |
|---|---|
| Fake-webcam test of the live app | The live loop, threshold, no-hands state and input size all work. Of 8 dataset clips on a pretend camera, three runs got 5 or 6 right, with 0 or 1 wrong. It uses dataset video, so it isn't a webcam test |
| Larger models | 128 and 256 units hit the 25-minute limit, so a longer run might beat the selected model |

### Unverified

| Task | Status |
|---|---|
| **Real webcam, 1,000-sign model** | Never tried |
| Real webcam, 50-sign model | Only your earlier informal try ("some of the words are being recognised") |
| Live webcam accuracy | Not measured for either model. The test scores come from fluent signers in dataset videos, so yours may well be lower |

### Not done

More seeds, the GRU comparison and `demo_webcam.py`. These were optional.

## Known issues

- **Rude words:** the 1,000-sign list came from word frequency, so it includes "bitch", "suck" and "damn". They can appear in the top-3 bars. If you'd rather avoid that risk with an audience, use the 50-sign model.
- **Challenge tab:** it wasn't tested with the 1,000-sign model, and it picks random words from the whole active list. That means obscure or rude words could come up, so use the 50-sign model for it.
- **Saved model choice:** if you pick or train another model in this browser, it stays selected next time, so check the dropdown before you start.
- **Single-sign scores are rough:** most signs have only 4 test clips, so don't make strong claims about one sign.
- **Test split:** it was capped at 4 clips per sign. Say so in your report.
- **"Not sure yet" with a high percentage:** this can happen because a word must also stay on top for 2 checks in a row. It's what stops wrong words flickering up.

## Before you submit

- Tell your guide about the change from WLASL to ASL Citizen, and that face features aren't used.
- **Put no dataset video frames in your report or slides.** The charts are fine.
- Ask your guide before pushing `app_model/` or `app_model_1000/` to GitHub, because the licence doesn't clearly cover trained models.
- Delete `~/signo_data` and any backups when the project is finished. The licence requires this.

## Figures for your report

| What | Where |
|---|---|
| All results and the model comparison | `results_1000/results.md` |
| Accuracy and loss curves | `results_1000/accuracy_curve.png`, `results_1000/loss_curve.png` |
| Confusion matrix (the 60 most-confused signs) | `results_1000/confusion_matrix.png` |
| Scores for every sign | `results_1000/per_class_metrics.csv` |
| Project description, architecture and limitations | `README.md` |
| Plain-language explanations | `BEGINNER_GUIDE.md` |
