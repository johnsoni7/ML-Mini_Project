# Signo: final status (version 2)

**Ready to demo. The 50-sign model is above your 80% target at 91.2%. The 1,000-sign model improved to 73.4% but did not reach 80%. Neither has been measured on a real webcam, so try them for 10 minutes first.**

## Start it

```bash
cd ~/Desktop/signo
.venv/bin/python -m http.server 8000 --bind 127.0.0.1
```

If it says the address is already in use, the server is already running, so carry on. Open http://localhost:8000 in Chrome and press **Cmd+Shift+R** once. You need internet, because TensorFlow.js and MediaPipe load from a CDN. It takes about 10 seconds to become ready.

## Demonstrate it

1. Pick your model from the dropdown at the top:
   - **Accurate ASL model (50 signs):** the most reliable choice, and it runs more smoothly.
   - **Large ASL model (1000 signs):** the default, to show the wider vocabulary.
2. Click **Wake up Signo!** and allow the camera.
3. Keep your head, shoulders and both hands in frame.
4. Sign one word at normal speed, then lower your hands. Try **family, friend, boy, learn, understand** first.
5. Read the caption and the top-3 bars.

The five-minute talk and 22 viva answers are in `DEMO_SCRIPT.md`.

## Results

| Model | Test top-1 | Test top-5 | Validation top-1 | Random guessing (top-1) | Test clips |
|---|---:|---:|---:|---:|---:|
| **50 signs, version 2** | **91.2%** | **97.6%** | 95.2% | 2% | 614 |
| **1,000 signs, version 2** | **73.4%** | **92.0%** | 79.4% | 0.1% | 3,983 |
| 50 signs, version 1 | 76.7% | 95.9% | 82.3% | 2% | 614 |
| 1,000 signs, version 1 | 61.4% | 84.4% | 69.5% | 0.1% | 3,983 |

Test signers never appear in training. **Don't describe 91.2% as the 1,000-sign score, or 92.0% as top-1.** State which model and which measure you mean.

## What was checked

### Completed

| Task | Result |
|---|---|
| Model selection | Validation only, across 19 configurations for 1,000 signs and 4 runs for 50 signs |
| Test evaluation | Logged. Each split has now been evaluated twice |
| TensorFlow.js export | Both succeeded, with input `[null, 30, 100]` |
| Python vs TF.js | Same top-1 and top-3 order on 1,000 of 1,000 and 50 of 50 test sequences |
| Test videos through the browser | 1,000 signs: 60 clips, 76.7% top-1. 50 signs: 50 clips, 90% top-1 |
| Switching and fallback | Labels stay matched. A missing or corrupt model falls back to the next one |
| Teach | A saved Teach model survives a reload |
| Tests | All 23 pass |
| Storage | About 1 GB of the 10 GB budget. No new videos |
| Git | Committed locally. **Not pushed to GitHub** |

### Partly completed

| Task | Status |
|---|---|
| Live speed | Each prediction is about 107 to 132 ms, under the 220 ms limit. In the full live loop, predictions arrive every 225 to 266 ms (median). On 1,000 signs, the camera fell to 4 to 11 frames per second. On 50 signs it was 12.5 |
| Fake-webcam test | 1,000 signs: 5 to 7 of 8 right over four runs. 50 signs: 8 of 8 in one run, plus one extra wrong word. This uses dataset video, so it isn't a webcam test |
| Research | Three of the papers you listed couldn't be read, and `results/research_review.md` says so |
| Experiments | Larger and graph models didn't get fair training on this CPU, so they aren't proven worse |

### Unverified

| Task | Status |
|---|---|
| **Real webcam on either version 2 model** | Never tried |
| Live webcam accuracy | Not measured. Test scores come from fluent signers in dataset videos, so yours may well be lower |

## Known issues

- **Rude words:** the 1,000-sign list includes "bitch", "suck" and "damn", which can appear in the top-3 bars. The 50-sign list doesn't have this problem.
- **Challenge tab:** it picks random words from the active list, so use the 50-sign model for it.
- **Saved model choice:** a model you pick or train stays selected, so check the dropdown before you start.
- **Test split repeated:** each has been evaluated twice, so scores may be slightly optimistic. Say so if asked.
- **1,000-sign test split:** capped at 4 clips per sign, so single-sign scores are rough.

## Before you submit

- Use `PROJECT_REPORT_v2.pdf`, and check its course and section line, which I took from your sample.
- Tell your guide about the switch from WLASL to ASL Citizen, and that face features aren't used.
- **Put no dataset video frames in your report or slides.** The charts are fine.
- GitHub isn't updated with version 2. The repo there is public, so ask your guide about the licence before pushing models. To update it: `git push johnsoni7 feature/asl-citizen-model:main`
- Delete `~/signo_data` and any backups when the project is finished. The licence requires this.

## Where things are

| What | Where |
|---|---|
| Final report | `PROJECT_REPORT_v2.pdf` |
| Experiments and decisions | `results/optimization_report.md`, `results/optimization_comparison.csv` |
| Why the model made mistakes | `results/error_analysis.md` |
| Papers | `results/research_review.md` |
| Full results | `results_1000_v2/results.md`, `results_50_v2/results.md` |
| Plain-language explanation | `BEGINNER_GUIDE.md` |
