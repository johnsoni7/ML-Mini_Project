# Signo: five-minute demo script

> **Update, version 4 (3 November 2026).** The default 1,000-sign model is now version 4: **83.2% top-1 and 96.4% top-5** on 8,134 official test clips never used before, against 74.1% and 92.6% for version 2 on the same clips. On the earlier 3,983 test clips it scored 83.8% and 96.6%, against 73.4% and 92.0%. It is one temporal convolution network plus one BiLSTM, with the same 30 x 100 input, and takes 68 ms per prediction in the browser against 116 ms. Real webcam accuracy still has not been measured. See `results/FINAL_RESULTS.md`. Older numbers below describe earlier versions.

**Before you start:** plug in, close heavy apps, start the server, open Chrome and press Cmd+Shift+R once. Choose your model from the top bar (see below), wake the camera, and try 2 signs. Keep `results/optimization_report.md` open in a tab. **Show no dataset video frames**, because the licence forbids it. The charts are fine.

## Which signs to use

Check each sign against a video dictionary beforehand, then rehearse on your own webcam. Webcam accuracy hasn't been measured, so keep the ones that work for you. Each sign below has only 3 or 4 test clips, so these results are a guide, not a guarantee.

**Which model?** For the most reliable demo, use **Accurate ASL model (50 signs)**, which scored 91.2% top-1. Use **Large ASL model (1000 signs)**, at 73.4%, to show the wider vocabulary. Rehearse on both.

**Signs at 100% precision and recall on both models' test clips:** family, friend, boy, learn, understand.

**Avoid leading with:** fish and school (weakest on the 50-sign model). On the 1,000-sign model, thankyou and think scored 0% on their few test clips, and come scored 25%.

**Rude words:** the 1,000-sign list includes signs such as "bitch", "suck" and "damn", because it was chosen by word frequency. One can appear in the top-3 bars. If it does, say so plainly and move on, or use the 50-sign model.

## The script

**1. The problem (30 s).** "Millions of people use sign language, but few hearing people understand it. Interpreters aren't always available. Signo is a step towards bridging that gap."

**2. What Signo does (20 s).** "You sign a word to a webcam and it appears as text. It runs entirely in the browser, so no video leaves the device. It recognises isolated words, one at a time, from a list of 1,000 or a more accurate list of 50. It isn't a sentence translator."

**3. Architecture (30 s).** Show the README diagram. "Videos go through MediaPipe to get body and hand positions. Those become numbers, a BiLSTM learns from them, and we export it to run in the browser."

**4. MediaPipe (30 s).** Point at the skeleton on the video. "MediaPipe is a pre-trained model that finds points on the body and hands. We don't train it. Working from points instead of pixels means clothing and background matter much less."

**5. The 100 features (30 s).** "7 upper-body points, 21 per hand, each with x and y. That's 98, plus 2 flags for whether each hand is visible. Positions are measured from the centre of the chest in shoulder widths, so where you stand matters less."

**6. The BiLSTM (30 s).** "We give it 30 frames of those 100 numbers. It reads the movement forwards and backwards, and learns which patterns of motion belong to which sign."

**7. Results (45 s).** Show the table and accuracy curve. "We tested on signers the model never saw. On 1,000 signs, top-1 went from 61.4% to 73.4% and top-5 from 84.4% to 92.0%. On 50 signs, top-1 went from 76.7% to 91.2%. Random guessing gets 0.1% and 2%. We analysed the errors first and found that left-dominant signers scored far lower. Mirroring and fresh augmentation fixed that. All choices used validation data only."

**8. Live demo (60 s).** Sign 3 or 4 words, then lower your hands to show "No hands in view". If it misreads a sign, point out the top-3 bars and move on. A miss is normal, and being honest about it is better than pretending.

**9. Limitations (25 s).** "It doesn't see facial expressions, which matter in ASL. It has limited data per word, it only knows its fixed list, and webcam conditions differ from the dataset."

**10. Future work (20 s).** "More data, face landmarks, a 'not signing' class, and eventually continuous recognition."

## Viva questions

**1. Why MediaPipe?** It's a ready-made, accurate landmark detector that runs in real time in a browser and in Python. Using it in both places means the live input matches the training data.

**2. Why landmark-based recognition?** A clip becomes 30 x 100 numbers, so a small model can train on few examples. Our 1,000-sign model has 792,784 parameters and takes about 126 ms per prediction in the browser on this laptop.

**3. Why not raw RGB video?** One 640 x 480 frame has 921,600 values. With 11 to 17 training clips per word, a pixel model could learn clothing or backgrounds instead of signs. The trade-off is that landmarks lose some detail, and depend on MediaPipe finding the points.

**4. What are the 100 features?** 7 pose points, 21 left-hand points and 21 right-hand points, each with x and y (98 values), plus 2 hand-present flags. They're centred on the shoulder midpoint and divided by shoulder width.

**5. Why 30 frames?** The model needs a fixed input size and clips vary in length, so we pick 30 evenly spaced frames from the part where hands are visible. It's enough to capture the movement while keeping the model small. It was part of the project's fixed specification.

**6. Why a BiLSTM?** Signs are defined by movement over time, and an LSTM reads sequences and remembers earlier frames. It's also small enough to run live in a browser.

**7. What does bidirectional mean?** Two LSTMs read the sequence, one forwards and one backwards, and their results are combined. So the model can use both how a sign starts and how it ends.

**8. What is overfitting?** Memorising training data instead of learning general patterns. Version 1 scored 92.3% on training and 69.5% on validation. Version 4's convolution model scores 99.97% on clean training clips and 84.3% on validation, a gap of 15.7 points, so it still memorises to a degree. Earlier we quoted a gap of about 6 points, but that compared changed training clips with clean validation clips, so it was too generous. Measured the same way, the version 2 gap was about 18 points.

**9. What is data augmentation?** Slightly changing training clips so the model sees more variety. We apply new changes every epoch: mirroring, time warping, scale, rotation, position, jitter and brief hand dropout. It's applied to training data only, and works on the landmark numbers, not the video.

**10. Why separate training, validation and test?** Training teaches the weights, validation decides when to stop and which model to use, and test gives one final honest score. If you tune on the test set, its score stops being honest. Our test signers appear in neither of the other sets.

**11. What does top-5 accuracy mean?** The right word is among the model's five highest guesses. Ours is 92.0%, above our top-1 of 73.4%, so when the first guess is wrong the right answer is often close behind.

**12. What happens when hand detection fails?** Missing hands become zeros, and their flag becomes 0. If fewer than 5 frames show hands, the clip is rejected. That happened to 49 of 22,421 videos. In the app you see "No hands in view".

**13. Why is this not continuous translation?** It classifies one sign at a time from a fixed list. Continuous translation would need to find where signs begin and end within flowing signing, and handle sign language grammar. The Sentence box only collects separate words.

**14. Why ASL Citizen and not WLASL, as in your synopsis?** We could fetch single videos without downloading the 45.9 GB archive, and its official split shares no signers across sets. *(Confirm this change with your guide before the viva.)*

**15. Why no face features, when your synopsis mentions them?** The specification was frozen at 100 values with the nose as the only face point. This is a real limitation, since facial expressions matter in ASL.

**16. Why is 1,000-sign accuracy lower than 50-sign accuracy?** There are 20 times as many signs to choose between, with a similar number of training clips per sign. The two scores also use different test sets, so they aren't a direct comparison.

**17. How did you choose among your models?** We compared 19 configurations on validation clips only, then averaged two long-trained models (79.4%). A three-model version reached 80.3% but took about 291 ms per prediction, over our 220 ms limit, so we rejected it. One caveat: larger and graph models didn't get enough training on a CPU, so this doesn't prove they couldn't do better.

**18. Why is test (73.4%) lower than validation (79.4%)?** We chose the best epochs and models using validation, which makes that score optimistic, and the two sets contain different people. In version 1, the gap was mostly left-dominant signers, who make up 35% of test clips and 13.5% of validation clips. Mirroring fixed that. The test score is the one to quote.

**19. You limited the test set to 4 clips per sign. Isn't that a problem?** It saved about 5 GB and 1.5 hours. The clips were picked by a hash of the file name, not by results, and the test signers are unchanged. It makes single-sign scores coarse, since one clip is worth 25%, but the overall score still rests on 3,983 clips.

**21. How many times did you use the test set?** Twice per split: once for version 1 and once for version 2. Both are logged. Nothing was tuned after either score, but a second look could make the score slightly optimistic.

**22. Why did mirroring help so much?** A left-dominant signer's sign looks roughly like a right-dominant signer's sign flipped. Left-dominant signers went from 48.2% to 69.9% in a controlled comparison, while right-dominant signers dropped slightly, from 74.3% to 72.5%. We inferred handedness from hand visibility, because the dataset doesn't label it.

**20. Have you measured webcam accuracy?** No. The test scores come from dataset videos of fluent signers. We checked the live app loop using dataset video as a pretend camera, and the 50-sign model was tried informally on a real webcam, but we have no measured webcam accuracy for either model.
