# Signo: five-minute demo script

**Before you start:** plug in, close heavy apps, start the server, open Chrome and press Cmd+Shift+R once. Check that the top bar says **Large ASL model (1000 signs)**, wake the camera, and try 2 signs. Keep `results_1000/results.md` open in a tab. **Show no dataset video frames**, because the licence forbids it. The charts are fine.

## Which signs to use

Check each sign against a video dictionary beforehand, then rehearse on your own webcam. Webcam accuracy hasn't been measured, so keep the ones that work for you. Each sign below has only 3 or 4 test clips, so these results are a guide, not a guarantee.

| Use first (1,000-sign test results) | Precision | Recall |
|---|---:|---:|
| family | 100% | 100% |
| home | 100% | 100% |
| book | 80% | 100% |
| water | 80% | 100% |
| apple, love, teacher, understand | 100% | 75% |

**Avoid leading with:** stop, thankyou and think (0% recall on the test clips), and school, happy, sad, help, go and mother (25% recall).

**If the 1,000-sign model struggles on the day,** switch the dropdown to **Small ASL model (50 signs)**. Its top test signs are more, friend, know, home and big. This model scored 76.7% top-1 on its own test set, and it has fewer signs to confuse.

**Rude words:** the 1,000-sign list includes signs such as "bitch", "suck" and "damn", because it was chosen by word frequency. One can appear in the top-3 bars. If it does, say so plainly and move on.

## The script

**1. The problem (30 s).** "Millions of people use sign language, but few hearing people understand it. Interpreters aren't always available. Signo is a step towards bridging that gap."

**2. What Signo does (20 s).** "You sign a word to a webcam and it appears as text. It runs entirely in the browser, so no video leaves the device. It recognises isolated words, one at a time, from a list of 1,000. It isn't a sentence translator."

**3. Architecture (30 s).** Show the README diagram. "Videos go through MediaPipe to get body and hand positions. Those become numbers, a BiLSTM learns from them, and we export it to run in the browser."

**4. MediaPipe (30 s).** Point at the skeleton on the video. "MediaPipe is a pre-trained model that finds points on the body and hands. We don't train it. Working from points instead of pixels means clothing and background matter much less."

**5. The 100 features (30 s).** "7 upper-body points, 21 per hand, each with x and y. That's 98, plus 2 flags for whether each hand is visible. Positions are measured from the centre of the chest in shoulder widths, so where you stand matters less."

**6. The BiLSTM (30 s).** "We give it 30 frames of those 100 numbers. It reads the movement forwards and backwards, and learns which patterns of motion belong to which sign."

**7. Results (45 s).** Show the table and accuracy curve. "We trained on 14,626 clips from 35 signers and tested on 3,983 clips from 11 **different** signers. Top-1 is 61.4% and top-5 is 84.4%, against 0.1% and 0.5% for random guessing. We compared six model runs using validation data only and scored the test set once. The smallest model won, and training accuracy of 92% against 70% on validation shows it still overfits."

**8. Live demo (60 s).** Sign 3 or 4 words, then lower your hands to show "No hands in view". If it misreads a sign, point out the top-3 bars and move on. A miss is normal, and being honest about it is better than pretending.

**9. Limitations (25 s).** "It doesn't see facial expressions, which matter in ASL. It has limited data per word, it only knows its fixed list, and webcam conditions differ from the dataset."

**10. Future work (20 s).** "More data, face landmarks, a 'not signing' class, and eventually continuous recognition."

## Viva questions

**1. Why MediaPipe?** It's a ready-made, accurate landmark detector that runs in real time in a browser and in Python. Using it in both places means the live input matches the training data.

**2. Why landmark-based recognition?** A clip becomes 30 x 100 numbers, so a small model can train on few examples. Our 1,000-sign model has 110,872 parameters and takes about 69 ms per prediction in the browser on this laptop.

**3. Why not raw RGB video?** One 640 x 480 frame has 921,600 values. With 11 to 17 training clips per word, a pixel model could learn clothing or backgrounds instead of signs. The trade-off is that landmarks lose some detail, and depend on MediaPipe finding the points.

**4. What are the 100 features?** 7 pose points, 21 left-hand points and 21 right-hand points, each with x and y (98 values), plus 2 hand-present flags. They're centred on the shoulder midpoint and divided by shoulder width.

**5. Why 30 frames?** The model needs a fixed input size and clips vary in length, so we pick 30 evenly spaced frames from the part where hands are visible. It's enough to capture the movement while keeping the model small. It was part of the project's fixed specification.

**6. Why a BiLSTM?** Signs are defined by movement over time, and an LSTM reads sequences and remembers earlier frames. It's also small enough to run live in a browser.

**7. What does bidirectional mean?** Two LSTMs read the sequence, one forwards and one backwards, and their results are combined. So the model can use both how a sign starts and how it ends.

**8. What is overfitting?** Memorising training data instead of learning general patterns. We see it: at the kept epoch, training accuracy is 92.3% and validation is 69.5%. The 256-unit model shows it more strongly, with 99.0% on training and 66.9% on validation.

**9. What is data augmentation?** Creating slightly altered training copies. We make 4 per clip with small changes in scale, rotation, position and jitter. It's applied to training data only, and works on the landmark numbers, not the video.

**10. Why separate training, validation and test?** Training teaches the weights, validation decides when to stop and which model to use, and test gives one final honest score. If you tune on the test set, its score stops being honest. Our test signers appear in neither of the other sets.

**11. What does top-5 accuracy mean?** The right word is among the model's five highest guesses. Ours is 84.4%, above our top-1 of 61.4%, so when the first guess is wrong the right answer is often close behind.

**12. What happens when hand detection fails?** Missing hands become zeros, and their flag becomes 0. If fewer than 5 frames show hands, the clip is rejected. That happened to 49 of 22,421 videos. In the app you see "No hands in view".

**13. Why is this not continuous translation?** It classifies one sign at a time from a fixed list. Continuous translation would need to find where signs begin and end within flowing signing, and handle sign language grammar. The Sentence box only collects separate words.

**14. Why ASL Citizen and not WLASL, as in your synopsis?** We could fetch single videos without downloading the 45.9 GB archive, and its official split shares no signers across sets. *(Confirm this change with your guide before the viva.)*

**15. Why no face features, when your synopsis mentions them?** The specification was frozen at 100 values with the nose as the only face point. This is a real limitation, since facial expressions matter in ASL.

**16. Why is 1,000-sign accuracy lower than 50-sign accuracy?** There are 20 times as many signs to choose between, with a similar number of training clips per sign. The two scores also use different test sets, so they aren't a direct comparison.

**17. How did you choose among your models?** Six runs, compared on validation clips only. The 48-unit run from round 2 had the best validation top-1 (69.5%) and top-5 (90.2%), and it's the smallest. The test set was scored once, afterwards. One caveat: the larger round 2 models hit a 25-minute limit, so this doesn't prove a larger model could never do better.

**18. Why is test (61.4%) lower than validation (69.5%)?** We picked the best of six runs on validation, which makes that score a little optimistic, and the two sets contain different people. The test score is the one to quote.

**19. You limited the test set to 4 clips per sign. Isn't that a problem?** It saved about 5 GB and 1.5 hours. The clips were picked by a hash of the file name, not by results, and the test signers are unchanged. It makes single-sign scores coarse, since one clip is worth 25%, but the overall score still rests on 3,983 clips.

**20. Have you measured webcam accuracy?** No. The test scores come from dataset videos of fluent signers. We checked the live app loop using dataset video as a pretend camera, and the 50-sign model was tried informally on a real webcam, but we have no measured webcam accuracy for either model.
