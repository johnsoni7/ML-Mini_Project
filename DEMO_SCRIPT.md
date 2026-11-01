# Signo: five-minute demo script

**Before you start:** plug in, close heavy apps, start the server, open Chrome, wake the camera, and try 2 signs. Keep `results/results.md` open in a tab. **Show no dataset video frames**, because the licence forbids it. The charts are fine.

Check your signs against a video dictionary beforehand. These are from the test results, so try them first: **more, friend, know, home, big**. Avoid leading with **no, stop, happy, again, school**.

## The script

**1. The problem (30 s).** "Millions of people use sign language, but few hearing people understand it. Interpreters aren't always available. Signo is a step towards bridging that gap."

**2. What Signo does (20 s).** "You sign a word to a webcam and it appears as text. It runs entirely in the browser, so no video leaves the device. It recognises isolated words, one at a time. It isn't a sentence translator."

**3. Architecture (30 s).** Show the README diagram. "Videos go through MediaPipe to get body and hand positions. Those become numbers, a BiLSTM learns from them, and we export it to run in the browser."

**4. MediaPipe (30 s).** Point at the skeleton on the video. "MediaPipe is a pre-trained model that finds points on the body and hands. We don't train it. Working from points instead of pixels means clothing and background matter much less."

**5. The 100 features (30 s).** "7 upper-body points, 21 per hand, each with x and y. That's 98, plus 2 flags for whether each hand is visible. Positions are measured from the centre of the chest in shoulder widths, so where you stand matters less."

**6. The BiLSTM (30 s).** "We give it 30 frames of those 100 numbers. It reads the movement forwards and backwards, and learns which patterns of motion belong to which sign."

**7. Results (45 s).** Show the table and accuracy curve. "We trained on 729 clips from 29 signers, and tested on 614 clips from 11 **different** signers. Top-1 is 76.7% and top-5 is 95.9%, against 2% and 10% for random guessing. Training accuracy is about 99% and validation about 80%, so it does overfit. We kept validation and test separate. The original code mixed them, which we found and fixed."

**8. Live demo (60 s).** Sign 3 or 4 words. Lower your hands to show "No hands in view". If it misreads a sign, point out the top-3 bars and move on. A miss is normal, and being honest about it is better than pretending.

**9. Limitations (25 s).** "It doesn't see facial expressions, which matter in ASL. It has limited data per word, it only knows its fixed list, and webcam conditions differ from the dataset."

**10. Future work (20 s).** "More data, face landmarks, a 'not signing' class, and eventually continuous recognition."

## Viva questions

**1. Why MediaPipe?** It's a ready-made, accurate landmark detector that runs in real time in a browser and in Python. Using it in both places means the live input matches the training data.

**2. Why landmark-based recognition?** A clip becomes 30 x 100 numbers, so a small model can train on few examples. Our model has 64,322 parameters and trained in 26 seconds on a laptop CPU.

**3. Why not raw RGB video?** One 640 x 480 frame has 921,600 values. With about 15 training clips per word, a pixel model could learn clothing or backgrounds instead of signs. The trade-off is that landmarks lose some detail, and depend on MediaPipe finding the points.

**4. What are the 100 features?** 7 pose points, 21 left-hand points and 21 right-hand points, each with x and y (98 values), plus 2 hand-present flags. They're centred on the shoulder midpoint and divided by shoulder width.

**5. Why 30 frames?** The model needs a fixed input size and clips vary in length, so we pick 30 evenly spaced frames from the part where hands are visible. It's enough to capture the movement while keeping the model small. It was part of the project's fixed specification.

**6. Why a BiLSTM?** Signs are defined by movement over time, and an LSTM reads sequences and remembers earlier frames. It's also small enough to run live in a browser.

**7. What does bidirectional mean?** Two LSTMs read the sequence, one forwards and one backwards, and their results are combined. So the model can use both how a sign starts and how it ends.

**8. What is overfitting?** Memorising training data instead of learning general patterns. We see it: training accuracy is about 99% and validation about 80%. Validation loss was lowest at epoch 27 (0.839) and rose to 1.401 by epoch 42. So we kept epoch 27.

**9. What is data augmentation?** Creating slightly altered training copies. We make 4 per clip with small changes in scale, rotation, position and jitter. It's applied to training data only, and works on the landmark numbers, not the video.

**10. Why separate training, validation and test?** Training teaches the weights, validation decides when to stop, and test gives one final honest score. If you tune on the test set, its score stops being honest. Our test signers appear in neither of the other sets.

**11. What does top-5 accuracy mean?** The right word is among the model's five highest guesses. Ours is 95.9%, far above our top-1 of 76.7%, so when it's wrong the right answer is usually close behind.

**12. What happens when hand detection fails?** Missing hands become zeros, and their flag becomes 0. If fewer than 5 frames show hands, the clip is rejected. That happened to 6 of 1,535 videos. In the app you see "No hands in view".

**13. Why is this not continuous translation?** It classifies one sign at a time from a fixed list. Continuous translation would need to find where signs begin and end within flowing signing, and handle sign language grammar. The Sentence box only collects separate words.

**14. Why ASL Citizen and not WLASL, as in your synopsis?** We could fetch single videos without downloading the 45.9 GB archive, and its official split shares no signers across sets. *(Confirm this change with your guide before the viva.)*

**15. Why no face features, when your synopsis mentions them?** The specification was frozen at 100 values with the nose as the only face point. This is a real limitation, since facial expressions matter in ASL.
