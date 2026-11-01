# Signo: a beginner's guide

This guide explains how Signo works using its real code, so you can explain and defend it in your viva. Read it from top to bottom once, then use the numbered sections for revision.

## The whole system in one picture

```mermaid
flowchart TD
    subgraph Training["Training (Python, done once)"]
        A["ASL Citizen videos<br/>50 signs, 1,535 clips"] --> B["MediaPipe Holistic<br/>finds body and hand points"]
        B --> C["Feature extraction<br/>100 numbers per frame"]
        C --> D["Trim and resample<br/>30 frames x 100 numbers"]
        D --> E["BiLSTM training<br/>train and validation clips"]
        E --> F["Test evaluation<br/>unseen signers"]
        E --> G["Export to TensorFlow.js<br/>app_model/"]
    end
    subgraph Live["Live demo (browser, on your laptop)"]
        H["Webcam"] --> I["MediaPipe Holistic"]
        I --> J["Same 100 numbers per frame"]
        J --> K["Last 2.2 seconds<br/>becomes 30 x 100"]
        K --> L["BiLSTM in TensorFlow.js"]
        G --> L
        L --> M["Smoothing and threshold"]
        M --> N["Word, top 3, confidence bars"]
    end
```

## 1. What problem is Signo trying to solve?

Most hearing people don't understand sign language, which creates a barrier for deaf and hard-of-hearing people in places like classrooms and hospitals. Signo takes a step towards closing that gap: you sign a word to a webcam and the word appears as text on screen.

## 2. What is sign-language recognition?

It means getting a computer to work out which sign a person made by watching them. Signo recognises American Sign Language (ASL) words from a fixed list of 50, stored in `python/vocab_asl_citizen.txt`.

ASL is its own language, and it isn't the same as Indian Sign Language. Signo only knows ASL, so someone signing ISL won't be recognised correctly.

## 3. Isolated recognition vs continuous translation

- **Isolated recognition (what Signo does):** one sign at a time, chosen from a fixed list. Think of flashcards.
- **Continuous translation (what Signo does not do):** following natural signing, where signs flow together, and turning it into grammatical sentences. Think of a live interpreter.

The "Sentence" box in the app can be misleading. It only collects separate recognised words in order. It doesn't understand grammar, and it isn't translation.

## 4. What is MediaPipe Holistic?

MediaPipe Holistic is a ready-made model from Google. Show it one image and it tells you where a person's body and hands are. **We don't train it; we only use it.**

Signo uses it twice with the same settings: `new_holistic()` in `python/extract_wlasl.py` for the dataset videos, and `new Holistic(...)` in `index.html` for the webcam. Using the same settings in both places matters, because the live input should look like the training data.

## 5. What are landmarks and keypoints?

The two words mean the same thing here: a named point on the body, given as a position in the image. For example, landmark 15 is the left wrist. `x = 0.5, y = 0.3` means halfway across the image and 30% of the way down.

MediaPipe finds 33 body points and 21 points on each hand (the wrist plus 4 per finger). Signo keeps 7 of the body points: nose, both shoulders, both elbows and both wrists (`POSE_IDX` in `common.py`).

## 6. Why numbers instead of the raw video?

| | Raw video | Landmarks |
|---|---|---|
| Data per frame | 640 x 480 x 3 = 921,600 numbers | 100 numbers |
| Affected by clothes, room, skin tone, lighting | Directly | Much less, as long as MediaPipe finds the points |
| Training examples needed | Many more | Fewer |

We only have about 15 training clips per sign. A model working from raw pixels could easily learn the wrong things, such as "the person with the red shirt signs HELLO". Landmarks remove most of that distracting detail and leave just the movement.

The cost is that some information is lost. Without face landmarks, the model can't see facial expressions, which carry real meaning in ASL.

## 7. What is a feature vector?

It's a plain list of numbers describing one thing. In Signo, one video frame becomes one list of 100 numbers, built by `extract_features()` in `common.py`.

## 8. Why 30 frames?

A neural network like this one needs the same input size every time, but clips vary in length. A slow sign might have 120 frames and a fast one 40. So `to_seq()` always picks 30 evenly spaced frames. A 120-frame clip keeps about every fourth frame, and a 40-frame clip keeps most of them.

Thirty is enough to capture the shape of a movement while keeping the model small. Before resampling, `trim()` cuts away the parts where no hand is visible, keeping 2 extra frames on each side.

## 9. Why does each frame have 100 features?

| Part | Points | x and y each | Numbers |
|---|---|---:|---:|
| Upper body | 7 | x 2 | 14 |
| Left hand | 21 | x 2 | 42 |
| Right hand | 21 | x 2 | 42 |
| Is the left / right hand visible? (1 or 0) | | | 2 |
| **Total** | | | **100** |

**Why the coordinates aren't used directly.** A person standing near the camera looks bigger than one further away. A person on the left of the picture has different x values from one on the right. To make them comparable, every point is converted like this:

```python
f[k] = (p[i].x - cx) / sc   # cx = midpoint between the shoulders, sc = shoulder width
```

So a position now means "how far from the centre of the chest, measured in shoulder widths". Whoever you are and wherever you stand, your hand raised to your chin gives similar numbers.

**About the mirror.** The app shows your video flipped, like a mirror, but only with CSS (`transform: scaleX(-1)`). The model always gets the unflipped coordinates, the same as in training.

## 10. What is a BiLSTM?

An LSTM (Long Short-Term Memory) is a neural network layer that reads a sequence one step at a time and keeps a running memory. In Signo it reads frame 1, updates its memory, reads frame 2, updates again, and so on through frame 30.

BiLSTM means *bidirectional* LSTM. Signo's model (`build()` in `python/train.py`):

```text
Input: 30 frames x 100 numbers
  -> Bidirectional LSTM (48 memory units each way): reads the movement
  -> Dropout (30%): randomly switches off parts during training to prevent memorising
  -> Dense (48 neurons): combines what was learned
  -> Dense with softmax (50 outputs): one probability per sign, adding up to 100%
```

## 11. What does bidirectional mean?

There are two LSTMs. One reads frames 1 to 30, and the other reads frames 30 to 1. Their results are joined together.

Why bother? How a sign ends can help make sense of how it began. HELLO and SORRY finish in very different places, for example. A forward-only reader only learns the ending at the very end, whereas the backward reader starts with it.

## 12. How does it learn movement patterns?

Nobody writes a rule like "HELLO starts at the forehead". Learning goes like this:

1. The model starts with random internal numbers (weights), so its guesses are random.
2. It sees a clip and outputs 50 probabilities.
3. The **loss** measures how wrong it was. Low probability for the correct sign means high loss.
4. An algorithm called Adam nudges all the weights slightly to reduce that loss.
5. Repeat thousands of times.

Over time, the weights come to respond to useful patterns, such as a hand moving away from the chin, or two hands repeatedly coming together. So it learns from the *change in position over time*, not from one frozen pose.

## 13. Training, validation and test sets

Think of a student preparing for an exam:

| Set | Analogy | Used for | Clips | Signers |
|---|---|---|---:|---:|
| Training | Textbook | Weights learn from these | 731 selected | 35 |
| Validation | Practice exam | Deciding when to stop and which version to keep | 188 selected | 6 |
| Test | Final exam | One honest score at the very end | 616 selected | 11 |

**The bug we fixed.** The original `train.py` mixed validation and test into one set, then used that set both to decide when to stop and to report the final score. That's like practising on the final exam and then quoting your mark. It makes the score look better than it should. Now `train.py` never scores test clips, and a separate `evaluate.py` does it and logs each attempt.

**Why signer-independent matters.** The three sets contain completely different people (overlap measured as 0). So the test score answers the question "how well does it work on people it has never seen?", which is what happens when you use it yourself.

## 14. Epochs, loss, accuracy and overfitting

- **Epoch:** one full pass through the training data.
- **Loss:** how wrong the model is, with confidence taken into account. Lower is better. This is what training actually reduces.
- **Accuracy:** the percentage of clips where the top guess was right.
- **Overfitting:** the model memorises the training clips instead of learning general patterns. It looks like training accuracy rising while validation accuracy stalls and validation loss starts climbing.

**How Signo fights overfitting:** dropout, data augmentation, and early stopping. Early stopping means that if validation loss hasn't improved for 15 epochs, training stops and the best version is kept.

## 15. What is data augmentation?

It means creating slightly altered training copies. `augment()` in `common.py` makes 4 copies of each training clip, with random changes to:

- size (up to 10% either way)
- rotation (up to about 6 degrees either way)
- position (a small shift)
- each point (small random jitter, imitating MediaPipe's own noise)

It's like seeing the same sign from slightly different people and camera angles. It's only applied to **training** clips, because validation and test must stay real. Note that these copies are made from the numbers, so they aren't new real data, and we don't count them as extra clips.

## 16. Top-1 and top-5 accuracy

- **Top-1:** the model's first guess is correct.
- **Top-5:** the correct sign is somewhere in its five highest guesses.

If you sign HELP and the model ranks `please, help, sorry, more, want`, that counts as wrong for top-1 and right for top-5. Top-5 shows whether the model is "close", which is useful when many signs look alike.

For comparison, random guessing among 50 signs gets 2% top-1 and 10% top-5.

## 17. How does the trained model recognise a sign through a webcam?

All of this happens on your laptop. No video leaves the browser.

1. **Camera:** the browser gets webcam frames.
2. **Landmarks:** MediaPipe finds the points in each frame.
3. **Features:** `extract()` in `index.html` makes the same 100 numbers as Python. A test runs both versions and checks they match.
4. **Memory:** the app keeps the last 3 seconds of frames.
5. **Input:** about 5 times a second, it takes the last 2.2 seconds, trims them to where hands are visible, and resamples to 30 frames.
6. **Prediction:** the BiLSTM outputs 50 probabilities.
7. **Smoothing:** the last 3 predictions are averaged to reduce flicker.
8. **Decision:** a word only appears if it's above the threshold (default 70%, the "Picky level" slider) and stays top for 2 checks in a row. Otherwise you see "Not sure yet". With no hands visible you see "No hands in view".

## How the files fit together

| File | What it does |
|---|---|
| `python/common.py` | The frozen feature spec. **Don't change it.** |
| `python/fetch_asl_citizen.py` | Downloads only the videos we need |
| `python/extract_asl_citizen.py` | Videos to 30 x 100 number grids |
| `python/train.py` | Trains the model using train and validation clips only |
| `python/evaluate.py` | Scores the chosen model on the test set and makes the figures |
| `python/export_tfjs.py` | Converts the model for the browser |
| `index.html` | The whole web app |
| `app_model/` | The trained model the browser loads |
| `results/` | The real scores and figures |
