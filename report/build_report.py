"""Build PROJECT_REPORT.pdf. Layout follows the lab's report format. Run make_figures.py first."""
from PIL import Image as PILImage
from reportlab.lib import colors
from reportlab.lib.enums import TA_CENTER, TA_JUSTIFY, TA_LEFT
from reportlab.lib.pagesizes import A4
from reportlab.lib.styles import ParagraphStyle as PS
from reportlab.platypus import BaseDocTemplate, Frame, HRFlowable, Image, KeepTogether, ListFlowable, ListItem, PageTemplate, Paragraph, Spacer, Table, TableStyle

W = A4[0] - 114
F = "report/figures/"
GREY = colors.Color(.706, .706, .706)
body = PS("body", fontName="Times-Roman", fontSize=10, leading=13.2, alignment=TA_JUSTIFY, spaceAfter=6)
h1 = PS("h1", fontName="Times-Bold", fontSize=12.5, leading=15, spaceBefore=8, spaceAfter=5, keepWithNext=1)
h2 = PS("h2", fontName="Times-Bold", fontSize=11, leading=14, spaceBefore=5, spaceAfter=4, keepWithNext=1)
cap = PS("cap", fontName="Times-Italic", fontSize=9, leading=11, alignment=TA_CENTER, spaceBefore=3, spaceAfter=8)
cell = PS("cell", fontName="Times-Roman", fontSize=8.8, leading=10.5, alignment=TA_CENTER)
cellL = PS("cellL", parent=cell, alignment=TA_LEFT)
head = PS("head", parent=cell, fontName="Times-Bold")
headL = PS("headL", parent=head, alignment=TA_LEFT)
title = PS("title", fontName="Times-Bold", fontSize=15, leading=19, alignment=TA_CENTER, spaceAfter=3)
sub = PS("sub", fontName="Times-Italic", fontSize=11.5, leading=15, alignment=TA_CENTER, spaceAfter=7)
meta = PS("meta", fontName="Times-Roman", fontSize=10, leading=13.5, alignment=TA_CENTER)
names = PS("names", parent=meta, fontName="Times-Bold", spaceBefore=3)


def P(t, s=body):
    return Paragraph(t, s)


def H(t):
    return P(t, h1)


def H2(t):
    return P(t, h2)


def fig(name, width, caption):
    w, h = PILImage.open(F + name).size
    return KeepTogether([Image(F + name, width=width, height=width * h / w), P(caption, cap)])


def pair(a, b, caption):
    wa, ha = PILImage.open(a).size
    wb, hb = PILImage.open(b).size
    each = (W - 6) / 2
    t = Table([[Image(a, width=each, height=each * ha / wa), Image(b, width=each, height=each * hb / wb)]], colWidths=[W / 2, W / 2])
    t.setStyle(TableStyle([("LEFTPADDING", (0, 0), (-1, -1), 0), ("RIGHTPADDING", (0, 0), (-1, -1), 0), ("ALIGN", (0, 0), (-1, -1), "CENTER")]))
    return KeepTogether([t, P(caption, cap)])


def bullets(items):
    return ListFlowable([ListItem(P(i, PS("b", parent=body, spaceAfter=3)), leftIndent=30, bulletColor=colors.black) for i in items], bulletType="bullet", start="\u2022", bulletFontName="Times-Roman", bulletFontSize=10, bulletOffsetY=0, leftIndent=30, bulletDedent=12)


def table(rows, widths, caption, bold_rows=(), left_cols=(0,)):
    data = []
    for i, r in enumerate(rows):
        row = []
        for j, c in enumerate(r):
            if i == 0:
                st = headL if j in left_cols else head
            elif i in bold_rows:
                st = headL if j in left_cols else head
            else:
                st = cellL if j in left_cols else cell
            row.append(P(c, st))
        data.append(row)
    t = Table(data, colWidths=widths, repeatRows=1)
    t.setStyle(TableStyle([("GRID", (0, 0), (-1, -1), .5, colors.black), ("BACKGROUND", (0, 0), (-1, 0), colors.Color(.902, .902, .902)), ("VALIGN", (0, 0), (-1, -1), "MIDDLE"), ("TOPPADDING", (0, 0), (-1, -1), 3), ("BOTTOMPADDING", (0, 0), (-1, -1), 3), ("LEFTPADDING", (0, 0), (-1, -1), 4), ("RIGHTPADDING", (0, 0), (-1, -1), 4)]))
    return KeepTogether([t, P(caption, cap)])


def decorate(c, d):
    c.saveState()
    c.setFont("Times-Roman", 9.5)
    c.setFillColor(colors.Color(.35, .35, .35))
    c.drawCentredString(A4[0] / 2, 28, f"- {d.page} -")
    if d.page > 1:
        c.setFont("Times-Italic", 9)
        c.drawString(59.5, A4[1] - 69, "Signo: Real-Time Sign Language Recognition Using Landmarks & BiLSTM")
        c.drawRightString(A4[0] - 59.5, A4[1] - 69, "ML Lab Project Report")
        c.setStrokeColor(GREY)
        c.setLineWidth(.75)
        c.line(57, A4[1] - 74, A4[0] - 57, A4[1] - 74)
    c.restoreState()


s = []
s += [P("Signo: Real-Time Isolated Sign Language Recognition", title),
      P("Landmark-Based Temporal Deep Learning in the Browser", sub),
      HRFlowable(width="100%", thickness=1, color=GREY, spaceAfter=9),
      P("5th Semester B.Tech. Computer Science &amp; Engineering &nbsp;|&nbsp; Section A", meta),
      P("Machine Learning Laboratory (CSE 3152) &nbsp;|&nbsp; Manipal Institute of Technology, Manipal", meta),
      P("Submitted By: John Soni Thomas (CSE-A, Roll No. 40, Reg. No. 240905338)<br/>Mokshagna Y (CSE-A, Roll No. 42, Reg. No. 240905354)", names),
      Spacer(1, 7), HRFlowable(width="100%", thickness=.5, color=colors.Color(.85, .85, .85), spaceAfter=8)]

s += [H("Abstract"), P("Most hearing people do not understand sign language, which creates a communication barrier in places like classrooms and hospitals. In this lab project, we built Signo, a browser application that recognises isolated American Sign Language (ASL) words from a webcam, with all processing on the user's own device. MediaPipe Holistic converts each video frame into 100 numbers describing the hands and upper body, and a Bidirectional LSTM (BiLSTM) classifies how those numbers change over 30 frames. We trained on the ASL Citizen dataset using its official split, in which no signer appears in more than one set. While building the pipeline, we found and fixed a data leakage bug in the original training code, which had merged the validation and test sets. We compared six BiLSTM runs on validation data only, then scored the selected model once on the test set. On 1,000 signs it reached <b>61.4% top-1 and 84.4% top-5 accuracy</b> on 3,983 clips from 11 unseen signers, compared with 0.1% and 0.5% for random guessing. A smaller 50-sign model reached 76.7% and 95.9% on its own test set and serves as a fallback. Exported to TensorFlow.js, the 1,000-sign model matched Python's top prediction on all 1,000 test sequences checked. Live webcam accuracy has not been measured, and we report this openly as a limitation.")]

s += [H("1. Introduction"), P("Sign languages are complete languages, and they rely on movement over time rather than on single poses. Interpreters are not always available, so a tool that recognises signs could help in everyday situations. Many earlier computer-vision projects focused on static fingerspelled letters, which do not reflect how most signing works."),
      P("Building a useful word-level recogniser has two main challenges. First, public word-level datasets such as WLASL [2] and ASL Citizen [1] have few videos per word, so a model working directly from pixels can easily memorise clothing and backgrounds instead of signs. Second, a score only means something if the model is tested on people it has never seen, because that is what happens when a new person stands in front of the camera. This project addresses both challenges, using body landmarks instead of pixels and a strict separation of signers across the data splits."),
      P("Signo performs <b>isolated sign recognition, not continuous translation</b>. It recognises one word at a time from a fixed list. It does not follow natural signing or produce grammatical sentences.")]

s += [H("2. Project Objectives"), bullets([
    "<b>Objective 1:</b> Convert sign videos into normalised landmark sequences using MediaPipe Holistic, with feature extraction that behaves identically in Python and in the browser.",
    "<b>Objective 2:</b> Train a temporal deep learning model to classify isolated signs, and evaluate it honestly using top-1 accuracy, top-5 accuracy, per-class precision and recall, and unseen signers.",
    "<b>Objective 3:</b> Run the trained model live in a browser, showing the recognised word, the top three guesses and their confidence, and showing no word when the model is unsure.",
])]

s += [H("3. Dataset Description &amp; Anti-Leakage Data Engineering"),
      P("We used ASL Citizen [1], a community-sourced dataset from Microsoft Research containing 83,399 videos of 2,731 signs from 52 signers. We downloaded only the videos we needed, using HTTP Range requests, so the 45.9 GB archive never had to be fetched in full. About 13 GB was stored, outside the code repository. The dataset's licence does not allow its videos to be redistributed or substantially reproduced, so no dataset video frames appear in this report."),
      P("We selected 1,000 signs before any training, so accuracy could not influence the choice. The list keeps 50 common demonstration signs and adds the most common English words among the remaining signs. Of 22,421 selected clips, 22,372 were usable. The other 49 were rejected because too few frames showed a hand.")]
s += [fig("splits.png", W * .78, "Figure 1: Usable clips and signers in each split of the 1,000-sign dataset."),
      fig("clips_per_sign.png", W * .78, "Figure 2: Clips available per sign. Training data is limited, with 11 to 17 clips for each sign.")]
s += [H2("Preventing Data Leakage"),
      P("When we inspected the original training code, we found a serious flaw: it merged the validation and test sets into one, then used that set both to decide when to stop training and to report the final score. This is like practising on the final exam and then quoting the mark. We made three changes to fix it:"),
      bullets([
          "<b>Separate jobs:</b> <font face='Courier' size='8.5'>train.py</font> now uses only training and validation clips. A separate script, <font face='Courier' size='8.5'>evaluate.py</font>, scores the test set, logs each attempt and refuses to repeat it by default.",
          "<b>Official split:</b> we kept the dataset's signer assignments, and the training script stops if any signer appears in more than one split. The measured overlap is 0.",
          "<b>Automated tests:</b> a regression test guards against the merged split returning. Other tests check that the Python and browser feature code produce the same output.",
      ]),
      P("<b>One disclosed change:</b> to save about 5 GB of downloads and 1.5 hours of processing, we capped the test split at 4 clips per sign and kept every training and validation clip. The clips were picked using a hash of the file name, not by content or results, and the test signers did not change. Seventeen signs have 3 test clips because an extraction failed.")]

s += [H("4. Methodology &amp; Model Architectures"),
      P("The system has two connected parts, as shown in Figure 3: an offline training pipeline in Python and a live recognition pipeline in the browser. Both use the same feature definition."),
      fig("pipeline.png", W * .96, "Figure 3: The training pipeline and the live browser pipeline."),
      H2("Branch A: Landmark Feature Pipeline"),
      P("MediaPipe [3] Holistic is a pre-trained model that finds points on the body and hands. We use it without training it. Each frame becomes 100 numbers, as listed in Table 1."),
      table([["Part", "Points", "Values"], ["Upper body: nose, shoulders, elbows, wrists", "7", "14"], ["Left hand", "21", "42"], ["Right hand", "21", "42"], ["Left and right hand-present flags", "", "2"], ["Total", "", "100"]], [W * .56, W * .15, W * .15], "Table 1: The 100 features computed for each frame.", bold_rows=(5,)),
      P("Coordinates are centred on the midpoint between the shoulders and divided by shoulder width, so a person's position and distance from the camera matter less. Each clip is trimmed to the part where a hand is visible, keeping 2 extra frames on each side, then resampled to 30 evenly spaced frames. The model's input is therefore a 30 x 100 grid."),
      H2("Branch B: Bidirectional LSTM Classifier"),
      P("An LSTM [4] reads a sequence one step at a time and keeps a running memory, which suits signs because their meaning lies in movement. A bidirectional network [5] reads the frames in both directions, so it can use both how a sign begins and how it ends. The selected model is small enough to run live in a browser:"),
      P("<font face='Courier' size='8.2'>Input (30, 100) &rarr; Bidirectional LSTM (48) &rarr; Dropout (0.3) &rarr; Dense (48, ReLU) &rarr; Softmax (1,000)</font>", PS("code", parent=body, alignment=TA_CENTER)),
      P("It has 110,872 parameters. We trained it with TensorFlow on a laptop CPU in 14.8 minutes, using the Adam optimiser with an initial learning rate of 0.003, batch size 64 and seed 0. The learning rate was halved whenever validation loss stalled for 3 epochs. Each training clip had 4 augmented copies, with small random changes in size, rotation, position and individual points, giving 73,130 training sequences. Augmented copies were made from the landmark numbers and were used for training only."),
      P("We trained in two rounds. Round 1 tried 48, 128 and 256 units and kept the epoch with the lowest validation loss. Its logs showed validation accuracy still rising after validation loss had turned upwards, so that rule was keeping weak epochs. Round 2 therefore kept the epoch with the highest validation accuracy, added the falling learning rate, and used stronger dropout (0.5) on the two larger models.")]

s += [H("5. Experimental Results &amp; Performance Comparison"),
      P("All six runs were compared on the same 3,763 validation clips. The test set played no part in this choice."),
      table([["Run", "Units", "Dropout", "Epoch kept", "Val Top-1", "Val Top-5", "Val Loss", "Train Acc.", "Parameters"],
             ["Round 2 (Selected)", "48", "0.3", "29 of 39", "69.5%", "90.2%", "1.618", "92.3%", "110,872"],
             ["Round 2", "128", "0.5", "27 of 31", "68.7%", "89.3%", "1.968", "98.4%", "396,392"],
             ["Round 2", "256", "0.5", "12 of 12", "66.9%", "88.1%", "2.251", "99.0%", "1,119,464"],
             ["Round 1", "48", "0.3", "14 of 22", "66.0%", "89.9%", "1.498", "79.4%", "110,872"],
             ["Round 1", "256", "0.4", "4 of 8", "59.6%", "85.2%", "1.929", "85.7%", "1,119,464"],
             ["Round 1", "128", "0.3", "3 of 11", "56.3%", "83.9%", "1.808", "69.7%", "396,392"]],
            [W * .19, W * .07, W * .09, W * .11, W * .1, W * .1, W * .09, W * .1, W * .15], "Table 2: Validation comparison of all six runs (seed 0).", bold_rows=(1,)),
      fig("runs.png", W * .9, "Figure 4: Training and validation accuracy at each run's kept epoch. A wider gap indicates more overfitting."),
      P("We selected the 48-unit model from Round 2, which has the highest validation top-1 and top-5 accuracy and the fewest parameters. We then evaluated it once on the test set. Table 3 gives the final results, and they were not tuned afterwards."),
      table([["Model", "Val Top-1", "Val Top-5", "Test Top-1", "Test Top-5", "Mean Precision", "Mean Recall", "Test Clips", "Chance Top-1"],
             ["BiLSTM, 1,000 signs (Proposed)", "69.5%", "90.2%", "61.4%", "84.4%", "65.0%", "61.5%", "3,983", "0.1%"],
             ["BiLSTM, 50 signs (Fallback)", "82.3%", "96.2%", "76.7%", "95.9%", "79.3%", "76.7%", "614", "2.0%"]],
            [W * .28, W * .08, W * .08, W * .09, W * .09, W * .1, W * .09, W * .09, W * .1], "Table 3: Final performance. The two models use different test sets, so their scores are not directly comparable.", bold_rows=(1,)),
      fig("curves.png", W * .9, "Figure 5: Loss and accuracy of the selected model by epoch. The dashed line marks the kept epoch."),
      P("<b>Key findings from the experiments:</b>", PS("kf", parent=body, spaceAfter=3)),
      P("1. <b>Far above chance.</b> The model's 61.4% test top-1 accuracy compares with 0.1% for random guessing, on signers it never saw. At 84.4%, top-5 accuracy shows that the right sign is often among the leading guesses even when the first guess is wrong."),
      P("2. <b>Larger models did not help within the time available.</b> In Round 2, more units brought higher training accuracy (up to 99.0%) but lower validation accuracy. However, the 128-unit and 256-unit runs hit a 25-minute limit, and the 256-unit run's best epoch was its last. This result does not show that a larger model could never do better."),
      P("3. <b>The selected model overfits.</b> Its training accuracy of 92.3% compares with 69.5% on validation. With only 11 to 17 real clips per sign, this is not surprising."),
      P("4. <b>Test accuracy is 8.1 points below validation.</b> We chose the best of six runs using validation data, which makes that score somewhat optimistic. The validation set also contains only 5 signers, and the test set contains 11 different ones. The test score is the figure to rely on."),
      P("5. <b>Per-sign scores vary widely.</b> As Figure 6 shows, 198 signs reached 100% recall and 45 reached 0%. Each test clip usually moves a sign's recall by 25%, so single-sign scores are rough. Among common words, <i>family</i> and <i>home</i> scored 100% precision and recall, and <i>stop</i>, <i>thankyou</i> and <i>think</i> scored 0%."),
      P("6. <b>Many errors look reasonable.</b> The most frequent confusions include <i>favorite</i> read as <i>color</i>, <i>haha</i> as <i>three</i>, <i>think</i> as <i>mind</i>, <i>stars</i> as <i>star</i>, <i>sit</i> as <i>chair</i> and <i>equal</i> as <i>even</i>. Several of these pairs are related in meaning. No single confusion accounts for more than 4 clips."),
      pair(F + "recall.png", F + "threshold.png", "Figure 6: Left, the number of signs at each test recall. Right, the effect of the confidence threshold on validation clips."),
      fig("matrix.png", W * .76, "Figure 7: Test confusion matrix for 30 signs taken from the most frequent confusions. Errors involving other signs are not shown.<br/>The full 1,000 x 1,000 matrix is in <font face='Courier' size='8'>results_1000/confusion_matrix.csv</font>.")]

s += [H("6. Browser Integration &amp; Real-Time Verification"),
      P("We exported the model to TensorFlow.js. The app loads it automatically on startup and checks its input shape, output size and a trial prediction before using it. About 5 times per second, it takes the last 2.2 seconds of frames, trims them, resamples them to 30 frames and runs the model. Each prediction took about 67 ms on our laptop in a test browser session."),
      P("To prevent flickering and overconfident answers, the last 3 predictions are averaged. A word appears only if its confidence passes the threshold (70% by default) and it stays on top for 2 consecutive checks. Otherwise the app shows \"Not sure yet\", and with no hands visible it shows \"No hands in view\". On validation clips, the 70% threshold gives a word for 75% of clips, and 81% of those words are correct. At 90%, these figures become 56% and 89%, as shown in Figure 6."),
      table([["Check", "Result", "Status"],
             ["TF.js vs Python on identical test sequences", "Same top-1 and top-3 order on 1,000 of 1,000. Largest probability difference under 0.000002", "Passed"],
             ["Test videos through browser MediaPipe, features and model", "60 clips: 65% top-1 and 85% top-5. Python scored 65% top-1 on the same clips. Same prediction on 53 of 60", "Passed"],
             ["Model switching and fallback", "Labels and output size stay matched. If the 1,000-sign model is missing or corrupt, the app opens with the 50-sign model", "Passed"],
             ["Live app loop with test videos as a simulated camera", "Input size correct (3,000 values). Over three runs of 8 clips: 5 or 6 right, 0 or 1 wrong, and the rest gave no word. Six signs were picked for good test results, so this is not an accuracy score", "Passed as a functional check"],
             ["Real webcam, 50-sign model", "Informal trial: some words were recognised correctly. No counts were recorded", "Not measured"],
             ["Real webcam, 1,000-sign model", "No measured trial", "Not measured"]],
            [W * .32, W * .53, W * .15], "Table 4: Verification of the browser application.", left_cols=(0, 1)),
      fig("app_screenshot.png", W * .62, "Figure 8: The Signo interface. The camera feed here is Chrome's built-in test pattern, so the app correctly reports that no hands are in view."),
      ]

s += [H("7. Practical Deployment: Two-Model Browser Architecture"),
      P("Signo is a static web page, so it needs no application server. Video stays on the user's device, which matters because sign videos show people's faces. An internet connection is needed only to load TensorFlow.js and MediaPipe from their content delivery network. The app holds two models side by side:"),
      P("- <b>Default (1,000 signs):</b> the larger vocabulary, with lower accuracy because there are more signs to confuse.<br/>- <b>Fallback (50 signs):</b> a smaller vocabulary with higher test accuracy, selectable from a menu and loaded automatically if the larger model fails."),
      H2("Known Limitations"),
      bullets([
          "<b>Live accuracy is unknown:</b> all test scores come from fluent signers in dataset videos. A beginner's signing at a webcam may score lower.",
          "<b>No facial information:</b> the nose is the only face point used, although facial expressions carry real meaning in ASL.",
          "<b>Limited evidence:</b> each final model comes from a single training run and seed, so we report no mean or standard deviation.",
          "<b>No \"not signing\" class:</b> the app relies on hand visibility and the threshold, so any hand movement is treated as an attempted sign.",
          "<b>Fixed, unfiltered vocabulary:</b> signs outside the list are forced towards the closest known sign. The list was chosen by word frequency and includes some offensive words.",
      ])]

s += [H("8. Conclusion &amp; Future Work"),
      P("In this project, we built a working pipeline from raw sign videos to word-level recognition in a browser. We found and fixed a validation and test leakage bug, kept signers separate across splits, chose the model using validation data only and scored the test set once. The selected BiLSTM reached 61.4% top-1 and 84.4% top-5 accuracy on 1,000 signs from unseen signers, and the browser version matched Python's top prediction on all 1,000 sequences checked. The main open question is how well it works on a real webcam, which we have not yet measured."),
      P("For future work, we plan to measure live webcam accuracy with real users, add more training clips per sign, repeat training with several seeds, give larger models longer training runs, add face landmarks and a \"not signing\" class, and eventually move towards continuous recognition.")]

s += [H("References"), bullets([
    "<b>[1]</b> A. Desai et al., 'ASL Citizen: A Community-Sourced Dataset for Advancing Isolated Sign Language Recognition', Advances in Neural Information Processing Systems (NeurIPS), Datasets and Benchmarks Track, 2023.",
    "<b>[2]</b> D. Li, C. Rodriguez Opazo, X. Yu, and H. Li, 'Word-level Deep Sign Language Recognition from Video: A New Large-scale Dataset and Methods Comparison', IEEE/CVF Winter Conference on Applications of Computer Vision (WACV), 2020.",
    "<b>[3]</b> C. Lugaresi et al., 'MediaPipe: A Framework for Building Perception Pipelines', arXiv:1906.08172, 2019.",
    "<b>[4]</b> S. Hochreiter and J. Schmidhuber, 'Long Short-Term Memory', Neural Computation, vol. 9, no. 8, 1997.",
    "<b>[5]</b> M. Schuster and K. K. Paliwal, 'Bidirectional Recurrent Neural Networks', IEEE Transactions on Signal Processing, vol. 45, no. 11, 1997.",
])]

doc = BaseDocTemplate("PROJECT_REPORT.pdf", pagesize=A4, title="Signo: Real-Time Isolated Sign Language Recognition", author="John Soni Thomas; Mokshagna Y", subject="Machine Learning Laboratory project report")
pad = {"leftPadding": 2.5, "rightPadding": 2.5, "topPadding": 0, "bottomPadding": 0}
doc.addPageTemplates([
    PageTemplate(id="first", frames=[Frame(57, 46, W, A4[1] - 46 - 56, **pad)], onPage=decorate, autoNextPageTemplate="rest"),
    PageTemplate(id="rest", frames=[Frame(57, 46, W, A4[1] - 46 - 84, **pad)], onPage=decorate),
])
doc.build(s)
print("built")
