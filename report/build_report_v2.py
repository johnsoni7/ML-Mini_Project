"""Build PROJECT_REPORT_v2.pdf. Same layout as the first report. Run make_figures.py and make_figures_v2.py first."""
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
        c.drawString(59.5, A4[1] - 69, "Signo Version 2: Sign Language Recognition Using Landmarks & BiLSTM")
        c.drawRightString(A4[0] - 59.5, A4[1] - 69, "ML Lab Project Report")
        c.setStrokeColor(GREY)
        c.setLineWidth(.75)
        c.line(57, A4[1] - 74, A4[0] - 57, A4[1] - 74)
    c.restoreState()



s = []
s += [P("Signo: Real-Time Isolated Sign Language Recognition", title),
      P("Improving Generalisation in a Landmark-Based Browser Recogniser (Version 2)", sub),
      HRFlowable(width="100%", thickness=1, color=GREY, spaceAfter=9),
      P("5th Semester B.Tech. Computer Science &amp; Engineering &nbsp;|&nbsp; Section A", meta),
      P("Machine Learning Laboratory (CSE 3152) &nbsp;|&nbsp; Manipal Institute of Technology, Manipal", meta),
      P("Submitted By: John Soni Thomas (CSE-A, Roll No. 40, Reg. No. 240905338)<br/>Mokshagna Y (CSE-A, Roll No. 42, Reg. No. 240905354)", names),
      Spacer(1, 7), HRFlowable(width="100%", thickness=.5, color=colors.Color(.85, .85, .85), spaceAfter=8)]

s += [H("Abstract"), P("Signo is a browser application that recognises isolated American Sign Language (ASL) words from a webcam, using MediaPipe Holistic landmarks and a Bidirectional LSTM (BiLSTM). Version 1 of our 1,000-sign model scored 61.4% top-1 on unseen signers. In this project, we investigated why it made mistakes and improved it through controlled experiments, without changing the 30 x 100 input. Error analysis showed that left-dominant signers were recognised far less well (46.7% against 73.1% on validation), and that the model was memorising its training clips (92.3% training accuracy against 69.5% on validation). Applying new random changes to clips in every epoch, including left-right mirroring, addressed both problems. We compared 19 configurations on validation data only, then averaged two long-trained models. <b>The final 1,000-sign model reached 73.4% top-1 and 92.0% top-5 accuracy on 3,983 clips from 11 unseen signers, and the 50-sign model reached 91.2% top-1 and 97.6% top-5 on 614 clips.</b> Each prediction takes well under our 220 ms limit, although the full live loop with MediaPipe sometimes takes longer. We did not reach 80% on 1,000 signs. Each test split has now been evaluated twice, and live webcam accuracy has still not been measured.")]

s += [H("1. Introduction"), P("Word-level sign recognition is difficult for two reasons. Signs are defined by movement over time, and public datasets offer few videos per word, so models can easily memorise the training examples. A score is only meaningful when the model is tested on people it has never seen, because that is what happens when a new person stands in front of the camera."),
      P("Our first version showed a large drop between validation accuracy (69.5%) and test accuracy (61.4%), and a large gap between training and validation accuracy. Instead of simply building a bigger network, we examined the errors first, and then tested one change at a time to see what actually helped."),
      P("Signo performs <b>isolated sign recognition, not continuous translation</b>. It recognises one word at a time from a fixed list. It does not follow natural signing or produce grammatical sentences.")]

s += [H("2. Project Objectives"), bullets([
    "<b>Objective 1:</b> Find out why the first model makes mistakes, using evidence from its validation errors and from data quality checks.",
    "<b>Objective 2:</b> Improve generalisation through controlled experiments, keeping the 30 x 100 input and using validation data only to make every choice.",
    "<b>Objective 3:</b> Deploy the best model in the browser within a 220 ms limit per prediction, keeping the earlier models as working fallbacks.",
])]

s += [H("3. Dataset Description &amp; Data Analysis"),
      P("We used ASL Citizen [1], which contains 83,399 videos of 2,731 signs from 52 signers, with an official split in which no signer appears in more than one set. The vocabularies were fixed before any training, so accuracy could not influence them. The dataset's licence does not allow its videos to be redistributed or substantially reproduced, so no dataset video frames appear in this report. No new videos were downloaded for this version."),
      table([["", "1,000-sign model", "50-sign model"], ["Usable clips", "22,372 of 22,421", "1,529 of 1,535"], ["Train / validation / test clips", "14,626 / 3,763 / 3,983", "729 / 186 / 614"], ["Train / validation / test signers", "35 / 5 / 11", "29 / 5 / 11"], ["Signers shared between splits", "0", "0"], ["Test clips per sign", "3 or 4 (capped to save storage)", "All official clips"]], [W * .4, W * .3, W * .3], "Table 1: Data used by each model."),
      H2("What the Error Analysis Found"),
      bullets([
          "<b>Handedness:</b> we inferred each signer's dominant hand from how often each hand is visible. Left-dominant signers supply 17.0% of training clips, 13.5% of validation clips and 35.3% of test clips. Version 1 scored 46.7% on them on validation. Reweighting its validation score to the test's mix predicts 63.8%, close to its actual 61.4%, so this probably explains most of its validation-to-test gap.",
          "<b>Memorising:</b> version 1 reached 92.3% on training clips and 69.5% on validation, using a fixed set of augmented copies.",
          "<b>Little class imbalance:</b> each sign has 11 to 17 training clips. We therefore did not use class weights or focal loss, because they address a problem we do not have.",
          "<b>Clean data:</b> we found no duplicate clips, no shared signers, and no broken normalisation. Clips with hands visible in under half their frames do score lower, but they are only 3.27% of validation clips.",
          "<b>Similar signs:</b> common confusions include <i>assist</i> and <i>assistant</i>, <i>star</i> and <i>stars</i>, and <i>holiday</i> and <i>vacation</i>. Some pairs may be the same or nearly the same sign. Dafnis et al. [2] corrected WLASL's labels for this reason. We kept our labels unchanged, because merging them would change the exam.",
      ]),
      H2("Protecting the Test Set"),
      P("All model choices used validation clips only. The test set had already been scored once for version 1, so each model's final score here is a <b>second, logged evaluation</b> of its split, and repeated evaluation can make a score slightly optimistic. We made no changes after seeing either final score. Every evaluation is recorded in a <font face='Courier' size='8.5'>test_log.json</font> file.")]

s += [H("4. Methodology &amp; Model Architectures"),
      P("The pipeline, shown in Figure 1, is unchanged from version 1. The frozen specification in <font face='Courier' size='8.5'>common.py</font> gives 100 numbers per frame: 7 upper-body points, 21 points per hand (each with x and y) and 2 hand-present flags, centred on the shoulders and scaled by shoulder width. Each clip is trimmed and resampled to 30 frames."),
      fig("pipeline.png", W * .84, "Figure 1: The training pipeline and the live browser pipeline."),
      H2("Branch A: Improved Training"),
      P("Version 1 made 4 fixed changed copies of each clip. Version 2 makes <b>new random changes every epoch</b>, so the model almost never sees exactly the same input twice and has to learn the sign itself. The changes applied to training clips only are:"),
      bullets([
          "<b>Mirroring (50% of clips):</b> flip left and right and swap the hands, so a right-dominant signer's clip resembles a left-dominant signer's. Published ASL work uses this too [2].",
          "<b>Time warping:</b> use a random stretch of at least 80% of the clip, resampled unevenly, to imitate different signing speeds.",
          "<b>Geometry and jitter:</b> small random changes to size, rotation and position, and small noise on each point.",
          "<b>Hand dropout:</b> blank a hand for up to 4 frames in 30% of clips, to imitate MediaPipe briefly losing it.",
      ]),
      P("The learning rate falls smoothly along a cosine curve, gradients are clipped, and the epoch with the best validation accuracy is kept."),
      H2("Branch B: Bidirectional LSTM Classifier"),
      P("An LSTM [3] reads a sequence one step at a time with a running memory, and a bidirectional network [4] reads it in both directions. Each member of the final 1,000-sign model has this structure:"),
      P("<font face='Courier' size='8.2'>Input (30, 100) &rarr; Bidirectional LSTM (128) &rarr; Dropout (0.4) &rarr; Dense (128, ReLU) &rarr; Dropout (0.4) &rarr; Softmax</font>", PS("code", parent=body, alignment=TA_CENTER)),
      P("The final model averages the outputs of two such networks trained separately, because they make somewhat different mistakes. Both are packed into one TensorFlow.js model of 792,784 parameters, so the browser code and its features did not change. The 50-sign model similarly combines one 128-unit network and one 64-unit network, with 369,828 parameters.")]

s += [H("5. Experimental Results &amp; Performance Comparison"),
      P("We trained on a laptop CPU using a fixed seed for each run. Figure 2 shows controlled comparisons, each changing one thing from the control. With 3,763 validation clips, differences of 1 to 2 points on one seed are close to noise."),
      fig("v2_ablation.png", W * .9, "Figure 2: Validation top-1 accuracy when one thing changes at a time."),
      table([["Change tested", "Result on validation", "Decision"],
             ["Mirroring", "Overall +1.4 points. Left-dominant signers 48.2% &rarr; 69.9%. Right-dominant 74.3% &rarr; 72.5%", "Keep"],
             ["Time warping", "About +2 points", "Keep"],
             ["Longer training (180 epochs)", "78.0%, up from 72.2% at about 60 epochs", "Keep"],
             ["Averaging two long models", "79.4%, up from 78.0%. Takes about 107 to 132 ms", "Keep"],
             ["Averaging three long models", "80.3%, but takes about 291 ms, over the limit", "Reject"],
             ["Mirror averaging at prediction time", "No gain for the long model (77.9% against 78.0%)", "Reject"],
             ["Wrist-relative hands", "70.6% against 72.2%", "Reject"],
             ["AdamW weight decay", "70.7% against 72.2%", "Reject"],
             ["Label smoothing", "73.3% against 72.2%, within noise. Used in the three-model option only", "Not adopted"],
             ["Half the training clips", "65.7%, so more real data should help", "Future work"],
             ["CNN + two BiLSTMs; 256 units", "Still improving when the time ran out (60.1%, 56.0%)", "Inconclusive"],
             ["Lightweight graph model (ST-GCN style)", "About 5.5 minutes per epoch on this CPU, so it never really trained", "Inconclusive"]],
            [W * .32, W * .54, W * .14], "Table 2: Summary of the experiments. All 19 configurations are in results/optimization_comparison.csv.", left_cols=(0, 1)),
      table([["Model", "Val Top-1", "Val Top-5", "Test Top-1", "Test Top-5", "Mean Precision", "Mean Recall", "Mean F1", "Latency"],
             ["Version 2, 1,000 signs (Proposed)", "79.4%", "95.0%", "73.4%", "92.0%", "75.4%", "73.4%", "72.5%", "107-132 ms"],
             ["Version 1, 1,000 signs", "69.5%", "90.2%", "61.4%", "84.4%", "65.0%", "61.5%", "60.5%", "67 ms"],
             ["Version 2, 50 signs", "95.2%", "98.4%", "91.2%", "97.6%", "91.7%", "91.3%", "91.1%", "107 ms"],
             ["Version 1, 50 signs", "82.3%", "96.2%", "76.7%", "95.9%", "79.3%", "76.7%", "76.7%", "66 ms"]],
            [W * .29, W * .08, W * .08, W * .09, W * .09, W * .1, W * .09, W * .09, W * .09], "Table 3: Final performance. Random guessing scores 0.1% top-1 on 1,000 signs and 2% on 50 signs. The 1,000-sign and 50-sign models use different test sets.", bold_rows=(1,)),
      fig("v2_before_after.png", W * .8, "Figure 3: Test accuracy of version 1 and version 2."),
      fig("v2_curves.png", W * .82, "Figure 4: Loss and accuracy of the long-trained member of the final 1,000-sign model. The dashed line marks the kept epoch."),
      P("<b>Key findings from the experiments:</b>", PS("kf", parent=body, spaceAfter=3)),
      P("1. <b>Better training mattered more than a bigger network.</b> Most of the gain came from changing how the model is trained. The gap between training and validation accuracy fell from about 23 points to about 6 for the long-trained member."),
      P("2. <b>Mirroring closed the left-handed gap.</b> As Figure 5 shows, validation accuracy for left-dominant signers rose from 46.7% to 76.6%, close to the 79.8% for right-dominant signers."),
      fig("v2_signers.png", W * .82, "Figure 5: Validation accuracy for each signer. Dominant hand is inferred from hand visibility."),
      P("3. <b>The validation-to-test gap narrowed, but remains.</b> It fell from 8.1 to 6.0 points. Handedness no longer explains it, since reweighting predicts 78.7% against the actual 73.4%. Choosing the best epoch and combination on validation makes that score optimistic, and the validation set has only 5 signers, whose individual scores range from 66.0% to 84.1%."),
      P("4. <b>Accuracy above 80% on 1,000 signs was not achieved.</b> We expect more real training clips to be the most useful next step, since training on half the clips scored about 6.5 points lower. Processing a further download was not possible in the time available, and we estimated that it would add only 1 to 2 points."),
      P("5. <b>Per-sign scores are rough on the 1,000-sign test.</b> Most signs have only 4 test clips, so each clip moves recall by 25%. In total, 368 signs reached 100% recall and 29 reached 0%. Compared with version 1, 446 signs improved and 113 got worse."),
      pair(F + "v2_recall.png", F + "v2_threshold.png", "Figure 6: Left, the number of signs at each test recall. Right, the effect of the confidence threshold on validation clips."),
      fig("v2_matrix.png", W * .62, "Figure 7: Test confusion matrix for 30 signs taken from the most frequent confusions of the 1,000-sign model. Errors involving other signs are not shown.<br/>The full matrix is in <font face='Courier' size='8'>results_1000_v2/confusion_matrix.csv</font>.")]

s += [H("6. Browser Integration &amp; Real-Time Verification"),
      P("Both new models were exported to TensorFlow.js and added to the app. The 1,000-sign model is the default, and the dropdown switches models. A word appears only when confidence passes the threshold (70% by default) and it stays on top for 2 consecutive checks. On validation clips, the 70% threshold gives a word for 69% of clips, and 93% of those words are correct. This is a useful property for a demonstration, but it is not the same as 93% overall accuracy, and we have not measured it on test clips or on a webcam."),
      table([["Check", "Result", "Status"],
             ["TF.js vs Python, 1,000 signs", "Same top-1 and top-3 order on 1,000 of 1,000 test sequences. Largest probability difference under 0.000002", "Passed"],
             ["TF.js vs Python, 50 signs", "Same top-1 and top-3 order on 50 of 50 test sequences", "Passed"],
             ["Test videos through the browser, 1,000 signs", "60 clips: 76.7% top-1, 88.3% top-5. Python 78.3% on the same clips. Same prediction on 55 of 60", "Passed"],
             ["Test videos through the browser, 50 signs", "50 clips: 90% top-1, 92% top-5. Python 86% on the same clips. Same prediction on 46 of 50", "Passed"],
             ["Prediction speed", "Each prediction: about 107 to 132 ms (1,000 signs) and 106 to 109 ms (50 signs). In the live loop with MediaPipe, predictions arrive every 225 to 266 ms (median), with 4 to 11 frames per second for 1,000 signs and 12.5 for 50", "Partly met"],
             ["Live app loop, test videos as a simulated camera", "8 clips. 1,000 signs, over four runs: 5 to 7 right, 0 or 1 wrong. 50 signs, one run: 8 right, 1 extra wrong word. Six signs were picked for good earlier results, so this is not an accuracy score", "Passed as a functional check"],
             ["Model switching and fallback", "Labels and output size stay matched. If the 1,000-sign model is missing or corrupt, the app still opens", "Passed"],
             ["Real webcam", "Version 2 not yet measured. An informal trial of version 1 recognised some words", "Not measured"]],
            [W * .31, W * .54, W * .15], "Table 4: Verification of the browser application.", left_cols=(0, 1)),
      fig("app_screenshot_v2.png", W * .48, "Figure 8: The Signo interface. The camera feed here is Chrome's built-in test pattern, so the app correctly reports that no hands are in view.")]

s += [H("7. Practical Deployment: Choosing a Model"),
      P("Signo runs entirely on the user's device, so no video is uploaded. An internet connection is needed only to load TensorFlow.js and MediaPipe. The two main models serve different purposes:"),
      P("- <b>Large ASL model (1,000 signs, default):</b> a wide vocabulary, with 73.4% top-1 and 92.0% top-5 on unseen signers.<br/>- <b>Accurate ASL model (50 signs):</b> a small vocabulary, with 91.2% top-1 and 97.6% top-5 on unseen signers."),
      H2("Known Limitations"),
      bullets([
          "<b>Live accuracy is unknown:</b> all test scores come from fluent signers in dataset videos. A beginner's signing at a webcam may score lower.",
          "<b>Repeated test evaluation:</b> each test split has now been evaluated twice, so the scores may be slightly optimistic.",
          "<b>Limited experiments:</b> most comparisons used one seed and a limited time on a CPU. Larger, convolutional and graph models were not given a fair amount of training, so we cannot say they are worse.",
          "<b>No facial information:</b> the nose is the only face point used, although facial expressions carry real meaning in ASL.",
          "<b>Handedness is inferred:</b> mirroring could also harm the few signs whose meaning depends on actual left or right direction, and we did not test those separately.",
          "<b>Fixed, unfiltered vocabulary:</b> signs outside the list are forced towards the closest known sign. The 1,000-sign list was chosen by word frequency and includes some offensive words.",
      ])]

s += [H("8. Conclusion &amp; Future Work"),
      P("In this project, we improved Signo by first finding out why it failed. Error analysis pointed to poor recognition of left-dominant signers and to memorising of training clips, and controlled experiments confirmed that fresh augmentation with mirroring, longer training and a small model combination address them. Test top-1 rose from 61.4% to 73.4% on 1,000 signs and from 76.7% to 91.2% on 50 signs, on signers the models never saw, with each prediction well under our 220 ms limit. The 1,000-sign model did not reach 80%, and live webcam accuracy remains the main open question."),
      P("For future work, we plan to measure webcam accuracy with real users, add more real training clips (for example from WLASL [5], after checking labels and signers), review similar-looking sign labels, repeat training with more seeds, give larger and graph models fair training on a GPU, and add a \"not signing\" class.")]

s += [H("References"), bullets([
    "<b>[1]</b> A. Desai et al., 'ASL Citizen: A Community-Sourced Dataset for Advancing Isolated Sign Language Recognition', Advances in Neural Information Processing Systems (NeurIPS), Datasets and Benchmarks Track, 2023.",
    "<b>[2]</b> K. M. Dafnis, E. Chroni, C. Neidle, and D. N. Metaxas, 'Bidirectional Skeleton-Based Isolated Sign Recognition using Graph Convolutional Networks', Language Resources and Evaluation Conference (LREC), 2022.",
    "<b>[3]</b> S. Hochreiter and J. Schmidhuber, 'Long Short-Term Memory', Neural Computation, vol. 9, no. 8, 1997.",
    "<b>[4]</b> M. Schuster and K. K. Paliwal, 'Bidirectional Recurrent Neural Networks', IEEE Transactions on Signal Processing, vol. 45, no. 11, 1997.",
    "<b>[5]</b> D. Li, C. Rodriguez Opazo, X. Yu, and H. Li, 'Word-level Deep Sign Language Recognition from Video: A New Large-scale Dataset and Methods Comparison', IEEE/CVF Winter Conference on Applications of Computer Vision (WACV), 2020.",
])]

doc = BaseDocTemplate("PROJECT_REPORT_v2.pdf", pagesize=A4, title="Signo Version 2: Real-Time Isolated Sign Language Recognition", author="John Soni Thomas; Mokshagna Y", subject="Machine Learning Laboratory project report")
pad = {"leftPadding": 2.5, "rightPadding": 2.5, "topPadding": 0, "bottomPadding": 0}
doc.addPageTemplates([
    PageTemplate(id="first", frames=[Frame(57, 46, W, A4[1] - 46 - 56, **pad)], onPage=decorate, autoNextPageTemplate="rest"),
    PageTemplate(id="rest", frames=[Frame(57, 46, W, A4[1] - 46 - 84, **pad)], onPage=decorate),
])
doc.build(s)
print("built")
