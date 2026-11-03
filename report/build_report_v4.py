"""Build PROJECT_REPORT_v4.pdf in the lab's report format. Run make_figures_v4.py first. Numbers are read from saved result files."""
import csv
import json

import numpy as np
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
code = PS("code", parent=body, alignment=TA_CENTER)
TT = "<font face='Courier' size='8.5'>%s</font>"


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
    wa, ha = PILImage.open(F + a).size
    wb, hb = PILImage.open(F + b).size
    each = (W - 6) / 2
    t = Table([[Image(F + a, width=each, height=each * ha / wa), Image(F + b, width=each, height=each * hb / wb)]], colWidths=[W / 2, W / 2])
    t.setStyle(TableStyle([("LEFTPADDING", (0, 0), (-1, -1), 0), ("RIGHTPADDING", (0, 0), (-1, -1), 0), ("ALIGN", (0, 0), (-1, -1), "CENTER")]))
    return KeepTogether([t, P(caption, cap)])


def bullets(items):
    return ListFlowable([ListItem(P(i, PS("b", parent=body, spaceAfter=3)), leftIndent=30, bulletColor=colors.black) for i in items], bulletType="bullet", start="•", bulletFontName="Times-Roman", bulletFontSize=10, leftIndent=30)


def table(rows, widths, caption, bold_rows=(), left_cols=(0,), split=False):
    data = []
    for i, r in enumerate(rows):
        row = []
        for j, c in enumerate(r):
            bold = i == 0 or i in bold_rows
            row.append(P(c, (headL if j in left_cols else head) if bold else (cellL if j in left_cols else cell)))
        data.append(row)
    t = Table(data, colWidths=widths, repeatRows=1)
    t.setStyle(TableStyle([("GRID", (0, 0), (-1, -1), .5, colors.black), ("BACKGROUND", (0, 0), (-1, 0), colors.Color(.902, .902, .902)), ("VALIGN", (0, 0), (-1, -1), "MIDDLE"), ("TOPPADDING", (0, 0), (-1, -1), 3), ("BOTTOMPADDING", (0, 0), (-1, -1), 3), ("LEFTPADDING", (0, 0), (-1, -1), 4), ("RIGHTPADDING", (0, 0), (-1, -1), 4)]))
    return [t, P(caption, cap)] if split else KeepTogether([t, P(caption, cap)])


def decorate(c, d):
    c.saveState()
    c.setFont("Times-Roman", 9.5)
    c.setFillColor(colors.Color(.35, .35, .35))
    c.drawCentredString(A4[0] / 2, 28, f"- {d.page} -")
    if d.page > 1:
        c.setFont("Times-Italic", 9)
        c.drawString(59.5, A4[1] - 69, "Signo Version 4: Sign Language Recognition Using Landmarks & Temporal Convolution")
        c.drawRightString(A4[0] - 59.5, A4[1] - 69, "ML Lab Project Report")
        c.setStrokeColor(GREY)
        c.setLineWidth(.75)
        c.line(57, A4[1] - 74, A4[0] - 57, A4[1] - 74)
    c.restoreState()


rep = json.load(open("results/final_test/report.json"))
M = rep["models"]
new, old, tr = M["final"], M["baseline"], M["final_train_only"]
choice = json.load(open("research/final_choice.json"))
base = json.load(open("research/baseline.json"))
board = {r["name"]: r for r in csv.DictReader(open("experiments/leaderboard.csv"))}
pcs = list(csv.DictReader(open("results/final_test/final_fresh_per_class.csv")))
thr = json.load(open("results/final_test/threshold_fresh.json"))
lat = json.load(open("results/final_test/latency.json"))
LAT_OLD, LAT_NEW = lat["Deployed before this work (final_v2)"], lat["Final model (final_v4)"]
FRESH, USED = rep["sets"]["fresh"]["clips"], rep["sets"]["used"]["clips"]
gain = rep["paired_vs_baseline"]["final/fresh"]
gain_used = rep["paired_vs_baseline"]["final/used"]
pc = lambda v, d=1: f"{v * 100:.{d}f}%"
sg = lambda v: f"{v * 100:+.1f}"


def correct(d):
    y = np.load(f"{d}/val_y.npy")
    return dict(zip(np.load(f"{d}/val_videos.npy").tolist(), (np.load(f"{d}/val_probs.npy").argmax(1) == y).astype(float)))


def paired(a, b):
    A, B = correct(a), correct(b)
    k = sorted(set(A) & set(B))
    d = np.array([A[x] - B[x] for x in k])
    m = d[np.random.RandomState(0).randint(0, len(d), (2000, len(d)))].mean(1)
    return d.mean(), np.percentile(m, 2.5), np.percentile(m, 97.5)


def exp(name):
    return "experiments/" + board[name]["id"]


def pair_txt(a, b):
    m, lo, hi = paired(a, b)
    return f"{sg(m)} ({sg(lo)} to {sg(hi)})"


def gate_row(label, name, decision):
    r = board[name]
    return [label, pc(float(r["top1"])), pc(float(r["macro_f1"])), pair_txt(exp(name), exp("gate_base")), decision]


def run_row(label, name):
    r = board[name]
    return [label, r["epochs"], pc(float(r["top1"])), pc(float(r["top5"])), pc(float(r["macro_f1"])), f"{float(r['gap']) * 100:.1f}", f"{int(r['params']):,}"]


tcn, bil = board["full_tcn_fair"], board["full_mixup_always+moretime"]
s = []
s += [P("Signo: Real-Time Isolated Sign Language Recognition", title),
      P("Temporal Convolution and Mixup in a Landmark-Based Browser Recogniser (Version 4)", sub),
      HRFlowable(width="100%", thickness=1, color=GREY, spaceAfter=9),
      P("5th Semester B.Tech. Computer Science &amp; Engineering &nbsp;|&nbsp; Section A", meta),
      P("Machine Learning Laboratory (CSE 3152) &nbsp;|&nbsp; Manipal Institute of Technology, Manipal", meta),
      P("Submitted By: John Soni Thomas (CSE-A, Roll No. 40, Reg. No. 240905338)<br/>Mokshagna Y (CSE-A, Roll No. 42, Reg. No. 240905354)", names),
      Spacer(1, 7), HRFlowable(width="100%", thickness=.5, color=colors.Color(.85, .85, .85), spaceAfter=8)]

s += [H("Abstract"), P(f"Signo is a browser application that recognises isolated American Sign Language (ASL) words from a webcam, using MediaPipe Holistic landmarks and a neural network. Our previous 1,000-sign model, version 2, averaged two Bidirectional LSTMs. In this project, we diagnosed its errors and ran {len(board)} recorded experiments, with validation data alone deciding every choice. The diagnosis showed that the main limit is data per sign, so we tested methods that make a model generalise from few examples. Mixup, a temporal convolution network (TCN) and adding the validation clips to the final training gave real gains. Face landmarks, more frames, a frame embedding and motion input did not. The final model combines one TCN and one BiLSTM and keeps the same 30 x 100 input. To measure it fairly, we downloaded {FRESH:,} official test clips that had never been used, and scored them once after every choice was final. It reached <b>{pc(new['fresh']['top1'])} top-1 and {pc(new['fresh']['top5'])} top-5</b> on signers it never saw, against {pc(old['fresh']['top1'])} and {pc(old['fresh']['top5'])} for version 2 on the same clips, a gain of {gain['diff'] * 100:.1f} points. It is also faster in the browser, at {LAT_NEW} ms per prediction against {LAT_OLD} ms. Accuracy on a real webcam has not yet been measured.")]

s += [H("1. Introduction"), P("Word-level sign recognition is difficult for two reasons. Signs are defined by movement over time, and public datasets offer few videos per word, so models can easily memorise the training examples. A score is only meaningful when the model is tested on people it has never seen, because that is what happens when a new person stands in front of the camera."),
      P(f"Version 2 of Signo scored 73.4% top-1 on 1,000 signs, and earlier tests of convolution-based models had suggested they could not compete with the BiLSTM. In this project, we questioned both results. We rebuilt the experiment process so that each idea was tested fairly, measured uncertainty for each comparison, and kept the test clips out of every decision."),
      P("Signo performs <b>isolated sign recognition, not continuous translation</b>. It recognises one word at a time from a fixed list. It does not follow natural signing or produce grammatical sentences.")]

s += [H("2. Project Objectives"), bullets([
    "<b>Objective 1:</b> Find out why the existing 1,000-sign model makes mistakes, by auditing its data, labels, features and validation errors.",
    "<b>Objective 2:</b> Improve top-1 accuracy and macro F1 through controlled experiments on architecture, augmentation and features, informed by published work and decided on validation data only.",
    "<b>Objective 3:</b> Measure the final model once on unseen test clips, and deploy it in the browser only after it matches Python on identical inputs.",
])]

s += [H("3. Dataset Description &amp; Anti-Leakage Data Engineering"),
      P("We used ASL Citizen [1], which contains 83,399 videos of 2,731 signs from 52 signers, with an official split in which no signer appears in more than one set. We kept the same 1,000 signs as version 2. The licence does not allow its videos to be redistributed or substantially reproduced, so no dataset video frames appear in this report."),
      table([["Check", "Result"],
             ["Train / validation clips", "14,626 / 3,763, from 35 / 5 signers"],
             ["Test clips", f"{USED:,} used in earlier versions, and {FRESH:,} never used before, from the same 11 signers"],
             ["Signers shared between any two splits", "0"],
             ["Duplicate video IDs, clips with identical features, non-finite values", "0, 0, 0"],
             ["Training clips per sign", "11 to 17, a ratio of only 1.5, so class weighting has little to correct"],
             ["Label order", "Equal to the sorted training labels and to the browser's labels. Checked by an assertion in every run"],
             ["Feature pipeline", "Features rebuilt from raw landmarks match the stored values on 400 of 400 clips, largest difference 0.0007"]],
            [W * .42, W * .58], "Table 1: Audit of the 1,000-sign dataset.", left_cols=(0, 1), split=True),
      fig("v4_dataset.png", W * .86, "Figure 1: Left, usable clips in each set. Right, training clips per sign."),
      H2("What the Diagnosis Found"),
      bullets([
          "<b>Too little data per sign.</b> The official training split holds no more clips for these signs, and we already had all of them. On 5,000 training clips, a BiLSTM reaches 55.5%. On all 14,626, it reaches 78.0% with longer training.",
          "<b>Memorising.</b> Version 2's members score 97.5% and 95.4% on clean training clips against 79.4% on validation for the pair, so the real gap is about 18 points. <b>This corrects our version 2 report</b>, which quoted about 6 points by comparing augmented training clips with clean validation clips.",
          "<b>Lost hand tracking.</b> Version 2's top-1 falls from 84.5% when under 10% of a clip has no visible hand to 62.4% when over 30% does. This is a correlation, so we tested it directly.",
          "<b>Camera shape.</b> The features did not correct for video shape, and one test signer's clips are all 16:9, against almost none in training. Simulating 16:9 on validation dropped top-1 from 79.38% to 76.53%.",
          "<b>Earlier convolution tests were unfair.</b> They stopped at 34 to 72 epochs on a time budget, while the BiLSTM was still improving at 180.",
      ]),
      H2("Protecting the Test Set"),
      P(f"The {USED:,} test clips from earlier versions had already been scored twice, so they are not untouched. The official split holds more test clips for these signs, because version 1 capped its sample at 4 clips per sign to save storage. We downloaded the other 8,161 clips, of which {FRESH:,} were usable, and did not score them until every choice was final. The evaluation script logs its run and refuses to run again. Test clips are also removed from the data before any training process starts.")]

s += [H("4. Methodology &amp; Model Architectures"),
      P("The system has two connected parts, shown in Figure 2. MediaPipe [8] finds the body and hand points. Each frame gives the same 100 numbers as earlier versions: 7 upper-body points, 21 points per hand (each with x and y) and 2 hand-present flags, centred on the shoulders and scaled by shoulder width. Each clip is trimmed and resampled to 30 frames."),
      fig("v4_pipeline.png", W * .88, "Figure 2: The training pipeline and the live browser pipeline."),
      H2("Branch A: Automated Experiments and Mixup"),
      P(f"An unattended controller, {TT % 'auto_optimize.py'}, trained runs from a queue, scored each on validation and compared it with a reference, using a paired bootstrap on shared clips to give a 95% interval. It also queued follow-ups from rules, for example trying more epochs when the best epoch fell near the end. Quick tests on 5,000 clips came first, and promising ideas then ran on all clips."),
      P("<b>Mixup</b> [2] blends two training clips and their labels in a random proportion, so the model cannot simply memorise individual clips. We added it to the existing changes applied to training clips: mirroring, time warping, small changes to size, rotation and position, and briefly blanking a hand. Mixup makes the model under-confident, with mean confidence of 74% against 82% accuracy, which would make the app show fewer words. We corrected this with temperature scaling [3], fitted on validation and built into each member's last layer, so the browser code did not need to change."),
      H2("Branch B: Temporal Convolution Network and BiLSTM"),
      P("A TCN [4] uses convolution filters along time, with gaps between filter positions (dilation of 1, 2, 4, 1, 2, 4) that let later layers see longer stretches of movement. A BiLSTM [7] instead reads the frames in both directions with a running memory. Our first TCN stayed at chance level. Testing without augmentation showed a setup fault, so we normalised before each block and used a wider final layer with less dropout. The two members of the final model are:"),
      P("<font face='Courier' size='7.6'>TCN: Input (30, 100) &rarr; Dense (128) &rarr; 6 residual Conv1D blocks (128 filters, width 5)<br/>&rarr; Average + Max pooling &rarr; Dense (256, ReLU) &rarr; Softmax (1,000)</font>", code),
      P("<font face='Courier' size='7.6'>BiLSTM: Input (30, 100) + frame-to-frame change &rarr; Bidirectional LSTM (128)<br/>&rarr; Dense (128, ReLU) &rarr; Softmax (1,000)</font>", code),
      P(f"Their outputs are averaged in one TensorFlow.js model of {new['params']:,} parameters. We selected the combination by a rule fixed in advance: add the member that most improves validation top-1, and stop when a member adds under 0.3 points. Also fixed in advance, the chosen members were retrained with the same settings on training plus validation clips (18,389 clips from 40 signers), because the diagnosis pointed to data as the main limit. That is the deployed model. The training-only version is kept for comparison.")]

s += [H("5. Experimental Results &amp; Performance Comparison"),
      P("We trained on a laptop CPU. Table 2 lists the quick tests, each compared with the same control. An interval that excludes zero indicates a real difference on these clips."),
      table([["Change tested (5,000 clips)", "Val Top-1", "Macro F1", "Against control, points (95% interval)", "Decision"],
             ["Control: BiLSTM, 128 units", pc(float(board['gate_base']['top1'])), pc(float(board['gate_base']['macro_f1'])), "", ""],
             gate_row("Mixup + body variation", "gate_body_mix", "Test on all clips"),
             gate_row("Mixup", "gate_mixup", "Keep"),
             gate_row("Label smoothing", "gate_smooth", "Test on all clips"),
             gate_row("Motion input", "gate_velocity", "Test on all clips"),
             gate_row("Bridging hand gaps", "gate_interp", "Reject"),
             gate_row("Hand size and arm length variation", "gate_body_aug", "Reject"),
             gate_row("Blended frame sampling", "gate_linear", "Reject"),
             gate_row("45 frames instead of 30", "gate_t45", "Reject"),
             gate_row("Face points", "gate_face", "Reject"),
             gate_row("Face points + bridging", "gate_face_interp", "Reject"),
             gate_row("Frame embedding", "gate_embed", "Reject")],
            [W * .31, W * .1, W * .1, W * .29, W * .2], "Table 2: Quick tests on 5,000 training clips, on all 3,763 validation clips.", bold_rows=(1,), split=True),
      fig("v4_gates.png", W * .8, "Figure 3: The quick tests. The two convolution tests are left out, because they failed from a setup fault."),
      table([["Model (all 14,626 training clips)", "Epochs", "Val Top-1", "Val Top-5", "Macro F1", "Gap (points)", "Parameters"],
             run_row("TCN, mixup 50% (selected)", "full_tcn_fair"),
             run_row("TCN, mixup 100%, seed 1", "tcn_always_s1"),
             run_row("TCN, mixup 100%, seed 2", "tcn_always_s2"),
             run_row("BiLSTM, mixup 100%", "full_mixup_always+moretime"),
             run_row("BiLSTM, mixup 100%", "full_mixup_always"),
             run_row("BiLSTM, mixup 100%, seed 1", "full_mixup_always+seed1"),
             run_row("BiLSTM 192 units, mixup 50%", "full_mixup_u192"),
             run_row("BiLSTM, mixup 100%, motion (selected)", "always_velocity"),
             run_row("BiLSTM, mixup 100%, smoothing", "full_mixup_always+smooth"),
             run_row("BiLSTM, mixup 50%, smoothing", "full_mixup_smooth"),
             run_row("BiLSTM, mixup 50%", "full_mixup"),
             run_row("BiLSTM, mixup 50%, body variation", "full_mixup_body"),
             ["Earlier single BiLSTM (reference)", "180", "78.0%", "94.5%", "77.5%", "19.5", "396,392"]],
            [W * .37, W * .08, W * .1, W * .1, W * .1, W * .1, W * .15], "Table 3: Runs on all training clips. Gap is clean training accuracy minus validation accuracy. The 180-epoch TCN run stopped at 158 under a time cap. Stopped early and not listed: the two-layer BiLSTM (epoch 28), half-strength 300 epochs (epoch 175), half-strength motion (epoch 6), convolution with attention (epoch 87, 68.7%) and a 96-channel TCN (epoch 17)."),
      fig("v4_models.png", W * .84, "Figure 4: Validation top-1 of all full runs and the selected combination, on the same clips."),
      fig("v4_curves.png", W * .84, "Figure 5: Validation accuracy and loss during training for both selected members and the earlier BiLSTM."),
      P(f"Combining the TCN and the BiLSTM with motion input gave {pc(choice['val']['top1'], 2)} on validation, with macro F1 of {pc(choice['val']['macro_f1'], 2)}. Adding a third member added only 0.29 points, under the 0.3-point rule, so the model stayed at two. Table 4 gives the final results."),
      table([["Test set", "Model", "Top-1 (95% interval)", "Top-5", "Macro Precision", "Macro Recall", "Macro F1", "Weighted F1"]] +
            [[lab, name, f"{pc(m[k]['top1'])} ({pc(m[k]['top1_ci95'][0])} to {pc(m[k]['top1_ci95'][1])})", pc(m[k]['top5']), pc(m[k]['macro_precision']), pc(m[k]['macro_recall']), pc(m[k]['macro_f1']), pc(m[k]['weighted_f1'])] for k, lab in (("fresh", f"Unseen ({FRESH:,} clips)"), ("used", f"Earlier ({USED:,} clips)")) for name, m in (("Version 4 (Proposed)", new), ("Version 4, training clips only", tr), ("Version 2", old))],
            [W * .13, W * .2, W * .2, W * .08, W * .1, W * .09, W * .1, W * .1], f"Table 4: Final performance. Random guessing scores 0.1% top-1. The unseen clips give the unbiased score, and were scored once.", bold_rows=(1, 4), left_cols=(0, 1)),
      fig("v4_before_after.png", W * .82, "Figure 6: Version 2 and version 4 on both test sets."),
      P("<b>Key findings from the experiments:</b>", PS("kf", parent=body, spaceAfter=3)),
      P(f"1. <b>A real gain on unseen clips.</b> On the same {FRESH:,} clips, version 4 is {gain['diff'] * 100:.1f} points better than version 2 (95% interval {gain['ci95'][0] * 100:.1f} to {gain['ci95'][1] * 100:.1f}). On the earlier clips, the gain is {gain_used['diff'] * 100:.1f} points. Signs with recall of 50% or less fell from {old['fresh']['classes_recall_le_50']} to {new['fresh']['classes_recall_le_50']}."),
      P(f"2. <b>The TCN gave the largest gain.</b> It scored {pc(float(tcn['top1']), 2)} on validation, {pair_txt('experiments/' + tcn['id'], 'model_out/f_long_s0')} points against the earlier BiLSTM. It also beat the best BiLSTM here, by {pair_txt('experiments/' + tcn['id'], 'experiments/' + bil['id'])} points, which is narrow. The earlier conclusion against convolution came from unfair tests."),
      P(f"3. <b>Mixup was next.</b> Blending every batch added {pair_txt(exp('full_mixup_always'), 'model_out/f_long_s0')} points to the same BiLSTM, and a second seed gave {pc(float(board['full_mixup_always+seed1']['top1']), 2)} against {pc(float(board['full_mixup_always']['top1']), 2)}, so the gain is stable."),
      P(f"4. <b>More training data helped.</b> Adding the validation clips raised top-1 on the unseen clips from {pc(tr['fresh']['top1'])} to {pc(new['fresh']['top1'])}, a gain of 2.1 points (95% interval 1.5 to 2.7), which supports data being the main limit."),
      P(f"5. <b>Some published ideas did not transfer.</b> Face points cost 6.8 points, although top Kaggle solutions use them [5]. One possible cause is that face shape helps a model recognise the signer, since the training gap widened, but we did not test this. Frame embedding also lost here, against a published gain [6], though only on 5,000 clips with a BiLSTM. Motion input did not help by itself. The rule chose that BiLSTM because it combines well with the TCN, not because motion helps."),
      P(f"6. <b>The model still memorises.</b> The TCN scores {pc(float(tcn['train_top1_clean']), 2)} on clean training clips against {pc(float(tcn['top1']), 2)} on validation, a gap of {float(tcn['gap']) * 100:.1f} points. The gap has narrowed only somewhat from about 18."),
      P(f"7. <b>Every test signer improved.</b> As Figure 7 shows, the lowest signer rose from {pc(min(old['fresh']['by_signer'].values()))} to {pc(min(new['fresh']['by_signer'].values()))}, but signers still range up to {pc(max(new['fresh']['by_signer'].values()))}. The signer filmed in 16:9 rose from {pc(old['fresh']['by_signer']['P42'])} to {pc(new['fresh']['by_signer']['P42'])}. Correcting for video shape probably contributes, but we did not separate it from the model change."),
      fig("v4_signers.png", W * .84, "Figure 7: Top-1 for each test signer on the unseen clips, sorted by version 4 score. Signers are numbered for this chart."),
      P(f"8. <b>Validation was optimistic.</b> The training-only model scored {pc(choice['val']['top1'])} on validation and {pc(tr['fresh']['top1'])} on the unseen clips. Part of this 4.2-point drop is likely from many comparisons on the same validation clips, and part from different signers."),
      P(f"9. <b>Similar signs remain the next limit.</b> Top-5 is {(new['fresh']['top5'] - new['fresh']['top1']) * 100:.1f} points above top-1, and the top 100 confused pairs hold 34% of errors, including <i>earth</i> and <i>planet</i>, <i>assist</i> and <i>assistant</i>, <i>experience</i> and <i>experienced</i>, and <i>register</i> and <i>rule</i>. Some pairs may be the same or nearly the same sign. Each sign has only about 8 unseen test clips, so per-sign scores are rough."),
      table([["Best 10 signs", "F1", "Worst 10 signs", "F1"]] + [[a["sign"], pc(float(a["f1"])), b["sign"], pc(float(b["f1"]))] for a, b in zip(pcs[:10], pcs[-10:][::-1])],
            [W * .3, W * .2, W * .3, W * .2], "Table 5: Best and worst signs on the unseen clips. 178 signs tied at 100% F1, so the best list is only a sample.", left_cols=(0, 2)),
      pair("v4_recall.png", "v4_threshold.png", "Figure 8: Left, the number of signs at each recall on the unseen clips. Right, the effect of the confidence threshold on the same clips. The dashed line marks the app's default."),
      fig("v4_matrix.png", W * .56, "Figure 9: Test confusion matrix for 30 signs taken from the most frequent confusions. Errors involving other signs are not shown.<br/>The full matrix is in <font face='Courier' size='8'>results/final_test/final_fresh_confusion.csv</font>.")]

s += [H("6. Browser Integration &amp; Real-Time Verification"),
      P(f"The model was exported to TensorFlow.js and added to the app as the default, with version 2 still selectable. Export exposed two problems, which we fixed: TensorFlow.js could not read an array as a layer's starting value, and it computed the wrong output length for causal padding. An equivalent layer gives identical numbers, and the validation score did not change. A word appears only when confidence passes the threshold (70% by default) and it stays on top for 2 consecutive checks. On the unseen clips, the 70% threshold gives a word for {pc(thr['0.7']['shown'])} of clips, and {pc(thr['0.7']['correct'])} of those words are correct."),
      table([["Check", "Result", "Status"],
             ["TF.js vs Python on identical sequences", "Same top-1 and top-3 order on 300 of 300 test sequences. Largest probability difference under 0.000001", "Passed"],
             ["Test videos through the browser's whole pipeline", "40 clips: 82.5% top-1, 92.5% top-5. Python 85% on the same clips. Same prediction on 95% of clips", "Passed"],
             ["Prediction speed (idle machine)", f"{LAT_NEW} ms mean, 72 ms p95. Version 2: {LAT_OLD} ms mean, 119 ms p95", "Passed"],
             ["Live app loop, test videos as a simulated camera", "8 clips, first pass: 6 right, 1 wrong (book read as school), 1 with no word, and 1 extra wrong word. Median 217 ms between predictions and 17.4 camera frames per second in this run, against 4 to 11 for version 2. Six signs were picked for good earlier results, so this is not an accuracy score", "Passed as a functional check"],
             ["Model switching and fallback", "Labels and output size stay matched. If the new model is missing or corrupt, the app opens with the previous one", "Passed"],
             ["Automated tests", "All 23 existing tests pass", "Passed"],
             ["Camera-shape correction", "Formula checked in Python to within 0.0000004. 4:3 video is unchanged. Not yet run on a real widescreen camera", "Partly checked"],
             ["Real webcam accuracy", "Version 4 not yet measured", "Not measured"]],
            [W * .29, W * .55, W * .16], "Table 6: Verification of the browser application.", left_cols=(0, 1)),
      fig("v4_speed.png", W * .78, "Figure 10: Mean prediction time in Chrome with WebGL."),
      fig("app_screenshot_v4.png", W * .5, "Figure 11: The Signo interface. The camera feed here is Chrome's built-in test pattern, so the app correctly reports that no hands are in view.")]

s += [H("7. Practical Deployment: Choosing a Model"),
      P("Signo runs entirely on the user's device, so no video is uploaded. An internet connection is needed only to load TensorFlow.js and MediaPipe. The model files total 7.2 MB. The main models serve different purposes:"),
      P(f"- <b>Large ASL model (1,000 signs, default):</b> version 4, with {pc(new['fresh']['top1'])} top-1 and {pc(new['fresh']['top5'])} top-5 on unseen signers.<br/>- <b>Accurate ASL model (50 signs):</b> unchanged from version 2, with 91.2% top-1 and 97.6% top-5 on its own test set. It is an easier task and a different test set, so the two are not directly comparable.<br/>- <b>Previous large ASL model:</b> version 2, kept for comparison."),
      H2("Known Limitations"),
      bullets([
          "<b>Live accuracy is unknown:</b> all test scores come from fluent signers in dataset videos. A beginner's signing at a webcam may score lower.",
          "<b>Same test signers:</b> both test sets come from the same 11 signers, so the score may not hold for very different signers or cameras.",
          f"<b>Earlier test clips are not untouched:</b> the {USED:,} clips have now been scored in three versions. Quote the unseen clips' score.",
          "<b>Single seeds in the final model:</b> each member comes from one seed, though other seeds gave similar results. The final retrained model has no validation score of its own.",
          "<b>Limited search:</b> many runs used one seed, and we did not test pretraining on the other 1,731 signs or give graph networks a fair test.",
          "<b>No facial expression:</b> the nose is the only face point used, although facial expressions carry real meaning in ASL. Adding face points lowered accuracy in our test.",
          "<b>Fixed, unfiltered vocabulary:</b> signs outside the list are forced towards the closest known sign. The list was chosen by word frequency and includes some offensive words.",
      ])]

s += [H("8. Conclusion &amp; Future Work"),
      P(f"In this project, we improved Signo's 1,000-sign model by first finding its main limit, then testing each idea with measured uncertainty. A temporal convolution network and mixup gave most of the gain, and training on the validation clips added more, all while keeping the same 30 x 100 input. On {FRESH:,} test clips never used before, top-1 rose from {pc(old['fresh']['top1'])} to {pc(new['fresh']['top1'])} and macro F1 from {pc(old['fresh']['macro_f1'])} to {pc(new['fresh']['macro_f1'])}, and predictions became faster. Just as usefully, several appealing ideas were measured and rejected, and one earlier conclusion against convolution models was overturned."),
      P("For future work, we plan to measure webcam accuracy with real users, test the camera-shape correction on a widescreen camera, try pretraining on the other 1,731 ASL Citizen signs, review near-duplicate sign labels, and add a \"not signing\" class.")]

s += [H("References"), bullets([
    "<b>[1]</b> A. Desai et al., 'ASL Citizen: A Community-Sourced Dataset for Advancing Isolated Sign Language Recognition', Advances in Neural Information Processing Systems (NeurIPS), Datasets and Benchmarks Track, 2023.",
    "<b>[2]</b> H. Zhang, M. Cisse, Y. N. Dauphin, and D. Lopez-Paz, 'mixup: Beyond Empirical Risk Minimization', International Conference on Learning Representations (ICLR), 2018.",
    "<b>[3]</b> C. Guo, G. Pleiss, Y. Sun, and K. Q. Weinberger, 'On Calibration of Modern Neural Networks', International Conference on Machine Learning (ICML), 2017.",
    "<b>[4]</b> S. Bai, J. Z. Kolter, and V. Koltun, 'An Empirical Evaluation of Generic Convolutional and Recurrent Networks for Sequence Modeling', arXiv:1803.01271, 2018.",
    "<b>[5]</b> H. Sohn, 'Google Isolated Sign Language Recognition, 1st Place Solution', Kaggle, 2023. github.com/hoyso48/Google---Isolated-Sign-Language-Recognition-1st-place-solution",
    "<b>[6]</b> T. Vandendriessche, M. De Coster, A. Lejon, and J. Dambre, 'Representing Signs as Signs: One-Shot ISLR to Facilitate Functional Sign Language Technologies', arXiv:2502.20171, 2025.",
    "<b>[7]</b> S. Hochreiter and J. Schmidhuber, 'Long Short-Term Memory', Neural Computation, vol. 9, no. 8, 1997.",
    "<b>[8]</b> C. Lugaresi et al., 'MediaPipe: A Framework for Building Perception Pipelines', arXiv:1906.08172, 2019.",
])]

doc = BaseDocTemplate("PROJECT_REPORT_v4.pdf", pagesize=A4, title="Signo Version 4: Real-Time Isolated Sign Language Recognition", author="John Soni Thomas; Mokshagna Y", subject="Machine Learning Laboratory project report")
pad = {"leftPadding": 2.5, "rightPadding": 2.5, "topPadding": 0, "bottomPadding": 0}
doc.addPageTemplates([
    PageTemplate(id="first", frames=[Frame(57, 46, W, A4[1] - 46 - 56, **pad)], onPage=decorate, autoNextPageTemplate="rest"),
    PageTemplate(id="rest", frames=[Frame(57, 46, W, A4[1] - 46 - 84, **pad)], onPage=decorate),
])
doc.build([x for item in s for x in (item if isinstance(item, list) else [item])])
print("built")
