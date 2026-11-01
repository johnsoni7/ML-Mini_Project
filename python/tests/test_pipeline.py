"""Run with: .venv/bin/python -m pytest python/tests -q
The dataset-based tests are skipped automatically if ~/signo_data is absent. Synthetic data here only tests code paths."""
import json
import os
import re
import shutil
import subprocess
import sys
from types import SimpleNamespace as NS

import numpy as np
import pytest

ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.insert(0, os.path.join(ROOT, "python"))

import common  # noqa: E402
from common import D, POSE_IDX, T, augment, extract_features, to_seq, trim  # noqa: E402

DATASET = os.path.expanduser("~/signo_data/asl_citizen/signo_dataset.json")


def lm(rng, n):
    return NS(landmark=[NS(x=float(a), y=float(b)) for a, b in rng.rand(n, 2)])


def fake_result(rng, left=True, right=True):
    return NS(pose_landmarks=lm(rng, 33), left_hand_landmarks=lm(rng, 21) if left else None, right_hand_landmarks=lm(rng, 21) if right else None)


def to_js(res):
    conv = lambda h: None if h is None else [{"x": p.landmark_x, "y": p.landmark_y} for p in [NS(landmark_x=q.x, landmark_y=q.y) for q in h.landmark]]
    return {"poseLandmarks": conv(res.pose_landmarks), "leftHandLandmarks": conv(res.left_hand_landmarks), "rightHandLandmarks": conv(res.right_hand_landmarks)}


# ---------- frozen specification ----------
def test_spec_constants():
    assert T == 30 and D == 100 and POSE_IDX == [0, 11, 12, 13, 14, 15, 16]
    assert 7 * 2 + 21 * 2 + 21 * 2 + 2 == D


def test_common_py_is_unchanged_since_initial_commit():
    orig = subprocess.run(["git", "show", "3d69284:python/common.py"], cwd=ROOT, capture_output=True, check=True).stdout
    assert open(os.path.join(ROOT, "python/common.py"), "rb").read() == orig


# ---------- feature extraction ----------
def test_feature_shape_and_normalisation():
    rng = np.random.RandomState(0)
    r = fake_result(rng)
    f, hands = extract_features(r)
    assert f.shape == (D,) and hands is True
    a, b = r.pose_landmarks.landmark[11], r.pose_landmarks.landmark[12]
    sc = np.hypot(a.x - b.x, a.y - b.y)
    cx, cy = (a.x + b.x) / 2, (a.y + b.y) / 2
    assert f[0] == pytest.approx((r.pose_landmarks.landmark[0].x - cx) / sc, rel=1e-5)
    # the shoulders end up mirror images of one another, one half a shoulder width from the centre
    assert np.hypot(f[2], f[3]) == pytest.approx(0.5, abs=1e-5)
    assert f[2] == pytest.approx(-f[4], abs=1e-5)
    assert f[98] == 1 and f[99] == 1


def test_missing_hand_is_zeros_and_flag_off():
    f, hands = extract_features(fake_result(np.random.RandomState(1), left=False, right=True))
    assert hands is True and f[98] == 0 and f[99] == 1
    assert np.all(f[14:56] == 0) and np.any(f[56:98] != 0)


def test_no_pose_or_tiny_shoulders_gives_none():
    rng = np.random.RandomState(2)
    r = fake_result(rng)
    r.pose_landmarks = None
    assert extract_features(r) is None
    r = fake_result(rng)
    r.pose_landmarks.landmark[12].x, r.pose_landmarks.landmark[12].y = r.pose_landmarks.landmark[11].x, r.pose_landmarks.landmark[11].y
    assert extract_features(r) is None


def test_trim_and_resample():
    f = lambda i: np.full(D, i, np.float32)
    frames = [(f(i), 5 <= i <= 14) for i in range(30)]
    out = trim(frames)
    assert [v[0] for v in out] == list(range(3, 17))  # hand span padded by 2 frames either side
    assert trim([(f(i), i < 4) for i in range(10)]) is None  # fewer than 5 hand frames
    seq = to_seq(out)
    assert seq.shape == (T, D) and seq[0, 0] == 3 and seq[-1, 0] == 16
    assert to_seq([f(7)]).shape == (T, D)


def test_augment_keeps_shape_and_missing_hands_zero():
    x = np.ones((T, D), np.float32)
    x[:, 14:56] = 0
    x[:, 98] = 0
    o = augment(x, np.random.RandomState(0))
    assert o.shape == (T, D) and np.all(o[:, 14:56] == 0) and np.all(o[:, 98] == 0) and np.all(o[:, 99] == 1)


# ---------- Python and the browser must compute the same numbers ----------
def browser_functions():
    html = open(os.path.join(ROOT, "index.html"), encoding="utf-8").read()
    consts = re.search(r"const T = 30;.*?const D = [^\n]*\n", html, re.S).group(0)
    funcs = re.search(r"function extract\(res\).*?(?=function augment\()", html, re.S).group(0)
    return consts + funcs


@pytest.mark.skipif(shutil.which("node") is None, reason="node not installed")
def test_browser_and_python_features_match():
    rng = np.random.RandomState(3)
    clips, expected = [], []
    for c in range(25):
        n = int(rng.randint(5, 140))
        results = [fake_result(rng, left=bool(rng.rand() > 0.5), right=bool(rng.rand() > 0.5)) for _ in range(n)]
        clips.append([to_js(r) for r in results])
        frames = [extract_features(r) for r in results]
        frames = [fr for fr in frames if fr is not None]
        tr = trim(frames)
        expected.append(None if tr is None else to_seq(tr).ravel().tolist())
    js = browser_functions() + """
const clips = JSON.parse(require('fs').readFileSync(0, 'utf8'));
const out = clips.map(c => { const fr = c.map(extract).filter(Boolean); const tr = trim(fr); return tr ? Array.from(toSeq(tr)) : null; });
console.log(JSON.stringify(out));"""
    p = subprocess.run(["node", "-e", js], input=json.dumps(clips), capture_output=True, text=True, check=True)
    got = json.loads(p.stdout)
    assert sum(e is not None for e in expected) > 15
    for e, g in zip(expected, got):
        assert (e is None) == (g is None)
        if e is not None:
            assert len(g) == T * D
            assert np.max(np.abs(np.array(e) - np.array(g))) < 1e-6


@pytest.mark.skipif(shutil.which("node") is None, reason="node not installed")
def test_python_and_js_rounding_pick_same_frames():
    """Python's round() sends x.5 to the even number and JS Math.round sends it up, so prove they never disagree here."""
    py = [[int(round(t * (n - 1) / (T - 1))) for t in range(T)] for n in range(1, 2001)]
    js = "const o=[];for(let n=1;n<=2000;n++){const r=[];for(let t=0;t<30;t++)r.push(Math.round(t*(n-1)/29));o.push(r)}console.log(JSON.stringify(o))"
    assert json.loads(subprocess.run(["node", "-e", js], capture_output=True, text=True, check=True).stdout) == py


# ---------- splitting and leakage ----------
def tiny(n_per=10, k=3):
    labels, splits, signers, videos = [], [], [], []
    for c in range(k):
        for i in range(n_per):
            s = "train" if i < 6 else "val" if i < 8 else "test"
            labels.append(f"w{c}"), splits.append(s), signers.append({"train": "A", "val": "B", "test": "C"}[s]), videos.append(f"{c}-{i}")
    return labels, splits, signers, videos


def test_official_splits_preserved_and_disjoint():
    from train import assign_splits, split_indices, validate_splits

    labels, splits, signers, videos = tiny()
    out, policy = assign_splits(labels, splits)
    assert out == splits and policy == "official"
    idx = split_indices(out)
    assert not set(idx["train"]) & set(idx["test"]) and not set(idx["val"]) & set(idx["test"]) and not set(idx["train"]) & set(idx["val"])
    assert sum(len(v) for v in idx.values()) == len(labels)
    s = validate_splits(labels, out, signers, videos)
    assert s["counts"] == {"train": 18, "val": 6, "test": 6} and s["signer_overlap"] == {"train_val": 0, "train_test": 0, "val_test": 0}


def test_val_and_test_are_separate_not_one_heldout_pool():
    """Regression test for the original bug, where everything that wasn't train became one shared held-out set."""
    from train import split_indices

    labels, splits, _, _ = tiny()
    idx = split_indices(splits)
    assert len(idx["val"]) == 6 and len(idx["test"]) == 6 and not set(idx["val"]) & set(idx["test"])


def test_missing_splits_are_deterministic_and_three_way():
    from train import assign_splits

    labels = [f"w{c}" for c in range(3) for _ in range(20)]
    a, policy = assign_splits(labels, [None] * 60, seed=1)
    b, _ = assign_splits(labels, [None] * 60, seed=1)
    assert a == b and policy == "stratified_random" and {"train", "val", "test"} == set(a)


@pytest.mark.parametrize("break_it, msg", [
    (lambda l, s, g, v: s.__setitem__(slice(None), ["train"] * len(s)), "empty"),
    (lambda l, s, g, v: v.__setitem__(1, v[0]), "more than once"),
    (lambda l, s, g, v: g.__setitem__(s.index("test"), "A"), "more than one split"),
    (lambda l, s, g, v: [s.__setitem__(i, "train") for i, y in enumerate(l) if y == "w0"], "missing from val or test"),
])
def test_validation_rejects_bad_splits(break_it, msg):
    from train import validate_splits

    labels, splits, signers, videos = tiny()
    break_it(labels, splits, signers, videos)
    with pytest.raises(ValueError, match=msg):
        validate_splits(labels, splits, signers, videos)


def test_unknown_split_name_rejected():
    from train import assign_splits

    with pytest.raises(ValueError):
        assign_splits(["a"], ["holdout"])


def test_topk_hand_checked():
    from train import topk

    p = np.array([[0.7, 0.2, 0.1], [0.1, 0.2, 0.7]])
    assert topk(p, np.array([0, 0]), 1) == 0.5 and topk(p, np.array([0, 1]), 2) == 1.0


def test_model_shape():
    from train import build

    m = build(7)
    assert m.input_shape == (None, T, D) and m.output_shape == (None, 7)
    assert [type(x).__name__ for x in m.layers] == ["Bidirectional", "Dropout", "Dense", "Dense"]
    assert build(7, "gru").output_shape == (None, 7)


def test_end_to_end_train_then_evaluate_once(tmp_path):
    """Pipeline plumbing on SYNTHETIC numbers: never reported as a result."""
    rng = np.random.RandomState(0)
    labels, splits, signers, videos = tiny()
    samples = [{"label": l, "split": s, "signer": g, "video_id": v, "data": (rng.rand(T, D) + int(l[1])).round(3).tolist()} for l, s, g, v in zip(labels, splits, signers, videos)]
    data = tmp_path / "d.json"
    data.write_text(json.dumps({"seqLen": T, "featureDim": D, "labels": sorted(set(labels)), "samples": samples}))
    run, res = tmp_path / "run", tmp_path / "res"
    env = {**os.environ, "TF_CPP_MIN_LOG_LEVEL": "2"}
    py = sys.executable
    subprocess.run([py, "python/train.py", "--data", str(data), "--out", str(run), "--epochs", "2", "--aug", "1"], cwd=ROOT, env=env, check=True, capture_output=True)
    assert {"model.keras", "labels.json", "history.json", "val_report.json", "config.json"} <= set(os.listdir(run))
    assert "test" not in json.dumps(json.load(open(run / "val_report.json")))
    subprocess.run([py, "python/evaluate.py", "--run", str(run), "--data", str(data), "--results", str(res)], cwd=ROOT, env=env, check=True, capture_output=True)
    assert {"report.json", "per_class_metrics.csv", "confusion_matrix.csv", "confusion_matrix.png", "loss_curve.png", "accuracy_curve.png", "test_log.json"} <= set(os.listdir(res))
    rep = json.load(open(res / "report.json"))
    assert rep["counts"]["test"] == 6 and 0 <= rep["test_top1"] <= 1
    again = subprocess.run([py, "python/evaluate.py", "--run", str(run), "--data", str(data), "--results", str(res)], cwd=ROOT, env=env, capture_output=True, text=True)
    assert again.returncode != 0 and "already evaluated" in again.stderr


# ---------- the real extracted dataset ----------
@pytest.mark.skipif(not os.path.exists(DATASET), reason="dataset not on this machine")
def test_real_dataset_integrity():
    from train import load, validate_splits

    d = load([DATASET])
    assert d["X"].shape[1:] == (T, D)
    s = validate_splits(d["labels"], d["splits"], d["signers"], d["videos"])
    assert s["classes"] == 50 and not any(s["signer_overlap"].values())
    flags = d["X"][:, :, 98:]
    assert set(np.unique(flags).tolist()) <= {0.0, 1.0}
    assert (flags.sum(axis=2) > 0).mean() > 0.5  # hands are actually visible in most frames
