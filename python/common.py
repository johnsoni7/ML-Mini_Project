"""Shared feature spec. MUST stay identical to the constants/functions in index.html.

Per frame (100 floats):
  7 pose points (nose, L/R shoulder, L/R elbow, L/R wrist)  -> 14
  left hand 21 points                                       -> 42
  right hand 21 points                                      -> 42
  [left_present, right_present] flags                       -> 2
All x/y are centred on the shoulder midpoint and divided by shoulder width.
Missing hands are zeros. Each clip is trimmed to the span where a hand is visible
and then resampled (nearest frame) to T=30 frames.
"""
import numpy as np

T = 30
POSE_IDX = [0, 11, 12, 13, 14, 15, 16]
D = (len(POSE_IDX) + 21 + 21) * 2 + 2  # 100


def extract_features(results):
    """MediaPipe Holistic results -> (feature_vector[D], hands_visible) or None."""
    pose = results.pose_landmarks
    if pose is None:
        return None
    p = pose.landmark
    a, b = p[11], p[12]
    sc = float(np.hypot(a.x - b.x, a.y - b.y))
    if sc < 0.03:
        return None
    cx, cy = (a.x + b.x) / 2, (a.y + b.y) / 2
    f = np.zeros(D, dtype=np.float32)
    k = 0
    for i in POSE_IDX:
        f[k] = (p[i].x - cx) / sc
        f[k + 1] = (p[i].y - cy) / sc
        k += 2
    L, R = results.left_hand_landmarks, results.right_hand_landmarks
    for h in (L, R):
        for i in range(21):
            if h is not None:
                f[k] = (h.landmark[i].x - cx) / sc
                f[k + 1] = (h.landmark[i].y - cy) / sc
            k += 2
    f[k] = 1.0 if L is not None else 0.0
    f[k + 1] = 1.0 if R is not None else 0.0
    return f, (L is not None or R is not None)


def trim(frames):
    """frames: list of (feat, hands). Returns list of feats trimmed to hand-visible span, or None."""
    idx = [i for i, (_, h) in enumerate(frames) if h]
    if len(idx) < 5:
        return None
    lo, hi = max(0, idx[0] - 2), min(len(frames), idx[-1] + 3)
    return [f for f, _ in frames[lo:hi]]


def to_seq(feats):
    """list of feature vectors -> array (T, D) by nearest-frame resampling."""
    n = len(feats)
    out = np.zeros((T, D), dtype=np.float32)
    for t in range(T):
        out[t] = feats[int(round(t * (n - 1) / (T - 1)))]
    return out


def augment(x, rng=np.random):
    """Same idea as the browser: scale / rotate / shift / jitter present points only."""
    s = 0.9 + rng.rand() * 0.2
    dx, dy = (rng.rand() - 0.5) * 0.2, (rng.rand() - 0.5) * 0.2
    r = (rng.rand() - 0.5) * 0.2
    c, sn = np.cos(r), np.sin(r)
    o = np.zeros_like(x)
    for t in range(T):
        for j in range(49):
            present = j < 7 or (x[t, 98] > 0 if j < 28 else x[t, 99] > 0)
            if not present:
                continue
            px, py = x[t, j * 2], x[t, j * 2 + 1]
            o[t, j * 2] = (px * c - py * sn) * s + dx + (rng.rand() - 0.5) * 0.03
            o[t, j * 2 + 1] = (px * sn + py * c) * s + dy + (rng.rand() - 0.5) * 0.03
        o[t, 98], o[t, 99] = x[t, 98], x[t, 99]
    return o
