"""Tracking-based dead-ball mask for the experimental hysteresis decoder."""

import xml.etree.ElementTree as ET

import numpy as np
import pandas as pd

FPS = 25
PX, PY = 105.3, 68.0
GK_INDICES = (23, 45)
HALVES = ((11185, 90651), (110808, 190197))
KICKOFF_OFFSET_FRAMES = 88

STILL_WINDOW = 12
STILL_METERS = 1.0
STILL_MIN_S = 3.0
SLOW_PLAYER_MPS = 1.5
MISSING_MIN_S = 5.0
OUTSIDE_MIN_S = 1.0
CLOSE_GAP_S = 1.0
KEEP_MIN_S = 3.0
SMOOTH_FRAMES = 51


def runs_of(mask):
    m = np.concatenate([[False], mask, [False]])
    d = np.diff(m.astype(np.int8))
    return list(zip(np.where(d == 1)[0], np.where(d == -1)[0] - 1))


def keep_long(mask, min_frames):
    out = np.zeros_like(mask)
    for a, b in runs_of(mask):
        if b - a + 1 >= min_frames:
            out[a:b + 1] = True
    return out


def close_gaps(mask, max_gap):
    out = mask.copy()
    runs = runs_of(mask)
    for (_, b0), (a1, _) in zip(runs[:-1], runs[1:]):
        if a1 - b0 - 1 <= max_gap:
            out[b0:a1] = True
    return out


def second_half_kickoff_frame(tagging_path):
    root = ET.parse(tagging_path).getroot()
    starts = []
    for inst in root.iter("instance"):
        text = {}
        for lab in inst.findall("label"):
            text[(lab.findtext("group") or "").strip()] = (lab.findtext("text") or "").strip()
        if text.get("Type", "").lower() == "kick off (start)" and text.get("Half", "").lower().startswith("2"):
            starts.append(int(round(float(inst.findtext("start")) * FPS)))
    if len(starts) != 1:
        raise ValueError("expected exactly one second-half Kick Off (Start), found %d" % len(starts))
    return starts[0] + KICKOFF_OFFSET_FRAMES


def tracking_dead_mask(P, B):
    n = len(B)
    bx, by = B[:, 0] * PX, B[:, 1] * PY
    valid = ~np.isnan(bx) & ~np.isnan(by)
    outside = valid & ((B[:, 0] < 0) | (B[:, 0] > 1) | (B[:, 1] < 0) | (B[:, 1] > 1))

    w = STILL_WINDOW
    disp = np.full(n, np.nan)
    disp[w:n - w] = np.hypot(bx[2 * w:] - bx[:n - 2 * w], by[2 * w:] - by[:n - 2 * w])
    still = np.isfinite(disp) & (disp < STILL_METERS)

    outfield = [i for i in range(P.shape[1]) if i not in GK_INDICES]
    speed = np.nanmean(P[:, outfield, 2], axis=1)
    speed = pd.Series(speed).rolling(SMOOTH_FRAMES, center=True, min_periods=10).median().to_numpy()
    slow = np.isfinite(speed) & (speed < SLOW_PLAYER_MPS)

    a = keep_long(outside, int(OUTSIDE_MIN_S * FPS))
    b = keep_long(still, int(STILL_MIN_S * FPS))
    d = keep_long(~valid & slow, int(MISSING_MIN_S * FPS))
    mask = close_gaps(a | b | d, int(CLOSE_GAP_S * FPS))
    return keep_long(mask, int(KEEP_MIN_S * FPS))


def build_dead_mask(P, B, tagging_path=None, pre_kickoff_dead=True):
    mask = tracking_dead_mask(P, B)
    if pre_kickoff_dead:
        if tagging_path is None:
            raise ValueError("pre_kickoff_dead needs the Smart Tagging path")
        kickoff = second_half_kickoff_frame(tagging_path)
        mask[HALVES[1][0]:kickoff] = True
    graded = np.zeros(len(B), bool)
    for a, b in HALVES:
        graded[a:b + 1] = True
    return mask & graded
