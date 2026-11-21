"""Synthetic canonical pose sequences for the avatar demo and tests.

These are hand-authored, clearly-labelled test motions (arm raise, handshape
cycle, head nod, brow raise). They are NOT American Sign Language and are never
used for model training or for claims about signing quality.
"""

from __future__ import annotations

import math
from typing import Dict, List, Tuple

import numpy as np

NUM_POINTS = 137
FPS = 24.0

# Proportions in shoulder-width units (canonical space).
UPPER_ARM = 0.62
FOREARM = 0.55
HEAD_CY = 0.47
HEAD_RX = 0.22
HEAD_RY = 0.30

SENTENCE_TEXTS = {
    "arm_raise": ["Raise the left arm up", "Lower the arm back down"],
    "handshape_cycle": ["Open hand", "Close the hand", "Point the index finger", "Open again"],
    "head_nod": ["Nod the head yes", "Return to neutral"],
    "brow_raise": ["Raise the eyebrows", "Furrow the brows"],
    "session": [
        "Raise the left arm up",
        "Open and close the hand",
        "Nod the head yes",
        "Raise the eyebrows",
    ],
}


def _rot(v: np.ndarray, angle: float) -> np.ndarray:
    c, s = math.cos(angle), math.sin(angle)
    return np.array([c * v[0] - s * v[1], s * v[0] + c * v[1]], dtype=np.float64)


def _ellipse(cx: float, cy: float, rx: float, ry: float, a0: float, a1: float, n: int) -> np.ndarray:
    angles = np.linspace(a0, a1, n)
    return np.stack([cx + rx * np.cos(angles), cy + ry * np.sin(angles)], axis=-1)


def _face_points(
    brow_raise: float,
    brow_furrow: float,
    mouth_open: float,
    smile: float,
    gaze_x: float,
    gaze_y: float,
    head_roll: float,
    head_nod: float,
) -> np.ndarray:
    """Build 70 face points (iBUG-68 + pupils) in canonical coordinates."""
    cy = HEAD_CY - 0.08 * head_nod
    rx = HEAD_RX
    ry = HEAD_RY * (1.0 - 0.10 * abs(head_nod))

    f = np.zeros((70, 2), dtype=np.float64)

    # Jaw 0-16: left temple -> chin -> right temple (x is signer-left).
    jaw = _ellipse(0.0, cy, rx, ry, math.radians(200), math.radians(340), 17)
    f[0:17] = jaw

    # Brows 17-26. Face-local 17-21 = signer's right (image -x).
    brow_y = cy + 0.115 + 0.045 * brow_raise
    brow_inner = 0.045 - 0.020 * brow_furrow
    for side, (x0, x1) in ((-1, (-0.19, -brow_inner)), (1, (brow_inner, 0.19))):
        start = 17 if side < 0 else 22
        xs = np.linspace(x0, x1, 5)
        arch = np.sin(np.linspace(0, math.pi, 5)) * 0.022
        f[start:start + 5, 0] = xs
        f[start:start + 5, 1] = brow_y + arch + (0.012 * brow_raise if side > 0 else 0.0)

    # Nose bridge 27-30 and base 31-35.
    f[27:31, 0] = 0.0
    f[27:31, 1] = np.linspace(cy + 0.09, cy - 0.02, 4)
    f[31:36, 0] = np.linspace(-0.055, 0.055, 5)
    f[31:36, 1] = cy - 0.035

    # Eyes 36-47: 6 points each, right then left.
    eye_dx = 0.105
    for start, cx in ((36, -eye_dx), (42, eye_dx)):
        pts = _ellipse(cx, cy + 0.055, 0.072, 0.030, 0, 2 * math.pi, 6)[:-0] 
        f[start:start + 6] = _ellipse(cx, cy + 0.055, 0.072, 0.030, math.radians(15), math.radians(345), 6)

    # Outer lips 48-59 and inner lips 60-67.
    mouth_cy = cy - 0.175 - 0.015 * mouth_open
    f[48:60] = _ellipse(0.0, mouth_cy, 0.095 + 0.012 * smile, 0.050 + 0.030 * mouth_open,
                        math.radians(180), math.radians(540), 12)
    f[60:68] = _ellipse(0.0, mouth_cy, 0.062, 0.018 + 0.028 * mouth_open,
                        math.radians(180), math.radians(540), 8)

    # Pupils 68-69.
    f[68] = [-eye_dx + 0.018 * gaze_x, cy + 0.055 + 0.012 * gaze_y]
    f[69] = [eye_dx + 0.018 * gaze_x, cy + 0.055 + 0.012 * gaze_y]

    # Head roll around the neck; head nod is already approximated above.
    f = np.stack([_rot(p, head_roll) for p in f], axis=0)
    return f


def _hand_points(
    wrist: np.ndarray,
    forearm_dir: np.ndarray,
    side: int,
    curl: float,
    spread: float,
    thumb_opposition: float,
    palm_flip: float = 0.0,
) -> np.ndarray:
    """Build 21 hand points in canonical coordinates.

    ``side`` is +1 for the signer's left hand and -1 for the right. Fingers
    point along ``forearm_dir``; the lateral axis is perpendicular. Curl folds
    the chain in the image plane (a 2D approximation of flexion).
    """
    u = forearm_dir / (np.linalg.norm(forearm_dir) + 1e-9)
    v = np.array([-u[1], u[0]]) * side  # lateral, away from thumb for the index side

    pts = np.zeros((21, 2), dtype=np.float64)
    pts[0] = wrist

    # Palm frame: MCP row is 0.13 ahead of the wrist.
    mcp_base = wrist + u * 0.13
    lateral = {"index": 0.070, "middle": 0.024, "ring": -0.024, "pinky": -0.070}
    segment = {
        "index": (0.085, 0.052, 0.042),
        "middle": (0.092, 0.058, 0.046),
        "ring": (0.086, 0.054, 0.043),
        "pinky": (0.070, 0.046, 0.038),
    }
    order = ["index", "middle", "ring", "pinky"]
    for i, name in enumerate(order):
        start = 5 + i * 4
        lat = lateral[name] + spread * (0.012 * (2 - i))
        p = mcp_base + v * lat
        pts[start] = p
        direction = u
        for j, seg in enumerate(segment[name]):
            bend = math.radians(105 * curl) * (1.0 if j > 0 else 0.45)
            direction = _rot(direction, -bend * side)
            p = p + direction * seg
            pts[start + 1 + j] = p

    # Thumb 1-4: leaves the wrist toward the index side and forward.
    thumb_dir = _rot(u, math.radians(side * (55 - 35 * thumb_opposition)))
    thumb_dir = thumb_dir + v * (0.25 * (1 - thumb_opposition)) * -side
    thumb_dir /= np.linalg.norm(thumb_dir) + 1e-9
    p = wrist + u * 0.035 + v * (-0.035)
    pts[1] = p
    for j, seg in enumerate((0.075, 0.048, 0.040)):
        bend = math.radians(45 * curl + 25 * thumb_opposition)
        thumb_dir = _rot(thumb_dir, -bend * side)
        p = p + thumb_dir * seg
        pts[2 + j] = p

    return pts


def _build_frame(
    phase: float,
    *,
    arm_raise: float,
    hand_curl: float,
    hand_spread: float,
    thumb_opposition: float,
    head_nod: float,
    head_roll: float,
    brow_raise: float,
    brow_furrow: float,
    mouth_open: float,
    smile: float,
    gaze_x: float,
    gaze_y: float,
) -> np.ndarray:
    p = np.zeros((NUM_POINTS, 2), dtype=np.float64)

    # --- torso ---------------------------------------------------------
    p[1] = [0.0, 0.0]                       # neck
    p[5] = [0.5, -0.06]                     # left shoulder
    p[2] = [-0.5, -0.06]                    # right shoulder
    p[8] = [0.0, -1.28]                     # mid hip
    p[12] = [0.18, -1.28]
    p[9] = [-0.18, -1.28]
    p[13] = [0.20, -2.05]                   # left knee
    p[10] = [-0.20, -2.05]
    p[14] = [0.20, -2.82]
    p[11] = [-0.20, -2.82]
    p[19] = [0.24, -2.92]
    p[20] = [0.16, -2.94]
    p[21] = [0.20, -2.86]
    p[22] = [-0.24, -2.92]
    p[23] = [-0.16, -2.94]
    p[24] = [-0.20, -2.86]

    # --- arms (2D forward kinematics) ---------------------------------
    arm_targets = {
        # side: (shoulder angle from +x, elbow flexion, wrist add-on)
        "left": (-70.0 + 130.0 * arm_raise, 18.0 + 32.0 * arm_raise),
        "right": (-110.0 + 20.0 * arm_raise, 14.0 + 20.0 * arm_raise),
    }
    wrists: Dict[str, np.ndarray] = {}
    forearms: Dict[str, np.ndarray] = {}
    for side, (angle_deg, flex_deg) in arm_targets.items():
        sh = p[5] if side == "left" else p[2]
        theta = math.radians(angle_deg)
        elbow = sh + UPPER_ARM * np.array([math.cos(theta), math.sin(theta)])
        phi = theta + math.radians(flex_deg) * (1 if side == "left" else -1)
        wrist = elbow + FOREARM * np.array([math.cos(phi), math.sin(phi)])
        if side == "left":
            p[6], p[7] = elbow, wrist
        else:
            p[3], p[4] = elbow, wrist
        wrists[side] = wrist
        forearms[side] = wrist - elbow

    # --- head placement ------------------------------------------------
    head_shift = np.array([0.0, -0.10 * head_nod])
    face = _face_points(
        brow_raise, brow_furrow, mouth_open, smile, gaze_x, gaze_y, head_roll, head_nod
    ) + head_shift
    p[25:95] = face
    # Body head landmarks follow the head pose (nose, eyes, ears).
    p[0] = face[30] + np.array([0.0, 0.01])
    p[15] = face[38]   # body REye = face-local right eye
    p[16] = face[44]   # body LEye = face-local left eye
    p[17] = face[0] + np.array([-0.02, 0.02])
    p[18] = face[16] + np.array([0.02, 0.02])

    # --- hands ---------------------------------------------------------
    p[95:116] = _hand_points(wrists["left"], forearms["left"], +1, hand_curl, hand_spread, thumb_opposition)
    p[116:137] = _hand_points(wrists["right"], forearms["right"], -1, hand_curl, hand_spread, thumb_opposition)

    return p


def _phase_params(kind: str, u: float) -> Dict[str, float]:
    """Map normalized time ``u`` in [0,1] to motion parameters."""
    base = dict(
        arm_raise=0.0, hand_curl=0.15, hand_spread=0.0, thumb_opposition=0.2,
        head_nod=0.0, head_roll=0.0, brow_raise=0.0, brow_furrow=0.0,
        mouth_open=0.1, smile=0.2, gaze_x=0.0, gaze_y=0.0,
    )
    if kind == "arm_raise":
        tri = 1.0 - abs(2.0 * u - 1.0)
        base.update(arm_raise=tri, mouth_open=0.1 + 0.15 * tri)
    elif kind == "handshape_cycle":
        base.update(arm_raise=0.55, hand_curl=(1 - math.cos(4 * math.pi * u)) / 2,
                    hand_spread=0.4 * math.sin(2 * math.pi * u),
                    thumb_opposition=(1 - math.cos(2 * math.pi * u)) / 2,
                    brow_raise=0.25)
    elif kind == "head_nod":
        base.update(arm_raise=0.1, head_nod=math.sin(2 * math.pi * u),
                    brow_raise=0.2 + 0.2 * math.sin(2 * math.pi * u), smile=0.3)
    elif kind == "brow_raise":
        base.update(arm_raise=0.15)
        if u < 0.5:
            base.update(brow_raise=math.sin(math.pi * u * 2), mouth_open=0.25)
        else:
            base.update(brow_furrow=math.sin(math.pi * (u - 0.5) * 2), smile=-0.2, gaze_x=-0.5)
    else:  # session: four 25% phases
        q = (u * 4) % 1.0
        stage = min(3, int(u * 4))
        if stage == 0:
            base.update(_phase_params("arm_raise", q))
        elif stage == 1:
            base.update(_phase_params("handshape_cycle", q))
        elif stage == 2:
            base.update(_phase_params("head_nod", q))
        else:
            base.update(_phase_params("brow_raise", q))
    return base


def build_sequence(kind: str = "session", fps: float = FPS) -> Tuple[np.ndarray, List[Dict[str, object]]]:
    """Return ``[T,137,2]`` canonical points and sentence boundaries."""
    if kind not in SENTENCE_TEXTS:
        raise ValueError(f"Unknown synthetic sequence '{kind}'")
    durations = {"arm_raise": 4.0, "handshape_cycle": 4.0, "head_nod": 3.0,
                 "brow_raise": 3.0, "session": 8.0}
    duration = durations[kind]
    T = int(round(duration * fps))
    frames = np.zeros((T, NUM_POINTS, 2), dtype=np.float32)
    for t in range(T):
        u = t / max(1, T - 1)
        frames[t] = _build_frame(t, **_phase_params(kind, u))

    texts = SENTENCE_TEXTS[kind]
    boundaries = []
    n = len(texts)
    for i, text in enumerate(texts):
        start = int(round(i * T / n))
        end = int(round((i + 1) * T / n))
        boundaries.append({"index": i, "startFrame": start, "endFrame": end, "text": text})
    return frames, boundaries


def build_all(fps: float = FPS) -> Dict[str, Tuple[np.ndarray, List[Dict[str, object]]]]:
    return {kind: build_sequence(kind, fps) for kind in SENTENCE_TEXTS}
