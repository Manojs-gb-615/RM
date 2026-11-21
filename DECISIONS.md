# DECISIONS.md — Three.js ASL Avatar Renderer

Status: implemented. Every decision below is reflected in code. Assumptions that
could not be verified in this checkout are marked **UNVERIFIED**.

## 1. Data flow

```
How2Sign OpenPose JSON ──► ml/convert_openpose.py ──► [T,274] .npy (SignLingo order)
                                                       x/1280, y/720, [0,1]
                                                              │
                                                              ▼
                         backend/services/canonical_pose.py (adapter)
                         - layout reorder -> openpose137 (canonical)
                         - invert normalization -> pixels -> shoulder units
                         - neck origin, +x signer-left, +y up, +z toward camera
                         - mask zeros, interpolate/hold missing
                                                              │
                                                              ▼
                         demo_data/*.npy + manifest.json  (ground truth or synthetic)
                                                              │
                    GET /api/poses/{id}?format=bin|json ◄──────┘
                                                              │
                                                              ▼
   BROWSER: decode ─► One Euro filter (body/hands) ─► 2D→3D solver
            ─► bone-local quaternions ─► VRM normalized humanoid + FaceRig
            ─► Three.js render (or procedural fallback mannequin)
                                                              │
                                                              ▼
                              evaluation harness (reprojection, jitter, flips, ...)
```

The backend never retargets and never touches the generation model.

## 2. Repository discrepancies found (important)

### 2.1 Keypoint order is NOT the task's canonical order

The repository's real 137-point layout is produced by
`SignLingo-AI-Mini/ml/convert_openpose.py`:

| Source slice | Region | Count |
|---|---|---|
| 0–24 | OpenPose BODY_25 | 25 |
| 25–45 | left hand | 21 |
| 46–66 | right hand | 21 |
| 67–136 | face (OpenPose 70 = iBUG-68 + 2 pupils) | 70 |

The task specifies body → face → left hand → right hand. Both are now explicit:

- `layouts/openpose137.json` — canonical target layout (task order).
- `layouts/signlingo_openpose137.json` — the repository's actual order plus a
  `sourceToCanonical[137]` index map and a `regions` description.

`backend/services/canonical_pose.py` applies the map. Both layout files are
covered by `tests/test_pose_layouts.py`. No layout guessing at runtime.

### 2.2 Normalization is anisotropic, not simple 0–1

`convert_openpose.py` writes `x = px / 1280`, `y = py / 720`, clipped to `[0,1]`
(the frame is 1280×720). That means equal pixel distances do not have equal
normalized distances. The adapter therefore:

1. Optionally inverts the research model's `(pose - mean)/std`
   (`src/data/how2sign_dataset.py`, `norm_stats.npz`) when those stats are
   supplied.
2. Converts back to pixel space with the source frame size
   (`source_width`, `source_height`; default 1280×720).
3. Flips y to point up: `Y = (1 - y_norm) * height`.
4. Centres on the neck (index 1) and divides by the **per-clip median shoulder
   width** measured in pixel space (indices 2 and 5), so a single shoulder-width
   unit is constant for the clip.
5. Sets z = 0; depth is inferred later in the browser.

Generated model output has no confidence channel. The adapter takes
`source="generated"` and treats every point as valid (only exact `(0,0)`
sentinel pairs are still treated as missing, matching the converter).

### 2.3 Real data is absent from this checkout — demo uses synthetic by default

`SignLingo-AI-Mini/data/real/poses/` is empty (gitignored), there is no
`models/retrieval_index.pkl`, and `C:/how2sign` does not exist on this machine.
`scripts/prepare_demo_poses.py` therefore searches for real OpenPose JSON,
`[T,274]` `.npy` files and a metadata CSV; if none are found it writes a clearly
labelled synthetic sequence and sets `source: "synthetic"` in the manifest. The
viewer displays a warning for synthetic data. Real-clip evaluation numbers are
**not** claimed. This is recorded in `docs/AVATAR_EVAL.md` as well.

### 2.4 The ContextSign frontend was config-only

`frontend/` contained `package.json`, `vite.config.ts`, `tsconfig*.json` and
`tailwind.config.js` but no `index.html` or `src/`. The new avatar module
therefore lives in `frontend/src/avatar/` and the demo page is scaffolded in
`frontend/` (TypeScript + Vite + Three.js, no React). The existing stick figure
drawing logic is ported to framework-free TypeScript
(`src/avatar/stickFigure.ts`) using the same connections and colours as
`SignLingo-AI-Mini/frontend/src/components/PoseAnimationViewer.tsx`.

### 2.5 Navigation is not part of this layer

The avatar layer is framework-agnostic. The demo page is a single vanilla page;
it does not add routing to the research UI.

## 3. Canonical space contract

- Origin: neck (body index 1).
- Unit: 1 = median shoulder width of the clip (indices 2–5).
- +x: the signer's anatomical left, as seen in the image (not mirrored by
  default).
- +y: up.
- +z: toward the camera (the signer faces the viewer).
- Data frames are `float32[T,137,2]`, x/y only. Missing points are encoded with a
  parallel validity mask internally; the wire format nulls no values, and a
  manifest field describes whether a sequence is ground truth (zeros = missing)
  or generated (all valid).

**Anatomical mapping:** canonical index 95–115 is the signer's left hand and
116–136 the signer's right hand. The avatar's humanoid `left*` bones receive the
signer's left data. No mirroring by default; `mirror` is an optional viewer
toggle that swaps hand data and negates x.

## 4. Depth (2D → 3D)

Weak perspective / orthographic camera. For a segment of known rest length `L`
and observed 2D displacement `(dx, dy)`:

```
d2 = sqrt(dx^2 + dy^2)
d2c = min(d2, L * (1 - 1e-4))          # clamp noise
dz = ±sqrt(max(0, L^2 - d2c^2))
```

The sign is chosen by:

1. **Prior:** signing happens in front of the torso, so forearms and hands
   prefer +z. Upper arm prior is a configurable smaller bias (`upperArmForward`
   = 0.25 by default).
2. **Continuity:** the candidate whose z is closer to the previous frame's z is
   preferred; a sign flip needs a configurable advantage
   (`depthFlipHysteresis`, default 0.35 in shoulder-width units).

Both arms are solved independently with two-bone analytic IK. Shoulder and elbow
limits are configurable; default elbow flexion is clamped to 5°–175°.

This is a heuristic, not a full 3D reconstruction. It cannot resolve true depth;
see `LIMITATIONS`.

## 5. Hands

- Palm normal comes from the 2D signed area of `(wrist, indexMCP, pinkyMCP)`
  combined with handedness. The sign convention is configurable
  (`palmSignLeft`, `palmSignRight`); defaults are `+1`/`-1` and were validated
  only against synthetic data (**UNVERIFIED** on real signers).
- Finger curl = projected chain length / expected chain length plus 2D joint
  angles. Curl drives the avatar's proximal/intermediate/distal rotations.
- Finger spread = angle between adjacent MCP directions.
- Thumb uses a separate opposition/curl estimate from thumb tip vs index MCP.
- Missing hand points fall back to the last valid frame, then to a neutral hand.
  A `handConfidence` flag is emitted and the viewer can show a warning.

VRM finger bones are driven through `humanoid.getNormalizedBoneNode`. If the
avatar lacks a finger chain, that chain is skipped and logged, never silently
rotated.

## 6. Torso and head

- Torso stays camera-facing. Roll comes from the shoulder line; pitch from the
  shoulder-mid to hip-mid line; yaw is a small heuristic from nose offset
  relative to the shoulder midpoint. These are deliberately small and clamped.
- Head rotation is solved with a weak-perspective POSIT/Procrustes fit using
  seven stable points (nose, chin, both eyes, both ears, mouth centre) against a
  generic 3D face model in shoulder-width units. Fallback to eye-line roll +
  nose-offset yaw when fewer than four points are valid.
- The fit result is smoothed with slerp; the head position is clamped so it
  cannot detach from the neck.

## 7. Filtering and smoothing

- **One Euro filter** on canonical keypoints before retargeting. Separate
  parameters for body and hands: hands use a higher `minCutoff`/`beta`
  (more responsive) so fingerspelling is not blurred.
- **Slerp smoothing** on bone rotations after retargeting, with a shorter
  half-life for hands than body.
- Gap handling: gaps ≤ `maxGapFrames` (default 5) are linearly interpolated;
  longer gaps hold the last valid value. Leading/trailing gaps hold the first/last
  valid value.
- Outlier rejection: a joint moving faster than `maxJointSpeed` shoulder
  widths/frame is treated as a spike and replaced by the previous value.

## 8. Frame rate and playback

- FPS is read from manifest metadata. Default 24 only when metadata is absent.
- Rendering runs at display refresh rate; positions lerp and rotations slerp
  between data frames using `alpha = (t - frameStart) / frameDuration`.
- Sentence boundaries come from `manifest.sentences[].startFrame/endFrame`.
  Crossfade is 7 frames (configurable 6–8) at boundaries. No rest pose is
  inserted between sentences of the same session; the rest pose is used only at
  session start/end. This matches the E1–E3 context-window experiments where
  discourse is continuous.

## 9. Avatar and FaceRig

- Primary avatar: VRM 1.0 via `@pixiv/three-vrm` (pinned). VRM 0.x is also
  accepted by the loader; three-vrm normalises its facing direction. The loader
  logs the detected VRM version. Facing correction is applied through
  three-vrm's own scene rotation, **not** by rotating our animation.
- `FaceRig` probes the avatar at load:
  1. **ARKit-52 morph targets** on the face mesh (`browInnerUp`,
     `browDownLeft/Right`, `browOuterUpLeft/Right`, `eyeBlinkLeft/Right`,
     `jawOpen`, `mouthFunnel`, `mouthPucker`, `mouthSmileLeft/Right`,
     `mouthFrownLeft/Right`, `cheekPuff`, `eyeLookIn/Out/Up/Down`).
  2. **VRM expression presets** (`aa`, `ih`, `ou`, `ee`, `oh`, `blink`,
     `lookUp/Down/Left/Right`, `happy`, `angry`, `sad`).
  3. Custom brow expressions are detected by name matching
     (`brow*`, `eyebrow*`).
  Unsupported channels are reported in the UI. If no brow channel exists, a
  visible warning is shown: ASL yes/no vs wh-question marking cannot be
  rendered faithfully.
- If no VRM is supplied, a **procedural mannequin** with the same humanoid
  skeleton and simple face markers is rendered, so the demo, evaluation harness
  and tests run without any avatar asset.
- The avatar file is never downloaded automatically. Place it at
  `frontend/public/avatars/default.vrm` or use the file picker.

## 10. Orientation from the repo

The existing app draws body indices 0–24 as openpose BODY_25, left hand
25–45, right hand 46–66 and face 67–136 (see `PoseAnimationViewer.tsx`). The
adapter and the ported stick figure use that same source order before converting
to canonical order, so the side-by-side comparison remains aligned.

## 11. Stack and versions

- TypeScript, Vite, Three.js, `@pixiv/three-vrm` (versions pinned in
  `frontend/package.json`).
- Vitest for unit/golden tests; `tsx` for the Node evaluation runner.
- Retargeting is entirely browser-side (pure math in `frontend/src/avatar/`),
  reusable from Node for evaluation. The backend only serves canonical data.
- No React dependency in the avatar module or demo.

## 12. Evaluation choices

- Reprojection compares **FK joint positions** (not the solver's target joints)
  against input canonical keypoints, using the same orthographic projection.
  Using target joints would trivially report ±0 for arms and would not measure
  retargeting error.
- Units: shoulder widths. Report mean and p95 per body part.
- Jitter: mean absolute second difference (jerk proxy) and high-frequency energy
  (`var(x - 3-frame moving average)`), before and after filtering, plus the
  lag (frames) of the best cross-correlation between raw and filtered wrist
  trajectories.
- Limb flips: sign changes of the elbow's side of the shoulder–wrist line,
  excluding near-collinear frames; wrist speed spikes above the outlier
  threshold.
- Self-intersection: hand sphere vs torso capsule (shoulder-mid→hip-mid) and
  head sphere; count contiguous episodes.
- Bone-length consistency: coefficient of variation of each FK bone length per
  clip; max across bones is reported.
- Golden regression: retarget a fixed synthetic fixture and compare FK joint
  positions to a committed JSON snapshot with a tolerance of `1e-4` shoulder
  widths. Any silent change to the retargeting math fails the test.
- Results are generated by `npx tsx scripts/eval.ts` into
  `avatar_eval_results.json` and `docs/AVATAR_EVAL.md`. Only numbers produced by
  that command are reported.

## 13. Things explicitly NOT done / unverified

- No real How2Sign poses are present, so "real clip" metrics are unavailable in
  this checkout. The preparation script supports them and the evaluation will
  include them automatically.
- No VRM asset is committed. The VRM path (bone names, ARKit morph probing,
  VRM0/VRM1 facing) is implemented against the pinned library API but **cannot
  be visually verified here**; the procedural path is what runs in the demo and
  tests.
- Palm orientation and elbow depth are heuristic. No claim of ASL correctness.
- No signer evaluation was performed for this renderer.

## 14. Configuration surface

All tunables live in `frontend/src/avatar/config.ts` (filters, depth priors,
limits, camera, crossfade, face gains) and are exposed in the viewer's debug
panel. Backend layout/normalization options are arguments to
`backend/services/canonical_pose.py` and `scripts/prepare_demo_poses.py`.
