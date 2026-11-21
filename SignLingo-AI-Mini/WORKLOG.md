# WORKLOG — SignLingo AI Mini

**Window:** 2026-11-20 → 2026-11-21
**Scope:** Close the four gaps (real data, trained ranker, browser-verified frame sync,
automated + human evaluation) and prove each one by running it.

This is the complete engineering log: commands attempted, failures, root causes,
metrics, and evidence. Anything not verified is marked PENDING/BLOCKED rather than done.

---

## Executive summary

| Gap | Result | Evidence |
|---|---|---|
| 1. Real How2Sign data | **Done (subset)** — 450 val clips converted to `[T,274]`; index built | `data/real/metadata.csv`, `models/retrieval_index.pkl`, `tests/test_convert_openpose.py` |
| 2. Gradient-trained ranker | **Done** — dual encoder trained with InfoNCE; integrated into retrieval | `ml/ranker.py`, `models/ranker.pt`, `models/ranker_train_log.csv`, `tests/test_ranker.py` |
| 3. Browser-verified frame sync | **Done** — Playwright + Edge, skeleton tracks video ±1 frame | `frontend/tests/e2e/studio.spec.ts` (2 passed), screenshots 01–06 |
| 4a. Automated evaluation | **Done** — Recall@1/5/10, MRR, p50/p95 latency, baseline vs trained | `ml/evaluate.py`, `docs/eval_results.md`, `models/eval_results.json` |
| 4b. Human signer evaluation | **PENDING** — protocol + form + aggregator ready; no signers available | `eval/PROTOCOL.md`, `eval/rating_form.csv`, `eval/aggregate_ratings.py` |

**Final verification run:** `python -m pytest tests -q` → **31 passed**;
`npm run build` → **0 errors**; `npx playwright test` → **2 passed**.

**Honest headline:** the trained ranker does **not** beat the fixed baseline on the
non-trivial context-only ablation. The full-protocol numbers (~1.0) are near-trivial
because the query caption is itself in the index. No ASL-accuracy claim is made.

---

## Day 1 — 2026-11-20

### 22:13 — Initial commit (`dcf208f`)

Repository seeded with two apps: `ContextSign AI` and `SignLingo-AI-Mini`.

### Prototype build → commit `130e539` (23:03)

- `ml/dataset.py`: loader for `[T,274]` poses; synthetic generator for pipeline tests.
- `ml/retrieval.py`: SRT/VTT parsing; text-only and context-aware weighted cosine retrieval;
  sentence-embedding model cache; index save/load.
- `ml/inference.py`: backend wrapper with configurable index path and synthetic flag.
- `ml/train.py`: index builder with honest split evaluation (`self_top1`, `heldout_mean_score`).
- `ml/setup_data.py`: `real` / `synthetic` / `template` modes.
- `backend/main.py`: 8 FastAPI endpoints, lifespan startup.
- `frontend/`: React + Vite + Tailwind single page, canvas skeleton viewer.
- Tests: **19 passed**; frontend build passed.

### Gap audit → commit `4d9a2ce` (23:08)

Found and fixed layout/sync gaps that the first pass had missed:

- Layout forced to **LEFT video / RIGHT skeleton / BOTTOM transcript + controls**.
- Video play/pause drives the animation (not just transcript clicks).
- Added `tests/test_api_flow.py` exercising the real API path.
- Removed unused Vite template assets and fixed the favicon path.
- Tests re-run: **19 passed**.

### Honest limitations recorded at end of Day 1

- No real pose data on disk (`data/poses` empty) → index was synthetic only.
- No gradient training (fixed cosine weights).
- UI was built and API-tested but never opened in a real browser.
- No evaluation by ASL signers.

---

## Day 2 — 2026-11-21

### Gap 1 — Real How2Sign 2D keypoints

**Attempt A (blocked):** official How2Sign keypoint archives are hosted on Google Drive.
All links returned *"Too many users have viewed or downloaded this file recently"* (quota).
No `gdown` workaround succeeds because the limit is server-side.

**Attempt B (worked):** public Hugging Face mirror `Saintbook/how2sign_keypoints`.

- Downloaded `val_2D_keypoints.tar.gz` (**1.24 GB**) via `huggingface_hub`.
- Extracted to `C:\Users\manoj\Downloads\how2sign_keypoints\extracted\openpose_output\`
  → `json/` (1,739 clips) + `video/` (1,739 MP4s).
- Annotations: `C:\Users\manoj\Downloads\how2sign_realigned_val.csv` (tab-separated).

**Converter — `ml/convert_openpose.py`**

- Reads OpenPose BODY_25 + left/right hand + 70 face landmarks → **137 keypoints × 2 = [T,274]**.
- Layout: body 0–24, left hand 25–45, right hand 46–66, face 67–136.
- Selects the highest-confidence person per frame; sorts frames numerically; pads/truncates
  to the annotation window; normalizes x by width, y by height (1280×720).
- Missing keypoints stored as `(0,0)`; the viewer skips them.
- Resume-friendly (skips existing `.npy`); `--workers 8` with `ThreadPoolExecutor`.
- Emits `metadata.csv` with `sample_id,sentence,video_id,narrative_id,start_time,end_time,prev_sample_id,frames`.

**Conversion results:** first serial pass produced 259 clips in ~20 min; resumed with 8 workers
to **450 clips** in `data/real/poses/` and `data/real/metadata.csv` (82 KB, committed).
450 clips is the practical subset for this milestone; the train split (≈31k clips) was not processed.

**One-command pipeline — `ml/build_real_index.py`:** convert → train ranker → build index
(→ evaluation doc), with `--limit`, `--epochs`, `--seed`, `--data-dir`, `--skip-train`.
Refuses synthetic data.

### Gap 2 — Gradient-trained ranking model

**`ml/ranker.py`**

- Pose features: 137 keypoints → mean + std over time → **548-d** vector (missing points masked).
- Text tower: frozen `all-MiniLM-L6-v2` 384-d → Linear(384→256) → GELU → Linears → **128-d**, L2-normalized.
- Pose tower: 548 → 256 → **128-d**, L2-normalized.
- Symmetric **InfoNCE** loss with learnable logit scale.
- 80/20 train/val split by sample index; fixed seed 42; AdamW; per-epoch loss CSV;
  best-validation checkpoint saved.

**Training run:** 450 clips, 40 epochs. `train_loss` fell to **0.178**; validation loss
rose to **8.16** → clear overfitting, best-val weights retained. Artifacts:
`models/ranker.pt` (1.23 MB), `models/ranker_train_log.csv`.

**Retrieval integration — `ml/retrieval.py`**

- Vectorized scoring matrices (text, context, pose; baseline and trained variants).
- Trained score mix: **0.6 text + 0.2 context + 0.2 pose**; `_has_pose` guards missing poses.
- `score_all(..., include_current=True)`; `save()/load()` persist `use_trained` +
  `trained_checkpoint` (fixed a duplicate `index` pickle key).
- The 76 MB index is generated locally and gitignored; the 1.2 MB ranker checkpoint is committed.

**CLI / backend plumbing**

- `ml/train.py`: `--ranker`, `--train-ranker`, `--epochs`, `--seed`, `--no-trained`;
  synthetic builds redirect to `models/synthetic_index.pkl`.
- `ml/inference.py`: `RANKER_PATH` env / `ranker_path` arg; auto-uses trained engine when the
  checkpoint exists; `status()` exposes `is_trained` and `ranker_path`.
- `frontend/src/api.ts`: status type updated; header shows “trained InfoNCE ranker”.

### Gap 4a — Automated evaluation

**`ml/evaluate.py`** reports per protocol/mode/model: Recall@1/5/10, MRR, p50/p95 latency.
Protocols: `full` (query included) and `ctx-only` (current sentence hidden).
Writes `models/eval_results.json` + `docs/eval_results.md`.

| Protocol | Mode | Model | R@1 | R@5 | R@10 | MRR | p50 ms | p95 ms |
|---|---|---|---|---|---|---|---|---|
| full | text_only | baseline | 1.000 | 1.000 | 1.000 | 1.000 | 29.8 | 42.0 |
| full | text_only | trained | 1.000 | 1.000 | 1.000 | 1.000 | 32.0 | 42.1 |
| full | context_aware | baseline | 1.000 | 1.000 | 1.000 | 1.000 | 66.2 | 88.1 |
| full | context_aware | trained | 0.989 | 1.000 | 1.000 | 0.994 | 68.2 | 91.0 |
| ctx-only | context_aware | baseline | 0.044 | 0.256 | 0.422 | 0.159 | 31.2 | 39.5 |
| ctx-only | context_aware | trained | 0.044 | 0.256 | 0.300 | 0.147 | 32.6 | 44.4 |

**Interpretation (no spin):** full-protocol retrieval is trivial because the exact caption
is indexed. The hard ablation hides the current sentence; there the trained model matches
baseline R@1 (0.044) but is worse at R@10/MRR. This is a negative result and is reported as such.

### Gap 3 — Browser-verified frame-level sync

**`PoseAnimationViewer.tsx`**

- New `driven` prop (video-controlled mode), `data-testid="pose-canvas"`,
  `data-frame` / `data-total` on the canvas, visible frame counter.
- Skips `(0,0)` missing keypoints instead of drawing to the corner.
- Fixed an initial-render bug: a parent re-render reset the frame to 0. Root cause was an
  unstable `onFrameChange` callback in the effect deps; fixed with `renderRef` and
  `useCallback` so the reset runs only when `poses/width/height` change.

**`StudioPage.tsx`**

- `requestVideoFrameCallback` (rAF fallback) maps `video.currentTime` to
  `floor((t - start) / duration * nFrames)`.
- Handles `seeking`, `seeked`, `ratechange`, `play`, `pause`, `ended`, `loadedmetadata`,
  `timeupdate`; `playbackRate` follows the speed selector; video `loop`.
- `togglePlay`, `restartPlayback`, `scrubTo`; transcript sentence click seeks the video.
- `data-testid`s added for layout, video, inputs, generate button, sentences, and the
  active sentence (`data-start/end/frames`).
- Pose viewer aspect ratio fixed from `360×360` to **360×203** (16:9) after screenshots
  showed a vertically stretched skeleton; comparison viewers `170×96`.
- “SYNTHETIC DATA.” banner and “trained InfoNCE ranker” badge wired to API status.

**Playwright**

- Installed `@playwright/test` 1.65.1; `frontend/playwright.config.ts` uses
  `channel: 'msedge'` because the installed Playwright Chromium lacks H.264.
- Two `webServer` entries: uvicorn (from project root) + Vite.
- Fixtures: `sample.mp4` (real How2Sign clip, 902 KB) and `sample.srt` (3 real captions:
  0–2.5 s, 2.5–5 s, 5–8 s).
- Test covers: layout boxes, upload, generate, playback frame sync ±1, seek ±1, rate change,
  sentence-click seek, absence of synthetic banner, and a mocked-status test that asserts
  the synthetic banner is visible.

**Failures hit and fixed**

1. `__dirname is not defined in ES module scope` → replaced with
   `path.dirname(fileURLToPath(import.meta.url))`.
2. Frame mismatch (`frame=0`, expected `10`, total `32`) → the render effect reset to frame 0
   on every parent re-render; fixed with `renderRef` + stable `useCallback(handleFrameChange)`.

**Final E2E:** `2 passed (21.2s)`. Screenshots:
`01-initial`, `02-transcript-loaded`, `03-after-search`, `04-during-playback`,
`05-after-sentence-click`, `06-synthetic-banner`.

### Gap 4b — Human evaluation kit (PENDING)

- `eval/PROTOCOL.md`: blinded protocol, ≥3 Deaf ASL signers, 20 stratified sentences,
  both retrieval modes, randomized order, 1–5 scales for sign accuracy / naturalness /
  sync quality, plus free-text error notes.
- `eval/rating_form.csv`: header + two `EXAMPLE_` rows (ignored by the aggregator).
- `eval/aggregate_ratings.py`: validates scores, skips examples/invalid rows, prints
  mean/SD/n per mode; prints **PENDING HUMAN EVALUATION** when no valid rows exist.
- `tests/test_ratings.py`: verifies aggregation, example/invalid skipping, and pending mode.

**No signer ratings were collected.** This remains the only unmet acceptance item and is
ethically required before claiming ASL correctness.

### Verification runs (final state)

```text
python -m pytest tests -q      → 31 passed in 11.36s
npm run build                  → built in 1.19s, 0 TypeScript errors
npx playwright test            → 2 passed (21.2s)
python eval/aggregate_ratings.py --input eval/rating_form.csv --output eval/summary.csv
                               → PENDING HUMAN EVALUATION
```

---

## Root causes worth remembering

1. **Google Drive quota** blocks the official keypoints; the HF mirror is the practical path.
2. **Playwright Chromium has no H.264** → videos render black; use installed Edge (`channel:'msedge'`).
3. **Effect dependency instability** reset animation frames mid-playback; `renderRef` + memoized
   callback fixed it. This is why the frame-sync assertion initially failed.
4. **Duplicate pickle key** in the index save path caused the restored index to differ from the
   live one; removed.
5. **Aspect ratio distortion**: normalized coordinates assume 16:9; drawing into a square canvas
   stretched the skeleton. Canvas is now 16:9.

---

## Decisions and tradeoffs

- Used the **validation split (450 clips)** to prove the full real-data path quickly, instead of
  downloading/processing the ~22 GB train split. The pipeline accepts the train split unchanged.
- Committed the **ranker checkpoint (1.2 MB)** and loss log as training evidence; did not commit
  the **76 MB index** (regenerable; gitignored).
- Kept synthetic data behind explicit flags and a visible banner; never used for reported metrics.
- Reported the **negative** trained-vs-baseline ablation instead of selecting a flattering metric.
- MMR (not accuracy) is used as the ranking summary; human ASL correctness is explicitly out of scope.

## Blockers / open items

- Human ASL-signer evaluation (kit ready; no signers in this environment).
- Train split not processed (only val split).
- Trained ranker does not improve the context-only ablation.
- Full-protocol metrics are near-trivial by construction; treat them as sanity checks only.

## Reproduction

```bash
# data
python -c "from huggingface_hub import hf_hub_download; hf_hub_download(repo_id='Saintbook/how2sign_keypoints', repo_type='dataset', filename='val_2D_keypoints.tar.gz', local_dir='C:/how2sign')"
tar -xzf C:/how2sign/val_2D_keypoints.tar.gz -C C:/how2sign/extracted

# convert + train + index
python ml/build_real_index.py --openpose-json C:/how2sign/extracted/openpose_output/json \
    --csv C:/how2sign/how2sign_realigned_val.csv \
    --out-dir data/real --index models/retrieval_index.pkl \
    --ranker models/ranker.pt --limit 450 --epochs 40

# evaluate
python ml/evaluate.py --data-dir data/real --index models/retrieval_index.pkl --ranker models/ranker.pt

# tests
python -m pytest tests -q
cd frontend && npm run build && npm run test:e2e
```

## Evidence index

- Data: `data/real/metadata.csv`, `data/real/DATASET_SOURCE.txt`
- Model: `models/ranker.pt`, `models/ranker_train_log.csv`
- Metrics: `docs/eval_results.md`, `models/eval_results.json`
- Browser: `frontend/tests/e2e/studio.spec.ts`, `frontend/tests/e2e/screenshots/01..06`
- Human kit: `eval/PROTOCOL.md`, `eval/rating_form.csv`, `eval/aggregate_ratings.py`

## Commit series (Day 2)

| Commit | Theme |
|---|---|
| `4983250` | chore: ignore generated indexes, pose arrays, Playwright artifacts |
| `3a43609` | feat(data): real How2Sign OpenPose conversion + one-command pipeline |
| `f3687f0` | feat(ml): InfoNCE dual-encoder ranker + training checkpoint |
| `cc85e5b` | feat(eval): recall/MRR/latency baseline-vs-trained evaluation |
| `ce08751` | feat(ui): frame-accurate sync + Playwright screenshots |
| `ef18ddb` | feat(eval): pending ASL signer protocol/form/aggregator |
| _final doc commit_ | docs: README, status, and this XL worklog |

---

## Scorecard

| Acceptance criterion | Met? | Proof |
|---|---|---|
| Real How2Sign poses on disk | Yes (450 val clips) | metadata + npy local |
| Trained by gradient descent (encoders/InfoNCE) | Yes | loss CSV 0.178; checkpoint |
| Train/val split, fixed seed, logged curves | Yes | 80/20, seed 42, CSV |
| Recall@1/5/10, MRR, latency vs baseline | Yes | table above |
| Real headless browser screenshots | Yes | 01–06 |
| Skeleton frame tracks video ±1 frame | Yes | Playwright assertions |
| Seek / pause / rate / loop handled | Yes | Playwright + code |
| Human eval protocol + form + aggregator | Yes | eval/ + tests |
| Human ratings collected | **No — PENDING** | explicit marker |
| Synthetic data impossible to mistake | Yes | banner test + explicit flags |
