# SignLingo AI Mini

**Lightweight context-aware ASL pose retrieval from short educational videos.**

> Research prototype. This system **retrieves** the closest ASL pose clip from a
> dataset — it does **not** generate novel ASL. Retrieved poses have not been
> evaluated by qualified ASL signers. Unmatched sentences fall back to caption-only.

---

## What it does

1. Upload a short educational MP4.
2. Upload its English transcript (`.srt` / `.vtt`).
3. For each sentence, retrieve the best-matching ASL pose clip using a small
   text encoder, optionally conditioned on the previous sentence + previous pose.
4. Animate the retrieved 2D skeleton (body, arms, hands, fingers) on a canvas.
5. Play the video and animation side by side; compare **text-only** vs
   **context-aware** retrieval.

Retrieval uses a **trained dual encoder** (small text projection + pose tower,
trained with InfoNCE on real sentence/clip pairs) or the fixed weighted cosine
**baseline** for comparison. No Transformer or VAE is trained.

---

## Verified status (commands actually run)

| Item | Status |
|---|---|
| Real How2Sign 2D keypoints | ✅ 450 val clips converted ([T,274]) |
| Real retrieval index | ✅ `models/retrieval_index.pkl` (450 clips, local) |
| Trained InfoNCE ranker | ✅ `models/ranker.pt` + loss log |
| Backend full flow | ✅ retrieve / retrieve_all / compare via TestClient |
| Frontend build | ✅ `npm run build`, 0 TypeScript errors |
| Headless browser E2E | ✅ Playwright + Edge, 2 tests, 5 screenshots |
| Frame-level sync | ✅ frame tracks `video.currentTime` within ±1 frame |
| pytest suite | ✅ 31 passed |
| ASL-signer evaluation | ⏳ **PENDING** (kit ready in `eval/`) |

Synthetic poses remain available only through an explicit command
(`ml/setup_data.py synthetic`, `ml/train.py --synthetic`) and are flagged with a
red **SYNTHETIC DATA** banner.

---

## Project structure

```
SignLingo-AI-Mini/
├── frontend/                 React + Vite + Tailwind (port 5173)
│   └── tests/e2e/            Playwright test + screenshots + fixtures
├── backend/main.py           FastAPI server (port 8000)
├── ml/
│   ├── dataset.py            Dataset loader
│   ├── retrieval.py          Retrieval engine (trained + baseline)
│   ├── ranker.py             Dual encoder, InfoNCE training, checkpoint
│   ├── inference.py          Backend wrapper
│   ├── convert_openpose.py   OpenPose JSON -> [T,274] .npy + metadata
│   ├── build_real_index.py   One-command real pipeline
│   ├── train.py              Build index (optional training)
│   ├── evaluate.py           Recall@1/5/10, MRR, latency
│   └── setup_data.py         Real/synthetic/template data helpers
├── eval/                     Human evaluation kit (PENDING)
├── data/real/                Real converted metadata (poses gitignored)
├── models/                   ranker.pt, index, logs, eval JSON (gitignored)
├── tests/                    pytest suite
└── requirements.txt
```

---

## Prerequisites

- Python 3.10+ (tested 3.12), Node.js 18+ (tested 22)
- `pip install -r requirements.txt`
- `cd frontend && npm install`

---

## Real How2Sign data: exact steps

Official source: <https://how2sign.github.io/> — "B-F-H 2D Keypoints clips
(frontal view)". The official Google Drive links are quota-limited; a public HF
mirror of the same archives is used here.

```bash
# 1. Download the validation keypoints (1.2 GB) + the val CSV
python -c "from huggingface_hub import hf_hub_download; \
hf_hub_download(repo_id='Saintbook/how2sign_keypoints', repo_type='dataset', \
filename='val_2D_keypoints.tar.gz', local_dir='C:/how2sign')"

# 2. Extract
mkdir C:/how2sign/extracted
tar -xzf C:/how2sign/val_2D_keypoints.tar.gz -C C:/how2sign/extracted

# 3. One command: convert 450 clips -> train ranker -> build real index
python ml/build_real_index.py \
    --openpose-json C:/how2sign/extracted/openpose_output/json \
    --csv C:/how2sign/how2sign_realigned_val.csv \
    --out-dir data/real --index models/retrieval_index.pkl \
    --ranker models/ranker.pt --limit 450 --epochs 40
```

Already have `[T, 274]` `.npy` files? Use `python ml/setup_data.py real --csv ...
--pose-dir ... --output data/real/metadata.csv`, then
`python ml/build_real_index.py --data-dir data/real`.

The converter (`ml/convert_openpose.py`) packs OpenPose BODY_25 + both hands +
70 face landmarks into 137 keypoints × 2 coords, normalizes by frame size, and
writes `data/real/poses/<SENTENCE_NAME>.npy` + `data/real/metadata.csv`.
It resumes if interrupted (`--workers 8` for parallel conversion).

---

## Trained ranking model

`ml/ranker.py` trains a **dual encoder** by gradient descent:

- Text tower: frozen `all-MiniLM-L6-v2` 384-d → MLP → 128-d (L2-normalized).
- Pose tower: 548-d pooled keypoints (mean + std) → MLP → 128-d.
- Loss: symmetric **InfoNCE** with learnable temperature.
- 80/20 train/val split, fixed seed 42, AdamW, per-epoch loss CSV.

```bash
python ml/train.py --data-dir data/real --ranker models/ranker.pt \
    --train-ranker --epochs 40 --output models/retrieval_index.pkl
```

Artifacts: `models/ranker.pt`, `models/ranker_train_log.csv`,
`models/retrieval_index.pkl`. The index stores trained clip embeddings and
loads the checkpoint at inference.

---

## Evaluation (real val split, n=450, val=90)

```bash
python ml/evaluate.py --data-dir data/real \
    --index models/retrieval_index.pkl --ranker models/ranker.pt
```

| Protocol | Mode | Model | R@1 | R@5 | R@10 | MRR | p50 ms | p95 ms |
|---|---|---|---|---|---|---|---|---|
| full | text_only | baseline | 1.000 | 1.000 | 1.000 | 1.000 | 29.8 | 42.0 |
| full | text_only | trained | 1.000 | 1.000 | 1.000 | 1.000 | 32.0 | 42.1 |
| full | context_aware | baseline | 1.000 | 1.000 | 1.000 | 1.000 | 66.2 | 88.1 |
| full | context_aware | trained | 0.989 | 1.000 | 1.000 | 0.994 | 68.2 | 91.0 |
| ctx-only | context_aware | baseline | 0.044 | 0.256 | 0.422 | 0.159 | 31.2 | 39.5 |
| ctx-only | context_aware | trained | 0.044 | 0.256 | 0.300 | 0.147 | 32.6 | 44.4 |

**Honest reading:** the `full` protocol includes the query caption in the index,
so exact-caption retrieval is near-trivial (≈1.0 for both). The non-trivial
`ctx-only` ablation (current sentence hidden) is hard, and the trained ranker did
**not** beat the fixed baseline there. These are retrieval metrics, not ASL
correctness. Raw numbers: `models/eval_results.json`, `docs/eval_results.md`.

---

## Run the demo

```bash
# terminal 1 (project root)
uvicorn backend.main:app --reload --port 8000

# terminal 2
cd frontend && npm run dev
```

Open http://localhost:5173. The backend loads `models/retrieval_index.pkl`
(real data) by default. Env overrides: `DATA_DIR`, `POSE_DIR`, `INDEX_PATH`,
`RANKER_PATH`.

Synthetic demo (testing only, red banner):

```bash
python ml/setup_data.py synthetic --output-dir data/synthetic --n 30
python ml/train.py --synthetic --train-ranker --epochs 10
$env:INDEX_PATH="models/synthetic_index.pkl"; uvicorn backend.main:app --port 8000
```

---

## Tests and headless browser verification

```bash
python -m pytest tests -q          # 31 passed
cd frontend && npm run build       # 0 errors
npm run test:e2e                   # Playwright + installed Edge (channel: msedge)
```

Playwright asserts the LEFT/RIGHT/BOTTOM layout, uploads a real How2Sign clip +
SRT, generates poses, checks that the skeleton frame equals
`floor(((currentTime - start) / duration) * nFrames)` within ±1 during playback
and after seeking, changes playback rate, clicks a transcript sentence (video
seeks), and verifies the synthetic banner via a mocked status response.

Screenshots (in `frontend/tests/e2e/screenshots/`):
`01-initial.png`, `02-transcript-loaded.png`, `03-after-search.png`,
`04-during-playback.png`, `05-after-sentence-click.png`, `06-synthetic-banner.png`.

---

## Human evaluation kit (PENDING)

- `eval/PROTOCOL.md` — blinded protocol for ≥3 ASL signers.
- `eval/rating_form.csv` — sign accuracy / naturalness / sync quality (1–5) + comments.
- `eval/aggregate_ratings.py` — mean, SD, n per mode; prints `PENDING HUMAN EVALUATION`
  while empty.

No ASL-accuracy claim is made until this is completed by qualified signers.

---

## Limitations

- Retrieval only; no novel ASL generation.
- 450 validation clips; captions are mostly unique, making full-protocol retrieval trivial.
- Poses are automatic OpenPose estimates, not manually verified ground truth.
- Trained ranker did not improve the hard context-only ablation.
- Not suitable for accessibility use without human signer evaluation.

## Citations

- How2Sign: Duarte et al., CVPR 2021 — https://how2sign.github.io/ (CC BY-NC 4.0)
- Sentence Transformers: Reimers & Gurevych, EMNLP 2019 — https://www.sbert.net/
