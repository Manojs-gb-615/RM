# SignLingo AI Mini

**Lightweight Context-Aware ASL Pose Retrieval from Short Educational Videos.**

> Research prototype. This system **retrieves** the closest ASL pose clip from a small
> dataset — it does **not** generate novel ASL. Retrieved poses have not been evaluated by
> qualified ASL signers. Unmatched sentences fall back to caption-only.

---

## What it does

1. Upload a short educational MP4 (30–60 s).
2. Upload its English transcript (`.srt` / `.vtt`).
3. For each sentence, retrieve the best-matching ASL pose clip using a lightweight
   sentence-embedding model, optionally conditioned on the previous sentence + previous pose.
4. Animate the retrieved 2D skeleton (body, arms, hands, fingers) on a canvas.
5. Play the video and animation side by side; compare **text-only** vs **context-aware** retrieval.

Retrieval is a fixed weighted cosine ranking, not a trained Transformer / VAE. Here
"training" means building the sentence-embedding index from real examples; no gradient
training is performed. Section 3 of the brief explicitly permits a simple similarity ranker.

---

## Verified status (what was actually run)

| Item | Status |
|---|---|
| Python ML pipeline (synthetic end-to-end) | ✅ runs |
| `ml/setup_data.py` (real + synthetic + template) | ✅ runs |
| Backend full flow (parse → retrieve → 274-d pose) | ✅ `retrieve_all` + `compare` tested via `TestClient` |
| Frontend build (`npm run build`) | ✅ 0 TypeScript errors |
| Test suite (`python -m pytest tests`) | ✅ 19 passed |
| Browser UI (visual video + skeleton playback) | ⚠️ build + API verified; not visually tested |
| **Real How2Sign pose data** | ❌ not present on disk (blocker) |
| Real trained index (`models/retrieval_index.pkl`) | ❌ not created (needs real poses) |
| Human / ASL-signer evaluation | ❌ not conducted |

Synthetic poses are only for pipeline testing and are clearly flagged in the UI. They are
**never** valid ASL output.

---

## Project structure

```
SignLingo-AI-Mini/
├── frontend/            React + Vite + Tailwind (port 5173)
├── backend/
│   └── main.py          FastAPI server (port 8000)
├── ml/
│   ├── dataset.py       Dataset loader + synthetic test data
│   ├── retrieval.py     Context-aware retrieval + SRT/VTT parsing
│   ├── inference.py     Inference wrapper for the backend
│   ├── train.py         Build + evaluate + save the retrieval index
│   └── setup_data.py    Build metadata.csv from How2Sign CSVs / make synthetic data
├── notebooks/
│   └── train_signlingo.ipynb   Google Colab training notebook
├── data/
│   ├── poses/           .npy pose files ([T, 274]) — not included
│   └── metadata.csv     Sample metadata (header template committed)
├── models/              retrieval_index.pkl (generated)
├── tests/               pytest suite
├── requirements.txt
└── README.md
```

---

## Prerequisites

- Python 3.10+ (tested on 3.12)
- Node.js 18+ and npm (tested on Node 22)
- `pip install -r requirements.txt`
- `cd frontend && npm install`

---

## Dataset setup

The standard How2Sign release does **not** ship pose files. Each researcher must extract
2D MediaPipe body+hand keypoints and save them as `[T, 274]` `.npy` arrays named by
`sentence_id` (e.g. `--7E8rIUe34_19-6-rgb_front.npy`).

1. Register at https://how2sign.github.io/ and download the realigned CSVs + pose files.
2. Put pose `.npy` files in `data/poses/`.
3. Build the manifest:

```bash
python ml/setup_data.py real \
    --csv /path/to/how2sign_realigned_train.csv \
    --pose-dir data/poses \
    --output data/metadata.csv
```

`metadata.csv` columns:
```
sample_id,sentence,video_id,narrative_id,start_time,end_time,prev_sample_id,frames
```

`setup_data.py` matches CSV rows to pose files by sentence id/name, then by video id +
timestamp range, validates every pose is `[T, 274]`, and derives `prev_sample_id` within
each video. Rows without a valid pose are skipped and reported.

> No pose data? See the blocker note below. Do **not** invent ASL poses.

---

## Training / building the index

Real data (CPU-friendly, no GPU needed after the small embedding model is cached):

```bash
python ml/train.py --data-dir data --pose-dir data/poses --eval
```

Outputs `models/retrieval_index.pkl` (and `models/eval_results.json` when `--eval`).

### Synthetic smoke test (testing only)

```bash
python ml/setup_data.py synthetic --output-dir data/synthetic --n 30
python ml/train.py --data-dir data/synthetic --pose-dir data/synthetic/poses \
    --output data/synthetic/retrieval_index.pkl --eval
```

The synthetic dataset writes a `SYNTHETIC_DATASET.txt` marker and the backend exposes
`is_synthetic: true`, which shows a red warning banner in the UI.

### Google Colab

Open `notebooks/train_signlingo.ipynb`, set `USE_SYNTHETIC = False` and your paths,
run all cells, then download `retrieval_index.pkl` into `models/`.

---

## Evaluation honesty

`ml/train.py --eval` reports two clearly separated numbers:

- **`self_top1`** — with the full index (query included), is the top result the query
  itself? This is a pipeline sanity check (~1.0 expected), **not** generalisation accuracy.
- **`heldout_mean_score`** — index built on a train split, mean top-1 cosine score on
  held-out samples. A similarity proxy, **not** accuracy.

No accuracy percentage is claimed. ASL correctness requires evaluation by qualified signers.

---

## Run the demo

Open two terminals.

**Backend** (run from `SignLingo-AI-Mini/`):

```bash
uvicorn backend.main:app --reload --port 8000
```

**Frontend:**

```bash
cd frontend
npm run dev
```

Open http://localhost:5173. The Vite dev server proxies `/api` to `http://localhost:8000`.

Optional environment variables for the backend:

```bash
DATA_DIR=data                 # folder containing metadata.csv + poses/
POSE_DIR=data/poses           # optional explicit pose folder
INDEX_PATH=models/retrieval_index.pkl
```

If no index exists but pose files do, the backend builds one automatically at startup.

---

## API

| Method | Path | Description |
|---|---|---|
| GET | `/api/health` | health check |
| GET | `/api/status` | ready state, counts, `is_synthetic` |
| GET | `/api/dataset/samples` | dataset samples |
| GET | `/api/dataset/samples/{id}/pose` | pose array for a sample |
| POST | `/api/retrieve` | retrieve best pose for a sentence |
| POST | `/api/compare` | text-only vs context-aware |
| POST | `/api/transcript/parse` | parse `.srt` / `.vtt` |
| POST | `/api/transcript/retrieve_all` | retrieve poses for every segment |

---

## Tests

```bash
python -m pytest tests -q
```

Covers dataset shapes, text-only and context-aware retrieval, index save/load (including the
synthetic flag), evaluation bounds, inference end-to-end, and backend endpoints.

---

## Known blocker

**Real How2Sign pose `.npy` files are not available in this environment.** Until they are
extracted and placed in `data/poses/` (and `metadata.csv` built with `ml/setup_data.py real`),
the app runs only on synthetic test data, which is not ASL.

---

## Limitations

- Retrieval only — no novel ASL generation.
- Small dataset (20–50 clips for the prototype).
- Poses are MediaPipe approximations, not manually annotated ground truth.
- Context awareness depends on dataset diversity.
- Not suitable for accessibility use without human evaluation.

## Citations

- How2Sign: Duarte et al., CVPR 2021 — https://how2sign.github.io/
- Sentence Transformers: Reimers & Gurevych, EMNLP 2019 — https://www.sbert.net/
