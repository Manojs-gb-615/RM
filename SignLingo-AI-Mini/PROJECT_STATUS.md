# PROJECT_STATUS.md — SignLingo AI Mini

Updated: 2026-11-20

---

## Component status

| Component | Status | Notes |
|---|---|---|
| `ml/dataset.py` | ✅ DONE | Loader + synthetic test data + synthetic marker detection |
| `ml/retrieval.py` | ✅ DONE | text_only + context_aware, SRT/VTT parsing, model cache, save/load |
| `ml/inference.py` | ✅ DONE | Backend wrapper, configurable index path, synthetic flag |
| `ml/train.py` | ✅ DONE | Builds index, honest split evaluation |
| `ml/setup_data.py` | ✅ DONE | `real` / `synthetic` / `template` modes |
| `backend/main.py` | ✅ DONE | 8 endpoints, lifespan startup, pydantic v2 validators |
| `frontend/` | ✅ DONE | React + Tailwind single page, canvas skeleton viewer |
| Frontend build | ✅ PASSED | `npm run build` → 0 errors |
| Colab notebook | ✅ DONE | Honest evaluation; `USE_SYNTHETIC` defaults to False |
| Tests | ✅ PASSED | `python -m pytest tests -q` → 15 passed |
| Backend end-to-end | ✅ VERIFIED | parse → retrieve → 274-d pose (synthetic data) |

---

## Data status

| Item | Status |
|---|---|
| How2Sign CSVs | Available in `Downloads` (outside this repo) |
| How2Sign pose `.npy` files | ❌ NOT AVAILABLE locally |
| `data/metadata.csv` | ⚠️ header-only template committed |
| Real retrieval index | ❌ NOT BUILT (no pose data) |
| Synthetic test data | Generated on demand + clearly flagged |

---

## Verified commands

```bash
# unit tests
python -m pytest tests -q                      # 15 passed

# synthetic pipeline (testing only)
python ml/setup_data.py synthetic --output-dir data/synthetic --n 30
python ml/train.py --data-dir data/synthetic --pose-dir data/synthetic/poses \
    --output data/synthetic/retrieval_index.pkl --eval

# frontend
cd frontend && npm run build                    # passes
```

---

## Startup commands

```bash
# 1. install
pip install -r requirements.txt
cd frontend && npm install && cd ..

# 2. build index from real data (once poses exist)
python ml/setup_data.py real --csv <how2sign.csv> --pose-dir data/poses --output data/metadata.csv
python ml/train.py --data-dir data --pose-dir data/poses --eval

# 3. run (from SignLingo-AI-Mini/)
uvicorn backend.main:app --reload --port 8000
cd frontend && npm run dev
# open http://localhost:5173
```

---

## What is NOT done

- Real How2Sign pose data download/extraction (blocker; requires registration).
- Real index build (blocked by pose data).
- Human evaluation by qualified ASL signers (not conducted; ethically required before real use).

## Honest limitations

- Retrieval only, not generation.
- Retrieval quality untested on real data.
- No certified ASL accuracy evaluation.
- Prototype only — not for accessibility use without expert evaluation.
- Synthetic poses are random sinusoids and are flagged as such; never ASL.
