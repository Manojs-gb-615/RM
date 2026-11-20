# PROJECT_STATUS.md — SignLingo AI Mini

Generated: 2026-11-20

---

## Component Status

| Component | Status | Notes |
|---|---|---|
| Project structure | ✅ DONE | All dirs created |
| `ml/dataset.py` | ✅ DONE | Loader + synthetic data |
| `ml/retrieval.py` | ✅ DONE | text_only + context_aware, .srt/.vtt parser |
| `ml/inference.py` | ✅ DONE | FastAPI wrapper |
| `ml/train.py` | ✅ DONE | Builds retrieval index |
| `backend/main.py` | ✅ DONE | 8 API endpoints |
| `frontend/` | ✅ DONE | React + Tailwind, TypeScript build passes |
| Frontend build | ✅ DONE | `npm run build` → 0 errors |
| Colab notebook | ✅ DONE | `notebooks/train_signlingo.ipynb` |
| README.md | ✅ DONE | Full setup instructions |

---

## Data Status

| Item | Status |
|---|---|
| How2Sign CSVs | Available (Downloads folder) |
| How2Sign pose .npy files | ❌ NOT AVAILABLE locally |
| `data/metadata.csv` | ❌ NOT CREATED (requires pose files) |
| Retrieval index (`models/retrieval_index.pkl`) | ❌ NOT TRAINED (no pose data) |
| Synthetic index (for testing) | Can be generated via `python ml/train.py --synthetic` |

---

## Test Results

### Python import test
- Status: **PENDING** (requires numpy/sentence-transformers install)
- Command: `python -c "from ml.dataset import make_synthetic_dataset; print('OK')"`

### Frontend build
- Status: ✅ **PASSED**
- Command: `npm run build` (from `frontend/`)
- Output: 0 TypeScript errors, bundle built successfully in 3.00s

### Backend import test
- Status: **PENDING** (requires fastapi/uvicorn install)

---

## Startup Commands (once dependencies installed)

```bash
# 1. Install Python deps
pip install -r requirements.txt

# 2. Build retrieval index (synthetic for testing, real data for production)
python ml/train.py --synthetic --eval

# 3. Start backend (from project root)
uvicorn backend.main:app --reload --port 8000

# 4. Start frontend (in another terminal)
cd frontend && npm run dev

# 5. Open browser
# http://localhost:5173
```

---

## What is NOT done

- Real How2Sign pose data download (requires registration at https://how2sign.github.io/)
- Real model training (blocked by pose data)
- Human evaluation by qualified ASL signers (NOT CONDUCTED, ethically mandatory before real use)
- Backend API end-to-end smoke test (pending pip install)

---

## Honest Limitations

- System performs pose **retrieval** only, not generation
- Retrieval quality untested on real data
- No certified ASL accuracy evaluation performed
- Prototype only — not suitable for accessibility use without expert evaluation
