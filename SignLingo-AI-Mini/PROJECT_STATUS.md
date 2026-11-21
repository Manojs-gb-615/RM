# PROJECT_STATUS.md — SignLingo AI Mini

Updated: 2026-11-21

---

## Component status

| Component | Status | Notes |
|---|---|---|
| `ml/convert_openpose.py` | ✅ DONE | OpenPose JSON → [T,274] .npy + metadata; resume + parallel |
| `ml/build_real_index.py` | ✅ DONE | One command: convert → train → index |
| `ml/ranker.py` | ✅ DONE | Dual encoder, symmetric InfoNCE, checkpoint + loss log |
| `ml/retrieval.py` | ✅ DONE | Trained + baseline rankers, vectorized scoring, save/load |
| `ml/evaluate.py` | ✅ DONE | Recall@1/5/10, MRR, p50/p95 latency, baseline vs trained |
| `ml/dataset.py` | ✅ DONE | Loader + synthetic marker detection |
| `ml/inference.py` | ✅ DONE | Loads trained index/checkpoint; reports `is_trained` |
| `ml/train.py` | ✅ DONE | `--train-ranker`, `--synthetic` gating, baseline fallback |
| `ml/setup_data.py` | ✅ DONE | `real` / `synthetic` / `template` modes |
| `backend/main.py` | ✅ DONE | 8 endpoints, lifespan startup |
| `frontend/` | ✅ DONE | LEFT video / RIGHT skeleton / BOTTOM transcript + controls |
| Frame-level sync | ✅ DONE | `requestVideoFrameCallback` (rAF fallback); seek/rate/pause/loop |
| Colab notebook | ✅ DONE | `USE_SYNTHETIC` defaults to False |
| Tests | ✅ PASSED | `python -m pytest tests -q` → 31 passed |
| Backend end-to-end | ✅ VERIFIED | retrieve, retrieve_all, compare (real index + synthetic tests) |
| Playwright E2E | ✅ PASSED | 2 tests, screenshots 01–06 |
| ASL-signer evaluation | ⏳ PENDING | Kit in `eval/` |

---

## Data status

| Item | Status |
|---|---|
| How2Sign CSVs | ✅ in `Downloads` (outside repo) |
| How2Sign OpenPose keypoints | ✅ val archive downloaded + extracted (1,739 clips) |
| Converted real poses | ✅ 450 clips in `data/real/poses/` (gitignored) |
| `data/real/metadata.csv` | ✅ 450 rows committed |
| Real retrieval index | ✅ `models/retrieval_index.pkl` (gitignored, local) |
| Trained ranker | ✅ `models/ranker.pt` (gitignored) + loss log |
| Synthetic test data | Explicit `--synthetic` only, flagged in UI |

---

## Verified commands

```bash
# convert + train + build (450 real val clips)
python ml/build_real_index.py --data-dir data/real --epochs 40

# evaluation
python ml/evaluate.py --data-dir data/real \
    --index models/retrieval_index.pkl --ranker models/ranker.pt

# tests
python -m pytest tests -q                      # 31 passed
cd frontend && npm run build                    # 0 errors
npm run test:e2e                                # 2 passed

# human ratings (pending)
python eval/aggregate_ratings.py --input eval/rating_form.csv --output eval/summary.csv
```

---

## Evaluation result (honest)

Full-protocol retrieval is near-trivial because the query caption is in the index
(baseline and trained both ≈1.0). In the non-trivial context-only ablation, the
trained ranker did **not** beat the fixed baseline (R@1 0.044 vs 0.044; R@10 0.300
vs 0.422). See `docs/eval_results.md`.

## What is NOT done

- Human evaluation by qualified ASL signers (kit ready; no ratings yet).
- Trained ranker does not improve the hard context-only task.
- Only the validation split (450 clips) is used; train split (31k clips) not processed.

## Honest limitations

- Retrieval only, not generation.
- Poses are automatic OpenPose estimates, not verified ASL ground truth.
- Captions are mostly unique, making standard retrieval metrics uninformative.
- Prototype only — not for accessibility use without expert evaluation.
- Synthetic poses are random sinusoids; never presented as ASL.
