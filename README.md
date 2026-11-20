# ContextSign AI — Context-Aware ASL Pose Generation

> **Research prototype.** No training results exist on real data. All executed tests use synthetic data only.
> See [FINAL_AUDIT.md](FINAL_AUDIT.md) for a full honest audit of project status.

---

## Table of Contents
1. [Project Description](#project-description)
2. [Research Motivation](#research-motivation)
3. [Architecture Overview](#architecture-overview)
4. [Dataset](#dataset)
5. [Experiments](#experiments)
6. [Setup Instructions](#setup-instructions)
7. [Quick Start](#quick-start)
8. [Project Structure](#project-structure)
9. [Limitations Disclaimer](#limitations-disclaimer)
10. [Citation](#citation)

---

## Project Description

**ContextSign AI** is a research system for generating American Sign Language (ASL) pose sequences from English text, conditioned on *discourse context* — the prior ASL sentences that precede the current one in a conversation or lecture.

The key research question is:

> *Does providing the model with previously generated ASL pose sequences improve the quality, coherence, and temporal consistency of newly generated ASL poses?*

The system uses a **context-aware Conditional Variational Autoencoder (CVAE)** architecture combining:

- A **DistilBERT text encoder** to represent the English sentence to be signed
- A **Transformer-based pose context encoder** to encode prior ASL pose sequences
- A **Cross-Attention Fusion** module to integrate text and context
- A **CVAE** for learning a structured latent distribution over pose sequences
- A **Transformer decoder** to generate the full pose sequence frame by frame

---

## Research Motivation

Sign language interpreting typically occurs within discourse — multi-sentence conversations, lectures, and narratives. However, most ASL generation models treat each sentence in isolation, ignoring:

1. **Referent tracking** — ASL uses spatial locations to track entities across sentences. A prior sentence may establish that "JOHN" occupies a particular signing space; subsequent sentences must respect this.
2. **Temporal consistency** — Transitions between sentences should be physically plausible (no sudden joint discontinuities).
3. **Discourse coherence** — Signing style (speed, signing space utilisation) is influenced by genre and conversational history.

ContextSign AI is the first systematic study of how varying the **context window size** (0, 1, or 2–3 prior sentences) affects ASL pose generation quality.

---

## Architecture Overview

```
English Text
    │
    ▼
[DistilBERT Tokenizer + Encoder]  ──►  text_feat  [B, 256]
    
Prior ASL Poses (window × T × 274)
    │
    ▼
[Pose Context Encoder]            ──►  ctx_feat   [B, 256]
    │
    ▼
[CrossAttentionFusion]            ──►  fused_feat [B, 256]
    │   (text as Query, context as Key/Value)
    ▼
[CVAE Encoder]  (at training time: also takes target pose as input)
    │
    ▼  z  [B, 128]
    │
    ▼
[Transformer Decoder + causal mask]
    │
    ▼
Predicted Pose Sequence  [B, T, 274]
```

For the full architecture with tensor shapes and experiment variants, see [ARCHITECTURE.md](ARCHITECTURE.md).

---

## Dataset

**How2Sign** (Duarte et al., 2021) — a large-scale multi-view American Sign Language production dataset.

| Split | Sentences | Duration |
|-------|-----------|----------|
| Train | ~31,128   | ~79h     |
| Val   | ~1,741    | ~4h      |
| Test  | ~2,322    | ~6h      |

- **Source**: Educational "How To" instructional videos with English captions
- **Registration required**: [https://how2sign.github.io/](https://how2sign.github.io/)
- **Pose format used**: MediaPipe Holistic — 137 keypoints × 2 (x,y) = **274 dimensions** per frame

> ⚠️ **Important**: The standard How2Sign release includes video files and CSV annotations only. Pose `.npy` files must be extracted by the researcher using MediaPipe. See [DATASET_SETUP.md](DATASET_SETUP.md) for details.

**CSVs available locally** at:
```
c:\Users\manoj\Downloads\how2sign_realigned_train.csv
c:\Users\manoj\Downloads\how2sign_realigned_val.csv
c:\Users\manoj\Downloads\how2sign_realigned_test.csv
```

---

## Experiments

| ID | Name | Description | Status |
|----|------|-------------|--------|
| E1 | Text-only baseline | No context; text → poses | Implemented, synthetic-tested |
| E2 | Context-aware (1 prev) | Text + 1 prior ASL seq | Implemented, synthetic-tested |
| E3 | Longer context (2–3 prev) | Text + 2–3 prior ASL seqs | Implemented, synthetic-tested |
| E4 | Text-history control | Text + prior texts (no prior poses) | Implemented, synthetic-tested |
| E5 | Zero-context probe | E2/E3 evaluated with context zeroed | Implemented, synthetic-tested |
| E6 | Shuffled-context probe | E2/E3 evaluated with wrong-video context | Implemented, synthetic-tested |
| E7 | Fusion ablation | CrossAttention vs Concatenation | Implemented, synthetic-tested |
| E8 | Variational ablation | E2 without CVAE | Implemented, synthetic-tested |
| E9 | Audio extension | FUTURE WORK — no aligned audio | Not implemented |

> **None of these experiments have been trained on real How2Sign pose data.**
> See [EXPERIMENTS.md](EXPERIMENTS.md) for full details and hypotheses.

---

## Setup Instructions

### Prerequisites

- Python 3.12
- NVIDIA GPU with CUDA (tested on RTX 2050 4GB)
- PyTorch 2.13 with CUDA
- Transformers 5.16
- How2Sign pose `.npy` files (see [DATASET_SETUP.md](DATASET_SETUP.md))

### Virtual Environment

The project virtual environment is located at:
```
c:\Users\manoj\Downloads\context_aware_howsign_project\.venv\
```

Activate it in PowerShell:
```powershell
c:\Users\manoj\Downloads\context_aware_howsign_project\.venv\Scripts\Activate.ps1
```

Or use the Python interpreter directly:
```
c:\Users\manoj\Downloads\context_aware_howsign_project\.venv\Scripts\python.exe
```

### Install Dependencies

```powershell
# From project root
c:\Users\manoj\Downloads\context_aware_howsign_project\.venv\Scripts\pip.exe install -r requirements.txt
```

**Core dependencies:**
```
torch>=2.0
transformers>=4.30
numpy
scipy
fastapi
uvicorn
pydantic
```

---

## Quick Start

> **Replace** `PYTHON` with the full path:
> `c:\Users\manoj\Downloads\context_aware_howsign_project\.venv\Scripts\python.exe`

### 1. Smoke Test (Synthetic Data — Works Now)

Runs a full forward/backward pass on randomly generated synthetic pose data to verify model components are correct:

```powershell
# From Research directory
$PYTHON = "c:\Users\manoj\Downloads\context_aware_howsign_project\.venv\Scripts\python.exe"

& $PYTHON -m pytest src/ -v -k "smoke"
# Or run a specific smoke test script if present:
& $PYTHON scripts/smoke_test.py
```

### 2. Train on Real Data (Requires Pose .npy Files)

```powershell
$PYTHON = "c:\Users\manoj\Downloads\context_aware_howsign_project\.venv\Scripts\python.exe"

# E1: Text-only baseline
& $PYTHON train.py --experiment E1 --data_dir path/to/pose_npy_dir --epochs 50

# E2: Context-aware (1 prior)
& $PYTHON train.py --experiment E2 --data_dir path/to/pose_npy_dir --context_window 1 --epochs 50

# E3: Longer context (2-3 prior)
& $PYTHON train.py --experiment E3 --data_dir path/to/pose_npy_dir --context_window 3 --epochs 50
```

> ⚠️ **This requires pose `.npy` files which are NOT included in the standard How2Sign release.**
> See [DATASET_SETUP.md](DATASET_SETUP.md) for extraction instructions.

### 3. Start the Backend API Server

```powershell
$PYTHON = "c:\Users\manoj\Downloads\context_aware_howsign_project\.venv\Scripts\python.exe"

# From Research directory
& $PYTHON -m uvicorn backend.main:app --host 0.0.0.0 --port 8000 --reload
```

The API will be available at: `http://localhost:8000`
Interactive docs: `http://localhost:8000/docs`

> ⚠️ Endpoints requiring a trained model will return `503 Model Not Ready` until checkpoints are loaded.

### 4. Start the Frontend

```powershell
# If using a Node.js frontend (from frontend directory):
cd c:\Users\manoj\OneDrive\Dokumen\Desktop\Research\frontend
npm install
npm run dev
```

---

## Project Structure

```
Research/
├── README.md                   ← This file
├── ARCHITECTURE.md             ← Detailed model architecture
├── DATASET_SETUP.md            ← How2Sign dataset setup guide
├── TRAINING_GUIDE.md           ← Step-by-step training instructions
├── EXPERIMENTS.md              ← All experiment descriptions
├── EVALUATION_REPORT.md        ← Honest evaluation report
├── API_DOCUMENTATION.md        ← Backend API reference
├── PROJECT_EXPLANATION.md      ← Plain English explanation for professors
├── VIVA_QUESTIONS.md           ← 45+ Q&A for viva preparation
├── RESEARCH_GAPS.md            ← Research positioning and gaps
├── FINAL_AUDIT.md              ← Honest audit of project status
│
├── src/
│   ├── __init__.py
│   ├── models/
│   │   ├── __init__.py
│   │   ├── text_encoder.py          ← DistilBERT text encoder
│   │   ├── pose_context_encoder.py  ← Transformer pose context encoder
│   │   ├── fusion.py                ← CrossAttentionFusion, ConcatFusion
│   │   ├── variational_encoder.py   ← CVAE encoder, KL loss
│   │   ├── decoder.py               ← Transformer pose decoder
│   │   └── contextsign_model.py     ← Full ContextSignModel
│   ├── data/
│   │   ├── __init__.py
│   │   └── how2sign_dataset.py      ← Dataset classes, normalization
│   ├── training/
│   │   ├── __init__.py
│   │   ├── losses.py                ← masked_mse_loss, combined_loss
│   │   └── trainer.py               ← Training loop, early stopping
│   ├── evaluation/
│   │   ├── __init__.py
│   │   ├── metrics.py               ← MPE, DTW, smoothness, bootstrap CI
│   │   ├── evaluator.py             ← Evaluation runner
│   │   └── human_eval_protocol.md   ← Human evaluation protocol
│   └── inference/
│       └── __init__.py
│
└── [backend/ and frontend/ as separate modules]
```

---

## Limitations Disclaimer

This project is a **research prototype** at an early stage. The following must be clearly understood:

1. **No real training has occurred.** All model weights are randomly initialized. No How2Sign pose data has been used for training.
2. **No checkpoints exist.** The model cannot generate meaningful ASL poses in its current state.
3. **Smoke tests use synthetic data only.** Unit tests confirm code correctness, not model quality.
4. **Pose files are not included** in the standard How2Sign distribution and must be extracted.
5. **Human evaluation has not been conducted.** Any human evaluation protocol described is a plan, not a result.
6. **No claims about ASL quality can be made** without trained models evaluated by qualified ASL signers.

See [FINAL_AUDIT.md](FINAL_AUDIT.md) for the complete honest audit.

---

## Citation

If you use this work, please cite:

```bibtex
@misc{contextsign2026,
  title  = {ContextSign AI: Context-Aware ASL Pose Generation via Transformer-CVAE},
  author = {[Author Name]},
  year   = {2026},
  note   = {Research prototype. No training results on real data.}
}

@inproceedings{duarte2021how2sign,
  title     = {How2Sign: A Large-scale Multimodal Dataset for Continuous American Sign Language},
  author    = {Duarte, Amanda and Palaskar, Shruti and Ventura, Lucas and Ghadiyaram, Deepti and DeHaan, Kenneth and Metze, Florian and Torres, Jordi and Giro-i-Nieto, Xavier},
  booktitle = {Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition},
  pages     = {2735--2744},
  year      = {2021}
}
```
