# ContextSign AI — System Architecture

> This document describes the designed architecture. No real-data training has occurred.
> All components are implemented in `src/models/`.

---

## Table of Contents
1. [System Overview](#system-overview)
2. [Data Flow Diagram](#data-flow-diagram)
3. [Component Details](#component-details)
   - [Text Encoder](#1-text-encoder)
   - [Pose Context Encoder](#2-pose-context-encoder)
   - [Cross-Attention Fusion](#3-cross-attention-fusion)
   - [CVAE Encoder](#4-cvae-encoder)
   - [Transformer Decoder](#5-transformer-decoder)
4. [Tensor Shapes Reference](#tensor-shapes-reference)
5. [Experiment Variants Table](#experiment-variants-table)
6. [Loss Functions](#loss-functions)
7. [KL Annealing Schedule](#kl-annealing-schedule)

---

## System Overview

```
┌─────────────────────────────────────────────────────────────────────┐
│                        ContextSign AI                               │
│                                                                     │
│  Input: English sentence  +  Prior ASL pose sequences               │
│  Output: Predicted ASL pose sequence [T, 274]                       │
│                                                                     │
│  ┌────────────────┐    ┌──────────────────────┐                    │
│  │  Text Encoder  │    │  Pose Context Encoder │                    │
│  │  (DistilBERT)  │    │  (Transformer)        │                    │
│  └───────┬────────┘    └──────────┬───────────┘                    │
│          │  [B, 256]              │  [B, 256]                       │
│          └──────────┬─────────────┘                                 │
│                     │                                               │
│          ┌──────────▼──────────┐                                   │
│          │  CrossAttentionFusion│                                   │
│          │  (text=Q, ctx=K,V)  │                                   │
│          └──────────┬──────────┘                                   │
│                     │  [B, 256]                                     │
│                     │                                               │
│          ┌──────────▼──────────┐  ◄── target pose (training only)  │
│          │    CVAE Encoder     │                                    │
│          │  μ, log_σ → z~N    │                                    │
│          └──────────┬──────────┘                                   │
│                     │  z [B, 128]                                   │
│                     │                                               │
│          ┌──────────▼──────────┐                                   │
│          │  Transformer Decoder│                                    │
│          │  (causal mask)      │                                    │
│          └──────────┬──────────┘                                   │
│                     │                                               │
│          Predicted Poses  [B, T, 274]                               │
└─────────────────────────────────────────────────────────────────────┘
```

---

## Data Flow Diagram

```
INPUT
=====

English sentence: "He walked to the store."
  │
  ▼
Tokenizer (DistilBERT)
  │
  ▼  tokens [B, L]  (L = sequence length, padded to max_len)
  │
  ▼
DistilBERT forward()
  │
  ▼  hidden_states [B, L, 768]
  │
  ▼  Mean pooling over sequence dim
  │
  ▼  sentence_embedding [B, 768]
  │
  ▼  Linear projection (768 → 256) + LayerNorm + GELU
  │
  ▼
text_feat  [B, 256]
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

Prior ASL poses: window × T × 274
(window = number of prior sentences, T = frames per sentence)
  │
  ▼  Flatten: [B, window, T, 274] → [B, window, T*274]
  │
  ▼  Linear projection per frame or per window
  │
  ▼  Input to Transformer Encoder with positional encoding
  │
  ▼  Transformer Encoder (multi-head self-attention)
     [B, window, d_model=256]
  │
  ▼  Aggregate (mean pool over window dim)
  │
  ▼
ctx_feat  [B, 256]
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

FUSION
======

text_feat  [B, 256]  ──► Query
ctx_feat   [B, 256]  ──► Key, Value
  │
  ▼  Multi-Head Cross-Attention
     Q: [B, 1, 256]
     K, V: [B, 1, 256]
     Attention output: [B, 1, 256]
  │
  ▼  Squeeze + residual
  │
  ▼
fused_feat  [B, 256]
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

CVAE ENCODER (training only)
=============================

Target pose: [B, T, 274]
  │
  ▼  Flatten and project to 256
  │
  ▼  Concat with fused_feat  → [B, 512]
  │
  ▼  MLP encoder
  │
  ▼  μ [B, 128],  log_σ [B, 128]
  │
  ▼  Reparameterisation: z = μ + ε * exp(log_σ),  ε ~ N(0,1)
  │
  ▼
z  [B, 128]

(At inference: z ~ N(0, I) directly, CVAE encoder skipped)
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

DECODER
=======

z  [B, 128]  +  fused_feat  [B, 256]
  │
  ▼  Project and expand to [B, T, 256] as decoder memory
  │
  ▼  Transformer Decoder
     - Causal mask ensures frame t depends only on frames < t
     - Positional encoding added to frame queries
     - Cross-attention over (z + fused_feat) memory
  │
  ▼  Linear output projection: 256 → 274  (per frame)
  │
  ▼
Predicted pose sequence  [B, T, 274]
```

---

## Component Details

### 1. Text Encoder

**File**: `src/models/text_encoder.py`

| Property | Value |
|----------|-------|
| Base model | `distilbert-base-uncased` |
| Frozen during training | Configurable (default: partially frozen) |
| Hidden size | 768 |
| Output projection | Linear(768, 256) → LayerNorm → GELU |
| Pooling | Mean over non-padding token positions |
| Output | `text_feat` ∈ ℝ^{B×256} |

DistilBERT is a distilled version of BERT with 6 layers instead of 12, achieving 97% of BERT's performance at 60% the size. It is used here because:
- Reasonable sentence-level representations
- Lower memory footprint than full BERT (important for RTX 2050 4GB)
- Pre-trained on large English corpora

### 2. Pose Context Encoder

**File**: `src/models/pose_context_encoder.py`

| Property | Value |
|----------|-------|
| Input | Prior poses `[B, window, T, 274]` |
| Frame projection | Linear(274, 128) per frame |
| Temporal encoder | Transformer Encoder (2 layers, 4 heads) |
| Output per sentence | Mean-pooled over T → [B, window, 128] |
| Cross-sentence encoder | Transformer Encoder over window dim |
| Output projection | Linear(128, 256) |
| Output | `ctx_feat` ∈ ℝ^{B×256} |

When `window = 0` (E1 / E5 / E8), `ctx_feat` is set to the zero vector, and fusion degenerates to passing `text_feat` directly.

### 3. Cross-Attention Fusion

**File**: `src/models/fusion.py`

**CrossAttentionFusion** (default for E2, E3):

```
Q = text_feat.unsqueeze(1)      [B, 1, 256]
K = ctx_feat.unsqueeze(1)       [B, 1, 256]
V = ctx_feat.unsqueeze(1)       [B, 1, 256]

attn_output = MultiheadAttention(Q, K, V)   [B, 1, 256]
fused_feat = LayerNorm(text_feat + attn_output.squeeze(1))  [B, 256]
```

**ConcatFusion** (used in E7 ablation):

```
fused_feat = Linear(cat([text_feat, ctx_feat], dim=-1))  [B, 512 → 256]
fused_feat = LayerNorm(fused_feat)                       [B, 256]
```

The rationale for cross-attention as the default: the text query should selectively attend to relevant contextual information, rather than a simple concatenation which treats both equally.

### 4. CVAE Encoder

**File**: `src/models/variational_encoder.py`

| Property | Value |
|----------|-------|
| Input | `[fused_feat; flattened_target_pose]` → [B, 256+256] |
| Hidden layers | MLP: 512 → 512 → 256 |
| Output | `μ` [B, 128], `log_σ` [B, 128] |
| Latent dim | 128 |
| Reparameterisation | `z = μ + exp(0.5 * log_σ) * ε` |
| Prior | `p(z) = N(0, I)` |
| KL divergence | `-0.5 * sum(1 + log_σ - μ² - exp(log_σ))` |

At inference, the CVAE encoder is bypassed: `z ~ N(0, I)` is sampled directly.

### 5. Transformer Decoder

**File**: `src/models/decoder.py`

| Property | Value |
|----------|-------|
| Memory input | `[z; fused_feat]` projected to 256 |
| Decoder layers | 4 |
| Attention heads | 8 |
| Feed-forward dim | 1024 |
| Positional encoding | Sinusoidal |
| Causal mask | Yes (autoregressive generation) |
| Output projection | Linear(256, 274) per frame |
| Output | Predicted poses ∈ ℝ^{B×T×274} |

---

## Tensor Shapes Reference

| Stage | Tensor | Shape |
|-------|--------|-------|
| Input text | `input_ids` | [B, L] |
| DistilBERT output | `last_hidden_state` | [B, L, 768] |
| After projection | `text_feat` | [B, 256] |
| Prior poses | `context_poses` | [B, window, T, 274] |
| After context encoder | `ctx_feat` | [B, 256] |
| After fusion | `fused_feat` | [B, 256] |
| CVAE mu/log_sigma | `mu`, `log_sigma` | [B, 128] each |
| Latent vector | `z` | [B, 128] |
| Decoder input | `z + fused` projected | [B, T, 256] |
| Decoder output | `pose_logits` | [B, T, 256] |
| Final projection | `pred_poses` | [B, T, 274] |
| Target pose | `target_poses` | [B, T, 274] |
| Padding mask | `pose_mask` | [B, T] bool |

**Notation:**
- `B` = batch size
- `L` = text sequence length (max 128 tokens)
- `T` = number of frames per pose sequence (variable; padded to max in batch)
- `window` = number of prior sentences (0, 1, or 2–3)
- `274` = 137 keypoints × 2 coordinates (x, y) — MediaPipe Holistic

---

## Experiment Variants Table

| Exp | Text Encoder | Pose Ctx Encoder | Fusion | CVAE | Context Window | Retrained |
|-----|:---:|:---:|:---:|:---:|:---:|:---:|
| E1 | ✓ | ✗ | — | ✓ | 0 | Yes |
| E2 | ✓ | ✓ | CrossAttn | ✓ | 1 | Yes |
| E3 | ✓ | ✓ | CrossAttn | ✓ | 2–3 | Yes |
| E4 | ✓ | ✓ (text only) | CrossAttn | ✓ | 1 (text) | Yes |
| E5 | ✓ | ✓ (zeroed) | CrossAttn | ✓ | 1 | No (uses E2 weights) |
| E6 | ✓ | ✓ (shuffled) | CrossAttn | ✓ | 1 | No (uses E2 weights) |
| E7 | ✓ | ✓ | Concat | ✓ | 1 | Yes |
| E8 | ✓ | ✓ | CrossAttn | ✗ | 1 | Yes |
| E9 | ✓ | ✓ + Audio | CrossAttn | ✓ | 1 | FUTURE |

**Notes:**
- E5 and E6 are **probe experiments** (no retraining): they evaluate the same E2 checkpoint under distribution-shifted context inputs.
- E4 replaces the pose context with BERT embeddings of prior English captions.
- E7 substitutes CrossAttentionFusion with ConcatFusion to test whether attention structure helps.
- E8 replaces the CVAE latent space with a simple deterministic mapping (AE-style).

---

## Loss Functions

**Combined Loss:**

```
L_total = L_recon + β(t) * L_KL
```

**Reconstruction Loss (masked MSE):**

```
L_recon = Σ_t [mask_t * ||pred_pose_t - target_pose_t||²] / Σ_t mask_t
```

Where `mask_t = 1` if frame `t` is not padding.

**KL Divergence Loss:**

```
L_KL = -0.5 * mean(1 + log_σ - μ² - exp(log_σ))
```

**E8 (no CVAE):**

```
L_total = L_recon  (no KL term)
```

**Component losses** (for analysis only, not backpropagated separately):

```
L_face    = MSE on face keypoints      (indices 0–467)
L_pose    = MSE on body keypoints      (indices 468–501)
L_left_hand  = MSE on left hand        (indices 502–521)
L_right_hand = MSE on right hand       (indices 522–541)
```

> *Note: exact index ranges depend on the MediaPipe Holistic landmark ordering used during pose extraction.*

---

## KL Annealing Schedule

To prevent **posterior collapse** (a common CVAE failure mode where the model learns to ignore the latent variable), the KL weight `β(t)` is annealed:

```
         ⎧ 0                                    if t < warmup_steps
β(t) =   ⎨ min(β_max, β_max * (t - warmup) /   if warmup_steps ≤ t < anneal_end
         ⎩         anneal_steps)
         ⎩ β_max                                otherwise
```

**Default hyperparameters:**

| Parameter | Default Value |
|-----------|---------------|
| `warmup_steps` | 1000 |
| `anneal_steps` | 5000 |
| `β_max` | 1.0 |
| Final β | 1.0 |

**Rationale:** Starting with β = 0 forces the model to learn meaningful latent representations via reconstruction. Gradually increasing β then brings the latent distribution toward the standard Gaussian prior.

The annealing schedule is logged during training and can be monitored in TensorBoard or Weights & Biases if configured.
