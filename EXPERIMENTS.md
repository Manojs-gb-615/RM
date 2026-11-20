# ContextSign AI — Experiments

> **Status**: All experiments (E1–E8) are **implemented** and **smoke-tested on synthetic data**.
> **No experiment has been trained on real How2Sign pose data.**
> E9 is future work and has NOT been implemented.

---

## Status Overview Table

| ID | Name | Hypothesis | Implementation | Synthetic Test | Real Training | Evaluated | Notes |
|----|------|-----------|---------------|----------------|--------------|-----------|-------|
| E1 | Text-only baseline | — | ✅ IMPLEMENTED | ✅ TESTED | ❌ NOT RUN | ❌ PENDING | Control |
| E2 | Context-aware (1 prev) | Context helps | ✅ IMPLEMENTED | ✅ TESTED | ❌ NOT RUN | ❌ PENDING | Main |
| E3 | Longer context (2–3 prev) | More context = better | ✅ IMPLEMENTED | ✅ TESTED | ❌ NOT RUN | ❌ PENDING | Main |
| E4 | Text-history control | Text context < Pose context | ✅ IMPLEMENTED | ✅ TESTED | ❌ NOT RUN | ❌ PENDING | Control |
| E5 | Zero-context probe | Model depends on context | ✅ IMPLEMENTED | ✅ TESTED | ❌ N/A | ❌ PENDING | No retrain |
| E6 | Shuffled-context probe | Wrong context hurts | ✅ IMPLEMENTED | ✅ TESTED | ❌ N/A | ❌ PENDING | No retrain |
| E7 | Fusion ablation | CrossAttn > Concat | ✅ IMPLEMENTED | ✅ TESTED | ❌ NOT RUN | ❌ PENDING | Retrained |
| E8 | Variational ablation | CVAE adds value | ✅ IMPLEMENTED | ✅ TESTED | ❌ NOT RUN | ❌ PENDING | Retrained |
| E9 | Audio extension | Audio improves ASL | ❌ NOT IMPLEMENTED | ❌ N/A | ❌ N/A | ❌ PENDING | Future work |

---

## Experiment Descriptions

---

### E1: Text-Only Baseline

**Purpose**: Establish a lower-bound baseline with no discourse context.

**Hypothesis**: Without context information, the model produces pose sequences that are plausible for the given sentence in isolation, but may be inconsistent with adjacent sentences in a discourse.

**Design**:
- Input: English sentence only
- Context window: 0 (no prior poses used)
- Pose context encoder: disabled (output = zero vector)
- Fusion: identity (text_feat passed directly to CVAE)
- CVAE: enabled
- Decoder: full Transformer decoder

**What it tests**: Establishes the minimum competence benchmark. All other experiments should be compared against E1.

**Comparison role**: Primary baseline. E2 and E3 must outperform E1 on pose quality AND discourse consistency metrics for the context hypothesis to be supported.

**Training command**:
```powershell
$PYTHON train.py --experiment E1 --context_window 0 --batch_size 16 --epochs 50 --output_dir checkpoints/E1
```

---

### E2: Context-Aware (1 Prior Sentence)

**Purpose**: Main experimental condition. Test whether providing the immediately preceding ASL pose sequence improves generation quality.

**Hypothesis**: The model conditioned on 1 prior pose sequence will produce higher-quality, more temporally consistent, and more spatially coherent pose sequences than E1.

**Design**:
- Input: English sentence + 1 prior ASL pose sequence
- Context window: 1
- Pose context encoder: Transformer encoder over prior frames
- Fusion: CrossAttentionFusion (text as Q, context as K/V)
- CVAE: enabled
- Decoder: full Transformer decoder

**What it tests**: Whether pose-level discourse context is useful for ASL generation.

**Key distinction from E5/E6**: E2 is a **trained model** that has learned to use context. E5 and E6 evaluate the **same trained E2 weights** under different context manipulations.

**Training command**:
```powershell
$PYTHON train.py --experiment E2 --context_window 1 --fusion cross_attention --batch_size 16 --epochs 50 --output_dir checkpoints/E2
```

---

### E3: Longer Context (2–3 Prior Sentences)

**Purpose**: Test whether providing more discourse history further improves generation quality.

**Hypothesis**: Models conditioned on 2–3 prior sentences will capture longer-range discourse dependencies (referent tracking, discourse arc) and outperform E2 on longer-distance coherence metrics.

**Design**:
- Input: English sentence + 2–3 prior ASL pose sequences
- Context window: 3 (variable; shorter at discourse starts)
- Pose context encoder: Transformer encoder over all prior frames + cross-sequence attention
- Fusion: CrossAttentionFusion
- CVAE: enabled
- Decoder: full Transformer decoder

**What it tests**: Whether larger context windows provide diminishing returns, improvement, or even degradation (due to memory constraints or noise from irrelevant earlier context).

**Expected challenge**: For RTX 2050 (4 GB), context window=3 significantly increases VRAM usage. Batch size must be reduced.

**Training command**:
```powershell
$PYTHON train.py --experiment E3 --context_window 3 --fusion cross_attention --batch_size 8 --epochs 50 --output_dir checkpoints/E3
```

---

### E4: Text-History Control

**Purpose**: Determine whether the benefit of context (if any) in E2/E3 comes specifically from prior *pose* information, or whether prior *text* information alone would be equally useful.

**Hypothesis**: Prior text captions contain some discourse information (referent naming, topic continuity) but lack the spatial and temporal specifics encoded in pose data. E4 will outperform E1 but underperform E2/E3.

**Design**:
- Input: English sentence + 1 prior English caption (text, not poses)
- Context representation: Prior text processed through the same DistilBERT encoder as the main text
- Fusion: CrossAttentionFusion (current text as Q, prior text embedding as K/V)
- CVAE: enabled
- **No pose context encoder** in this configuration

**What it tests**: The unique contribution of pose-level context, disentangled from text-level context.

**Expected insight**: If E4 ≈ E2, then the model is primarily using text-level discourse features, and pose context is redundant. If E2 > E4, pose information provides unique spatial/temporal benefits.

**Training command**:
```powershell
$PYTHON train.py --experiment E4 --context_window 1 --context_type text --batch_size 16 --epochs 50 --output_dir checkpoints/E4
```

---

### E5: Zero-Context Probe

**Purpose**: Determine how much the trained E2 model depends on context at inference time.

**Hypothesis**: If E2 genuinely learned to use context, then zeroing the context at test time will produce qualitatively different (and likely worse) output than providing real context.

**Design**:
- **Model**: E2 checkpoint (pre-trained with 1 prior context) — **NO RETRAINING**
- **Evaluation manipulation**: At test time, the prior pose sequence is replaced with an all-zeros tensor
- **Comparison**: Compare E5 output quality vs. E2 output quality using the same checkpoint

**What it tests**: The model's dependency on context at inference time. This is a **masked-feature probe**, not an independent model.

**Key distinction**: E5 is NOT a separate model. It is the trained E2 model evaluated under a distribution-shifted input.

**Evaluation command** (requires E2 checkpoint):
```powershell
$PYTHON evaluate.py --experiment E5 --checkpoint checkpoints/E2/best_model.pt --zero_context --output_dir results/E5
```

---

### E6: Shuffled-Context Probe

**Purpose**: Test whether the model uses context in a discourse-aware manner or simply as a statistical regulariser.

**Hypothesis**: If the model genuinely tracks discourse-specific features (referent locations, signing style), providing context from a *different video* should degrade output quality compared to providing true context.

**Design**:
- **Model**: E2 checkpoint — **NO RETRAINING**
- **Evaluation manipulation**: At test time, the prior pose sequence is drawn from a randomly selected *different* video (not the same discourse)
- **Comparison**: E6 (shuffled context) vs. E2 (true context) vs. E5 (no context)

**What it tests**: Whether context helps in a discourse-specific way (E2 >> E6) or only by providing general pose statistics (E2 ≈ E6 > E5).

**Key distinction**: Like E5, this is a **probe evaluation** using the E2 checkpoint, not a new model.

**Evaluation command** (requires E2 checkpoint):
```powershell
$PYTHON evaluate.py --experiment E6 --checkpoint checkpoints/E2/best_model.pt --shuffle_context --output_dir results/E6
```

---

### E7: Fusion Ablation (Concatenation vs. Cross-Attention)

**Purpose**: Determine whether the cross-attention fusion mechanism is specifically beneficial compared to a simpler concatenation baseline.

**Hypothesis**: Cross-attention allows the model to selectively attend to relevant contextual features based on the current text query, providing more flexible and informative fusion than simple concatenation.

**Design**:
- **Same as E2** but with `ConcatFusion` instead of `CrossAttentionFusion`
- Context window: 1
- Fusion: `ConcatFusion(text_feat, ctx_feat) → Linear(512, 256) → LayerNorm`
- CVAE: enabled
- **Requires full retraining** (new model, not E2 weights)

**What it tests**: The architectural choice of fusion mechanism. If E2 (CrossAttn) outperforms E7 (Concat), cross-attention is specifically beneficial.

**Training command**:
```powershell
$PYTHON train.py --experiment E7 --context_window 1 --fusion concat --batch_size 16 --epochs 50 --output_dir checkpoints/E7
```

---

### E8: Variational Ablation (No CVAE)

**Purpose**: Determine whether the CVAE component contributes to generation quality compared to a deterministic mapping.

**Hypothesis**: The CVAE latent space provides useful variability in generated poses and may learn a more structured representation. Without it, the model may converge to mode-averaged (blurry) predictions.

**Design**:
- **Same as E2** but without the CVAE encoder
- The encoder is replaced with a deterministic MLP (no reparameterisation, no KL loss)
- Loss: reconstruction loss only (MSE, no KL term)
- **Requires full retraining**

**What it tests**: The unique contribution of the variational component. If E2 (CVAE) outperforms E8 (no CVAE), the latent variable structure is beneficial.

**Training command**:
```powershell
$PYTHON train.py --experiment E8 --context_window 1 --fusion cross_attention --no_cvae --batch_size 16 --epochs 50 --output_dir checkpoints/E8
```

---

### E9: Audio Extension (FUTURE WORK)

**Purpose**: Extend the context representation to include audio prosody features (pitch, rhythm, energy), which may correlate with ASL signing style.

**Status**: ❌ **NOT IMPLEMENTED** — No aligned audio feature files are available.

**Hypothesis**: Audio prosodic features (extracted from the instructor's speech in How2Sign videos) may provide additional temporal and stylistic cues that improve ASL generation.

**Design (conceptual)**:
- Add an Audio Encoder (e.g., wav2vec 2.0 or simple MFCC encoder)
- Fuse audio features alongside text and pose context
- Requires extracting and aligning audio features with How2Sign sentences

**Blockers**:
1. Aligned audio feature files are not available in the standard How2Sign release
2. Audio extraction requires raw video + FFmpeg + audio processing pipeline
3. Audio encoder architecture not implemented in `src/models/`

**What would be needed**:
- Download raw How2Sign videos
- Extract audio segments aligned to `START_REALISED`/`END_REALISED` timestamps
- Compute MFCCs or use pre-trained audio encoder
- Add `AudioEncoder` class to `src/models/`
- Update `ContextSignModel` to include audio branch
- Re-run data pipeline with audio features

---

## Comparison Matrix

| Comparison | What it isolates |
|-----------|-----------------|
| E2 vs. E1 | Effect of pose context (1 prior) |
| E3 vs. E2 | Effect of longer context window |
| E4 vs. E2 | Pose context vs. text context |
| E5 vs. E2 | Model's context dependency at inference |
| E6 vs. E2 | Discourse-specificity of context use |
| E7 vs. E2 | CrossAttention fusion vs. Concat fusion |
| E8 vs. E2 | CVAE vs. deterministic encoder |

---

## Evaluation Protocol

For each experiment (once trained), evaluation involves:

1. **Automatic metrics** on the test set:
   - Mean Pose Error (MPE)
   - Component errors (face, body, hands separately)
   - Dynamic Time Warping (DTW) distance
   - Temporal smoothness score
   - Sequence length ratio

2. **Statistical significance**:
   - Bootstrap confidence intervals (95%, 1000 resamples)
   - Paired t-tests between E1 baseline and each experiment

3. **Human evaluation** (requires qualified ASL signers):
   - See `src/evaluation/human_eval_protocol.md`
   - **Has NOT been conducted**

> See [EVALUATION_REPORT.md](EVALUATION_REPORT.md) for metric definitions and report templates.
