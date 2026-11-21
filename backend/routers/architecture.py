"""System architecture specifications router."""

from fastapi import APIRouter

from backend.schemas import ArchitectureResponse

router = APIRouter(tags=["Architecture"])

ARCHITECTURE_COMPONENTS = [
    {
        "name": "TextEncoder",
        "type": "DistilBERT + Linear Projection",
        "input_shape": "[B, L] (token IDs)",
        "output_shape": "[B, 256] (text_feat)",
        "parameters": "distilbert-base-uncased (66M) + Linear(768, 256) + LayerNorm + GELU",
        "description": (
            "Encodes English text input into contextual token embeddings, "
            "mean-pools over sequence tokens, and projects to the 256-dimensional latent space."
        ),
    },
    {
        "name": "PoseContextEncoder",
        "type": "Hierarchical Spatio-Temporal Transformer Encoder",
        "input_shape": "[B, window, T, 274] (prior ASL poses)",
        "output_shape": "[B, 256] (ctx_feat)",
        "parameters": "Linear(274, 128) -> 2-layer Transformer (4 heads) -> Linear(128, 256)",
        "description": (
            "Encodes sequential prior ASL utterances across temporal frame and multi-sentence "
            "window dimensions to maintain continuous signing context and co-articulation."
        ),
    },
    {
        "name": "CrossAttentionFusion",
        "type": "Multi-Head Cross-Attention",
        "input_shape": "Query: text_feat [B, 1, 256], Key/Value: ctx_feat [B, 1, 256]",
        "output_shape": "[B, 256] (fused_feat)",
        "parameters": "8 attention heads, embed_dim=256, LayerNorm, FFN(256->512->256)",
        "description": (
            "Enables text representation to selectively attend to relevant prior signing dynamics, "
            "yielding a unified context-aware conditioning feature."
        ),
    },
    {
        "name": "CVAE",
        "type": "Conditional Variational Autoencoder",
        "input_shape": "fused_feat [B, 256] + target_pose [B, T, 274] (training)",
        "output_shape": "z [B, 128] (latent Gaussian representation)",
        "parameters": "MLP Encoder (512->512->256), mu/log_sigma heads (128-d), KL annealing",
        "description": (
            "Captures the one-to-many non-deterministic mapping between spoken English and ASL. "
            "At test time, z is sampled directly from N(0, I)."
        ),
    },
    {
        "name": "TransformerDecoder",
        "type": "Autoregressive Causal Transformer Decoder",
        "input_shape": "Memory: [z; fused_feat] [B, T, 256], Target queries with sinusoidal PE",
        "output_shape": "[B, T, 274] (predicted 2D keypoints)",
        "parameters": "4 layers, 8 heads, d_model=256, FFN=1024, Linear(256, 274)",
        "description": (
            "Generates temporally smooth MediaPipe Holistic landmark trajectories (137 keypoints * 2) "
            "conditioned on the fused text-context memory and latent vector."
        ),
    },
]

DATA_FLOW_STEPS = [
    "1. Tokenizer converts English sentence into token IDs [B, L].",
    "2. TextEncoder processes token IDs through DistilBERT and projects to text_feat [B, 256].",
    "3. PoseContextEncoder encodes prior sign poses [B, window, T, 274] into ctx_feat [B, 256].",
    "4. CrossAttentionFusion executes multi-head cross-attention (Q=text, K,V=ctx) -> fused_feat [B, 256].",
    "5. Latent vector z [B, 128] is sampled from standard Gaussian N(0, I) during inference.",
    "6. Conditioning memory [z; fused_feat] is constructed and expanded across temporal queries [B, T, 256].",
    "7. TransformerDecoder generates output landmark frames [B, T, 274] using causal attention.",
]

MODEL_CONFIG = {
    "hidden_dim": 256,
    "latent_dim": 128,
    "pose_dim": 274,
    "context_window": 1,
    "max_pose_frames": 128,
    "max_text_tokens": 64,
    "num_decoder_layers": 4,
    "num_decoder_heads": 8,
    "decoder_ffn_dim": 1024,
    "dropout": 0.1,
    "fusion_mode": "cross_attention",
    "base_text_model": "distilbert-base-uncased",
    "keypoints_count": 137,
    "coordinates_per_keypoint": 2,
}


@router.get(
    "/architecture",
    response_model=ArchitectureResponse,
    summary="Get model architecture and component specifications",
)
async def get_architecture() -> ArchitectureResponse:
    """Return formal specification of the ContextSign AI architecture, tensor shapes, and pipeline."""
    return ArchitectureResponse(
        components=ARCHITECTURE_COMPONENTS,
        data_flow=DATA_FLOW_STEPS,
        model_configuration=MODEL_CONFIG,
    )
