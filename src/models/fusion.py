"""Feature fusion mechanisms: Cross-Attention and Concatenation."""

import torch
import torch.nn as nn


class CrossAttentionFusion(nn.Module):
    """Fuses text representations and context representations using cross-attention."""

    def __init__(self, hidden_dim: int = 256, num_heads: int = 8, dropout: float = 0.1):
        super().__init__()
        self.hidden_dim = hidden_dim
        self.cross_attn = nn.MultiheadAttention(
            embed_dim=hidden_dim,
            num_heads=num_heads,
            dropout=dropout,
            batch_first=True,
        )
        self.dropout = nn.Dropout(dropout)
        self.norm1 = nn.LayerNorm(hidden_dim)
        self.ffn = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim * 2),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim * 2, hidden_dim),
        )
        self.norm2 = nn.LayerNorm(hidden_dim)

    def forward(self, text_feat: torch.Tensor, context_feat: torch.Tensor) -> torch.Tensor:
        """Args:

        text_feat: [B, hidden_dim]
        context_feat: [B, hidden_dim]

        Returns:
            [B, hidden_dim] fused feature representation
        """
        # Query: text feature, Key & Value: context feature
        q = text_feat.unsqueeze(1)      # [B, 1, D]
        k = context_feat.unsqueeze(1)   # [B, 1, D]
        v = context_feat.unsqueeze(1)   # [B, 1, D]

        attn_out, _ = self.cross_attn(query=q, key=k, value=v)  # [B, 1, D]
        x = self.norm1(text_feat + self.dropout(attn_out.squeeze(1)))
        out = self.norm2(x + self.ffn(x))
        return out


class ConcatFusion(nn.Module):
    """Fuses text representations and context representations via concatenation and linear projection."""

    def __init__(self, hidden_dim: int = 256, dropout: float = 0.1):
        super().__init__()
        self.hidden_dim = hidden_dim
        self.net = nn.Sequential(
            nn.Linear(hidden_dim * 2, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, hidden_dim),
            nn.LayerNorm(hidden_dim),
        )

    def forward(self, text_feat: torch.Tensor, context_feat: torch.Tensor) -> torch.Tensor:
        """Args:

        text_feat: [B, hidden_dim]
        context_feat: [B, hidden_dim]

        Returns:
            [B, hidden_dim] fused feature representation
        """
        concat = torch.cat([text_feat, context_feat], dim=-1)  # [B, 2*D]
        return self.net(concat)


def get_fusion(mode: str = "cross_attention", hidden_dim: int = 256, **kwargs) -> nn.Module:
    """Factory function to get fusion module."""
    if mode == "cross_attention":
        num_heads = kwargs.get("num_heads", 8)
        dropout = kwargs.get("dropout", 0.1)
        return CrossAttentionFusion(hidden_dim=hidden_dim, num_heads=num_heads, dropout=dropout)
    elif mode == "concat":
        dropout = kwargs.get("dropout", 0.1)
        return ConcatFusion(hidden_dim=hidden_dim, dropout=dropout)
    else:
        raise ValueError(f"Unknown fusion mode '{mode}'. Choose 'cross_attention' or 'concat'.")
