"""Pose context encoder for previous sentence sign poses."""

import math
import torch
import torch.nn as nn


class PositionalEncoding(nn.Module):
    """Sinusoidal positional encoding."""

    def __init__(self, hidden_dim: int, max_len: int = 512):
        super().__init__()
        pe = torch.zeros(max_len, hidden_dim)
        position = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, hidden_dim, 2).float() * (-math.log(10000.0) / hidden_dim))
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        pe = pe.unsqueeze(0)  # [1, max_len, hidden_dim]
        self.register_buffer("pe", pe)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x is [B, T, D]
        seq_len = x.size(1)
        return x + self.pe[:, :seq_len, :]


class PoseContextEncoder(nn.Module):
    """Encodes historical pose sequences from prior context sentences into a compact representation."""

    def __init__(
        self,
        pose_dim: int = 274,
        hidden_dim: int = 256,
        num_heads: int = 8,
        num_layers: int = 2,
        dropout: float = 0.1,
        max_len: int = 128,
    ):
        super().__init__()
        self.pose_dim = pose_dim
        self.hidden_dim = hidden_dim
        self.max_len = max_len

        self.input_proj = nn.Sequential(
            nn.Linear(pose_dim, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.Dropout(dropout),
        )

        self.pos_encoder = PositionalEncoding(hidden_dim, max_len=max_len * 4)

        encoder_layer = nn.TransformerEncoderLayer(
            d_model=hidden_dim,
            nhead=num_heads,
            dim_feedforward=hidden_dim * 4,
            dropout=dropout,
            activation="relu",
            batch_first=True,
        )
        self.transformer = nn.TransformerEncoder(encoder_layer, num_layers=num_layers)
        self.output_norm = nn.LayerNorm(hidden_dim)

    def forward(self, context_poses: torch.Tensor) -> torch.Tensor:
        """Forward pass.

        Args:
            context_poses: Tensor of shape [B, context_window, T, pose_dim]

        Returns:
            Tensor of shape [B, hidden_dim]
        """
        B = context_poses.size(0)
        W = context_poses.size(1)

        if W == 0:
            return torch.zeros((B, self.hidden_dim), device=context_poses.device, dtype=context_poses.dtype)

        T = context_poses.size(2)
        # Reshape [B, W, T, pose_dim] -> [B, W * T, pose_dim]
        flattened_poses = context_poses.view(B, W * T, self.pose_dim)

        # Mask check: if entire sequence is zero (e.g. no valid context)
        is_all_zero = (flattened_poses.abs().sum(dim=-1).sum(dim=-1) == 0)  # [B]

        x = self.input_proj(flattened_poses)  # [B, W*T, hidden_dim]
        x = self.pos_encoder(x)
        encoded = self.transformer(x)  # [B, W*T, hidden_dim]

        # Mean pooling over valid sequence
        pooled = encoded.mean(dim=1)  # [B, hidden_dim]
        out = self.output_norm(pooled)

        # Zero out entries where context was entirely zero
        if is_all_zero.any():
            out = out.masked_fill(is_all_zero.unsqueeze(-1), 0.0)

        return out
