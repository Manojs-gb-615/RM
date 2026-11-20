"""Text encoder module using DistilBERT with linear projection to hidden_dim."""

import torch
import torch.nn as nn

try:
    from transformers import DistilBertConfig, DistilBertModel
    HF_AVAILABLE = True
except ImportError:
    HF_AVAILABLE = False


class TextEncoder(nn.Module):
    """Encodes text token sequences using DistilBERT and projects CLS token representation."""

    def __init__(
        self,
        hidden_dim: int = 256,
        model_name: str = "distilbert-base-uncased",
        freeze: bool = False,
    ):
        super().__init__()
        self.hidden_dim = hidden_dim
        self.model_name = model_name
        self.freeze = freeze

        bert_dim = 768
        if HF_AVAILABLE:
            try:
                self.transformer = DistilBertModel.from_pretrained(model_name)
                bert_dim = self.transformer.config.dim
            except Exception:
                # Fallback to random weights if offline or HuggingFace hub is unreachable
                config = DistilBertConfig(
                    vocab_size=30522,
                    dim=768,
                    n_layers=2,
                    n_heads=4,
                    hidden_dim=768 * 4,
                )
                self.transformer = DistilBertModel(config)
                bert_dim = 768
        else:
            # Fallback simple embedding + GRU if transformers is not installed
            self.transformer = nn.Embedding(30522, bert_dim)

        if freeze:
            for param in self.transformer.parameters():
                param.requires_grad = False

        self.projection = nn.Sequential(
            nn.Linear(bert_dim, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.Dropout(0.1),
        )

    def forward(
        self, input_ids: torch.Tensor, attention_mask: torch.Tensor
    ) -> torch.Tensor:
        """Forward pass.

        Args:
            input_ids: [B, T_text]
            attention_mask: [B, T_text]

        Returns:
            [B, hidden_dim] text feature representation
        """
        if HF_AVAILABLE and isinstance(self.transformer, DistilBertModel):
            outputs = self.transformer(input_ids=input_ids, attention_mask=attention_mask)
            # CLS token representation at index 0
            cls_token = outputs.last_hidden_state[:, 0, :]  # [B, 768]
        else:
            # Fallback embedding average
            embeds = self.transformer(input_ids)  # [B, T, D]
            mask_expanded = attention_mask.unsqueeze(-1).float()
            sum_embeds = (embeds * mask_expanded).sum(dim=1)
            lens = mask_expanded.sum(dim=1).clamp(min=1e-5)
            cls_token = sum_embeds / lens

        feat = self.projection(cls_token)  # [B, hidden_dim]
        return feat
