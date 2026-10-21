"""
Pre-trained Multilingual Transformer Bi-Encoder + Cross-Interaction Neural Network
Backbone: HuggingFace Pre-trained Transformer (`sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2`
          or `intfloat/multilingual-e5-small`, 118M parameters, 12 Transformer Encoder layers, d_model=384)
          supporting Vietnamese + English technical CVs and Job Descriptions.
"""

from typing import Dict, List, Optional, Tuple

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from transformers import AutoModel, AutoTokenizer

DEFAULT_TRANSFORMER_BACKBONE = "sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2"


def mean_pooling(last_hidden_state: torch.Tensor, attention_mask: torch.Tensor) -> torch.Tensor:
    """
    Attention-mask-aware Mean Pooling over Transformer token embeddings:
      e = sum(h_t * m_t) / max(sum(m_t), 1e-9)
    """
    input_mask_expanded = attention_mask.unsqueeze(-1).expand(last_hidden_state.size()).float()
    sum_embeddings = torch.sum(last_hidden_state * input_mask_expanded, dim=1)
    sum_mask = input_mask_expanded.sum(dim=1)
    sum_mask = torch.clamp(sum_mask, min=1e-9)
    return sum_embeddings / sum_mask


class TransformerFeatureExtractor:
    """
    Wraps HuggingFace AutoTokenizer + AutoModel (118M params) to encode raw Vietnamese/English
    CV and JD texts into 384-dim contextualized semantic embeddings.
    """

    def __init__(self, model_name: str = DEFAULT_TRANSFORMER_BACKBONE, device: Optional[str] = None):
        self.model_name = model_name
        self.device = device or ("cuda" if torch.cuda.is_available() else "cpu")
        self.tokenizer = AutoTokenizer.from_pretrained(model_name)
        self.backbone = AutoModel.from_pretrained(model_name).to(self.device)
        self.backbone.eval()
        self.hidden_size = int(self.backbone.config.hidden_size)
        self.total_params = sum(p.numel() for p in self.backbone.parameters())

    @torch.no_grad()
    def encode_texts(self, texts: List[str], batch_size: int = 32, max_length: int = 256) -> np.ndarray:
        all_embeddings = []
        for start in range(0, len(texts), batch_size):
            batch_texts = texts[start : start + batch_size]
            encoded = self.tokenizer(
                batch_texts,
                padding=True,
                truncation=True,
                max_length=max_length,
                return_tensors="pt",
            ).to(self.device)
            outputs = self.backbone(**encoded)
            pooled = mean_pooling(outputs.last_hidden_state, encoded["attention_mask"])
            pooled = F.normalize(pooled, p=2, dim=-1)
            all_embeddings.append(pooled.cpu().numpy().astype(np.float32))
        return np.vstack(all_embeddings)


class ResidualSemanticBlock(nn.Module):
    def __init__(self, dim: int, dropout: float = 0.15):
        super().__init__()
        self.fc1 = nn.Linear(dim, dim)
        self.ln1 = nn.LayerNorm(dim)
        self.fc2 = nn.Linear(dim, dim)
        self.ln2 = nn.LayerNorm(dim)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        h = F.gelu(self.ln1(self.fc1(x)))
        h = self.dropout(h)
        h = self.ln2(self.fc2(h))
        return F.gelu(x + h)


class EndToEndTransformerBiEncoder(nn.Module):
    """
    Full End-to-End Fine-tunable Transformer Bi-Encoder (for GPU / Colab training & inference).
    Unfreezes the top `unfreeze_layers` of the 12-layer Transformer backbone and jointly trains
    the Dual-Tower projection + Cross-Interaction Multi-Task Head.
    """

    def __init__(
        self,
        backbone_name: str = DEFAULT_TRANSFORMER_BACKBONE,
        latent_dim: int = 128,
        struct_dim: int = 6,
        unfreeze_layers: int = 2,
    ):
        super().__init__()
        self.backbone = AutoModel.from_pretrained(backbone_name)
        hidden_size = int(self.backbone.config.hidden_size)

        # Freeze lower layers, unfreeze top `unfreeze_layers` encoder blocks
        for param in self.backbone.parameters():
            param.requires_grad = False
        if hasattr(self.backbone, "encoder") and hasattr(self.backbone.encoder, "layer"):
            for layer in self.backbone.encoder.layer[-unfreeze_layers:]:
                for param in layer.parameters():
                    param.requires_grad = True

        self.head = CvJdBiEncoderHead(
            embed_dim=hidden_size,
            latent_dim=latent_dim,
            struct_dim=struct_dim,
        )

    def forward(
        self,
        cv_input_ids: torch.Tensor,
        cv_attention_mask: torch.Tensor,
        jd_input_ids: torch.Tensor,
        jd_attention_mask: torch.Tensor,
        x_struct: torch.Tensor,
    ) -> Dict[str, torch.Tensor]:
        cv_out = self.backbone(input_ids=cv_input_ids, attention_mask=cv_attention_mask)
        jd_out = self.backbone(input_ids=jd_input_ids, attention_mask=jd_attention_mask)
        e_cv = F.normalize(mean_pooling(cv_out.last_hidden_state, cv_attention_mask), p=2, dim=-1)
        e_jd = F.normalize(mean_pooling(jd_out.last_hidden_state, jd_attention_mask), p=2, dim=-1)
        return self.head(e_cv, e_jd, x_struct)


class CvJdBiEncoderHead(nn.Module):
    """
    Dual-Tower Semantic Projection + Cross-Interaction Multi-Task Head
    operating on Pre-trained Transformer contextual embeddings (dim=384) + SVD features.
    """

    def __init__(self, embed_dim: int = 384, latent_dim: int = 128, struct_dim: int = 6):
        super().__init__()
        self.embed_dim = embed_dim
        self.latent_dim = latent_dim
        self.struct_dim = struct_dim

        self.cv_proj = nn.Sequential(
            nn.Linear(embed_dim, latent_dim),
            nn.LayerNorm(latent_dim),
            nn.GELU(),
            ResidualSemanticBlock(latent_dim, dropout=0.15),
        )
        self.jd_proj = nn.Sequential(
            nn.Linear(embed_dim, latent_dim),
            nn.LayerNorm(latent_dim),
            nn.GELU(),
            ResidualSemanticBlock(latent_dim, dropout=0.15),
        )

        # Cross-Interaction Fusion: [u, v, |u - v|, u * v, raw_cos_sim(1), latent_cos_sim(1), struct(6)]
        fusion_dim = latent_dim * 4 + 2 + struct_dim
        self.fusion_net = nn.Sequential(
            nn.Linear(fusion_dim, 256),
            nn.LayerNorm(256),
            nn.GELU(),
            nn.Dropout(0.15),
            ResidualSemanticBlock(256, dropout=0.15),
            nn.Linear(256, 96),
            nn.LayerNorm(96),
            nn.GELU(),
        )
        self.multi_head = nn.Sequential(
            nn.Linear(96, 4),
            nn.Sigmoid(),
        )

    def forward(
        self, e_cv: torch.Tensor, e_jd: torch.Tensor, x_struct: torch.Tensor
    ) -> Dict[str, torch.Tensor]:
        raw_cos = torch.sum(F.normalize(e_cv, p=2, dim=-1) * F.normalize(e_jd, p=2, dim=-1), dim=-1, keepdim=True)
        u = F.normalize(self.cv_proj(e_cv), p=2, dim=-1)
        v = F.normalize(self.jd_proj(e_jd), p=2, dim=-1)
        latent_cos = torch.sum(u * v, dim=-1, keepdim=True)

        diff = torch.abs(u - v)
        prod = u * v
        fused = torch.cat([u, v, diff, prod, raw_cos, latent_cos, x_struct], dim=-1)
        h = self.fusion_net(fused)
        out = self.multi_head(h)
        return {
            "overall": out[:, 0],
            "skill": out[:, 1],
            "experience": out[:, 2],
            "keyword": out[:, 3],
            "cosine_similarity": latent_cos.squeeze(-1),
            "raw_transformer_cosine": raw_cos.squeeze(-1),
        }
