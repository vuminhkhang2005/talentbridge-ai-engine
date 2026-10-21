"""
PyTorch Siamese Bi-Encoder + Cross-Interaction Neural Network for CV-JD Matching
Architecture:
  1. Dual-Tower Semantic Encoder (`cv_tower`, `jd_tower`):
     Maps Hybrid Word+Char TF-IDF + SVD (LSA) text embeddings (dim=96) into a shared
     L2-normalized latent semantic space u_cv, v_jd in R^64.
  2. Cross-Interaction + Structured ATS Feature Fusion Head:
     Combines [u_cv, v_jd, |u_cv - v_jd|, u_cv * v_jd, cos_sim, struct_features(6)]
     to predict calibrated compatibility scores in [0, 1] (scaled to 0..100%):
       - overall_score
       - skill_score
       - exp_score
       - keyword_score
"""

import re
from typing import Dict, List

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F


def extract_struct_features(
    cv_text: str,
    jd_text: str,
    cv_skills: List[str],
    jd_req_skills: List[str],
    jd_opt_skills: List[str],
    cv_years: float,
    jd_req_years: float,
) -> np.ndarray:
    """
    Computes 6 normalized structural ATS features in [0, 1]:
      0: required skill coverage ratio
      1: optional skill coverage ratio
      2: candidate skill precision
      3: experience fulfillment ratio
      4: experience surplus/deficit normalized
      5: lexical token overlap ratio
    """
    cv_lower = cv_text.lower()
    cv_set = {s.strip().lower() for s in cv_skills if s.strip()}
    req_set = {s.strip().lower() for s in jd_req_skills if s.strip()}
    opt_set = {s.strip().lower() for s in jd_opt_skills if s.strip()}

    matched_req = sum(1 for s in req_set if s in cv_set or s in cv_lower)
    matched_opt = sum(1 for s in opt_set if s in cv_set or s in cv_lower)

    req_cov = matched_req / max(1, len(req_set))
    opt_cov = matched_opt / max(1, len(opt_set)) if opt_set else req_cov
    skill_prec = (matched_req + matched_opt) / max(1, len(cv_set)) if cv_set else req_cov

    if jd_req_years <= 0:
        exp_ratio = min(1.0, 0.88 + min(0.12, cv_years * 0.04))
    else:
        exp_ratio = min(1.0, cv_years / jd_req_years)

    exp_diff_norm = max(0.0, min(1.0, (cv_years - jd_req_years + 4.0) / 8.0))

    cv_tokens = set(re.findall(r"[a-zA-ZÀ-ỹ0-9+#.]{2,}", cv_lower))
    jd_tokens = set(re.findall(r"[a-zA-ZÀ-ỹ0-9+#.]{2,}", jd_text.lower()))
    lexical_overlap = len(cv_tokens & jd_tokens) / max(1, len(jd_tokens))

    return np.array(
        [
            min(1.0, req_cov),
            min(1.0, opt_cov),
            min(1.0, skill_prec),
            min(1.0, exp_ratio),
            exp_diff_norm,
            min(1.0, lexical_overlap * 1.5),
        ],
        dtype=np.float32,
    )


class ResidualBlock(nn.Module):
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


class CvJdBiEncoderNet(nn.Module):
    def __init__(self, text_embed_dim: int = 96, latent_dim: int = 64, struct_dim: int = 6):
        super().__init__()
        self.text_embed_dim = text_embed_dim
        self.latent_dim = latent_dim
        self.struct_dim = struct_dim

        # Dual-Tower projection for CV and JD
        self.cv_input = nn.Sequential(
            nn.Linear(text_embed_dim, latent_dim),
            nn.LayerNorm(latent_dim),
            nn.GELU(),
        )
        self.jd_input = nn.Sequential(
            nn.Linear(text_embed_dim, latent_dim),
            nn.LayerNorm(latent_dim),
            nn.GELU(),
        )
        self.shared_encoder = ResidualBlock(latent_dim, dropout=0.12)

        # Interaction vector: [u, v, |u - v|, u * v, cosine_sim(1), struct_features(6)]
        fusion_in_dim = latent_dim * 4 + 1 + struct_dim
        self.fusion_backbone = nn.Sequential(
            nn.Linear(fusion_in_dim, 128),
            nn.LayerNorm(128),
            nn.GELU(),
            nn.Dropout(0.12),
            nn.Linear(128, 64),
            nn.LayerNorm(64),
            nn.GELU(),
        )
        # Multi-task output heads: [overall, skill, experience, keyword]
        self.multi_head = nn.Sequential(
            nn.Linear(64, 4),
            nn.Sigmoid(),
        )

    def encode_cv(self, x_cv: torch.Tensor) -> torch.Tensor:
        h = self.shared_encoder(self.cv_input(x_cv))
        return F.normalize(h, p=2, dim=-1)

    def encode_jd(self, x_jd: torch.Tensor) -> torch.Tensor:
        h = self.shared_encoder(self.jd_input(x_jd))
        return F.normalize(h, p=2, dim=-1)

    def forward(
        self, x_cv: torch.Tensor, x_jd: torch.Tensor, x_struct: torch.Tensor
    ) -> Dict[str, torch.Tensor]:
        u = self.encode_cv(x_cv)
        v = self.encode_jd(x_jd)
        cos_sim = torch.sum(u * v, dim=-1, keepdim=True)  # [-1, 1]
        diff = torch.abs(u - v)
        prod = u * v
        fused = torch.cat([u, v, diff, prod, cos_sim, x_struct], dim=-1)
        hidden = self.fusion_backbone(fused)
        scores = self.multi_head(hidden)
        return {
            "overall": scores[:, 0],
            "skill": scores[:, 1],
            "experience": scores[:, 2],
            "keyword": scores[:, 3],
            "cosine_similarity": cos_sim.squeeze(-1),
        }
