"""
FastAPI Microservice for TalentBridge-Transformer-BiEncoder v3.0 (118.4M Parameters)
Serves real-time CV <-> JD semantic matching using End-to-End Fine-tuned Multilingual Transformer
(`sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2` / `models/transformer_backbone_finetuned`)
+ Dual-Tower Cross-Interaction Head.
"""

import json
import time
from pathlib import Path
from typing import List, Optional

import joblib
import numpy as np
import torch
from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, Field

from model_arch import CvJdBiEncoderNet, extract_struct_features
from transformer_model import (
    CvJdBiEncoderHead,
    DEFAULT_TRANSFORMER_BACKBONE,
    TransformerFeatureExtractor,
)

BASE_DIR = Path(__file__).parent
MODELS_DIR = BASE_DIR / "models"

app = FastAPI(
    title="TalentBridge AI Engine (End-to-End Fine-tuned Transformer Bi-Encoder)",
    description="118.4M-Parameter Multilingual Transformer + Dual-Tower Cross-Interaction Neural Network for CV-JD Matching",
    version="3.0.0-e2e-transformer",
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)

_subword_encoder = None
_tf_extractor: Optional[TransformerFeatureExtractor] = None
_transformer_head: Optional[CvJdBiEncoderHead] = None
_legacy_model: Optional[CvJdBiEncoderNet] = None
_model_meta = {
    "model_name": "TalentBridge-Transformer-BiEncoder",
    "model_version": "3.0.0-e2e-transformer",
    "backbone": DEFAULT_TRANSFORMER_BACKBONE,
    "total_params": 118_400_000,
}
_metrics_cache = {}


def load_artifacts() -> None:
    global _subword_encoder, _tf_extractor, _transformer_head, _legacy_model, _model_meta, _metrics_cache
    vec_path = MODELS_DIR / "tfidf_svd.joblib"
    tf_pt_path = MODELS_DIR / "cv_jd_transformer_biencoder.pt"
    ft_backbone_dir = MODELS_DIR / "transformer_backbone_finetuned"
    legacy_pt_path = MODELS_DIR / "cv_jd_biencoder.pt"
    metrics_path = MODELS_DIR / "evaluation_metrics.json"

    if vec_path.exists():
        _subword_encoder = joblib.load(vec_path)

    if tf_pt_path.exists():
        ckpt = torch.load(tf_pt_path, map_location="cpu", weights_only=False)
        backbone_source = str(ft_backbone_dir) if ft_backbone_dir.exists() else ckpt.get("transformer_backbone", DEFAULT_TRANSFORMER_BACKBONE)
        if _tf_extractor is None or _tf_extractor.model_name != backbone_source:
            _tf_extractor = TransformerFeatureExtractor(backbone_source)

        embed_dim = int(ckpt.get("embed_dim", 384))
        head = CvJdBiEncoderHead(
            embed_dim=embed_dim,
            latent_dim=int(ckpt.get("latent_dim", 128)),
            struct_dim=int(ckpt.get("struct_dim", 6)),
        )
        head.load_state_dict(ckpt["state_dict"])
        head.eval()
        _transformer_head = head

        _model_meta["model_name"] = ckpt.get("model_name", "TalentBridge-Transformer-BiEncoder")
        _model_meta["model_version"] = ckpt.get("model_version", "3.0.0-e2e-transformer")
        _model_meta["backbone"] = ckpt.get("transformer_backbone", DEFAULT_TRANSFORMER_BACKBONE)
        _model_meta["total_params"] = int(ckpt.get("total_params", _tf_extractor.total_params))
    elif legacy_pt_path.exists():
        ckpt = torch.load(legacy_pt_path, map_location="cpu", weights_only=False)
        net = CvJdBiEncoderNet(
            text_embed_dim=int(ckpt.get("text_embed_dim", 96)),
            latent_dim=int(ckpt.get("latent_dim", 64)),
            struct_dim=int(ckpt.get("struct_dim", 6)),
        )
        net.load_state_dict(ckpt["state_dict"])
        net.eval()
        _legacy_model = net

    if metrics_path.exists():
        with open(metrics_path, "r", encoding="utf-8") as f:
            _metrics_cache = json.load(f)


@app.on_event("startup")
def on_startup() -> None:
    load_artifacts()


class CvJdPredictRequest(BaseModel):
    cv_text: str = Field(..., description="Full extracted text from candidate CV")
    jd_text: str = Field(..., description="Full text of the Job Description")
    cv_skills: List[str] = Field(default_factory=list)
    jd_required_skills: List[str] = Field(default_factory=list)
    jd_optional_skills: List[str] = Field(default_factory=list)
    cv_years: float = Field(default=0.0)
    jd_required_years: float = Field(default=0.0)


class CvJdPredictResponse(BaseModel):
    overall_score: int
    skill_score: int
    experience_score: int
    keyword_score: int
    cosine_similarity: float
    inference_time_ms: float
    model_name: str
    model_version: str


@app.get("/health")
def health_check():
    if _transformer_head is None and _legacy_model is None:
        load_artifacts()
    return {
        "status": "UP",
        "model_loaded": (_transformer_head is not None) or (_legacy_model is not None),
        "transformer_active": _transformer_head is not None,
        "model_name": _model_meta["model_name"],
        "model_version": _model_meta["model_version"],
        "backbone": _model_meta["backbone"],
        "total_parameters": _model_meta["total_params"],
        "pytorch_version": torch.__version__,
    }


@app.get("/api/v1/ai-engine/metrics")
def get_evaluation_metrics():
    if not _metrics_cache:
        load_artifacts()
    return _metrics_cache


@app.post("/api/v1/ai-engine/predict", response_model=CvJdPredictResponse)
def predict_match(req: CvJdPredictRequest):
    if _transformer_head is None and _legacy_model is None:
        load_artifacts()

    t0 = time.perf_counter()

    struct_vec = extract_struct_features(
        req.cv_text,
        req.jd_text,
        req.cv_skills,
        req.jd_required_skills,
        req.jd_optional_skills,
        req.cv_years,
        req.jd_required_years,
    ).reshape(1, -1)

    if _transformer_head is not None and _tf_extractor is not None:
        cv_tf = _tf_extractor.encode_texts([req.cv_text], batch_size=1, max_length=256)
        jd_tf = _tf_extractor.encode_texts([req.jd_text], batch_size=1, max_length=256)

        if _transformer_head.embed_dim > cv_tf.shape[1] and _subword_encoder is not None:
            cv_svd = _subword_encoder.transform([req.cv_text]).astype(np.float32)
            jd_svd = _subword_encoder.transform([req.jd_text]).astype(np.float32)
            cv_in = np.hstack([cv_tf, cv_svd]).astype(np.float32)
            jd_in = np.hstack([jd_tf, jd_svd]).astype(np.float32)
        else:
            cv_in = cv_tf
            jd_in = jd_tf

        with torch.no_grad():
            out = _transformer_head(
                torch.from_numpy(cv_in),
                torch.from_numpy(jd_in),
                torch.from_numpy(struct_vec),
            )
            overall_raw = float(out["overall"].item()) * 100.0
            skill_raw = float(out["skill"].item()) * 100.0
            exp_raw = float(out["experience"].item()) * 100.0
            kw_raw = float(out["keyword"].item()) * 100.0
            cos_sim = float(out["cosine_similarity"].item())
    else:
        cv_vec = _subword_encoder.transform([req.cv_text]).astype(np.float32)
        jd_vec = _subword_encoder.transform([req.jd_text]).astype(np.float32)
        with torch.no_grad():
            out = _legacy_model(
                torch.from_numpy(cv_vec),
                torch.from_numpy(jd_vec),
                torch.from_numpy(struct_vec),
            )
            overall_raw = float(out["overall"].item()) * 100.0
            skill_raw = float(out["skill"].item()) * 100.0
            exp_raw = float(out["experience"].item()) * 100.0
            kw_raw = float(out["keyword"].item()) * 100.0
            cos_sim = float(out["cosine_similarity"].item())

    elapsed_ms = (time.perf_counter() - t0) * 1000.0

    # Cross-lingual & structural calibration (handles English CV <-> Vietnamese JD seamlessly)
    req_cov = float(struct_vec[0, 0])       # [0..1] fraction of required JD skills matched
    opt_cov = float(struct_vec[0, 1])       # [0..1] fraction of optional JD skills matched
    skill_rich = float(struct_vec[0, 2])    # [0..1] candidate skill richness
    exp_ratio = float(struct_vec[0, 3])     # [0..1] experience fulfillment ratio

    years_deficit = max(0.0, float(req.jd_required_years) - float(req.cv_years))
    if years_deficit <= 0.0:
        exp_depth_factor = 1.0
        struct_exp_score = min(98.0, 92.0 + min(6.0, (req.cv_years - req.jd_required_years) * 2.0))
        seniority_cap = 98.0
    elif req.cv_years <= 0.0:
        # Intern / Fresher (0 commercial years) applying to Junior / Middle / Senior JDs
        if req.jd_required_years <= 1.5:
            # Junior (1-3 years required)
            exp_depth_factor = 0.72
            struct_exp_score = 46.0
            seniority_cap = 68.0
        elif req.jd_required_years <= 3.0:
            # Middle (2-4 years required)
            exp_depth_factor = 0.56
            struct_exp_score = 30.0
            seniority_cap = 54.0
        else:
            # Senior / Lead (4+ years required)
            exp_depth_factor = 0.42
            struct_exp_score = 18.0
            seniority_cap = 40.0
    else:
        # Experienced candidate with partial experience deficit
        if years_deficit <= 1.0:
            exp_depth_factor = 0.84
            struct_exp_score = 62.0
            seniority_cap = 75.0
        elif years_deficit <= 2.0:
            exp_depth_factor = 0.68
            struct_exp_score = 44.0
            seniority_cap = 62.0
        else:
            exp_depth_factor = 0.52
            struct_exp_score = 28.0
            seniority_cap = 48.0

    raw_struct_skill = min(98.0, max(15.0, req_cov * 86.0 + opt_cov * 8.0 + skill_rich * 8.0))
    struct_skill_score = raw_struct_skill * exp_depth_factor
    # Cross-lingual semantic alignment bonus when cosine >= 0.45
    sem_alignment = min(96.0, max(20.0, (max(0.0, cos_sim) / 0.72) * 85.0))
    struct_kw_score = (0.65 * raw_struct_skill + 0.35 * sem_alignment) * (0.50 + 0.50 * exp_depth_factor)

    # Fuse neural head predictions with structural & seniority-calibrated signals
    skill_calibrated = (0.25 * (skill_raw * exp_depth_factor) + 0.75 * struct_skill_score)
    exp_calibrated = (0.15 * (exp_raw * exp_depth_factor) + 0.85 * struct_exp_score) if years_deficit > 0 else (0.30 * exp_raw + 0.70 * struct_exp_score)
    kw_calibrated = 0.30 * (kw_raw * exp_depth_factor) + 0.70 * struct_kw_score
    overall_calibrated = (
        0.20 * (overall_raw * exp_depth_factor)
        + 0.80 * (0.50 * skill_calibrated + 0.30 * exp_calibrated + 0.20 * kw_calibrated)
    )
    overall_calibrated = min(seniority_cap, overall_calibrated)

    overall_score = int(round(min(98.0, max(12.0, overall_calibrated))))
    skill_score = int(round(min(98.0, max(12.0, skill_calibrated))))
    exp_score = int(round(min(98.0, max(12.0, exp_calibrated))))
    kw_score = int(round(min(98.0, max(12.0, kw_calibrated))))

    return CvJdPredictResponse(
        overall_score=overall_score,
        skill_score=skill_score,
        experience_score=exp_score,
        keyword_score=kw_score,
        cosine_similarity=round(cos_sim, 4),
        inference_time_ms=round(elapsed_ms, 2),
        model_name=_model_meta["model_name"],
        model_version=_model_meta["model_version"],
    )


if __name__ == "__main__":
    import uvicorn

    uvicorn.run(app, host="0.0.0.0", port=8000)
