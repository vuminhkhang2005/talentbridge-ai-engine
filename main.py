"""
FastAPI Microservice for TalentBridge-BiEncoder (PyTorch) Inference
Serves real-time CV <-> JD semantic matching for the Spring Boot backend on port 8000.
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

BASE_DIR = Path(__file__).parent
MODELS_DIR = BASE_DIR / "models"

app = FastAPI(
    title="TalentBridge AI Engine (PyTorch Bi-Encoder)",
    description="Dual-Tower Siamese Neural Network + Cross-Interaction Fusion for Vietnamese CV-JD Matching",
    version="2.1.0-pytorch",
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)

# Global model state loaded on startup
_text_encoder = None
_pytorch_model: Optional[CvJdBiEncoderNet] = None
_model_meta = {
    "model_name": "TalentBridge-BiEncoder",
    "model_version": "2.1.0-pytorch",
}
_metrics_cache = {}


def load_artifacts() -> None:
    global _text_encoder, _pytorch_model, _model_meta, _metrics_cache
    vec_path = MODELS_DIR / "tfidf_svd.joblib"
    pt_path = MODELS_DIR / "cv_jd_biencoder.pt"
    metrics_path = MODELS_DIR / "evaluation_metrics.json"

    if vec_path.exists() and pt_path.exists():
        _text_encoder = joblib.load(vec_path)
        ckpt = torch.load(pt_path, map_location="cpu", weights_only=False)
        _model_meta["model_name"] = ckpt.get("model_name", "TalentBridge-BiEncoder")
        _model_meta["model_version"] = ckpt.get("model_version", "2.1.0-pytorch")

        net = CvJdBiEncoderNet(
            text_embed_dim=int(ckpt.get("text_embed_dim", 96)),
            latent_dim=int(ckpt.get("latent_dim", 64)),
            struct_dim=int(ckpt.get("struct_dim", 6)),
        )
        net.load_state_dict(ckpt["state_dict"])
        net.eval()
        _pytorch_model = net

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
    if _pytorch_model is None:
        load_artifacts()
    return {
        "status": "UP",
        "model_loaded": _pytorch_model is not None,
        "model_name": _model_meta["model_name"],
        "model_version": _model_meta["model_version"],
        "pytorch_version": torch.__version__,
    }


@app.get("/api/v1/ai-engine/metrics")
def get_evaluation_metrics():
    if not _metrics_cache:
        load_artifacts()
    return _metrics_cache


@app.post("/api/v1/ai-engine/predict", response_model=CvJdPredictResponse)
def predict_match(req: CvJdPredictRequest):
    if _pytorch_model is None or _text_encoder is None:
        load_artifacts()

    t0 = time.perf_counter()

    cv_vec = _text_encoder.transform([req.cv_text]).astype(np.float32)
    jd_vec = _text_encoder.transform([req.jd_text]).astype(np.float32)
    struct_vec = extract_struct_features(
        req.cv_text,
        req.jd_text,
        req.cv_skills,
        req.jd_required_skills,
        req.jd_optional_skills,
        req.cv_years,
        req.jd_required_years,
    ).reshape(1, -1)

    with torch.no_grad():
        x_cv = torch.from_numpy(cv_vec)
        x_jd = torch.from_numpy(jd_vec)
        x_st = torch.from_numpy(struct_vec)
        out = _pytorch_model(x_cv, x_jd, x_st)

        overall_raw = float(out["overall"].item()) * 100.0
        skill_raw = float(out["skill"].item()) * 100.0
        exp_raw = float(out["experience"].item()) * 100.0
        kw_raw = float(out["keyword"].item()) * 100.0
        cos_sim = float(out["cosine_similarity"].item())

    elapsed_ms = (time.perf_counter() - t0) * 1000.0

    # Clamp to realistic ATS bounds [10..98]
    overall_score = int(round(min(98.0, max(10.0, overall_raw))))
    skill_score = int(round(min(98.0, max(10.0, skill_raw))))
    exp_score = int(round(min(98.0, max(10.0, exp_raw))))
    kw_score = int(round(min(98.0, max(10.0, kw_raw))))

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
