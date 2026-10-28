# syntax=docker/dockerfile:1
# TalentBridge AI Engine — FastAPI + PyTorch (CPU) Bi-Encoder for CV ↔ JD matching
FROM python:3.12-slim

ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    PIP_NO_CACHE_DIR=1

WORKDIR /app

# CPU-only PyTorch wheel keeps the image small (no CUDA libraries)
COPY requirements.txt .
RUN pip install --index-url https://download.pytorch.org/whl/cpu "torch>=2.2.0" \
    && pip install -r requirements.txt

COPY main.py model_arch.py transformer_model.py ./
COPY models ./models

EXPOSE 8000
HEALTHCHECK --interval=20s --timeout=5s --start-period=60s --retries=5 \
  CMD python -c "import urllib.request; urllib.request.urlopen('http://127.0.0.1:8000/docs', timeout=4)" || exit 1

CMD ["python", "main.py"]
