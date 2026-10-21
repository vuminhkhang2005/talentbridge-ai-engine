# TalentBridge AI Engine (`TalentBridge-Transformer-BiEncoder v3.0`)

Microservice Học sâu kết hợp **Pre-trained Multilingual Transformer (`118M Parameters`, 12 Encoder Layers)** và **Dual-Tower Cross-Interaction Neural Network** chuyên dụng cho bài toán **Đánh giá mức độ phù hợp ngữ nghĩa giữa Hồ sơ ứng viên (CV) và Bản mô tả công việc (JD)**.

Repository này được tách độc lập khỏi hệ thống Web tuyển dụng chính ([`ai-recruitment-platform`](https://github.com/vuminhkhang2005/ai-recruitment-platform)) để phục vụ huấn luyện mô hình học sâu, đánh giá thực nghiệm khoa học (**Tiểu luận chuyên ngành / Khóa luận tốt nghiệp**) và cung cấp API suy luận thời gian thực qua **FastAPI**.

---

## 1. Cấu trúc Repository

```text
talentbridge-ai-engine/
├── dataset/
│   └── cv_jd_vietnamese_dataset.json        # 1,500 cặp (900 mẫu thật từ HuggingFace + 600 mẫu tiếng Việt song ngữ)
├── models/
│   ├── cv_jd_transformer_biencoder.pt       # Trọng số mạng Dual-Tower & Cross-Interaction Head (PyTorch)
│   ├── tfidf_svd.joblib                     # Pipeline Subword N-Gram SVD bổ trợ
│   └── evaluation_metrics.json              # Kết quả đo lường MAE, RMSE, Pearson, Spearman, Macro F1
├── notebooks/
│   └── Train_Transformer_Colab_GPU.ipynb    # Notebook huấn luyện trên GPU (Google Colab / Google AI Ultra)
├── reports/
│   └── figures/                             # Biểu đồ thực nghiệm 300 DPI phục vụ báo cáo Tiểu luận
│       ├── training_curves.png
│       ├── benchmark_comparison.png
│       └── prediction_scatter.png
├── build_real_dataset.py                    # Tải dữ liệu thật từ HuggingFace Hub + sinh mẫu tiếng Việt đa dạng
├── transformer_model.py                     # Kiến trúc Pre-trained Multilingual Transformer + Dual-Tower Fusion
├── train_transformer.py                     # Huấn luyện & Benchmark 4 phương pháp + xuất biểu đồ khoa học
├── main.py                                  # FastAPI Inference Microservice (chạy tại cổng 8000)
├── RESEARCH_REPORT_TIEU_LUAN.md             # Báo cáo nghiên cứu & bảng thực nghiệm dùng cho Tiểu luận chuyên ngành
└── requirements.txt                         # Danh sách thư viện Python
```

---

## 2. Hướng dẫn Huấn luyện & Đánh giá Thực nghiệm

### Chạy trên máy Local hoặc Server:
```bash
pip install -r requirements.txt
python build_real_dataset.py
python train_transformer.py
```

### Chạy trên Google Colab GPU (Gói Google AI Ultra — A100 / L4 / T4):
Mở trực tiếp file [`notebooks/Train_Transformer_Colab_GPU.ipynb`](./notebooks/Train_Transformer_Colab_GPU.ipynb) trên Google Colab để chạy huấn luyện với bộ tăng tốc phần cứng GPU.

---

## 3. Khởi chạy AI Inference Server (Port `8000`)

```bash
python main.py
```

### Các Endpoint chính:
- `GET http://localhost:8000/health` — Kiểm tra trạng thái nạp trọng số Transformer (`118.4M` parameters).
- `GET http://localhost:8000/api/v1/ai-engine/metrics` — Xem kết quả thực nghiệm so sánh 4 phương pháp trên tập Test (`n = 225`).
- `POST http://localhost:8000/api/v1/ai-engine/predict` — Dự đoán điểm phù hợp đa chiều (`overall_score`, `skill_score`, `experience_score`, `keyword_score`, `cosine_similarity`, `inference_time_ms`).
