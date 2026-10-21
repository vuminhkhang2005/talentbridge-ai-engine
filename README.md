# TalentBridge AI Engine (`TalentBridge-BiEncoder v2.1`)

Microservice Học sâu (**PyTorch Dual-Tower Bi-Encoder + Cross-Interaction Neural Network**) chuyên dụng cho bài toán **Đánh giá mức độ phù hợp ngữ nghĩa giữa Hồ sơ ứng viên (CV) và Bản mô tả công việc (JD) tiếng Việt**.

Repository này được tách độc lập khỏi hệ thống Web tuyển dụng chính ([`ai-recruitment-platform`](https://github.com/vuminhkhang2005/ai-recruitment-platform)) để phục vụ huấn luyện mô hình, đánh giá thực nghiệm khoa học (**Tiểu luận chuyên ngành / Khóa luận tốt nghiệp**) và cung cấp API suy luận thời gian thực qua **FastAPI**.

---

## 1. Cấu trúc Repository

```text
talentbridge-ai-engine/
├── dataset/
│   └── cv_jd_vietnamese_dataset.json   # Bộ dữ liệu 360 cặp (CV, JD) song ngữ Việt - Anh thuộc 8 lĩnh vực
├── models/
│   ├── cv_jd_biencoder.pt              # Trọng số mô hình PyTorch (CvJdBiEncoderNet) đã huấn luyện
│   ├── tfidf_svd.joblib                # Pipeline biểu diễn Word+Subword N-Gram TF-IDF + SVD
│   └── evaluation_metrics.json         # Kết quả đo lường MAE, RMSE, Pearson, Spearman, Macro F1
├── generate_dataset.py                 # Trình sinh & gán nhãn bộ dữ liệu CV-JD chuẩn hóa
├── model_arch.py                       # Kiến trúc mạng nơ-ron PyTorch (Residual Dual-Tower + Fusion Head)
├── train_and_evaluate.py               # Huấn luyện đa nhiệm (Multi-Task Huber Loss) & Benchmark 2 Baselines
├── main.py                             # FastAPI Inference Microservice (chạy tại cổng 8000)
├── RESEARCH_REPORT_TIEU_LUAN.md        # Báo cáo nghiên cứu & bảng thực nghiệm dùng cho Tiểu luận chuyên ngành
└── requirements.txt                    # Danh sách thư viện Python
```

---

## 2. Hướng dẫn Huấn luyện & Đánh giá Thực nghiệm

```bash
pip install -r requirements.txt
python train_and_evaluate.py
```

Quá trình huấn luyện sẽ tự động:
1. Sinh và nạp bộ dữ liệu `360` cặp `(CV, JD)` tiếng Việt chia theo tỷ lệ Stratified `70% Train / 15% Val / 15% Test`.
2. Huấn luyện mạng `CvJdBiEncoderNet` bằng thuật toán tối ưu `AdamW` + `CosineAnnealingLR`.
3. Chạy thực nghiệm so sánh với **Baseline 1 (Lexical Keyword Overlap)** và **Baseline 2 (TF-IDF Cosine Similarity)**.
4. Xuất trọng số ra `models/cv_jd_biencoder.pt` và báo cáo khoa học tại [`RESEARCH_REPORT_TIEU_LUAN.md`](./RESEARCH_REPORT_TIEU_LUAN.md).

---

## 3. Khởi chạy AI Inference Server (Port `8000`)

```bash
python main.py
```

### Các Endpoint chính:
- `GET http://localhost:8000/health` — Kiểm tra trạng thái nạp trọng số PyTorch.
- `GET http://localhost:8000/api/v1/ai-engine/metrics` — Xem kết quả thực nghiệm trên tập Test.
- `POST http://localhost:8000/api/v1/ai-engine/predict` — Dự đoán điểm phù hợp đa chiều (`overall_score`, `skill_score`, `experience_score`, `keyword_score`, `cosine_similarity`, `inference_time_ms`).
