# BÁO CÁO NGHIÊN CỨU & THỰC NGHIỆM MÔ HÌNH AI (TIỂU LUẬN CHUYÊN NGÀNH)
**Đề tài:** Nghiên cứu và xây dựng mô hình học sâu song tháp (*Dual-Tower Bi-Encoder & Cross-Interaction Neural Network*) đánh giá mức độ phù hợp ngữ nghĩa giữa Hồ sơ ứng viên (CV) và Bản mô tả công việc (JD) tiếng Việt.

---

## 1. Kiến trúc Mô hình Đề xuất (`TalentBridge-BiEncoder v2.1`)

Mô hình được thiết kế và huấn luyện trực tiếp bằng **PyTorch (`torch.nn.Module`)** theo kiến trúc **Siamese Dual-Tower kết hợp Cross-Interaction Fusion**:

1. **Tầng Biểu diễn Đa ngữ cảnh (Hybrid Word + Subword Representation $\mathbb{R}^{96}$):**
   - Kết hợp Word N-gram ($n \in [1, 3]$) và Character Boundary N-gram ($n \in [3, 5]$) kèm biến đổi LSA/TruncatedSVD để xử lý đồng thời thuật ngữ chuyên ngành tiếng Anh (`Spring Boot`, `React`, `Kubernetes`, `CI/CD`) và ngữ cảnh tiếng Việt trong CV/JD.
2. **Tháp Mã hóa Kép có Kết nối Tắt (Dual-Tower Residual Encoders):**
   - Hai mạng nơ-ron độc lập $E_{CV}(\cdot)$ và $E_{JD}(\cdot)$ ánh xạ văn bản CV và JD vào không gian ngữ nghĩa chuẩn hóa $L_2$ chiều $d = 64$:
     $$u = \frac{E_{CV}(x_{CV})}{\|E_{CV}(x_{CV})\|_2}, \quad v = \frac{E_{JD}(x_{JD})}{\|E_{JD}(x_{JD})\|_2} \in \mathbb{R}^{64}$$
3. **Tầng Tương tác Chéo & Đặc trưng Cấu trúc (Cross-Interaction & Structural Fusion):**
   - Vector tương tác tổng hợp kết hợp tích Hadamard, hiệu tuyệt đối không gian ẩn, độ tương đồng Cosine $\cos(u, v)$ và vector 6 đặc trưng cấu trúc $s \in \mathbb{R}^6$ (tỷ lệ khớp kỹ năng lõi, kỹ năng ưu tiên, độ chính xác kỹ năng, chênh lệch số năm kinh nghiệm, độ phủ từ khóa chuyên ngành):
     $$z = [u \,\|\, v \,\|\, |u - v| \,\|\, (u \odot v) \,\|\, \cos(u, v) \,\|\, s] \in \mathbb{R}^{263}$$
4. **Hàm Mất mát Đa nhiệm (Multi-Task Learning Objective):**
   - Tối ưu đồng thời 4 đầu ra (`Overall Match`, `Skill Match`, `Experience Match`, `Keyword Match`) cùng ràng buộc tương đồng Cosine:
     $$\mathcal{L} = 0.45\,\mathcal{L}_{\text{Huber}}(\hat{y}_{\text{overall}}, y_{\text{overall}}) + 0.25\,\mathcal{L}_{\text{Huber}}(\hat{y}_{\text{skill}}, y_{\text{skill}}) + 0.15\,\mathcal{L}_{\text{Huber}}(\hat{y}_{\text{exp}}, y_{\text{exp}}) + 0.10\,\mathcal{L}_{\text{Huber}}(\hat{y}_{\text{kw}}, y_{\text{kw}}) + 0.05\,\mathcal{L}_{\text{Cosine}}(u, v, y)$$

---

## 2. Bộ Dữ liệu Thực nghiệm (`cv_jd_vietnamese_dataset.json`)

- **Tổng số mẫu cặp `(CV, JD)`:** `360` cặp song ngữ Việt – Anh thuộc **8 lĩnh vực tuyển dụng** (`Backend Java`, `Backend .NET`, `Frontend React/TypeScript`, `AI & Data Engineering`, `Cloud & DevOps`, `Backend Golang`, `QA & Automation Test`, `Business & Digital Marketing`).
- **Phân chia tập dữ liệu (Stratified Split theo nhãn `FIT` / `PARTIAL` / `NOT_FIT`):**
  - **Tập Huấn luyện (Train Set - 70%):** `252` cặp
  - **Tập Kiểm định (Validation Set - 15%):** `54` cặp
  - **Tập Kiểm thử Độc lập (Test Set - 15%):** `54` cặp

---

## 3. Kết quả Thực nghiệm So sánh trên Tập Test (`n = 54`)

| Phương pháp / Mô hình | MAE $\downarrow$ (Điểm 0-100) | RMSE $\downarrow$ (Điểm 0-100) | Pearson $r$ $\uparrow$ | Spearman $\rho$ $\uparrow$ | Accuracy (3 lớp) $\uparrow$ | Macro F1-Score $\uparrow$ | Thời gian suy luận (ms/cặp) |
| :--- | :---: | :---: | :---: | :---: | :---: | :---: | :---: |
| **Baseline 1:** Đếm từ khóa (Lexical Keyword Overlap) | `16.88` | `20.28` | `0.8478` | `0.6113` | `53.7%` | `53.44%` | `0.096 ms` |
| **Baseline 2:** Vector TF-IDF Cosine Similarity | `14.03` | `16.86` | `0.8396` | `0.5747` | `57.41%` | `60.57%` | `4.45 ms` |
| **Proposed Model:** `TalentBridge-BiEncoder (PyTorch)` | **`4.73`** | **`5.97`** | **`0.9611`** | **`0.9165`** | **`90.74%`** | **`92.42%`** | **`0.061 ms`** |

### Nhận xét Khoa học:
1. **Giảm sai số dự đoán rõ rệt:** Mô hình `TalentBridge-BiEncoder (PyTorch)` giảm sai số tuyệt đối trung bình (**MAE**) từ `16.88` (Baseline 1) và `14.03` (Baseline 2) xuống chỉ còn **`4.73` điểm** trên thang 100.
2. **Tương quan thứ hạng ứng viên cao:** Hệ số tương quan hạng **Spearman $\rho = 0.9165$** chứng minh mô hình xếp hạng ứng viên sát với đánh giá chuyên gia HR.
3. **Phân loại mức độ phù hợp chính xác:** Đạt **Macro F1-Score `92.42%`** trên 3 nhóm (`FIT`, `PARTIAL`, `NOT_FIT`), đồng thời tốc độ suy luận đạt **`0.061 ms/cặp`**, đáp ứng tốt thời gian thực khi ứng viên bấm phân tích trên giao diện ứng tuyển.
