# STAT3013 – Patient Experience & Hospital Readmission Risk

**An Integrated Statistical, Machine Learning, and Time-Series Framework for Analyzing Patient Experience and Hospital Readmission Risk**

Dữ liệu CMS (HCAHPS + Unplanned Hospital Visits), phân tích ở cấp bệnh viện.

## Bài toán
- **Target hồi quy:** `READM_30_HF`, tỷ lệ tái nhập viện 30 ngày của bệnh nhân suy tim (%).
- **Target phân loại:** `READM_HF_High` = 1 nếu `READM_30_HF` > 21.3 (21.3 là trung vị của tập train).
- **Feature:** 8 điểm HCAHPS linear score (0–100):

| Cột | Measure ID | Ý nghĩa |
|---|---|---|
| Nurse_Communication | H_COMP_1_LINEAR_SCORE | Giao tiếp của điều dưỡng |
| Doctor_Communication | H_COMP_2_LINEAR_SCORE | Giao tiếp của bác sĩ |
| Medication_Communication | H_COMP_5_LINEAR_SCORE | Giao tiếp về thuốc |
| Discharge_Information | H_COMP_6_LINEAR_SCORE | Thông tin khi xuất viện |
| Cleanliness | H_CLEAN_LINEAR_SCORE | Mức độ sạch sẽ |
| Quietness | H_QUIET_LINEAR_SCORE | Mức độ yên tĩnh |
| Overall_Hospital_Rating | H_HSP_RATING_LINEAR_SCORE | Đánh giá tổng thể bệnh viện |
| Recommend_Hospital | H_RECMND_LINEAR_SCORE | Sẵn lòng giới thiệu bệnh viện |

## Dữ liệu
`data/processed.csv`: 2.850 bệnh viện × 16 cột, không thiếu giá trị. Chi tiết làm sạch trong `data/README_cleaning.txt`.
- Split cố định qua cột `Split`: train 2.280 / test 570 (random_state=42). **Không chia lại.**
- Đọc Facility ID dạng text: `pd.read_csv(..., dtype={"Facility ID": str})`.

## Cấu trúc repo
```
data/        processed.csv, README_cleaning.txt
notebooks/   notebook tiền xử lý, EDA, ...
src/         evaluate.py (dùng chung) + code từng model
models/      file .pkl (gửi TV4 làm SHAP)
results/     bảng chỉ số, kết quả CV, điểm ROC
figures/     hình cho báo cáo, mỗi model 1 thư mục con
report/      Full Report, IEEE paper, slides
```

## Hàm dùng chung – `src/evaluate.py`
Cả 4 model phân loại dùng cùng dữ liệu, cùng split, cùng bộ chỉ số:
```python
import sys; sys.path.insert(0, "src")
from evaluate import load_data, scale, evaluate, save_roc_points

X_train, X_test, y_train, y_test = load_data()          # phân loại
# X_train, X_test, y_train, y_test = load_data("READM_30_HF")   # hồi quy

model.fit(X_train, y_train)                              # KNN/MLP: chuẩn hóa trước (scale hoặc Pipeline)
res = evaluate(model, X_test, y_test, "Random Forest")   # Accuracy, Precision, Recall, F1, ROC-AUC, TN/FP/FN/TP
save_roc_points(model, X_test, y_test, "results/rf_roc_points.csv")   # cho ROC chung
```
Quy ước tên file: `src/<model>.py`, `models/<model>.pkl`, `results/<model>_test_metrics.csv`, `results/<model>_roc_points.csv`, `figures/<model>/`.

## Phân công
| TV | Model ML | Phần khác |
|---|---|---|
| TV1 | Random Forest | Thống kê mô tả, kiểm định |
| TV2 | XGBoost | Hồi quy |
| TV3 | MLP | ARIMA |
| TV4 | SHAP (giải thích RF/XGBoost) | SEM, bảng so sánh model |
| TV5 | KNN | Tiền xử lý, evaluate.py, repo, ghép báo cáo |

## Kết quả trên test set (570 bệnh viện)
| Model | Accuracy | Precision | Recall | F1 | ROC-AUC |
|---|---|---|---|---|---|
| Random Forest | | | | | |
| XGBoost | | | | | |
| MLP | | | | | |
| KNN (K=241, distance, Manhattan) | 0.591 | 0.571 | 0.433 | 0.492 | 0.600 |

## Chạy lại
```bash
pip install -r requirements.txt
python src/evaluate.py     # kiểm tra dữ liệu và split
python src/knn.py          # KNN: ~10–15 phút (CV 5 fold × 10 lần)
```

## Lưu ý
- Phân tích cấp bệnh viện, không dự đoán cho từng bệnh nhân.
- Kỳ dữ liệu khác nhau: HCAHPS 10/2024–09/2025, tái nhập viện 07/2023–06/2025.
- 8 feature chỉ tương quan −0.06 đến −0.20 với target, nên ROC-AUC khoảng 0.55–0.65 là bình thường.
