import joblib
import shap
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

# 1. Nạp mô hình Random Forest đã huấn luyện (rf.pkl)
model = joblib.load('rf.pkl')

features = [
    'Nurse_Communication',
    'Doctor_Communication',
    'Medication_Communication',
    'Discharge_Information',
    'Cleanliness',
    'Quietness',
    'Overall_Hospital_Rating',
    'Recommend_Hospital'
]

# Sử dụng danh sách đặc trưng lưu trực tiếp trong mô hình nếu có
if hasattr(model, 'feature_names_in_'):
    features = list(model.feature_names_in_)

# 2. Nạp dữ liệu kiểm thử đã tiền xử lý chung của nhóm (thay đường dẫn file thực tế nếu cần)
# Ví dụ: df_test = pd.read_csv('test_data.csv')
# X_test = df_test[features]
# Nếu chưa có file X_test thực tế, có thể tạm tạo dữ liệu mẫu đúng cấu trúc 8 cột để chạy thử:
rng = np.random.default_rng(42)
X_test = pd.DataFrame({
    'Nurse_Communication': rng.integers(1, 6, size=200),
    'Doctor_Communication': rng.integers(1, 6, size=200),
    'Medication_Communication': rng.integers(1, 6, size=200),
    'Discharge_Information': rng.integers(1, 6, size=200),
    'Cleanliness': rng.integers(1, 6, size=200),
    'Quietness': rng.integers(1, 6, size=200),
    'Overall_Hospital_Rating': rng.integers(0, 11, size=200),
    'Recommend_Hospital': rng.integers(1, 5, size=200)
})[features]

# 3. Khởi tạo SHAP TreeExplainer và tính giá trị SHAP cho lớp nguy cơ cao (class = 1)
explainer = shap.TreeExplainer(model)
shap_values = explainer.shap_values(X_test)

# Chuẩn hóa cấu trúc shap_values theo từng phiên bản thư viện shap
if isinstance(shap_values, list):
    sv_class1 = shap_values[1]
    base_value_1 = explainer.expected_value[1]
elif np.ndim(shap_values) == 3:
    sv_class1 = shap_values[:, :, 1]
    base_value_1 = (
        explainer.expected_value[1]
        if np.ndim(explainer.expected_value) > 0
        else explainer.expected_value
    )
else:
    sv_class1 = shap_values
    base_value_1 = (
        explainer.expected_value
        if np.ndim(explainer.expected_value) == 0
        else explainer.expected_value[0]
    )

# 4. Vẽ và lưu SHAP Bar Plot (Mean |SHAP|)
plt.figure(figsize=(9, 5.5))
shap.summary_plot(sv_class1, X_test, plot_type='bar', show=False)
plt.title('SHAP Bar Plot - Mức độ quan trọng trung bình của đặc trưng')
plt.tight_layout()
plt.savefig('shap_bar_plot.png', dpi=300, bbox_inches='tight')
plt.close()

# 5. Vẽ và lưu SHAP Summary Plot (Beeswarm)
plt.figure(figsize=(10, 6))
shap.summary_plot(sv_class1, X_test, show=False)
plt.title('SHAP Summary Plot (Beeswarm) - Tác động đặc trưng đến dự đoán')
plt.tight_layout()
plt.savefig('shap_summary_plot.png', dpi=300, bbox_inches='tight')
plt.close()

# 6. Vẽ và lưu biểu đồ giải thích dự đoán cá nhân (Waterfall Plot cho mẫu đầu tiên)
sample_idx = 0
exp_single = shap.Explanation(
    values=sv_class1[sample_idx],
    base_values=float(np.asarray(base_value_1).reshape(-1)[0]),
    data=X_test.iloc[sample_idx].values,
    feature_names=features
)

plt.figure(figsize=(9.5, 5.5))
shap.plots.waterfall(exp_single, show=False)
plt.title(f'Giải thích dự đoán cá nhân - Mẫu kiểm thử #{sample_idx}')
plt.tight_layout()
plt.savefig('shap_individual_explanation.png', dpi=300, bbox_inches='tight')
plt.close()

print("Hoàn tất tạo các biểu đồ SHAP: shap_bar_plot.png, shap_summary_plot.png, shap_individual_explanation.png")
