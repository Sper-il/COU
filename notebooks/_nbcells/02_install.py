# === 1. Cài đặt môi trường ===

# KHÔNG cài lại torch: Colab đã có sẵn torch build sẵn cho CUDA. Cài lại sẽ
# làm mất hỗ trợ GPU và notebook sẽ chạy rất chậm.
!pip -q install -q pandas numpy scikit-learn matplotlib PyYAML joblib seaborn

import torch
print("torch:", torch.__version__)
print("cuda khả dụng:", torch.cuda.is_available())
if torch.cuda.is_available():
    print("GPU:", torch.cuda.get_device_name(0))
    print("Dung lượng GPU:", round(torch.cuda.get_device_properties(0).total_memory / 1024**3, 1), "GB")
else:
    print("CẢNH BÁO: không có GPU. Notebook sẽ chạy được nhưng rất lâu.")
