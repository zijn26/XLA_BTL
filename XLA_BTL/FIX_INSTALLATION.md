# Hướng Dẫn Khắc Phục Lỗi Cài Đặt

## Vấn Đề 1: NumPy không tương thích với Python 3.12
**Lỗi**: `numpy==1.24.3` không hỗ trợ Python 3.12 (chỉ hỗ trợ Python 3.8-3.11)

**Giải pháp**: Cập nhật numpy lên version >= 1.26.0

## Vấn Đề 2: OpenCV không tương thích với NumPy 2.x
**Lỗi**: `opencv-python==4.8.1.78` được compile với NumPy 1.x, không tương thích với NumPy 2.x

**Giải pháp**: 
- **Option 1 (Khuyến nghị)**: Upgrade OpenCV lên version >= 4.12.0 (hỗ trợ NumPy 2.x)
- **Option 2**: Downgrade NumPy về < 2.0.0 nếu muốn giữ OpenCV cũ

## Giải Pháp Cuối Cùng (Đã áp dụng)

File `requirements.txt` đã được cập nhật với:
- `opencv-python>=4.12.0` - Hỗ trợ NumPy 2.x
- `numpy>=1.26.0` - Hỗ trợ Python 3.12 và tương thích với OpenCV mới
- `setuptools>=65.0.0` - Cần để build packages

## Cài Đặt

```powershell
# Cài đặt tất cả dependencies
pip install -r requirements.txt
```

## Kiểm Tra

```powershell
# Kiểm tra versions
python -c "import cv2; import numpy; print('OpenCV:', cv2.__version__); print('NumPy:', numpy.__version__)"

# Kiểm tra imports
python -c "import cv2; import numpy as np; import tensorflow as tf; import flask; print('✅ Tất cả imports thành công!')"
```

## Lưu Ý

1. **OpenCV 4.12.0+** yêu cầu **NumPy >= 2.0**, nhưng vẫn tương thích với Python 3.12
2. Nếu gặp lỗi với TensorFlow, có thể cần cập nhật TensorFlow lên version mới hơn
3. Luôn đảm bảo cài `setuptools` và `wheel` trước khi cài các package khác

## Troubleshooting

### Nếu vẫn gặp lỗi với NumPy:
```powershell
pip uninstall numpy
pip install "numpy>=1.26.0"
```

### Nếu vẫn gặp lỗi với OpenCV:
```powershell
pip uninstall opencv-python
pip install "opencv-python>=4.12.0"
```

### Nếu gặp lỗi với TensorFlow:
```powershell
pip install --upgrade tensorflow
```
