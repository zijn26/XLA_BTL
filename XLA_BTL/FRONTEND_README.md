# Frontend - Nhận Diện Trái Cây

## Mô tả
Frontend đẹp và hiện đại cho ứng dụng nhận diện trái cây sử dụng AI/Deep Learning.

## Tính năng
- ✅ Upload ảnh bằng cách kéo thả hoặc click chọn
- ✅ Preview ảnh trước khi upload
- ✅ Hiển thị kết quả dự đoán chính với độ tin cậy
- ✅ Hiển thị top 5 dự đoán với ranking
- ✅ Hiển thị tất cả các bước xử lý ảnh (6 bước)
- ✅ Giao diện responsive, đẹp mắt
- ✅ Loading indicator khi xử lý
- ✅ Error handling với thông báo rõ ràng

## Cách sử dụng

### 1. Khởi động Backend
Đảm bảo backend Flask đang chạy:
```bash
python app.py
```
Server sẽ chạy tại: `http://localhost:5000`

### 2. Mở Frontend
Có 2 cách:

**Cách 1: Mở trực tiếp file HTML**
- Double-click vào file `index.html`
- Hoặc mở bằng trình duyệt

**Cách 2: Sử dụng local server (khuyến nghị)**
```bash
# Sử dụng Python
python -m http.server 8000

# Hoặc sử dụng Node.js
npx http-server -p 8000
```
Sau đó mở trình duyệt và truy cập: `http://localhost:8000`

### 3. Sử dụng
1. Click vào nút "Chọn Ảnh" hoặc kéo thả ảnh vào vùng upload
2. Chờ hệ thống xử lý (sẽ hiển thị loading)
3. Xem kết quả dự đoán và các bước xử lý ảnh

## Cấu trúc File
```
├── index.html          # File HTML chính
├── styles.css          # File CSS styling
├── script.js           # File JavaScript xử lý logic
└── FRONTEND_README.md  # File hướng dẫn này
```

## Các thành phần chính

### index.html
- Cấu trúc HTML của trang web
- Chứa các section: upload, loading, results

### styles.css
- Styling đẹp mắt với gradient background
- Responsive design cho mobile
- Animations và transitions mượt mà

### script.js
- Xử lý upload file (drag & drop + click)
- Gọi API `/predict` từ backend
- Hiển thị kết quả và xử lý lỗi

## API Integration
Frontend gọi API tại:
- **Endpoint**: `http://localhost:5000/predict`
- **Method**: POST
- **Body**: FormData với key `file`
- **Response**: JSON với format:
  ```json
  {
    "success": true,
    "top_predictions": [...],
    "prediction": "Corn",
    "confidence": 95.5,
    "processed_images": {...},
    "message": "..."
  }
  ```

## Lưu ý
- Đảm bảo backend đang chạy trước khi sử dụng frontend
- Nếu gặp lỗi CORS, kiểm tra xem backend đã bật CORS chưa (đã có trong code)
- Có thể cần điều chỉnh `API_URL` trong `script.js` nếu backend chạy ở port khác

## Browser Support
- Chrome/Edge (khuyến nghị)
- Firefox
- Safari
- Opera

## Tùy chỉnh
Bạn có thể tùy chỉnh:
- Màu sắc trong `styles.css`
- API URL trong `script.js`
- Thêm/bớt emoji cho các loại trái cây trong `script.js`

