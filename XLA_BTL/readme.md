<!-- dữ liệu trả về từ api có dạng : -->
{
    "success": true,
    "top_predictions": [
        {
            "class": "Corn",
            "confidence": 95.5
        },
        {
            "class": "Apple",
            "confidence": 3.2
        },
        {
            "class": "Banana",
            "confidence": 1.0
        },
        {
            "class": "Orange",
            "confidence": 0.2
        },
        {
            "class": "Grape",
            "confidence": 0.1
        }
    ],
    "prediction": "Corn",  // Kết quả cao nhất (để tương thích)
    "confidence": 95.5,      // Độ tin cậy cao nhất (để tương thích)
    "processed_images": {...},
    "message": "Dự đoán tốt nhất: Corn với độ tin cậy 95.5%"
ví dụ cách gọi APi : // Ví dụ với JavaScript/Fetch
const formData = new FormData();
formData.append('file', imageFile);

fetch('http://localhost:5000/predict', {
    method: 'POST',
    body: formData
})
.then(response => response.json())
.then(data => {
    console.log('Kết quả:', data);
    // data.prediction: tên loại trái cây
    // data.confidence: độ tin cậy (%)
    // data.processed_images: các ảnh đã xử lý (base64)
});

<!-- Cách dùng  -->
Cách sử dụng:
Cài đặt dependencies:
cd XLA_BTL
pip install -r requirements.txt
Chạy server:
python app.py