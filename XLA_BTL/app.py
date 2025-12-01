# Fix encoding for Windows console
import sys
import io
if sys.platform == 'win32':
    sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding='utf-8')
    sys.stderr = io.TextIOWrapper(sys.stderr.buffer, encoding='utf-8')

from flask import Flask, request, jsonify
from flask_cors import CORS
import cv2
import numpy as np
import os
import base64
import pickle
import json
import h5py
from tensorflow import keras
import tensorflow as tf
from werkzeug.utils import secure_filename
sys.path.append(os.path.dirname(os.path.abspath(__file__)))
from tienxulyanh import (
    load_image, apply_histogram_equalization, convert_to_grayscale,
    apply_median_filter, apply_canny_edge_detection, segment_by_thresholding
)

app = Flask(__name__)
CORS(app)  # Cho phép CORS để frontend có thể gọi API

# Cấu hình
UPLOAD_FOLDER = 'uploads'
PROCESSED_FOLDER = 'processed_images'
ALLOWED_EXTENSIONS = {'png', 'jpg', 'jpeg', 'gif', 'bmp'}

# Tạo thư mục nếu chưa có
os.makedirs(UPLOAD_FOLDER, exist_ok=True)
os.makedirs(PROCESSED_FOLDER, exist_ok=True)

# Load model và label encoder
MODEL_PATH = 'fruit_recognition_cnn_2channel.h5'
LABEL_ENCODER_PATH = 'label_encoder (1).pkl'

model = None
label_encoder = None

def patch_model_config(model_path):
    """Patch model config để fix lỗi batch_shape và DTypePolicy"""
    try:
        import tempfile
        import shutil
        
        # Tạo file tạm
        temp_path = model_path + '.patched'
        
        # Copy file gốc
        shutil.copy2(model_path, temp_path)
        
        # Đọc và sửa config
        with h5py.File(temp_path, 'r+') as f:
            # Tìm config trong attributes
            if 'model_config' in f.attrs:
                config_str = f.attrs['model_config']
                if isinstance(config_str, bytes):
                    config_str = config_str.decode('utf-8')
                
                config = json.loads(config_str)
                
                # Hàm đệ quy để sửa config
                def fix_config(obj):
                    if isinstance(obj, dict):
                        # Sửa InputLayer - batch_shape
                        if obj.get('class_name') == 'InputLayer' and 'config' in obj:
                            if 'batch_shape' in obj['config']:
                                batch_shape = obj['config'].pop('batch_shape')
                                if batch_shape and len(batch_shape) > 1:
                                    obj['config']['input_shape'] = list(batch_shape[1:])
                        
                        # Sửa dtype - DTypePolicy thành string
                        if 'config' in obj:
                            config_dict = obj['config']
                            if 'dtype' in config_dict:
                                dtype_val = config_dict['dtype']
                                if isinstance(dtype_val, dict):
                                    # Nếu là DTypePolicy object, chuyển thành string
                                    if dtype_val.get('class_name') == 'DTypePolicy':
                                        dtype_name = dtype_val.get('config', {}).get('name', 'float32')
                                        config_dict['dtype'] = dtype_name
                                    elif 'class_name' in dtype_val:
                                        # Các dtype policy khác
                                        dtype_name = dtype_val.get('config', {}).get('name', 'float32')
                                        config_dict['dtype'] = dtype_name
                        
                        # Đệ quy cho các key khác
                        for key, value in obj.items():
                            fix_config(value)
                    elif isinstance(obj, list):
                        for item in obj:
                            fix_config(item)
                
                fix_config(config)
                
                # Ghi lại config đã sửa
                f.attrs['model_config'] = json.dumps(config).encode('utf-8')
                print("✅ Đã patch model config (batch_shape + DTypePolicy)!")
        
        return temp_path
    except Exception as e:
        print(f"⚠️  Không thể patch config: {e}")
        return model_path

def load_model_with_compatibility(model_path):
    """Load model với các cách tương thích khác nhau"""
    # Cách 1: Thử patch config trước rồi load với custom_objects
    try:
        patched_path = patch_model_config(model_path)
        
        # Tạo custom objects để xử lý DTypePolicy và batch_shape
        def fix_dtype_policy(config):
            """Fix DTypePolicy trong config"""
            if isinstance(config, dict) and 'dtype' in config:
                dtype_val = config['dtype']
                if isinstance(dtype_val, dict) and dtype_val.get('class_name') == 'DTypePolicy':
                    dtype_name = dtype_val.get('config', {}).get('name', 'float32')
                    config['dtype'] = dtype_name
            return config
        
        # Custom InputLayer để xử lý batch_shape
        class CompatibleInputLayer(tf.keras.layers.InputLayer):
            @classmethod
            def from_config(cls, config):
                config = fix_dtype_policy(config)
                if 'batch_shape' in config:
                    batch_shape = config.pop('batch_shape')
                    if batch_shape and len(batch_shape) > 1:
                        config['input_shape'] = tuple(batch_shape[1:])
                return super().from_config(config)
        
        # Custom Conv2D và các layer khác để xử lý DTypePolicy
        class CompatibleConv2D(tf.keras.layers.Conv2D):
            @classmethod
            def from_config(cls, config):
                config = fix_dtype_policy(config)
                return super().from_config(config)
        
        custom_objects = {
            'InputLayer': CompatibleInputLayer,
            'Conv2D': CompatibleConv2D,
        }
        
        model = keras.models.load_model(patched_path, compile=False, custom_objects=custom_objects)
        
        # Xóa file patched nếu khác file gốc
        if patched_path != model_path and os.path.exists(patched_path):
            try:
                os.remove(patched_path)
            except:
                pass
        return model, "patched_config_with_custom_objects"
    except Exception as e1:
        print(f"⚠️  Cách 1 (patch + custom) thất bại: {str(e1)[:150]}")
        
        # Cách 2: Load với custom_objects mà không patch
        try:
            # Tạo custom objects để xử lý DTypePolicy
            def create_compatible_layer(base_class):
                class CompatibleLayer(base_class):
                    @classmethod
                    def from_config(cls, config):
                        if isinstance(config, dict) and 'dtype' in config:
                            dtype_val = config['dtype']
                            if isinstance(dtype_val, dict):
                                if dtype_val.get('class_name') == 'DTypePolicy':
                                    config['dtype'] = dtype_val.get('config', {}).get('name', 'float32')
                                elif 'class_name' in dtype_val:
                                    config['dtype'] = dtype_val.get('config', {}).get('name', 'float32')
                        # Fix batch_shape cho InputLayer
                        if base_class == tf.keras.layers.InputLayer and 'batch_shape' in config:
                            batch_shape = config.pop('batch_shape')
                            if batch_shape and len(batch_shape) > 1:
                                config['input_shape'] = tuple(batch_shape[1:])
                        return super().from_config(config)
                return CompatibleLayer
            
            custom_objects = {
                'InputLayer': create_compatible_layer(tf.keras.layers.InputLayer),
                'Conv2D': create_compatible_layer(tf.keras.layers.Conv2D),
                'MaxPooling2D': create_compatible_layer(tf.keras.layers.MaxPooling2D),
                'Dense': create_compatible_layer(tf.keras.layers.Dense),
                'Flatten': create_compatible_layer(tf.keras.layers.Flatten),
                'Dropout': create_compatible_layer(tf.keras.layers.Dropout),
            }
            
            return keras.models.load_model(model_path, compile=False, custom_objects=custom_objects), "custom_objects_only"
        except Exception as e2:
            print(f"⚠️  Cách 2 thất bại: {str(e2)[:150]}")
            
            # Cách 3: Load bình thường (thử lần cuối)
            try:
                return keras.models.load_model(model_path, compile=False), "compile=False"
            except Exception as e3:
                print(f"⚠️  Cách 3 thất bại: {str(e3)[:150]}")
                raise Exception(f"Không thể load model. Lỗi cuối: {e3}")

def load_model_and_encoder():
    """Load model và label encoder khi khởi động app"""
    global model, label_encoder
    try:
        print("Đang load model...")
        
        # Load model với các cách tương thích
        model, method = load_model_with_compatibility(MODEL_PATH)
        print(f"✅ Model loaded successfully (sử dụng: {method})!")
        
        print("Đang load label encoder...")
        with open(LABEL_ENCODER_PATH, 'rb') as f:
            label_encoder = pickle.load(f)
        print("✅ Label encoder loaded successfully!")
        
    except Exception as e:
        print(f"❌ Lỗi khi load model/encoder: {e}")
        import traceback
        traceback.print_exc()
        print("\n" + "="*60)
        print("💡 Gợi ý khắc phục:")
        print("   1. Cài đặt TensorFlow 2.10.1: pip install tensorflow==2.10.1")
        print("   2. Hoặc rebuild model với TensorFlow version mới")
        print("="*60)
        raise

def allowed_file(filename):
    """Kiểm tra file có đúng định dạng không"""
    return '.' in filename and filename.rsplit('.', 1)[1].lower() in ALLOWED_EXTENSIONS

def image_to_base64(image_path):
    """Chuyển ảnh thành base64 string"""
    with open(image_path, 'rb') as f:
        return base64.b64encode(f.read()).decode('utf-8')

def process_image_with_steps(image_path):
    """
    Xử lý ảnh qua các bước giống full_preprocessing_pipeline và lưu kết quả sau mỗi bước.
    Logic giống hệt full_preprocessing_pipeline trong tienxulyanh.py nhưng có thêm chức năng lưu ảnh.
    
    Returns:
        - processed_images: dict chứa các ảnh đã xử lý (base64)
        - final_image: ảnh cuối cùng để dự đoán (2 channel: grayscale + edge)
        - grayscale_image: ảnh xám
        - saved_paths: dict chứa đường dẫn các file đã lưu
    """
    # Bước 0: Đọc ảnh
    original_image = load_image(image_path)
    if original_image is None:
        return None, None, None, None
    
    original_image = cv2.resize(original_image, (100, 100))
    
    # Tạo unique ID cho session này
    import time
    session_id = str(int(time.time() * 1000))
    
    processed_images = {}
    saved_paths = {}
    
    # I. Làm sạch ảnh và Tăng cường chất lượng
    
    # 1. Chuyển ảnh xám
    grayscale_image = convert_to_grayscale(original_image)
    gray_path = os.path.join(PROCESSED_FOLDER, f'{session_id}_1_grayscale.jpg')
    cv2.imwrite(gray_path, grayscale_image)
    processed_images['1_grayscale'] = image_to_base64(gray_path)
    saved_paths['1_grayscale'] = gray_path
    
    # 2. Cân bằng histogram
    histogram_img = apply_histogram_equalization(grayscale_image)
    hist_path = os.path.join(PROCESSED_FOLDER, f'{session_id}_2_histogram.jpg')
    cv2.imwrite(hist_path, histogram_img)
    processed_images['2_histogram'] = image_to_base64(hist_path)
    saved_paths['2_histogram'] = hist_path
    
    # 3. Loại bỏ nhiễu bằng Median Filter
    denoised_image = apply_median_filter(grayscale_image)
    denoise_path = os.path.join(PROCESSED_FOLDER, f'{session_id}_3_denoised.jpg')
    cv2.imwrite(denoise_path, denoised_image)
    processed_images['3_denoised'] = image_to_base64(denoise_path)
    saved_paths['3_denoised'] = denoise_path
    
    # II. Tách đối tượng (Segmentation)
    
    # 4. Phát hiện biên bằng Canny (sử dụng denoised_image)
    edge_image = apply_canny_edge_detection(denoised_image)
    edge_path = os.path.join(PROCESSED_FOLDER, f'{session_id}_4_edges.jpg')
    cv2.imwrite(edge_path, edge_image)
    processed_images['4_edges'] = image_to_base64(edge_path)
    saved_paths['4_edges'] = edge_path
    
    # 5. Phương pháp Otsu (sử dụng grayscale_image)
    otsu_img = segment_by_thresholding(grayscale_image)
    otsu_path = os.path.join(PROCESSED_FOLDER, f'{session_id}_5_otsu.jpg')
    cv2.imwrite(otsu_path, otsu_img)
    processed_images['5_otsu'] = image_to_base64(otsu_path)
    saved_paths['5_otsu'] = otsu_path
    
    # Ảnh đầu vào cho CNN (2 channel: grayscale + edge)
    # Tạo ảnh 2 channel từ grayscale và edge
    final_image = np.stack([grayscale_image, edge_image], axis=-1)
    final_path = os.path.join(PROCESSED_FOLDER, f'{session_id}_6_final_2channel.jpg')
    # Lưu ảnh 2 channel dưới dạng visualization (chỉ hiển thị channel đầu - grayscale)
    cv2.imwrite(final_path, grayscale_image)
    processed_images['6_final_input'] = image_to_base64(final_path)
    saved_paths['6_final_input'] = final_path
    
    return processed_images, final_image, grayscale_image, saved_paths

def predict_fruit(image_2channel, top_k=5):
    """
    Hàm dự đoán loại quả từ ảnh 2 channel đã được xử lý, trả về top k dự đoán hàng đầu.
    
    Parameters:
        image_2channel: numpy array shape (100, 100, 2) - ảnh đã được xử lý (grayscale + edge)
        top_k: số lượng kết quả top cao nhất cần trả về (mặc định 5)
    
    Returns:
        top_predictions: list các dict chứa 'class' và 'confidence' sắp xếp theo confidence giảm dần
    """
    if model is None or label_encoder is None:
        return None
    
    # Kiểm tra shape của ảnh đầu vào
    if image_2channel.shape != (100, 100, 2):
        print(f"⚠️  Warning: Image shape {image_2channel.shape} không đúng, cần (100, 100, 2)")
        return None
    
    # Chuẩn hóa (chia 255.0 giống lúc train) - QUAN TRỌNG
    input_img = image_2channel.astype('float32') / 255.0
    
    # Mở rộng chiều batch (1, 100, 100, 2)
    input_batch = np.expand_dims(input_img, axis=0)
    
    # Dự đoán
    predictions = model.predict(input_batch, verbose=0)
    probabilities = predictions[0]  # Lấy mảng xác suất cho một ảnh đầu vào
    
    # Sắp xếp các xác suất và lấy chỉ số của top k dự đoán hàng đầu
    top_k_indices = np.argsort(probabilities)[::-1][:top_k]
    top_k_probabilities = probabilities[top_k_indices]
    
    # Lấy tên lớp tương ứng cho top k dự đoán hàng đầu
    top_k_labels = label_encoder.inverse_transform(top_k_indices)
    
    # Tạo danh sách kết quả
    top_predictions = []
    for i in range(len(top_k_labels)):
        top_predictions.append({
            'class': str(top_k_labels[i]),
            'confidence': round(float(top_k_probabilities[i]) * 100, 2)  # Chuyển thành phần trăm
        })
    
    return top_predictions

@app.route('/health', methods=['GET'])
def health_check():
    """Endpoint kiểm tra server có hoạt động không"""
    return jsonify({
        'status': 'ok',
        'message': 'Server is running',
        'model_loaded': model is not None,
        'encoder_loaded': label_encoder is not None
    })

@app.route('/predict', methods=['POST'])
def predict():
    """
    Endpoint nhận ảnh và trả về kết quả dự đoán cùng các ảnh đã xử lý
    
    Request:
        - file: ảnh upload (multipart/form-data)
    
    Response:
        - prediction: tên loại trái cây
        - confidence: độ tin cậy
        - processed_images: dict các ảnh đã xử lý (base64)
        - error: thông báo lỗi (nếu có)
    """
    try:
        # Kiểm tra có file trong request không
        if 'file' not in request.files:
            return jsonify({'error': 'Không có file ảnh trong request'}), 400
        
        file = request.files['file']
        
        # Kiểm tra file có tên không
        if file.filename == '':
            return jsonify({'error': 'Không có file được chọn'}), 400
        
        # Kiểm tra định dạng file
        if not allowed_file(file.filename):
            return jsonify({'error': 'Định dạng file không được hỗ trợ. Chỉ chấp nhận: PNG, JPG, JPEG, GIF, BMP'}), 400
        
        # Lưu file tạm
        filename = secure_filename(file.filename)
        filepath = os.path.join(UPLOAD_FOLDER, filename)
        file.save(filepath)
        
        # Xử lý ảnh qua các bước (để hiển thị các bước xử lý cho frontend)
        processed_images, final_image, grayscale_image, saved_paths = process_image_with_steps(filepath)
        
        if final_image is None:
            return jsonify({'error': 'Không thể xử lý ảnh'}), 500
        
        # Dự đoán - sử dụng final_image đã được xử lý (tránh xử lý lại)
        top_predictions = predict_fruit(final_image, top_k=5)
        
        if top_predictions is None:
            return jsonify({'error': 'Không thể dự đoán. Model chưa được load'}), 500
        
        # Lấy kết quả cao nhất để hiển thị trong message
        top_result = top_predictions[0] if top_predictions else None
        
        # Xóa file tạm sau khi xử lý xong
        try:
            os.remove(filepath)
            # Xóa các file processed sau khi đã encode base64
            for path in saved_paths.values():
                if os.path.exists(path):
                    os.remove(path)
        except Exception as e:
            print(f"Lỗi khi xóa file tạm: {e}")
        
        # Trả về kết quả
        return jsonify({
            'success': True,
            'top_predictions': top_predictions,  # Danh sách top 5 kết quả
            'prediction': top_result['class'] if top_result else None,  # Kết quả cao nhất (để tương thích)
            'confidence': top_result['confidence'] if top_result else None,  # Độ tin cậy cao nhất (để tương thích)
            'processed_images': processed_images,
            'message': f'Dự đoán tốt nhất: {top_result["class"]} với độ tin cậy {top_result["confidence"]}%' if top_result else 'Không có kết quả'
        })
        
    except Exception as e:
        return jsonify({
            'error': f'Lỗi khi xử lý: {str(e)}'
        }), 500

if __name__ == '__main__':
    # Load model khi khởi động
    load_model_and_encoder()
    
    # Chạy server
    print("=" * 50)
    print("Flask Server đang khởi động...")
    print("=" * 50)
    print(f"Model: {MODEL_PATH}")
    print(f"Label Encoder: {LABEL_ENCODER_PATH}")
    print("=" * 50)
    print("Server đang chạy tại: http://localhost:5000")
    print("Endpoint dự đoán: POST http://localhost:5000/predict")
    print("Endpoint health check: GET http://localhost:5000/health")
    print("=" * 50)
    
    app.run(debug=True, host='0.0.0.0', port=5000)

