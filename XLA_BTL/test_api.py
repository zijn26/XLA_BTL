import requests
import json
import os
import base64
from pathlib import Path

# Cấu hình
API_URL = "http://localhost:5000"
PREDICT_ENDPOINT = f"{API_URL}/predict"
HEALTH_ENDPOINT = f"{API_URL}/health"

def test_health_check():
    """Test endpoint health check"""
    print("=" * 60)
    print("🔍 Đang kiểm tra health check...")
    print("=" * 60)
    
    try:
        response = requests.get(HEALTH_ENDPOINT, timeout=5)
        if response.status_code == 200:
            data = response.json()
            print("✅ Server đang hoạt động!")
            print(f"   - Status: {data.get('status')}")
            print(f"   - Model loaded: {data.get('model_loaded')}")
            print(f"   - Encoder loaded: {data.get('encoder_loaded')}")
            return True
        else:
            print(f"❌ Lỗi: Status code {response.status_code}")
            return False
    except requests.exceptions.ConnectionError:
        print("❌ Không thể kết nối đến server!")
        print("   Hãy chắc chắn server đang chạy: python app.py")
        return False
    except Exception as e:
        print(f"❌ Lỗi: {e}")
        return False

def test_predict(image_path):
    """Test endpoint predict với ảnh"""
    print("\n" + "=" * 60)
    print(f"🖼️  Đang test với ảnh: {image_path}")
    print("=" * 60)
    
    # Kiểm tra file có tồn tại không
    if not os.path.exists(image_path):
        print(f"❌ File không tồn tại: {image_path}")
        return False
    
    try:
        # Mở file và gửi request
        with open(image_path, 'rb') as f:
            files = {'file': (os.path.basename(image_path), f, 'image/jpeg')}
            
            print("📤 Đang gửi request đến server...")
            response = requests.post(PREDICT_ENDPOINT, files=files, timeout=30)
        
        # Kiểm tra response
        if response.status_code == 200:
            data = response.json()
            print(data)    
            if data.get('success'):
                print("\n✅ Dự đoán thành công!")
                print(f"   🍎 Loại trái cây: {data.get('prediction')}")
                print(f"   📊 Độ tin cậy: {data.get('confidence')}%")
                print(f"   💬 Message: {data.get('message')}")
                
                # Hiển thị thông tin về các ảnh đã xử lý
                processed_images = data.get('processed_images', {})
                if processed_images:
                    print(f"\n📸 Các ảnh đã xử lý ({len(processed_images)} bước):")
                    for step_name, img_base64 in processed_images.items():
                        img_size = len(img_base64) / 1024  # KB
                        print(f"   - {step_name}: {img_size:.2f} KB (base64)")
                
                # Lưu ảnh đã xử lý (tùy chọn)
                save_processed_images = input("\n💾 Bạn có muốn lưu các ảnh đã xử lý không? (y/n): ").lower()
                if save_processed_images == 'y':
                    save_images(data.get('processed_images', {}), image_path)
                
                return True
            else:
                print(f"❌ Lỗi từ server: {data.get('error')}")
                return False
        else:
            print(f"❌ Lỗi HTTP: Status code {response.status_code}")
            try:
                error_data = response.json()
                print(f"   Chi tiết: {error_data.get('error', 'Unknown error')}")
            except:
                print(f"   Response: {response.text}")
            return False
            
    except requests.exceptions.Timeout:
        print("❌ Request timeout! Server có thể đang xử lý quá lâu.")
        return False
    except requests.exceptions.ConnectionError:
        print("❌ Không thể kết nối đến server!")
        print("   Hãy chắc chắn server đang chạy: python app.py")
        return False
    except Exception as e:
        print(f"❌ Lỗi: {e}")
        import traceback
        traceback.print_exc()
        return False

def save_images(processed_images, original_image_path):
    """Lưu các ảnh đã xử lý ra file"""
    if not processed_images:
        print("   Không có ảnh để lưu.")
        return
    
    # Tạo thư mục output
    output_dir = "test_output"
    os.makedirs(output_dir, exist_ok=True)
    
    # Lấy tên file gốc (không có extension)
    original_name = Path(original_image_path).stem
    
    print(f"\n💾 Đang lưu ảnh vào thư mục: {output_dir}/")
    
    for step_name, img_base64 in processed_images.items():
        try:
            # Decode base64
            img_data = base64.b64decode(img_base64)
            
            # Tạo tên file
            output_path = os.path.join(output_dir, f"{original_name}_{step_name}.jpg")
            
            # Lưu file
            with open(output_path, 'wb') as f:
                f.write(img_data)
            
            print(f"   ✅ Đã lưu: {output_path}")
        except Exception as e:
            print(f"   ❌ Lỗi khi lưu {step_name}: {e}")

def find_images_in_directory(directory="."):
    """Tìm tất cả file ảnh trong thư mục"""
    image_extensions = {'.jpg', '.jpeg', '.png', '.gif', '.bmp'}
    images = []
    
    for file in os.listdir(directory):
        if Path(file).suffix.lower() in image_extensions:
            images.append(file)
    
    return sorted(images)

def main():
    """Hàm main để chạy test"""
    print("\n" + "=" * 60)
    print("🧪 TEST API - FRUIT RECOGNITION")
    print("=" * 60)
    
    # Test health check trước
    if not test_health_check():
        print("\n⚠️  Server không hoạt động. Vui lòng khởi động server trước!")
        print("   Chạy lệnh: python app.py")
        return
    
    # Tìm các file ảnh trong thư mục hiện tại
    current_dir = os.path.dirname(os.path.abspath(__file__))
    images = find_images_in_directory(current_dir)
    
    if not images:
        print("\n❌ Không tìm thấy file ảnh nào trong thư mục!")
        return
    
    print(f"\n📁 Tìm thấy {len(images)} file ảnh:")
    for i, img in enumerate(images, 1):
        print(f"   {i}. {img}")
    
    # Chọn ảnh để test
    if len(images) == 1:
        selected_image = images[0]
        print(f"\n✅ Tự động chọn: {selected_image}")
    else:
        try:
            choice = input(f"\n👉 Chọn ảnh để test (1-{len(images)}) hoặc 'all' để test tất cả: ").strip()
            
            if choice.lower() == 'all':
                # Test tất cả ảnh
                for img in images:
                    image_path = os.path.join(current_dir, img)
                    test_predict(image_path)
                    print("\n" + "-" * 60 + "\n")
                return
            else:
                idx = int(choice) - 1
                if 0 <= idx < len(images):
                    selected_image = images[idx]
                else:
                    print("❌ Lựa chọn không hợp lệ!")
                    return
        except (ValueError, KeyboardInterrupt):
            print("\n❌ Đã hủy.")
            return
    
    # Test với ảnh đã chọn
    image_path = os.path.join(current_dir, selected_image)
    test_predict(image_path)
    
    print("\n" + "=" * 60)
    print("✅ Test hoàn tất!")
    print("=" * 60)

if __name__ == "__main__":
    try:
        main()
    except KeyboardInterrupt:
        print("\n\n⚠️  Đã hủy bởi người dùng.")
    except Exception as e:
        print(f"\n❌ Lỗi không mong đợi: {e}")
        import traceback
        traceback.print_exc()

