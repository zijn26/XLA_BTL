import cv2
import numpy as np
import matplotlib.pyplot as plt
import os # Import os module to remove file

def load_image(image_path):
    """
    Đọc ảnh từ đường dẫn đã cho.

    Parameters:
    - image_path (str): Đường dẫn đến file ảnh.

    Returns:
    - image (numpy.ndarray): Ảnh đã được đọc dưới dạng mảng NumPy (BGR).
    - None: Nếu không thể đọc ảnh.
    """
    image = cv2.imread(image_path)
    if image is None:
        print(f"Lỗi: Không thể đọc ảnh từ đường dẫn: {image_path}")
    return image
def apply_histogram_equalization(image):
    """
    Cân bằng histogram để tăng độ tương phản của ảnh.
    Hiệu quả với ảnh có độ tương phản thấp.
    """
    # Chuyển đổi sang ảnh xám nếu là ảnh màu
    if len(image.shape) == 3:
        # Chuyển đổi sang không gian màu YCrCb
        ycrcb_img = cv2.cvtColor(image, cv2.COLOR_BGR2YCrCb)
        # Cân bằng histogram cho kênh Y (độ sáng)
        ycrcb_img[:, :, 0] = cv2.equalizeHist(ycrcb_img[:, :, 0])
        # Chuyển đổi lại không gian màu BGR
        equalized_img = cv2.cvtColor(ycrcb_img, cv2.COLOR_YCrCb2BGR)
    else:
        equalized_img = cv2.equalizeHist(image)

    return equalized_img
def restore_from_noise(image):
    """
    Khôi phục ảnh bị nhiễu bằng bộ lọc Non-Local Means.
    Đây là một kỹ thuật khôi phục ảnh tiên tiến, loại bỏ nhiễu
    trong khi vẫn giữ được các chi tiết và đường nét.
    """
    if len(image.shape) == 3:
        # Khôi phục ảnh màu
        restored_image = cv2.fastNlMeansDenoisingColored(image, None, 10, 10, 7, 21)
    else:
        # Khôi phục ảnh xám
        restored_image = cv2.fastNlMeansDenoising(image, None, 10, 7, 21)
    return restored_image
def convert_to_grayscale(image):
    """
    Chuyển đổi ảnh từ BGR sang ảnh xám (Grayscale).

    Parameters:
    - image (numpy.ndarray): Ảnh đầu vào (BGR).

    Returns:
    - grayscale_image (numpy.ndarray): Ảnh xám.
    """
    return cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

def apply_median_filter(image, kernel_size=3):
    """
    Áp dụng Bộ lọc trung vị (Median Filter) để loại bỏ nhiễu.

    Parameters:
    - image (numpy.ndarray): Ảnh đầu vào (nên là ảnh xám).
    - kernel_size (int): Kích thước của kernel (phải là số lẻ, ví dụ: 3, 5, 7).

    Returns:
    - denoised_image (numpy.ndarray):  ảnh đã được làm giảm nhiễu.
    """
    return cv2.medianBlur(image, kernel_size)

def apply_canny_edge_detection(image, lower_threshold=None, upper_threshold=None):
    """
    Phát hiện biên trong ảnh sử dụng thuật toán Canny.

    Parameters:
    - image (numpy.ndarray): Ảnh đầu vào (nên là ảnh xám, đã giảm nhiễu).
    - lower_threshold (int, optional): Ngưỡng dưới cho Canny.
    - upper_threshold (int, optional): Ngưỡng trên cho Canny.

    Returns:
    - edge_image (numpy.ndarray): Ảnh biên (ảnh nhị phân).
    """
    # Tự động xác định ngưỡng nếu không được cung cấp
    if lower_threshold is None or upper_threshold is None:
        median = np.median(image)
        lower = int(max(0, 0.7 * median))
        upper = int(min(255, 1.3 * median))
    else:
        lower = lower_threshold
        upper = upper_threshold

    return cv2.Canny(image, lower, upper)
def segment_by_thresholding(image):
    """
    Phân vùng ảnh bằng phương pháp ngưỡng Otsu.
    Tự động tìm ngưỡng tối ưu để tách đối tượng và nền.
    """
    if len(image.shape) == 3:
        gray_image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    else:
        gray_image = image

    # Áp dụng ngưỡng Otsu
    _, segmented_image = cv2.threshold(gray_image, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
    return segmented_image

def apply_morphological_opening(image, kernel_size=(3, 3), iterations=1):
    """
    Áp dụng phép Mở (Morphological Opening) để loại bỏ nhiễu nhỏ và làm mịn biên.

    Parameters:
    - image (numpy.ndarray): Ảnh đầu vào (thường là ảnh biên).
    - kernel_size (tuple): Kích thước kernel.
    - iterations (int): Số lần lặp lại phép toán.

    Returns:
    - opened_image (numpy.ndarray): Ảnh sau khi áp dụng phép mở.
    """
    kernel = np.ones(kernel_size, np.uint8)
    return cv2.morphologyEx(image, cv2.MORPH_OPEN, kernel, iterations=iterations)

def apply_morphological_closing(image, kernel_size=(3, 3), iterations=1):
    """
    Áp dụng phép Đóng (Morphological Closing) để kết nối các đối tượng và lấp đầy lỗ hổng.

    Parameters:
    - image (numpy.ndarray): Ảnh đầu vào.
    - kernel_size (tuple): Kích thước kernel.
    - iterations (int): Số lần lặp lại phép toán.

    Returns:
    - closed_image (numpy.ndarray): Ảnh sau khi áp dụng phép đóng.
    """
    kernel = np.ones(kernel_size, np.uint8)
    return cv2.morphologyEx(image, cv2.MORPH_CLOSE, kernel, iterations=iterations)

def extract_boundary(image, kernel_size=(3, 3)):
    """
    Tách biên của đối tượng từ ảnh nhị phân.

    Parameters:
    - image (numpy.ndarray): Ảnh nhị phân đầu vào.
    - kernel_size (tuple): Kích thước kernel dùng cho phép xói mòn (erosion).

    Returns:
    - boundary_image (numpy.ndarray): Ảnh chỉ chứa đường biên.
    """
    kernel = np.ones(kernel_size, np.uint8)
    eroded_image = cv2.erode(image, kernel)
    return cv2.subtract(image, eroded_image)

def display_processing_steps(original_image, steps):
    """
    Hiển thị ảnh gốc và kết quả của từng bước xử lý.

    Parameters:
    - original_image (numpy.ndarray): Ảnh gốc.
    - steps (dict): Một dictionary chứa tên và kết quả của các bước.
    """
    num_steps = len(steps)
    plt.figure(figsize=(15, 10))

    # Hiển thị ảnh gốc
    plt.subplot(2, 4, 1)
    plt.imshow(cv2.cvtColor(original_image, cv2.COLOR_BGR2RGB))
    plt.title('1. Ảnh gốc')
    plt.axis('off')

    # Hiển thị các bước xử lý
    for i, (title, img) in enumerate(steps.items()):
        plt.subplot(2, 4, i + 2)
        cmap = 'gray' if len(img.shape) == 2 else None
        plt.imshow(img, cmap=cmap)
        plt.title(f'{i + 2}. {title}')
        plt.axis('off')

    plt.tight_layout()
    plt.show()

def full_preprocessing_pipeline(image_path, display_steps=True):
    """
    Thực hiện toàn bộ quy trình tiền xử lý ảnh bằng cách gọi các hàm riêng lẻ.

    Parameters:
    - image_path (str): Đường dẫn đến file ảnh.
    - display_steps (bool): Có hiển thị các bước xử lý hay không.

    Returns:
    - grayscale_image (numpy.ndarray): Ảnh xám đã được xử lý.
    - final_image (numpy.ndarray): Ảnh đã được xử lý cuối cùng, sẵn sàng cho CNN.
    - None: Nếu có lỗi xảy ra.
    """
    # Bước 0: Đọc ảnh
    original_image = load_image(image_path)
    if original_image is None:
        return None, None # Return two Nones if image loading fails
    original_image = cv2.resize(original_image, (100, 100))
    processed_results = {}

    # I. Làm sạch ảnh và Tăng cường chất lượng

    # 1. Chuyển ảnh xám
    grayscale_image = convert_to_grayscale(original_image)
    processed_results['Ảnh xám'] = grayscale_image
    #  Cân bằng histogram
    histogram_img = apply_histogram_equalization(grayscale_image)
    processed_results['Cân bằng histogram'] = histogram_img

    # 2. Loại bỏ nhiễu bằng Median Filter
    denoised_image = apply_median_filter(grayscale_image)
    processed_results['Loại bỏ nhiễu (Median)'] = denoised_image

    # II. Tách đối tượng (Segmentation)

    # 3. Phát hiện biên bằng Canny
    edge_image = apply_canny_edge_detection(denoised_image)
    processed_results['Phát hiện biên (Canny)'] = edge_image

    otsu_img = segment_by_thresholding(grayscale_image)
    processed_results['Phương pháp otsu'] = otsu_img

    # 5c. Tách biên (Boundary Extraction)
    # final_image = extract_boundary(edge_image)
    processed_results['Ảnh đầu vào cho CNN'] = edge_image

    if display_steps:
        display_processing_steps(original_image, processed_results)

    return grayscale_image, edge_image

    # Đầu ra là ảnh dạng (height , with , 2 )