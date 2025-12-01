// API Configuration
const API_URL = 'http://localhost:5000';
const PREDICT_ENDPOINT = `${API_URL}/predict`;

// DOM Elements
const fileInput = document.getElementById('fileInput');
const uploadArea = document.getElementById('uploadArea');
const previewContainer = document.getElementById('previewContainer');
const previewImage = document.getElementById('previewImage');
const loading = document.getElementById('loading');
const resultsSection = document.getElementById('resultsSection');
const errorMessage = document.getElementById('errorMessage');
const errorText = document.getElementById('errorText');

// Fruit emoji mapping
const fruitEmojis = {
    'Apple': '🍎',
    'Banana': '🍌',
    'Orange': '🍊',
    'Grape': '🍇',
    'Corn': '🌽',
    'Mango': '🥭',
    'Pineapple': '🍍',
    'Strawberry': '🍓',
    'Watermelon': '🍉',
    'Kiwi': '🥝',
    'Peach': '🍑',
    'Cherry': '🍒'
};

// Step names mapping
const stepNames = {
    '1_grayscale': '1. Ảnh Xám',
    '2_histogram': '2. Cân Bằng Histogram',
    '3_denoised': '3. Loại Bỏ Nhiễu',
    '4_edges': '4. Phát Hiện Biên (Canny)',
    '5_otsu': '5. Phân Vùng (Otsu)',
    '6_final_input': '6. Ảnh Đầu Vào CNN'
};

// Initialize
fileInput.addEventListener('change', handleFileSelect);
uploadArea.addEventListener('dragover', handleDragOver);
uploadArea.addEventListener('drop', handleDrop);
uploadArea.addEventListener('dragleave', handleDragLeave);

// File selection handler
function handleFileSelect(event) {
    const file = event.target.files[0];
    if (file) {
        processFile(file);
    }
}

// Drag and drop handlers
function handleDragOver(event) {
    event.preventDefault();
    uploadArea.style.background = '#f0f0f0';
}

function handleDragLeave(event) {
    event.preventDefault();
    uploadArea.style.background = 'white';
}

function handleDrop(event) {
    event.preventDefault();
    uploadArea.style.background = 'white';
    const file = event.dataTransfer.files[0];
    if (file && file.type.startsWith('image/')) {
        processFile(file);
    } else {
        showError('Vui lòng chọn file ảnh hợp lệ!');
    }
}

// Process file
function processFile(file) {
    // Validate file type
    const validTypes = ['image/png', 'image/jpeg', 'image/jpg', 'image/gif', 'image/bmp'];
    if (!validTypes.includes(file.type)) {
        showError('Định dạng file không được hỗ trợ. Vui lòng chọn PNG, JPG, JPEG, GIF hoặc BMP.');
        return;
    }

    // Show preview
    const reader = new FileReader();
    reader.onload = (e) => {
        previewImage.src = e.target.result;
        previewContainer.style.display = 'block';
        document.querySelector('.upload-content').style.display = 'none';
    };
    reader.readAsDataURL(file);

    // Upload and predict
    uploadAndPredict(file);
}

// Upload and predict
async function uploadAndPredict(file) {
    hideError();
    hideResults();
    showLoading();

    const formData = new FormData();
    formData.append('file', file);

    try {
        const response = await fetch(PREDICT_ENDPOINT, {
            method: 'POST',
            body: formData
        });

        if (!response.ok) {
            const errorData = await response.json();
            throw new Error(errorData.error || 'Lỗi khi gọi API');
        }

        const data = await response.json();

        if (data.success) {
            displayResults(data);
        } else {
            throw new Error(data.error || 'Không thể dự đoán');
        }
    } catch (error) {
        console.error('Error:', error);
        showError(error.message || 'Có lỗi xảy ra khi xử lý ảnh. Vui lòng thử lại!');
    } finally {
        hideLoading();
    }
}

// Display results
function displayResults(data) {
    // Display main prediction
    const topPrediction = data.top_predictions[0];
    const fruitName = topPrediction.class;
    const emoji = fruitEmojis[fruitName] || '🍎';
    
    document.getElementById('predictionName').textContent = `${emoji} ${fruitName}`;
    document.getElementById('predictionConfidence').textContent = topPrediction.confidence;

    // Display top predictions list
    displayTopPredictions(data.top_predictions);

    // Display processed images
    displayProcessedImages(data.processed_images);

    // Show results section
    showResults();
}

// Display top predictions
function displayTopPredictions(predictions) {
    const predictionsList = document.getElementById('predictionsList');
    predictionsList.innerHTML = '';

    predictions.forEach((pred, index) => {
        const emoji = fruitEmojis[pred.class] || '🍎';
        const item = document.createElement('div');
        item.className = 'prediction-item';
        item.innerHTML = `
            <div class="prediction-item-name">
                <span class="prediction-item-rank">${index + 1}</span>
                <span>${emoji} ${pred.class}</span>
            </div>
            <div class="prediction-item-confidence">${pred.confidence}%</div>
        `;
        predictionsList.appendChild(item);
    });
}

// Display processed images
function displayProcessedImages(processedImages) {
    const grid = document.getElementById('processedImagesGrid');
    grid.innerHTML = '';

    // Sort by step order
    const sortedSteps = Object.keys(processedImages).sort();

    sortedSteps.forEach(stepKey => {
        const stepName = stepNames[stepKey] || stepKey;
        const imageBase64 = processedImages[stepKey];
        
        const item = document.createElement('div');
        item.className = 'processed-image-item';
        item.innerHTML = `
            <img src="data:image/jpeg;base64,${imageBase64}" alt="${stepName}">
            <p>${stepName}</p>
        `;
        grid.appendChild(item);
    });
}

// Remove image
function removeImage() {
    fileInput.value = '';
    previewContainer.style.display = 'none';
    document.querySelector('.upload-content').style.display = 'block';
    hideResults();
    hideError();
}

// Show/hide functions
function showLoading() {
    loading.style.display = 'block';
    resultsSection.style.display = 'none';
}

function hideLoading() {
    loading.style.display = 'none';
}

function showResults() {
    resultsSection.style.display = 'block';
    // Smooth scroll to results
    setTimeout(() => {
        resultsSection.scrollIntoView({ behavior: 'smooth', block: 'start' });
    }, 100);
}

function hideResults() {
    resultsSection.style.display = 'none';
}

function showError(message) {
    errorText.textContent = message;
    errorMessage.style.display = 'flex';
    // Auto hide after 5 seconds
    setTimeout(() => {
        hideError();
    }, 5000);
}

function hideError() {
    errorMessage.style.display = 'none';
}

