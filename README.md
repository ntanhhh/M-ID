# CCCD Extractor

Ứng dụng trích xuất thông tin từ CCCD sử dụng FastAPI và YOLOv8.

## Tính năng

- Phát hiện và phân loại CCCD cũ/mới
- Cắt ảnh sử dụng YOLOv11
- Trích xuất thông tin từ ảnh CCCD

## Cài đặt

1. Clone repository:
```bash
git clone https://github.com/ntanhhh/CCCD_Extractor.git
cd CCCD_Extractor
```

2. Cài đặt các thư viện cần thiết:
```bash
pip install -r requirements.txt
```

3. Tải các model cần thiết từ [Google Drive](https://drive.google.com/drive/folders/14t1fJQrsg2noPLxsUB854mmRwEP9vU6d?usp=sharing) và đặt vào thư mục `model` với cấu trúc sau:
```
model/
├── detect_ttin/
│   ├── cccd_cu.pt
│   └── cccd_moi.pt
├── detect_4goc/
│   └── cropper.pt
└── finetune_vietocr/
    ├── config.yml
    └── transformerocr.pth
```

## Cấu trúc dự án

```
CCCD_Extractor/
├── main.py                  # FastAPI server, main entry point
├── m_ocr.py                 # Multimodal OCR logic (VietOCR, Finetuned VietOCR, PaddleOCR)
├── croper.py                # Image cropping logic
├── requirements.txt         # Python dependencies
├── README.md                # Project documentation (this file)
│
├── model/                   # All model files (YOLO, VietOCR, PaddleOCR)
│   ├── detect_ttin/         # YOLO models for old/new CCCD detection
│   ├── detect_4goc/         # YOLO model for cropping
│   ├── finetune_vietocr/    # VietOCR weights and config
│
├── cropped_images/          # Output folder for cropped ID card images
├── temp_uploads/            # Temporary uploaded images from API
├── templates/               # HTML templates for web interface
│   └── index.html           # Main web interface
│
├── dataset/                 # Raw dataset images for training/testing 
├── cropped_dataset/         # Training data for OCR/detection
├── results/                 # Output results, logs, etc.
│

```

## Folder/Files Description
- **main.py**: FastAPI backend server.
- **m_ocr.py**: Main OCR logic, combines multiple OCR engines.
- **cropper.py**: Handles cropping of ID card images.
- **requirements.txt**: List of required Python packages.
- **model/**: All model weights/configs for detection and OCR.
- **cropped_images/**: Where cropped ID card images are saved.
- **temp_uploads/**: Where uploaded images are temporarily stored.
- **templates/**: HTML templates for the web interface.
- **dataset/**, **train_data/**: (Optional) For training/testing purposes.

## Notes
- You should create the folders `cropped_images/` and `temp_uploads/` before running the app.
- The `model/` folder should contain all necessary weights/configs for YOLO, VietOCR, PaddleOCR, and optionally EasyOCR.
- The app will auto-create output folders if they do not exist, but it's best to prepare the structure in advance.

## Sử dụng

1. Khởi động server:
```bash
uvicorn main:app --reload
```

2. Mở trình duyệt và truy cập:
```
http://localhost:8000
```

3. Upload ảnh CCCD và xem kết quả

## API Endpoints

- `POST /upload`: Upload ảnh CCCD
- `POST /crop`: Cắt ảnh CCCD
- `POST /extract`: Trích xuất thông tin từ ảnh CCCD

## Các model sử dụng

- Model phát hiện CCCD cũ: model/detect_ttin/cccd_cu.pt
- Model phát hiện CCCD mới: model/detect_ttin/cccd_moi.pt
- Model cắt ảnh: model/detect_4goc/cropper.pt
- Model OCR: model/finetune_vietocr/transformerocr.pth

## Giấy phép

MIT License 