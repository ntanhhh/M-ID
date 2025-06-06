# CCCD Extractor

Ứng dụng trích xuất thông tin từ CCCD sử dụng FastAPI và YOLOv8.

## Tính năng

- Phát hiện và phân loại CCCD cũ/mới
- Cắt ảnh sử dụng YOLOv8
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
├── main.py              # File chính chứa FastAPI endpoints
├── multimodal_id.py     # Module xử lý CCCD
├── requirements.txt     # Danh sách thư viện cần thiết
├── templates/          # Thư mục chứa file HTML
│   └── index.html      # Giao diện người dùng
├── static/            # Thư mục chứa file CSS và JavaScript
│   ├── css/
│   │   └── style.css  # File CSS
│   └── js/
│       └── script.js  # File JavaScript
├── uploads/           # Thư mục lưu ảnh upload
└── cropped_images/    # Thư mục lưu ảnh đã cắt
```

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

## Yêu cầu hệ thống

- Python 3.8+
- CUDA (khuyến nghị cho GPU)
- RAM: 8GB+
- Ổ cứng: 1GB+ trống

## Giấy phép

MIT License 