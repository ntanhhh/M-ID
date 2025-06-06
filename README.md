# ID Card Information Extraction

Ứng dụng web xử lý và trích xuất thông tin từ CMND/CCCD sử dụng FastAPI và YOLOv8.

## Tính năng

- Upload ảnh CMND/CCCD
- Cắt ảnh tự động sử dụng YOLOv8
- Trích xuất thông tin từ ảnh đã cắt
- Hỗ trợ cả CMND cũ và CCCD mới
- Giao diện web thân thiện với người dùng

## Cài đặt

1. Clone repository:
```bash
git clone [repository-url]
cd [repository-name]
```

2. Cài đặt các dependencies:
```bash
pip install -r requirements.txt
```

3. Tải các model cần thiết:
- YOLOv8 model cho CMND cũ: `runs/detect/train15/weights/best.pt`
- YOLOv8 model cho CCCD mới: `model/detect_ttin/best.pt`
- VietOCR model: `model/vgg_transformer.pth`

## Sử dụng

1. Khởi động server:
```bash
uvicorn main:app --reload
```

2. Mở trình duyệt và truy cập: `http://localhost:8000`

## Cấu trúc project

```
.
├── main.py              # FastAPI application
├── multimodal_id.py     # Core ID card processing logic
├── requirements.txt     # Project dependencies
├── templates/          # HTML templates
│   └── index.html      # Main web interface
├── model/              # Model files
│   ├── detect_ttin/    # YOLOv8 model for new ID cards
│   └── vgg_transformer.pth  # VietOCR model
└── runs/               # YOLOv8 training results
    └── detect/
        └── train15/    # YOLOv8 model for old ID cards
``` 