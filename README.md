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
git clone https://github.com/ntanhhh/CCCD_Extractor.git
cd CCCD_Extractor
```

2. Cài đặt các dependencies:
```bash
pip install -r requirements.txt
```

3. Tải các model cần thiết:
- YOLOv8 model cho CMND cũ: `model/detect_ttin/cccd_cu.pt`
- YOLOv8 model cho CCCD mới: `model/detect_ttin/cccd_moi.pt`
- VietOCR model đã fine-tune: `model/finetune_vietocr`

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
└── model/              # Model files
    ├── detect_ttin/    # YOLOv8 models
    │   ├── cccd_cu.pt  # Model for old ID cards
    │   └── cccd_moi.pt # Model for new ID cards
    └── finetune_vietocr/ # Fine-tuned VietOCR model
``` 