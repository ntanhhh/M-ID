# CCCD Extractor

Ứng dụng trích xuất thông tin từ ảnh căn cước công dân (CCCD) sử dụng FastAPI, YOLOv8 và các mô hình OCR.

## Chức năng chính
- Phát hiện và phân loại CCCD cũ/mới tự động
- Cắt vùng chứa CCCD trên ảnh đầu vào
- Trích xuất thông tin văn bản từ ảnh CCCD (số, họ tên, ngày sinh, giới tính, quê quán, nơi thường trú...)
- Giao diện web đơn giản, dễ sử dụng
- Hỗ trợ API cho tích hợp hệ thống khác

## Hướng dẫn cài đặt

1. **Clone mã nguồn:**
```bash
git clone https://github.com/ntanhhh/CCCD_Extractor.git
cd CCCD_Extractor
```

2. **Cài đặt thư viện Python:**
```bash
pip install -r requirements.txt
```

3. **Tải các model cần thiết:**
- Tải các file model từ [Google Drive](https://drive.google.com/drive/folders/14t1fJQrsg2noPLxsUB854mmRwEP9vU6d?usp=sharing)
- Đặt vào thư mục `model/` với cấu trúc:
```
model/
├── detect_ttin/
│   ├── cccd_cu.pt
│   └── cccd_moi.pt
├── detect_4goc/
│   └── best.pt
└── finetune_vietocr/
    ├── config.yml
    └── transformerocr.pth
```

4. **Tạo các thư mục cần thiết:**
- `cropped_images/` : Lưu ảnh CCCD đã cắt
- `temp_uploads/`   : Lưu ảnh upload tạm thời

## Cấu trúc thư mục dự án
```
CCCD_Extractor/
├── main.py            # Server FastAPI
├── m_ocr.py           # Xử lý OCR đa mô hình
├── cropper.py         # Cắt ảnh CCCD
├── requirements.txt   # Thư viện Python
├── README.md          # Tài liệu này
├── model/             # Chứa các file model
├── cropped_images/    # Ảnh CCCD đã cắt
├── temp_uploads/      # Ảnh upload tạm thời
├── templates/         # Giao diện web (index.html)
├── dataset/           # (Tùy chọn) Dữ liệu huấn luyện
├── results/           # (Tùy chọn) Kết quả, log
```

## Hướng dẫn sử dụng

1. **Chạy server:**
```bash
uvicorn main:app --reload
```

2. **Truy cập giao diện web:**
- Mở trình duyệt và vào địa chỉ: http://localhost:8000
- Upload ảnh CCCD, hệ thống sẽ tự động cắt và trích xuất thông tin

3. **Sử dụng API:**
- `POST /upload`   : Upload ảnh CCCD
- `POST /crop`     : Cắt vùng CCCD trên ảnh
- `POST /extract`  : Trích xuất thông tin từ ảnh CCCD đã cắt

## Yêu cầu model
- Phát hiện CCCD cũ: `model/detect_ttin/cccd_cu.pt`
- Phát hiện CCCD mới: `model/detect_ttin/cccd_moi.pt`
- Cắt ảnh CCCD: `model/detect_4goc/best.pt`
- OCR: `model/finetune_vietocr/transformerocr.pth` và `model/finetune_vietocr/config.yml`

## Lưu ý
- Nên chuẩn bị sẵn các thư mục `cropped_images/` và `temp_uploads/` trước khi chạy.
- Thư mục `model/` phải chứa đầy đủ các file model cần thiết.
- Ứng dụng sẽ tự tạo các thư mục đầu ra nếu chưa có.

## Giấy phép

Phần mềm phát hành theo giấy phép MIT. 