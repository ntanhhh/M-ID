import torch
import cv2
import numpy as np
from PIL import Image
from vietocr.tool.config import Cfg
from vietocr.tool.predictor import Predictor
from ultralytics import YOLO
import os
import re

def non_max_suppression_fast(boxes, labels, overlapThresh=0.3):
    if len(boxes) == 0:
        return [], []
    
    if boxes.dtype.kind == "i":
        boxes = boxes.astype("float")
    
    pick = []
    x1, y1, x2, y2 = boxes[:, 0], boxes[:, 1], boxes[:, 2], boxes[:, 3]
    area = (x2 - x1 + 1) * (y2 - y1 + 1)
    idxs = np.argsort(y2)
    
    while len(idxs) > 0:
        last = len(idxs) - 1
        i = idxs[last]
        pick.append(i)
        xx1 = np.maximum(x1[i], x1[idxs[:last]])
        yy1 = np.maximum(y1[i], y1[idxs[:last]])
        xx2 = np.minimum(x2[i], x2[idxs[:last]])
        yy2 = np.minimum(y2[i], y2[idxs[:last]])
        w, h = np.maximum(0, xx2 - xx1 + 1), np.maximum(0, yy2 - yy1 + 1)
        overlap = (w * h) / area[idxs[:last]]
        idxs = np.delete(idxs, np.concatenate(([last], np.where(overlap > overlapThresh)[0])))
    
    return boxes[pick].astype("int"), [labels[idx] for idx in pick]

class MultimodalIDSystem:
    def __init__(self):
        # Đưa vào cả 2 model
        self.old_id_model = YOLO("model/detect_ttin/cccd_cu.pt")
        self.new_id_model = YOLO("model/detect_ttin/cccd_moi.pt")
        
        # Tải VietOCR
        self.vietocr_detector = self.load_vietocr()
        
        # Tạo thư mục đầu ra
        self.cropped_dir = "cropped_images"
        os.makedirs(self.cropped_dir, exist_ok=True)

    def load_vietocr(self):
        config = Cfg.load_config_from_file('model/finetune_vietocr/config.yml')
        config['weights'] = 'model/finetune_vietocr/transformerocr.pth'
        config['device'] = 'cpu'
        return Predictor(config)

    def preprocess_image(self, image):
        # Thay đổi kích thước và cải thiện chất lượng ảnh
        image = cv2.resize(image, None, fx=1.5, fy=1.5, interpolation=cv2.INTER_CUBIC)
        image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        image = cv2.GaussianBlur(image, (5, 5), 0)
        return image

    def detect_id_type(self, image):
        # Thử cả hai model và trả về model có độ tin cậy cao hơn
        old_results = self.old_id_model(image)
        new_results = self.new_id_model(image)
        
        old_conf = old_results[0].boxes.conf.max().item() if len(old_results[0].boxes) > 0 else 0
        new_conf = new_results[0].boxes.conf.max().item() if len(new_results[0].boxes) > 0 else 0
        
        print(f"\nĐộ tin cậy phát hiện:")
        print(f"Model CCCD cũ: {old_conf:.2f}")
        print(f"Model CCCD mới: {new_conf:.2f}")
        
        if old_conf > new_conf:
            return 'old', old_results
        else:
            return 'new', new_results

    def crop_id_card(self, image, results):
        if len(results[0].boxes) == 0:
            return None
            
        # Lấy tất cả các box phát hiện và nhãn của chúng
        boxes = results[0].boxes.xyxy.cpu().numpy()
        cls_indices = results[0].boxes.cls.cpu().numpy().astype(int)
        labels = [results[0].names[i] for i in cls_indices]
        
        # Áp dụng non-max suppression
        final_boxes, final_labels = non_max_suppression_fast(boxes, labels)
        
        # Xử lý từng box phát hiện
        cropped_regions = []
        region_labels = []
        
        for box, label in zip(final_boxes, final_labels):
            x1, y1, x2, y2 = box
            # Cắt ảnh
            cropped = image[int(y1):int(y2), int(x1):int(x2)]
            cropped_regions.append(cropped)
            region_labels.append(label)
        
        return cropped_regions, region_labels

    def extract_text(self, image):
        # Tiền xử lý ảnh cho OCR
        processed_image = self.preprocess_image(image)
        processed_image = Image.fromarray(processed_image)
        
        # Trích xuất văn bản bằng VietOCR
        text = self.vietocr_detector.predict(processed_image)
        return text

    def process_id_card(self, image_path):
        print(f"\nĐang xử lý CCCD: {image_path}")
        
        # Đọc ảnh
        image = cv2.imread(image_path)
        if image is None:
            print("Lỗi: Không thể đọc ảnh")
            return None, "Không thể đọc ảnh"

        # Phát hiện loại CCCD và lấy kết quả
        id_type, results = self.detect_id_type(image)
        print(f"\nLoại CCCD đã phát hiện: {id_type}")
        
        # Cắt các vùng của CCCD
        cropped_regions, region_labels = self.crop_id_card(image, results)
        if cropped_regions is None:
            print("Lỗi: Không thể phát hiện CCCD")
            return None, "Không thể phát hiện CCCD"

        # Xử lý từng vùng đã cắt
        label_data = {}
        text_positions = {}  # Lưu văn bản với tọa độ y
        y_max_positions = {}  # Theo dõi vị trí y cao nhất cho mỗi nhãn
        
        print(f"\nĐang xử lý loại CCCD: {id_type}")
        print("\nCác vùng và nhãn đã phát hiện:")
        for i, (box, label) in enumerate(zip(results[0].boxes.xyxy, region_labels)):
            print(f"Vùng {i+1}: {label} tại y={box[1]}")
        
        for i, (cropped, label) in enumerate(zip(cropped_regions, region_labels)):
            # Lưu ảnh đã cắt
            output_path = os.path.join(self.cropped_dir, f"cropped_{i}_{os.path.basename(image_path)}")
            cv2.imwrite(output_path, cropped)
            print(f"\nĐang xử lý vùng {i+1} ({label})")

            # Trích xuất văn bản
            text = self.extract_text(cropped)
            print(f"Văn bản đã trích xuất: {text}")

            # Lấy tọa độ y của box phát hiện
            y_coord = results[0].boxes.xyxy[i][1].item()  # tọa độ y1
            print(f"Tọa độ y: {y_coord}")

            # Lưu văn bản với vị trí của nó
            if label not in text_positions:
                text_positions[label] = []
                y_max_positions[label] = 0
            text_positions[label].append((y_coord, text))
            y_max_positions[label] = max(y_max_positions[label], y_coord)

        # Sắp xếp và kết hợp văn bản cho mỗi nhãn
        for label, positions in text_positions.items():
            # Sắp xếp dựa trên tọa độ y (từ trên xuống dưới)
            positions.sort(key=lambda x: x[0])
            
            # Đảo ngược thứ tự cho cả POO và POR trong cả CCCD cũ và mới
            if label.lower() in ['poo', 'por']:
                positions.reverse()
            
            # Kết hợp văn bản dựa trên so sánh tọa độ y
            combined_text = ""
            for y_coord, text in positions:
                if y_coord > y_max_positions[label]:
                    combined_text = text + ", " + combined_text if combined_text else text
                else:
                    combined_text = combined_text + ", " + text if combined_text else text
            
            label_data[label] = combined_text
            print(f"Văn bản đã kết hợp cho {label}: {combined_text}")

        print("\nDữ liệu nhãn cuối cùng:")
        for label, text in label_data.items():
            print(f"Nhãn: '{label}', Văn bản: '{text}'")

        # Kết hợp tất cả văn bản theo thứ tự đúng
        ordered_labels = ["Id", "Name", "Date", "Sex", "Nation", "POO", "POR"]
        all_text = ""
        all_info = {}
        
        # Ánh xạ nhãn sang tên hiển thị tiếng Việt
        display_names = {
            "Id": "Số CMND/CCCD",
            "Name": "Tên",
            "Date": "Ngày sinh",
            "Sex": "Giới tính",
            "Nation": "Quốc tịch",
            "POO": "Quê quán",
            "POR": "Nơi thường trú"
        }

        # Tạo cả all_text và all_info
        for label in ordered_labels:
            # Thử cả chữ hoa và chữ thường
            label_lower = label.lower()
            value = label_data.get(label) or label_data.get(label_lower, "")
            if value:
                # Định dạng ngày tháng nếu là trường ngày
                if label == "Date":
                    value = value.replace("-", "/").replace(".", "/").replace(" ", "/")
                    while "//" in value:
                        value = value.replace("//", "/")
                
                display_name = display_names.get(label, label)
                all_text += f"{display_name}: {value}\n"
                all_info[label] = value.strip()

        # In thông tin đã phân tích
        print("\nThông tin đã phân tích:")
        print("-" * 50)
        for field, value in all_info.items():
            if value:  # Chỉ in các trường đã tìm thấy
                display_name = display_names.get(field, field.upper())
                print(f"{display_name}: {value}")
        print("-" * 50)

        return {
            'id_type': id_type,
            'cropped_image_paths': [os.path.join(self.cropped_dir, f"cropped_{i}_{os.path.basename(image_path)}") 
                                  for i in range(len(cropped_regions))],
            'extracted_text': all_text,
            'parsed_info': all_info
        }

# Ví dụ sử dụng
if __name__ == "__main__":
    system = MultimodalIDSystem()
    
    # Lấy danh sách ảnh trong thư mục cropped_images
    image_dir = "cropped_images"
    image_files = [f for f in os.listdir(image_dir) if f.lower().endswith(('.png', '.jpg', '.jpeg'))]
    
    if not image_files:
        print("Không tìm thấy ảnh trong thư mục cropped_images!")
        print("Vui lòng đặt ảnh CCCD của bạn vào thư mục 'cropped_images'.")
        exit(1)
    
    print(f"\nĐã tìm thấy {len(image_files)} ảnh để xử lý.")
    
    # Xử lý từng ảnh
    for image_file in image_files:
        image_path = os.path.join(image_dir, image_file)
        print(f"\n{'='*50}")
        print(f"Đang xử lý ảnh: {image_file}")
        print(f"{'='*50}")
        
        try:
            result = system.process_id_card(image_path)
            if result:
                print("\nKết quả cuối cùng:")
                print(f"Loại CCCD: {result['id_type']}")
                print(f"Đường dẫn ảnh đã cắt: {result['cropped_image_paths']}")
                print("\nVăn bản đã trích xuất:")
                print(result['extracted_text'])
                print("\nThông tin đã phân tích:")
                for field, value in result['parsed_info'].items():
                    if value:
                        print(f"{field}: {value}")
        except Exception as e:
            print(f"Lỗi khi xử lý {image_file}: {str(e)}")
            continue 