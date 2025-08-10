import torch
import cv2
import numpy as np
from PIL import Image
from vietocr.tool.config import Cfg
from vietocr.tool.predictor import Predictor
from ultralytics import YOLO
import os
import re
from paddleocr import PaddleOCR
from difflib import SequenceMatcher

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

class MOCRSystem:
    def __init__(self):
        self.old_id_model = YOLO("model/detect_ttin/cccd_cu.pt")
        self.new_id_model = YOLO("model/detect_ttin/cccd_moi.pt")
        
        self.vietocr_detector = self.load_vietocr()
        self.vietocr_base_detector = self.load_vietocr_base()
        self.paddleocr_detector = self.load_paddleocr()
        self.cropped_dir = "cropped_images"
        os.makedirs(self.cropped_dir, exist_ok=True)

    def load_vietocr(self):
        config = Cfg.load_config_from_file('model/finetune_vietocr/config.yml')
        config['weights'] = 'model/finetune_vietocr/transformerocr.pth'
        config['device'] = 'cpu'
        return Predictor(config)

    def load_vietocr_base(self):
        config = Cfg.load_config_from_name('vgg_transformer')
        config['weights'] = 'model/vgg_transformer.pth'
        config['device'] = 'cpu'
        return Predictor(config)

    def load_paddleocr(self):
        return PaddleOCR(use_angle_cls=True, lang='vi')

    def preprocess_image(self, image):
        # Resize and enhance image quality
        image = cv2.resize(image, None, fx=1.5, fy=1.5, interpolation=cv2.INTER_CUBIC)
        image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        image = cv2.GaussianBlur(image, (5, 5), 0)
        return image

    def detect_id_type(self, image):
        # Run both models and pick the one with higher confidence
        old_results = self.old_id_model(image)
        new_results = self.new_id_model(image)
        
        old_conf = old_results[0].boxes.conf.max().item() if len(old_results[0].boxes) > 0 else 0
        new_conf = new_results[0].boxes.conf.max().item() if len(new_results[0].boxes) > 0 else 0
        
        print(f"\nDetection confidence:")
        print(f"Old CCCD model: {old_conf:.2f}")
        print(f"New CCCD model: {new_conf:.2f}")
        
        if old_conf > new_conf:
            return 'Cũ', old_results
        else:
            return 'Mới', new_results

    def crop_id_card(self, image, results):
        if len(results[0].boxes) == 0:
            return None
        boxes = results[0].boxes.xyxy.cpu().numpy()
        confidences = results[0].boxes.conf.cpu().numpy()
        cls_indices = results[0].boxes.cls.cpu().numpy().astype(int)
        labels = [results[0].names[i] for i in cls_indices]

        cropped_regions = []
        region_labels = []
        final_boxes_list = []
        for i, (box, label) in enumerate(zip(boxes, labels)):
            x1, y1, x2, y2 = box
            cropped = image[int(y1):int(y2), int(x1):int(x2)]
            cropped_regions.append(cropped)
            region_labels.append(label)
            final_boxes_list.append(box)
            crop_filename = f"{self.cropped_dir}/crop_{label}_{i}.jpg"
            cv2.imwrite(crop_filename, cropped)
        return cropped_regions, region_labels, final_boxes_list

    def extract_text(self, image):
        # Preprocess image for OCR
        processed_image = self.preprocess_image(image)
        processed_image_pil = Image.fromarray(processed_image)
        
        # Extract text using fine-tuned VietOCR
        vietocr_text = self.vietocr_detector.predict(processed_image_pil)
        # Extract text using VietOCR base
        vietocr_base_text = self.vietocr_base_detector.predict(processed_image_pil)
        # Extract text using PaddleOCR
        paddleocr_results = self.paddleocr_detector.ocr(processed_image)
        paddleocr_text = ' '.join([result[1][0] for result in paddleocr_results[0] if result and len(result) > 1])
        
        # Align characters from all 3 models
        aligned_texts = self.align_texts(vietocr_text, vietocr_base_text, paddleocr_text)
        # Vote per character
        final_text = self.vote_characters(aligned_texts)
        return final_text

    def align_texts(self, text1, text2, text3):
        aligned1 = list(text1)
        aligned2 = []
        aligned3 = []

        # Align text2 to text1
        matcher2 = SequenceMatcher(None, text1, text2)
        idx2 = 0
        for tag, i1, i2, j1, j2 in matcher2.get_opcodes():
            if tag == 'equal':
                aligned2.extend(text2[j1:j2])
                idx2 = j2
            elif tag == 'replace' or tag == 'delete':
                aligned2.extend([' '] * (i2 - i1))
            elif tag == 'insert':
                # Ignore insertions; we align to text1
                pass

        # Align text3 to text1
        matcher3 = SequenceMatcher(None, text1, text3)
        idx3 = 0
        for tag, i1, i2, j1, j2 in matcher3.get_opcodes():
            if tag == 'equal':
                aligned3.extend(text3[j1:j2])
                idx3 = j2
            elif tag == 'replace' or tag == 'delete':
                aligned3.extend([' '] * (i2 - i1))
            elif tag == 'insert':
                pass

        # Ensure aligned sequences have equal length
        maxlen = len(aligned1)
        aligned2 += [' '] * (maxlen - len(aligned2))
        aligned3 += [' '] * (maxlen - len(aligned3))

        return [aligned1, aligned2, aligned3]

    def is_vietnamese_accented(self, char):
        accented_chars = "ăâđêôơưáàảãạấầẩẫậắằẳẵặéèẻẽẹếềểễệíìỉĩịóòỏõọốồổỗộớờởỡợúùủũụứừửữựýỳỷỹỵ"
        return char.lower() in accented_chars

    def vote_characters(self, aligned_texts):
        final_text = []
        for i in range(len(aligned_texts[0])):
            vietocr_char = aligned_texts[0][i]
            if self.is_vietnamese_accented(vietocr_char):
                final_text.append(vietocr_char)
                continue
            chars = [text[i] for text in aligned_texts if i < len(text)]
            # If at least two characters match, choose that one
            if len(set(chars)) == 1:
                final_text.append(chars[0])
            elif len(set(chars)) == 2:
                for c in set(chars):
                    if chars.count(c) >= 2:
                        final_text.append(c)
                        break
                else:
                    final_text.append(vietocr_char)  # fallback
            else:
                final_text.append(vietocr_char)
        return ''.join(final_text)

    def process_id_card(self, image_path):
        image = cv2.imread(image_path)
        if image is None:
            print("Error: Unable to read image")
            return None, "Unable to read image"

        # Detect ID type and get results
        id_type, results = self.detect_id_type(image)

        crop_result = self.crop_id_card(image, results)
        if crop_result is None:
            print("No crop region detected, skipping OCR.")
            return None, "Unable to detect ID card"
        else:
            cropped_regions, region_labels, final_boxes = crop_result
            print(f"Cropped {len(cropped_regions)} regions, starting OCR")

        # Process each cropped region
        label_data = {}
        y_positions = {} 
        
        for cropped, label, box in zip(cropped_regions, region_labels, final_boxes):
            # Preprocess image
            processed_image = self.preprocess_image(cropped)
            processed_image_pil = Image.fromarray(processed_image)

            # VietOCR finetune
            vietocr_text = self.vietocr_detector.predict(processed_image_pil)
            # VietOCR base
            vietocr_base_text = self.vietocr_base_detector.predict(processed_image_pil)
            # PaddleOCR
            paddleocr_results = self.paddleocr_detector.ocr(processed_image)
            paddleocr_text = ' '.join([result[1][0] for result in paddleocr_results[0] if result and len(result) > 1]) if paddleocr_results and len(paddleocr_results) > 0 else ""

            # Align and vote across 3 OCR models
            aligned_texts = self.align_texts(vietocr_text, vietocr_base_text, paddleocr_text)
            final_text = self.vote_characters(aligned_texts)

            y1, y2 = box[1], box[3]
            y_coord = (y1 + y2) / 2  # center of box along y-axis
            print(f"Label: {label}, y: {y_coord}, text: {final_text}")

            # Handle labels with _new or _old suffixes
            if label.endswith('_new') or label.endswith('_old'):
                base_label = label.replace('_new', '').replace('_old', '')
                if base_label in label_data:
                    if y_coord > y_positions[base_label]:
                        label_data[base_label] = label_data[base_label] + " " + final_text
                    else:
                        label_data[base_label] = final_text + " " + label_data[base_label]
                    y_positions[base_label] = max(y_positions[base_label], y_coord)
                else:
                    label_data[base_label] = final_text
                    y_positions[base_label] = y_coord
            else:
                if label in label_data:
                    if y_coord > y_positions[label]:
                        label_data[label] = label_data[label] + " " + final_text
                    else:
                        label_data[label] = final_text + " " + label_data[label]
                    y_positions[label] = max(y_positions[label], y_coord)
                else:
                    label_data[label] = final_text
                    y_positions[label] = y_coord

        # Combine all text in the desired order
        ordered_labels = ["Id", "Name", "Date", "Sex", "Nation", "POO", "POR"]
        all_text = ""
        all_info = {}
        
        # Map labels to Vietnamese display names
        display_names = {
            "Id": "Số CMND/CCCD",
            "Name": "Tên",
            "Date": "Ngày sinh",
            "Sex": "Giới tính",
            "Nation": "Quốc tịch",
            "POO": "Quê quán",
            "POR": "Nơi thường trú"
        }

        # Build both all_text and all_info
        for label in ordered_labels:
            label_lower = label.lower()
            value = label_data.get(label) or label_data.get(label_lower, "")
            if value:
                # Normalize date formatting if applicable
                if label == "Date":
                    value = value.replace("-", "/").replace(".", "/").replace(" ", "/")
                    while "//" in value:
                        value = value.replace("//", "/")
                
                display_name = display_names.get(label, label)
                all_text += f"{display_name}: {value}\n"
                all_info[label] = value.strip()

        return {
            'id_type': id_type,
            'extracted_text': all_text,
            'parsed_info': all_info
        }

# Ví dụ sử dụng
if __name__ == "__main__":
    system = MOCRSystem()
    
    # Get image list in cropped_images
    image_dir = "cropped_images"
    if not os.path.exists(image_dir):
        print(f"\nError: Folder not found {image_dir}!")
        print("Please run the cropping process first.")
        exit(1)
        
    image_files = [f for f in os.listdir(image_dir) if f.lower().endswith(('.png', '.jpg', '.jpeg'))]
    
    if not image_files:
        print(f"\nError: No images found in folder {image_dir}!")
        print("Please run the cropping process first.")
        exit(1)
    
    # Get the latest image from cropped_images
    latest_image = max(image_files, key=lambda x: os.path.getctime(os.path.join(image_dir, x)))
    image_path = os.path.join(image_dir, latest_image)

    try:
        result = system.process_id_card(image_path)
        if result:
            print(f"ID Type: {result['id_type']}")
            print(result['extracted_text'])
            for field, value in result['parsed_info'].items():
                if value:
                    print(f"{field}: {value}")
    except Exception as e:
        print(f"Error while processing image: {str(e)}") 
        