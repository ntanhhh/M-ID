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
        # Load both YOLOv8 models
        self.old_id_model = YOLO("runs/detect/train15/weights/best.pt")
        self.new_id_model = YOLO("model/detect_ttin/best.pt")
        
        # Load VietOCR
        self.vietocr_detector = self.load_vietocr()
        
        # Create output directories
        self.cropped_dir = "cropped_images"
        os.makedirs(self.cropped_dir, exist_ok=True)

    def load_vietocr(self):
        config = Cfg.load_config_from_name('vgg_transformer')
        config['weights'] = 'model/vgg_transformer.pth'
        config['device'] = 'cpu'
        return Predictor(config)

    def preprocess_image(self, image):
        # Resize and enhance image
        image = cv2.resize(image, None, fx=1.5, fy=1.5, interpolation=cv2.INTER_CUBIC)
        image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        image = cv2.GaussianBlur(image, (5, 5), 0)
        return image

    def detect_id_type(self, image):
        # Try both models and return the one with higher confidence
        old_results = self.old_id_model(image)
        new_results = self.new_id_model(image)
        
        old_conf = old_results[0].boxes.conf.max().item() if len(old_results[0].boxes) > 0 else 0
        new_conf = new_results[0].boxes.conf.max().item() if len(new_results[0].boxes) > 0 else 0
        
        print(f"\nDetection Confidence:")
        print(f"Old ID Model: {old_conf:.2f}")
        print(f"New ID Model: {new_conf:.2f}")
        
        if old_conf > new_conf:
            return 'old', old_results
        else:
            return 'new', new_results

    def crop_id_card(self, image, results):
        if len(results[0].boxes) == 0:
            return None
            
        # Get all detection boxes and their labels
        boxes = results[0].boxes.xyxy.cpu().numpy()
        cls_indices = results[0].boxes.cls.cpu().numpy().astype(int)
        labels = [results[0].names[i] for i in cls_indices]
        
        # Apply non-max suppression
        final_boxes, final_labels = non_max_suppression_fast(boxes, labels)
        
        print(f"\nFound {len(final_boxes)} detection regions after NMS")
        
        # Process each detection box
        cropped_regions = []
        region_labels = []
        
        for i, (box, label) in enumerate(zip(final_boxes, final_labels)):
            x1, y1, x2, y2 = box
            conf = results[0].boxes.conf[i].item()
            
            # Print detection box coordinates and confidence
            print(f"\nDetection Box {i+1} ({label}):")
            print(f"Confidence: {conf:.2f}")
            print(f"Top-left: ({x1}, {y1})")
            print(f"Bottom-right: ({x2}, {y2})")
            print(f"Width: {x2-x1}, Height: {y2-y1}")
            
            # Crop the image
            cropped = image[int(y1):int(y2), int(x1):int(x2)]
            cropped_regions.append(cropped)
            region_labels.append(label)
        
        return cropped_regions, region_labels

    def extract_text(self, image):
        # Preprocess image for OCR
        processed_image = self.preprocess_image(image)
        processed_image = Image.fromarray(processed_image)
        
        # Extract text using VietOCR
        text = self.vietocr_detector.predict(processed_image)
        return text

    def process_id_card(self, image_path):
        print(f"\nProcessing ID Card: {image_path}")
        
        # Read image
        image = cv2.imread(image_path)
        if image is None:
            print("Error: Could not read image")
            return None, "Could not read image"

        # Detect ID type and get results
        id_type, results = self.detect_id_type(image)
        print(f"\nDetected ID Type: {id_type}")
        
        # Crop the ID card regions
        cropped_regions, region_labels = self.crop_id_card(image, results)
        if cropped_regions is None:
            print("Error: Could not detect ID card")
            return None, "Could not detect ID card"

        # Process each cropped region
        label_data = {}
        text_positions = {}  # Store text with their y-coordinates
        y_max_positions = {}  # Track highest y position for each label
        
        print(f"\nProcessing ID type: {id_type}")
        print("\nDetected regions and labels:")
        for i, (box, label) in enumerate(zip(results[0].boxes.xyxy, region_labels)):
            print(f"Region {i+1}: {label} at y={box[1]}")
        
        for i, (cropped, label) in enumerate(zip(cropped_regions, region_labels)):
            # Save cropped image
            output_path = os.path.join(self.cropped_dir, f"cropped_{i}_{os.path.basename(image_path)}")
            cv2.imwrite(output_path, cropped)
            print(f"\nProcessing region {i+1} ({label})")

            # Extract text
            text = self.extract_text(cropped)
            print(f"Extracted Text: {text}")

            # Get the y-coordinate of the detection box
            y_coord = results[0].boxes.xyxy[i][1].item()  # y1 coordinate
            print(f"Y coordinate: {y_coord}")

            # Store text with its position
            if label not in text_positions:
                text_positions[label] = []
                y_max_positions[label] = 0
            text_positions[label].append((y_coord, text))
            y_max_positions[label] = max(y_max_positions[label], y_coord)

        # Sort and combine text for each label
        for label, positions in text_positions.items():
            # Sort based on y-coordinate (top to bottom)
            positions.sort(key=lambda x: x[0])
            
            # Reverse order for specific cases:
            # - POO for old ID cards
            # - Both POO and POR for new ID cards
            if (id_type == 'old' and label.lower() == 'poo') or \
               (id_type == 'new' and label.lower() in ['poo', 'por']):
                positions.reverse()
            
            # Combine text based on y-coordinate comparison
            combined_text = ""
            for y_coord, text in positions:
                if y_coord > y_max_positions[label]:
                    combined_text = text + ", " + combined_text if combined_text else text
                else:
                    combined_text = combined_text + ", " + text if combined_text else text
            
            label_data[label] = combined_text
            print(f"Combined text for {label}: {combined_text}")

        print("\nFinal label data:")
        for label, text in label_data.items():
            print(f"Label: '{label}', Text: '{text}'")

        # Combine all text in the correct order
        ordered_labels = ["Id", "Name", "Date", "Sex", "Nation", "POR", "POO"]
        all_text = ""
        all_info = {}
        
        # Map labels to Vietnamese display names
        display_names = {
            "Id": "Số CMND/CCCD",
            "Name": "Họ và tên",
            "Date": "Ngày sinh",
            "Sex": "Giới tính",
            "Nation": "Quốc tịch",
            "POO": "Quê quán",
            "POR": "Nơi thường trú"
        }

        print("\nChecking labels in label_data:")
        for label in ordered_labels:
            # Try both original case and lowercase
            label_lower = label.lower()
            found = False
            if label in label_data:
                print(f"Found label '{label}' in label_data")
                found = True
            elif label_lower in label_data:
                print(f"Found label '{label_lower}' in label_data")
                found = True
            else:
                print(f"Label '{label}' not found in label_data")

        # Create both all_text and all_info
        for label in ordered_labels:
            # Try both original case and lowercase
            label_lower = label.lower()
            if label in label_data and label_data[label]:
                value = label_data[label]
                display_name = display_names.get(label, label)
                all_text += f"{display_name}: {value}\n"
                all_info[label] = value.strip()
                print(f"Added to output - Label: '{label}', Value: '{value}'")
            elif label_lower in label_data and label_data[label_lower]:
                value = label_data[label_lower]
                display_name = display_names.get(label, label)
                all_text += f"{display_name}: {value}\n"
                all_info[label] = value.strip()
                print(f"Added to output - Label: '{label}' (from '{label_lower}'), Value: '{value}'")
        
        # Print combined parsed information
        print("\nCombined Parsed Information:")
        print("-" * 50)
        for field, value in all_info.items():
            if value:  # Only print fields that were found
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

# Example usage
if __name__ == "__main__":
    system = MultimodalIDSystem()
    
    # Get list of images in the cropped_images directory
    image_dir = "cropped_images"
    image_files = [f for f in os.listdir(image_dir) if f.lower().endswith(('.png', '.jpg', '.jpeg'))]
    
    if not image_files:
        print("No images found in the cropped_images directory!")
        print("Please place your ID card images in the 'cropped_images' directory.")
        exit(1)
    
    print(f"\nFound {len(image_files)} images to process.")
    
    # Process each image
    for image_file in image_files:
        image_path = os.path.join(image_dir, image_file)
        print(f"\n{'='*50}")
        print(f"Processing image: {image_file}")
        print(f"{'='*50}")
        
        try:
            result = system.process_id_card(image_path)
            if result:
                print("\nFinal Results:")
                print(f"ID Type: {result['id_type']}")
                print(f"Cropped Image Paths: {result['cropped_image_paths']}")
                print("\nExtracted Text:")
                print(result['extracted_text'])
                print("\nParsed Information:")
                for field, value in result['parsed_info'].items():
                    if value:
                        print(f"{field}: {value}")
        except Exception as e:
            print(f"Error processing {image_file}: {str(e)}")
            continue 