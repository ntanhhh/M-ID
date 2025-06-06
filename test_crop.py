import os
import cv2
import torch
import numpy as np
from ultralytics import YOLO
import uuid

def non_max_suppression_fast(boxes, labels, overlapThresh):
    if len(boxes) == 0:
        return []
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
        w = np.maximum(0, xx2 - xx1 + 1)
        h = np.maximum(0, yy2 - yy1 + 1)
        overlap = (w * h) / area[idxs[:last]]
        idxs = np.delete(idxs, np.concatenate(([last], np.where(overlap > overlapThresh)[0])))
    return boxes[pick].astype("int"), [labels[idx] for idx in pick]

def find_missing_corner(coords):
    corners = ['top_left', 'top_right', 'bottom_left', 'bottom_right']
    missing = [c for c in corners if c not in coords]
    return missing[0] if missing else None

def estimate_missing_corner(coords):
    missing = find_missing_corner(coords)
    if missing == 'top_left':
        coords['top_left'] = [2 * coords['bottom_left'][0] - coords['bottom_right'][0],
                              2 * coords['top_right'][1] - coords['bottom_right'][1]]
    elif missing == 'top_right':
        coords['top_right'] = [2 * coords['bottom_right'][0] - coords['bottom_left'][0],
                               2 * coords['top_left'][1] - coords['bottom_left'][1]]
    elif missing == 'bottom_left':
        coords['bottom_left'] = [2 * coords['top_left'][0] - coords['top_right'][0],
                                 2 * coords['bottom_right'][1] - coords['top_right'][1]]
    elif missing == 'bottom_right':
        coords['bottom_right'] = [2 * coords['top_right'][0] - coords['top_left'][0],
                                  2 * coords['bottom_left'][1] - coords['top_left'][1]]
    return coords

def perspective_transform(image, points):
    width = int(np.linalg.norm(np.array(points['top_right']) - np.array(points['top_left'])))
    height = int(np.linalg.norm(np.array(points['bottom_left']) - np.array(points['top_left'])))
    dest_points = np.float32([[0, 0], [width-1, 0], [width-1, height-1], [0, height-1]])
    source_points = np.float32([points['top_left'], points['top_right'], points['bottom_right'], points['bottom_left']])
    M = cv2.getPerspectiveTransform(source_points, dest_points)
    return cv2.warpPerspective(image, M, (width, height))

def crop_image(result, img):
    tensor = result.boxes.xyxy.cpu().numpy()
    class_names = result.names
    confidences = result.boxes.conf.cpu().numpy()
    classes = result.boxes.cls.cpu().numpy()

    valid_indices = [i for i, conf in enumerate(confidences) if conf >= 0.5]
    if not valid_indices:
        print("Không có bounding box nào đạt ngưỡng confidence 0.5. Giữ nguyên ảnh.")
        return img

    tensor = tensor[valid_indices]
    confidences = confidences[valid_indices]
    classes = classes[valid_indices]
    labels = [class_names[int(cls)] for cls in classes]

    final_boxes, final_labels = non_max_suppression_fast(tensor, labels, 0.3)

    if len(final_boxes) < 2:
        print("Detect được quá ít điểm (<2). Giữ nguyên ảnh.")
        return img

    # ✅ Chọn box có độ tin cậy cao nhất cho mỗi label
    boxes_by_label = {}
    for box, label, conf in zip(final_boxes, final_labels, confidences):
        if label not in boxes_by_label:
            boxes_by_label[label] = []
        boxes_by_label[label].append((box, conf))

    final_points = {}
    for label, box_conf_list in boxes_by_label.items():
        best_box, _ = max(box_conf_list, key=lambda x: x[1])  # Chọn box có confidence cao nhất
        center_point = [(best_box[0] + best_box[2]) / 2, (best_box[1] + best_box[3]) / 2]
        final_points[label] = center_point

    all_x_min = np.min([box[0] for box in final_boxes])
    all_y_min = np.min([box[1] for box in final_boxes])
    all_x_max = np.max([box[2] for box in final_boxes])
    all_y_max = np.max([box[3] for box in final_boxes])
    bbox_area = (all_x_max - all_x_min) * (all_y_max - all_y_min)
    img_area = img.shape[0] * img.shape[1]
    ratio = bbox_area / img_area

    required_labels = {'top_left', 'top_right', 'bottom_left', 'bottom_right'}
    has_all_corners = set(final_points.keys()) == required_labels

    print(f"Thông số phân tích:")
    print(f"- Diện tích vùng phát hiện: {ratio*100:.2f}% so với ảnh")

    if has_all_corners:
        if ratio > 0.85:
            print("Ảnh đã chụp quá sát, giữ nguyên ảnh, không cắt.")
            return img
        else:
            print("Đủ 4 góc và thỏa mãn các điều kiện. Tiến hành crop.")
            return perspective_transform(img, final_points)
    else:
        if ratio > 0.85:
            print("Ảnh đã chụp sát, không cần cắt thêm.")
            return img
        else:
            # Thử nội suy góc thiếu
            missing = find_missing_corner(final_points)
            if missing:
                print(f"Phát hiện thiếu góc {missing}, thử nội suy...")
                estimated_points = estimate_missing_corner(final_points.copy())
                print("Đã nội suy góc thiếu, tiến hành crop.")
                return perspective_transform(img, estimated_points)
            else:
                print("Ảnh không đủ góc, không thể crop. Giữ nguyên ảnh.")
                return img


def process_image(image_path, output_dir, model):
    img = cv2.imread(image_path)
    if img is None:
        print(f"Lỗi: Không thể đọc ảnh {image_path}")
        return None

    results = model(image_path)
    os.makedirs(output_dir, exist_ok=True)

    for r in results:
        cropped_img = crop_image(r, img)
        if cropped_img is not None:
            output_filename = f"cropped_{uuid.uuid4().hex}.jpg"
            output_path = os.path.join(output_dir, output_filename)
            cv2.imwrite(output_path, cropped_img)
            print(f"✅ Ảnh đã được cắt và lưu tại: {output_path}")
            return output_filename

    print(f"Không đủ góc để cắt ảnh: {image_path}")
    return None

if __name__ == "__main__":
    model = YOLO("runs/detect/train16/weights/best.pt")
    image_path = "dataset/cmnd/712_jpg.rf.58a8a082946c210088323ee8aa1a4237.jpg"
    output_dir = "cropped_images"
    process_image(image_path, output_dir, model)
