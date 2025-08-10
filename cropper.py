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
    # Check both new and old corner labels
    corners = ['top_left_new', 'top_right_new', 'bottom_left_new', 'bottom_right_new',
              'top_left_old', 'top_right_old', 'bottom_left_old', 'bottom_right_old']
    missing = [c for c in corners if c not in coords]
    return missing[0] if missing else None

def estimate_missing_corner(coords):
    missing = find_missing_corner(coords)
    if not missing:
        return coords

    # Determine corner type (new or old)
    corner_type = 'new' if 'new' in missing else 'old'
    base_corner = missing.replace('_new', '').replace('_old', '')

    # Choose reference and target corners
    if base_corner == 'top_left':
        ref1 = f'top_right_{corner_type}'
        ref2 = f'bottom_left_{corner_type}'
        target = f'bottom_right_{corner_type}'
    elif base_corner == 'top_right':
        ref1 = f'top_left_{corner_type}'
        ref2 = f'bottom_right_{corner_type}'
        target = f'bottom_left_{corner_type}'
    elif base_corner == 'bottom_left':
        ref1 = f'top_left_{corner_type}'
        ref2 = f'bottom_right_{corner_type}'
        target = f'top_right_{corner_type}'
    else:  # bottom_right
        ref1 = f'top_right_{corner_type}'
        ref2 = f'bottom_left_{corner_type}'
        target = f'top_left_{corner_type}'

    # Validate that reference corners exist
    if ref1 not in coords or ref2 not in coords or target not in coords:
        print(f"Not enough corners to interpolate corner {missing}")
        return coords

    # Estimate the missing corner by symmetry
    mid_x = (coords[ref1][0] + coords[ref2][0]) / 2
    mid_y = (coords[ref1][1] + coords[ref2][1]) / 2
    coords[missing] = [2 * mid_x - coords[target][0],
                      2 * mid_y - coords[target][1]]
    return coords

def perspective_transform(image, points, corner_type='new'):
    dest_points = np.float32([[0, 0], [500, 0], [500, 300], [0, 300]])
    
    # Ensure all required corners exist
    required_corners = [
        f'top_left_{corner_type}',
        f'top_right_{corner_type}',
        f'bottom_right_{corner_type}',
        f'bottom_left_{corner_type}'
    ]
    
    for corner in required_corners:
        if corner not in points:
            print(f"Missing corner {corner}, cannot apply perspective transform")
            return image

    source_points = np.float32([
        points[f'top_left_{corner_type}'],
        points[f'top_right_{corner_type}'],
        points[f'bottom_right_{corner_type}'],
        points[f'bottom_left_{corner_type}']
    ])
    
    try:
        M = cv2.getPerspectiveTransform(source_points, dest_points)
        return cv2.warpPerspective(image, M, (510, 310))
    except Exception as e:
        print(f"Error during perspective transform: {str(e)}")
        return image

def crop_image(result, img):
    tensor = result.boxes.xyxy.cpu().numpy()
    class_names = result.names
    confidences = result.boxes.conf.cpu().numpy()
    classes = result.boxes.cls.cpu().numpy()

    valid_indices = [i for i, conf in enumerate(confidences) if conf >= 0.5]
    if not valid_indices:
        print("No bounding box meets confidence >= 0.5. Keeping original image.")
        return img

    tensor = tensor[valid_indices]
    confidences = confidences[valid_indices]
    classes = classes[valid_indices]
    labels = [class_names[int(cls)] for cls in classes]

    final_boxes, final_labels = non_max_suppression_fast(tensor, labels, 0.3)

    if len(final_boxes) < 2:
        print("Too few points detected (<2). Keeping original image.")
        return img

    # Chọn box có độ tin cậy cao nhất cho mỗi label
    boxes_by_label = {}
    for box, label, conf in zip(final_boxes, final_labels, confidences):
        if label not in boxes_by_label:
            boxes_by_label[label] = []
        boxes_by_label[label].append((box, conf))

    final_points = {}
    for label, box_conf_list in boxes_by_label.items():
        best_box, _ = max(box_conf_list, key=lambda x: x[1])
        center_point = [(best_box[0] + best_box[2]) / 2, (best_box[1] + best_box[3]) / 2]
        final_points[label] = center_point

    all_x_min = np.min([box[0] for box in final_boxes])
    all_y_min = np.min([box[1] for box in final_boxes])
    all_x_max = np.max([box[2] for box in final_boxes])
    all_y_max = np.max([box[3] for box in final_boxes])
    bbox_area = (all_x_max - all_x_min) * (all_y_max - all_y_min)
    img_area = img.shape[0] * img.shape[1]
    ratio = bbox_area / img_area

    # Kiểm tra đủ góc cho cả new và old
    required_labels_new = {f'{corner}_new' for corner in ['top_left', 'top_right', 'bottom_left', 'bottom_right']}
    required_labels_old = {f'{corner}_old' for corner in ['top_left', 'top_right', 'bottom_left', 'bottom_right']}
    
    has_all_corners_new = set(final_points.keys()) >= required_labels_new
    has_all_corners_old = set(final_points.keys()) >= required_labels_old

    print(f"- Detected region area: {ratio*100:.2f}% of image")

    if has_all_corners_new or has_all_corners_old:
        if ratio > 0.85:
            print("Image is too close-up, keep original.")
            return img
        else:
            corner_type = 'new' if has_all_corners_new else 'old'
            print(f"All 4 corners ({corner_type}) found and conditions met. Proceed to crop.")
            return perspective_transform(img, final_points, corner_type)
    else:
        if ratio > 0.85:
            print("Image is close-up, no further crop needed.")
            return img
        else:
            # Try to estimate the missing corner (new or old)
            missing = find_missing_corner(final_points)
            if missing:
                print(f"Detected missing corner {missing}")
                estimated_points = estimate_missing_corner(final_points.copy())
                corner_type = 'new' if 'new' in missing else 'old'
                print(f"Estimated missing corner ({corner_type}). Proceed to crop.")
                return perspective_transform(img, estimated_points, corner_type)
            else:
                print("Insufficient corners to crop. Keeping original image.")
                return img

def process_image(image_path, output_dir, model):
    img = cv2.imread(image_path)
    if img is None:
        print(f"Error: Cannot read image {image_path}")
        return None

    results = model(image_path)
    os.makedirs(output_dir, exist_ok=True)

    for r in results:
        cropped_img = crop_image(r, img)
        if cropped_img is not None:
            output_filename = f"cropped_{uuid.uuid4().hex}.jpg"
            output_path = os.path.join(output_dir, output_filename)
            cv2.imwrite(output_path, cropped_img)
            print(f"Cropped image saved at: {output_path}")
            return output_filename

    print(f"Insufficient corners to crop image: {image_path}")
    return None

if __name__ == "__main__":
    model = YOLO("model/detect_4goc/cropper.pt")
    image_path = ""
    output_dir = "cropped_images"
    process_image(image_path, output_dir, model)
