import numpy as np
from typing import List, Tuple
import cv2


def prepare_detection_input(       # noqa: WPS210
    image: np.ndarray,
    input_size: np.ndarray,
) -> Tuple[np.ndarray, float, float]:
    """
    Resizes and normalizes the image for the detection model.

    Args:
        image (np.ndarray): Input image.
        input_size (np.ndarray): Model's required input size [batch_size, height, width, channels].

    Returns:
        Tuple[np.ndarray, float, float]:
            - Preprocessed input tensor for the detection model.
            - Horizontal scale factor.
            - Vertical scale factor.
    """
    model_h, model_w = input_size[1], input_size[2]

    resized = cv2.resize(image, (model_w, model_h))
    normalized = resized.transpose(2, 0, 1) / 255.0     # noqa: WPS432
    input_tensor = np.expand_dims(normalized, axis=0).astype(np.float32)

    scale_x = image.shape[1] / float(model_w)
    scale_y = image.shape[0] / float(model_h)

    return input_tensor, scale_x, scale_y


def filter_raw_detections(
    raw_detections: np.ndarray,
    confidence_threshold: float,
) -> List[np.ndarray]:
    """
    Filters raw detections based on confidence threshold and minimum element count.

    Args:
        raw_detections (np.ndarray): Array of raw detections.
        confidence_threshold (float): Minimum confidence score to keep a detection.

    Returns:
        List[np.ndarray]: Filtered list of valid detections.
    """
    filtered = []
    for det in raw_detections:
        # det is expected to have at least 6 elements: [x_center, y_center, w, h, conf, class_id]
        if len(det) < 6:
            continue
        if det[4] < confidence_threshold:
            continue
        filtered.append(det)
    return filtered


def scale_detection(       # noqa: WPS211
    x_center: float,
    y_center: float,
    width: float,
    height: float,
    scale_x: float,
    scale_y: float,
) -> Tuple[int, int, int, int]:
    """
    Scales center-based bounding box coordinates to corner-based coordinates and original image size.

    Args:
        x_center (float): Center x-coordinate.
        y_center (float): Center y-coordinate.
        width (float): Bounding box width.
        height (float): Bounding box height.
        scale_x (float): Horizontal scale factor.
        scale_y (float): Vertical scale factor.

    Returns:
        Tuple[int, int, int, int]: Corner coordinates (x_min, y_min, x_max, y_max) scaled to original size.
    """
    x_min = int((x_center - width / 2) * scale_x)
    y_min = int((y_center - height / 2) * scale_y)
    x_max = int((x_center + width / 2) * scale_x)
    y_max = int((y_center + height / 2) * scale_y)
    return x_min, y_min, x_max, y_max


def nms(boxes: np.ndarray, scores: np.ndarray, iou_thres: float = 0.45) -> List[int]:        # noqa: WPS210
    """
    Performs Non-Maximum Suppression (NMS) to filter overlapping bounding boxes.

    Args:
        boxes (np.ndarray): Array of bounding boxes with shape [N, 4] (x_min, y_min, x_max, y_max).
        scores (np.ndarray): Confidence scores for each bounding box.
        iou_thres (float): Intersection-over-Union (IoU) threshold for suppression.

    Returns:
        List[int]: Indices of bounding boxes to keep.
    """
    x1 = boxes[:, 0]
    y1 = boxes[:, 1]
    x2 = boxes[:, 2]
    y2 = boxes[:, 3]

    order = scores.argsort()[::-1]
    keep = []

    while order.size > 0:
        ind = order[0]
        keep.append(ind)

        remaining_order = order[1:]     # noqa: WPS204
        xx1 = np.maximum(x1[ind], x1[remaining_order])
        yy1 = np.maximum(y1[ind], y1[remaining_order])
        xx2 = np.minimum(x2[ind], x2[remaining_order])
        yy2 = np.minimum(y2[ind], y2[remaining_order])

        width = np.maximum(0, xx2 - xx1 + 1)
        height = np.maximum(0, yy2 - yy1 + 1)
        inter = width * height

        width_i = x2[ind] - x1[ind] + 1
        height_i = y2[ind] - y1[ind] + 1
        area_i = width_i * height_i

        width_others = x2[remaining_order] - x1[remaining_order] + 1
        height_others = y2[remaining_order] - y1[remaining_order] + 1
        area_others = width_others * height_others

        union = area_i + area_others - inter
        iou = inter / union

        remaining = np.where(iou <= iou_thres)[0]
        order = order[remaining + 1]

    return keep
