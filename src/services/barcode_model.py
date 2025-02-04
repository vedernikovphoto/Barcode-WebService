import numpy as np
import typing as tp
import torch
import onnxruntime as ort
import logging
import yaml
from pathlib import Path
from typing import List, Tuple
from src.services.utils.preprocess_ocr import preprocess_for_ocr
from src.services.utils.predict_utils_ocr import matrix_to_string
from src.services.utils.utils_detection import prepare_detection_input, filter_raw_detections, scale_detection, nms

logging.basicConfig(level=logging.INFO, format='{message}', style='{')
BOXES_KEY = 'boxes'


class BarcodePipeline:
    """
    Handles loading and inference for barcode detection and OCR models.
    """

    def __init__(self, config_path: str) -> None:
        """
        Initializes the BarcodePipeline and loads both models using a unified configuration.

        Args:
            config_path (str): Path to the unified configuration file.
        """
        with open(config_path, 'r') as config_file:
            config = yaml.safe_load(config_file)

        self.config_detection = config['services']['barcode_detection']
        self.config_ocr = config['services']['barcode_ocr']

        self.device_detection = torch.device(self.config_detection['device'])
        self.device_ocr = torch.device(self.config_ocr['device'])
        self.detection_model = self._load_detection_model()
        self.ocr_model = self._load_ocr_model()

    def predict(self, image: np.ndarray) -> tp.List[str]:
        """
        Orchestrates the barcode detection and OCR pipeline.

        Args:
            image (np.ndarray): Input image as a NumPy array (RGB).

        Returns:
            List[str]: A list of unique decoded barcode texts from the detected regions.
        """
        # Detect barcodes in the image
        try:
            detections = self._detect_barcodes(image)
        except RuntimeError as detection_error:
            raise RuntimeError(f'Error during barcode detection: {detection_error}')

        if not detections:
            return []  # Return an empty list if no barcodes are detected

        # Crop detected regions
        cropped_regions = self._crop_regions(image, detections)

        if not cropped_regions:
            raise RuntimeError('No regions were cropped despite detections.')

        # Perform OCR on cropped regions
        ocr_results = []
        for cropped_image in cropped_regions:
            try:        # noqa: WPS229
                ocr_text = self._run_ocr(cropped_image)
                ocr_results.append(ocr_text)
            except RuntimeError as ocr_error:
                ocr_results.append(f'Error during OCR: {ocr_error}')

        return list(set(ocr_results))

    def _load_detection_model(self):
        """
        Loads the ONNX model for barcode detection.

        Returns:
            ONNX InferenceSession object.
        """
        model_path = Path(self.config_detection['model_path'])
        if not model_path.exists():
            raise FileNotFoundError(f'Detection model file not found at {model_path}')

        try:        # noqa: WPS229
            session = ort.InferenceSession(str(model_path), providers=['CPUExecutionProvider'])
            logging.info(f'Loaded detection ONNX model from {model_path} on CPU')
            return session
        except Exception as exception_error:
            raise RuntimeError(f'Failed to load detection ONNX model: {exception_error}')

    def _load_ocr_model(self):
        """
        Loads the TorchScript model for OCR.

        Returns:
            PyTorch JIT model.
        """
        model_path = Path(self.config_ocr['model_path'])
        if not model_path.exists():
            raise FileNotFoundError(f'OCR model file not found at {model_path}')

        try:        # noqa: WPS229
            model = torch.jit.load(str(model_path), map_location=torch.device('cpu'))
            model.eval()
            logging.info(f'Loaded OCR PyTorch model from {model_path} on {self.device_ocr}')
            return model
        except Exception as exception_error:
            raise RuntimeError(f'Failed to load OCR PyTorch model: {exception_error}')

    def _detect_barcodes(self, image: np.ndarray) -> List[Tuple[int, int, int, int]]:       # noqa: WPS210
        """
        Runs the detection model and returns a list of final bounding boxes.
        """
        # Prepare input tensor and scaling factors
        input_tensor, scale_x, scale_y = prepare_detection_input(image, self.config_detection['input_size'])

        # Run inference
        outputs = self.detection_model.run(None, {'images': input_tensor})
        raw_detections = outputs[0][0]

        # Filter out low-confidence or malformed detections
        detections = filter_raw_detections(
            raw_detections,
            self.config_detection['conf_thres'],
        )

        # Create bounding boxes and confidence arrays
        boxes = []
        confidences = []
        for det in detections:
            (x_center, y_center), (width, height), (confidence, _) = (
                det[:2],
                det[2:4],
                det[4:6],
            )
            scaled_box = scale_detection(
                x_center,
                y_center,
                width,
                height,
                scale_x,
                scale_y,
            )
            boxes.append(scaled_box)
            confidences.append(float(confidence))

        # Perform Non-Maximum Suppression (NMS)
        indices = nms(
            np.array(boxes),
            np.array(confidences),
            self.config_detection['iou_thres'],
        )

        # Gather final boxes from the kept indices
        final_bboxes: List[Tuple[int, int, int, int]] = []
        for ind in indices:
            final_bboxes.append(boxes[ind])

        return final_bboxes

    def _crop_regions(self, image: np.ndarray, detections: List[Tuple]) -> List[np.ndarray]:        # noqa: WPS210
        """
        Crops regions of interest (barcodes) from the input image based on detections.

        Args:
            image (np.ndarray): The original image as a NumPy array.
            detections (List[Tuple[int, int, int, int]]): List of bounding boxes (x_min, y_min, x_max, y_max).

        Returns:
            List[np.ndarray]: A list of cropped images corresponding to valid detected barcode regions.
        """
        cropped_images = []

        for ind, (x_min, y_min, x_max, y_max) in enumerate(detections):
            # Validate bounding box coordinates
            if x_min >= x_max or y_min >= y_max:
                message = (
                    f'Skipping invalid bounding box at index {ind}:'
                    f'({x_min}, {y_min}, {x_max}, {y_max})'     # noqa: WPS326
                )
                logging.info(message)
                continue

            # Ensure coordinates are within image bounds
            x_min = max(0, x_min)
            y_min = max(0, y_min)
            x_max = min(image.shape[1], x_max)
            y_max = min(image.shape[0], y_max)

            # Crop the region
            cropped_region = image[y_min:y_max, x_min:x_max]

            # Check if cropped region has valid dimensions
            if cropped_region.size == 0:
                message = (
                    f'Skipping empty cropped region at index {ind} for bbox:'
                    f'for bbox: ({x_min}, {y_min}, {x_max}, {y_max})'       # noqa: WPS326
                )
                logging.info(message)
                continue

            cropped_images.append(cropped_region)

        if not cropped_images:
            logging.info('No valid regions were cropped.')

        return cropped_images

    def _run_ocr(self, cropped_image: np.ndarray) -> str:
        """
        Runs OCR on a cropped image of a barcode.

        Args:
            cropped_image (np.ndarray): Cropped image of a detected barcode.

        Returns:
            str: Decoded content of the barcode.
        """
        # Preprocess the cropped image for OCR
        try:
            preprocessed_image = preprocess_for_ocr(cropped_image)
        except Exception as execption_error:
            raise RuntimeError(f'Error during OCR preprocessing: {execption_error}')

        # Ensure the preprocessed image is on the correct device
        preprocessed_image = preprocessed_image.to(self.device_ocr)
        logging.info(f'device_ocr is {self.device_ocr}')
        self.ocr_model.to(self.device_ocr)

        # Run inference using the OCR model
        with torch.no_grad():
            ocr_output = self.ocr_model(preprocessed_image).cpu()

        # Decode OCR output to text
        vocab = self.config_ocr['vocab']

        return self._decode_ocr_output(ocr_output, vocab)

    def _decode_ocr_output(self, ocr_output: torch.Tensor, vocab: str) -> str:
        """
        Decodes the OCR model output using CTC decoding.

        Args:
            ocr_output (torch.Tensor): Output tensor from the OCR model of shape [time_steps, batch_size, num_classes].
            vocab (str): The vocabulary used for decoding.

        Returns:
            str: Decoded text from the OCR model output.
        """
        string_pred, _ = matrix_to_string(ocr_output, vocab)

        return string_pred[0] if string_pred else ''
