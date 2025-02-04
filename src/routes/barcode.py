import cv2
import numpy as np
from dependency_injector.wiring import Provide, inject
from fastapi import Depends, File

from src.containers.containers import AppContainer
from src.routes.routers import router
from src.services.barcode_model import BarcodePipeline


@router.post('/detect')
@inject
def detect_barcodes(
    image: bytes = File(...),
    service: BarcodePipeline = Depends(Provide[AppContainer.barcode_pipeline]),
):
    """
    Detect barcodes in the input image.

    Args:
        image (bytes): Uploaded image file.
        service (BarcodePipeline): Injected BarcodePipeline service.

    Returns:
        dict: Detected barcode bounding boxes.
    """
    img = cv2.imdecode(np.frombuffer(image, np.uint8), cv2.IMREAD_COLOR)
    detections = service._detect_barcodes(img)  # noqa: WPS437

    return {'detections': detections}


@router.post('/ocr')
@inject
def detect_and_ocr(
    image: bytes = File(...),
    service: BarcodePipeline = Depends(Provide[AppContainer.barcode_pipeline]),
):
    """
    Detect barcodes and perform OCR on the input image.

    Args:
        image (bytes): Uploaded image file.
        service (BarcodePipeline): Injected BarcodePipeline service.

    Returns:
        dict: Decoded barcode texts from the detected regions.
    """
    img = cv2.imdecode(np.frombuffer(image, np.uint8), cv2.IMREAD_COLOR)
    ocr_results = service.predict(img)  # Full pipeline: detection + OCR

    return {'barcodes': ocr_results}
