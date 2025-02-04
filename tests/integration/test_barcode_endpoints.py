from fastapi.testclient import TestClient
from http import HTTPStatus


def test_detect(client: TestClient, sample_image_bytes: bytes):
    """
    Tests the barcode detection endpoint.

    Sends an image to the `/barcode/detect` endpoint and verifies:
    1. The HTTP response status is 200 (OK).
    2. The response contains a 'detections' field, which is a list of bounding boxes.

    Args:
        client (TestClient): Instance of FastAPI test client.
        sample_image_bytes (bytes): Sample image file in bytes format.

    Raises:
        AssertionError: If the response status is not OK or 'detections' is not a list.
    """
    files = {
        'image': sample_image_bytes,
    }
    response = client.post('/barcode/detect', files=files)

    if response.status_code != HTTPStatus.OK:
        raise AssertionError(f'Expected status {HTTPStatus.OK}, got {response.status_code}')

    detections = response.json()['detections']
    if not isinstance(detections, list):
        raise AssertionError(f"Expected 'detections' to be a list, got {type(detections)}")


def test_ocr(client: TestClient, sample_image_bytes: bytes):
    """
    Tests the OCR endpoint for barcode regions.

    Sends an image to the `/barcode/ocr` endpoint and verifies:
    1. The HTTP response status is 200 (OK).
    2. The response contains a 'barcodes' field, which is a list of recognized strings.

    Args:
        client (TestClient): Instance of FastAPI test client.
        sample_image_bytes (bytes): Sample image file in bytes format.

    Raises:
        AssertionError: If the response status is not OK or 'barcodes' is not a list.
    """
    files = {
        'image': sample_image_bytes,
    }
    response = client.post('/barcode/ocr', files=files)

    if response.status_code != HTTPStatus.OK:
        raise AssertionError(f'Expected status {HTTPStatus.OK}, got {response.status_code}')

    barcodes = response.json()['barcodes']
    if not isinstance(barcodes, list):
        raise AssertionError(f"Expected 'barcodes' to be a list, got {type(barcodes)}")
