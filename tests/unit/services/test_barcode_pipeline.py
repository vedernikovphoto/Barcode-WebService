import numpy as np
from src.containers.containers import AppContainer


def test_predicts_not_fail(app_container: AppContainer, sample_image_np: np.ndarray):
    """
    Ensures that the prediction methods of the BarcodePipeline do not raise exceptions.

    This test verifies that the `predict` method can be called on a sample image
    without errors and returns a list.

    Args:
        app_container (AppContainer): Dependency injection container.
        sample_image_np (np.ndarray): Sample image as a NumPy array.

    Raises:
        AssertionError: If the predictions are not returned as a list.
    """
    barcode_pipeline = app_container.barcode_pipeline()
    predictions = barcode_pipeline.predict(sample_image_np)
    if isinstance(predictions, list) is False:
        raise AssertionError('Predictions should be a list.')


def test_detection_output_within_image_bounds(app_container: AppContainer, sample_image_np: np.ndarray):
    """
    Validates that detected bounding boxes are within the image boundaries.

    This test ensures that the coordinates of the bounding boxes returned by the
    `_detect_barcodes` method do not exceed the dimensions of the input image.

    Args:
        app_container (AppContainer): Dependency injection container.
        sample_image_np (np.ndarray): Sample image as a NumPy array.

    Raises:
        AssertionError: If any bounding box is out of image bounds.
    """
    barcode_pipeline = app_container.barcode_pipeline()
    detections = barcode_pipeline._detect_barcodes(sample_image_np)  # noqa: WPS437

    img_h, img_w = sample_image_np.shape[:2]
    _validate_bounding_boxes(detections, img_w, img_h)


def test_unique_predictions(app_container: AppContainer, sample_image_np: np.ndarray):
    """
    Ensures that the `predict` method returns unique predictions.

    This test checks that the list of predictions returned by the `predict` method
    does not contain duplicates.

    Args:
        app_container (AppContainer): Dependency injection container.
        sample_image_np (np.ndarray): Sample image as a NumPy array.

    Raises:
        AssertionError: If duplicate predictions are found.
    """
    barcode_pipeline = app_container.barcode_pipeline()
    predictions = barcode_pipeline.predict(sample_image_np)
    if len(predictions) != len(set(predictions)):
        raise AssertionError('Predictions contain duplicates.')


def _validate_bounding_boxes(detections, img_w, img_h):     # noqa: WPS231
    """
    Validates that all bounding boxes are within the image dimensions.

    This utility method checks that the bounding box coordinates do not extend
    beyond the boundaries of the image.

    Args:
        detections (list): List of bounding box coordinates (x_min, y_min, x_max, y_max).
        img_w (int): Width of the image.
        img_h (int): Height of the image.

    Raises:
        AssertionError: If any bounding box is out of bounds.
    """
    for x_min, y_min, x_max, y_max in detections:
        if x_min < 0 or x_min >= x_max or x_max > img_w:
            raise AssertionError(f'Bounding box x-coordinates out of bounds: {x_min}, {x_max}')
        if y_min < 0 or y_min >= y_max or y_max > img_h:
            raise AssertionError(f'Bounding box y-coordinates out of bounds: {y_min}, {y_max}')
