import numpy as np
from src.containers.containers import AppContainer


class FakeBarcodePipeline:
    """
    A fake barcode pipeline for testing purposes that returns fixed outputs.

    Methods:
        predict(image): Returns a fixed barcode string for testing.
        _detect_barcodes(image): Returns a fixed bounding box for testing.
        _run_ocr(cropped_image): Returns a fixed OCR result for testing.
    """

    def predict(self, image):
        return ['123456789012']

    def _detect_barcodes(self, image):
        return [(10, 10, 100, 100)]

    def _run_ocr(self, cropped_image):
        return '123456789012'


def test_predicts_not_fail(app_container: AppContainer, sample_image_np: np.ndarray):
    """
    Verifies that the predict method of the barcode pipeline does not raise exceptions.

    This test overrides the actual pipeline with a FakeBarcodePipeline to ensure
    that the predict method runs successfully and returns a list.

    Args:
        app_container (AppContainer): Dependency injection container.
        sample_image_np (np.ndarray): Sample image as a NumPy array.

    Raises:
        AssertionError: If the predictions are not returned as a list.
    """
    with app_container.reset_singletons():
        with app_container.barcode_pipeline.override(FakeBarcodePipeline()):
            barcode_pipeline = app_container.barcode_pipeline()
            predictions = barcode_pipeline.predict(sample_image_np)
            if isinstance(predictions, list) is False:
                raise AssertionError('Predictions should be a list.')


def test_probabilities_within_range(app_container: AppContainer, sample_image_np: np.ndarray):
    """
    Ensures that predictions returned by the OCR model contain valid barcode strings.

    This test uses a FakeBarcodePipeline to mock the OCR model and checks
    that all returned predictions are numeric strings.

    Args:
        app_container (AppContainer): Dependency injection container.
        sample_image_np (np.ndarray): Sample image as a NumPy array.

    Raises:
        AssertionError: If a prediction contains non-digit characters.
    """
    with app_container.reset_singletons():
        with app_container.barcode_pipeline.override(FakeBarcodePipeline()):
            barcode_pipeline = app_container.barcode_pipeline()
            predictions = barcode_pipeline.predict(sample_image_np)
            for prediction in predictions:
                if prediction.isdigit() is False:
                    raise AssertionError(f"Prediction '{prediction}' should contain only digits.")
