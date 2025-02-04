from copy import deepcopy
import numpy as np
from src.containers.containers import AppContainer


def test_predict_dont_mutate_initial_image(app_container: AppContainer, sample_image_np: np.ndarray):
    """
    Ensures that the `predict` method does not mutate the input image.

    This test verifies that the `BarcodePipeline._detect_barcodes` method operates
    on a copy of the input image and does not alter the original image array.

    Args:
        app_container (AppContainer): Dependency injection container.
        sample_image_np (np.ndarray): Sample image as a NumPy array.

    Raises:
        AssertionError: If the input image is mutated during the execution.
    """
    initial_image = deepcopy(sample_image_np)
    barcode_pipeline = app_container.barcode_pipeline()
    barcode_pipeline._detect_barcodes(sample_image_np)      # noqa: WPS437

    if not np.allclose(initial_image, sample_image_np):
        raise AssertionError('The input image was mutated during prediction.')
