from PIL import Image
import numpy as np
import torch
import logging
from src.services.utils.transforms_ocr import get_transforms_ocr
from src.services.utils.config_ocr import ConfigOCR

logging.basicConfig(level=logging.INFO, format='{message}', style='{')
MAX_PIXEL_VALUE = 255.0


def preprocess_for_ocr(image: np.ndarray) -> torch.Tensor:
    """
    Preprocesses an image for the OCR model.

    Args:
        image (np.ndarray): Input image in RGB format as a NumPy array.

    Returns:
        torch.Tensor: Preprocessed image as a PyTorch tensor ready for OCR inference.
    """
    config_path = 'config/config_ocr.yaml'
    config = ConfigOCR.from_yaml(config_path)

    transforms = get_transforms_ocr(
        aug_config=config.augmentation_params,
        width=config.data_config.width,
        height=config.data_config.height,
        text_size=config.data_config.text_size,
        vocab=config.data_config.vocab,
        preprocessing=True,
        augmentations=False,
        postprocessing=True,
    )

    logging.info(f'Config Width: {config.data_config.width}')
    logging.info(f'Config Height: {config.data_config.height}')

    if config.data_config.width <= 0 or config.data_config.height <= 0:
        raise ValueError('Invalid configuration: width or height must be greater than 0.')

    if image.shape[0] == 0 or image.shape[1] == 0:
        raise ValueError('Invalid input image: image dimensions must be greater than 0.')

    if not isinstance(image, np.ndarray):
        raise ValueError('Input image must be a NumPy array.')

    image = Image.fromarray(image).convert('RGB')  # Ensure it's in RGB format
    transformed_image = transforms(image=np.array(image), text='')['image']

    return transformed_image.unsqueeze(0)
