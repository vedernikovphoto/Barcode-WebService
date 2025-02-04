import cv2
import pytest
import logging
from fastapi.testclient import TestClient
from omegaconf import OmegaConf
from pathlib import Path

from app import create_app
from src.containers.containers import AppContainer
from src.routes import barcode as barcode_routes

TESTS_DIR = Path(__file__).absolute()
SESSION_SCOPE = 'session'
logger = logging.getLogger(__name__)


@pytest.fixture(scope=SESSION_SCOPE)
def sample_image_bytes():
    """
    Fixture to provide a sample image as bytes.

    Reads a sample image file from the 'images' directory and yields its contents as bytes.
    """
    image_path = TESTS_DIR.parent / 'images' / 'sample_barcode.jpg'
    with image_path.open('rb') as image_file:
        yield image_file.read()


@pytest.fixture
def sample_image_np():
    """
    Fixture to provide a sample image as a NumPy array.

    Reads a sample image file, converts it from BGR to RGB format, and returns it as a NumPy array.
    """
    image_path = TESTS_DIR.parent / 'images' / 'sample_barcode.jpg'
    img = cv2.imread(str(image_path))
    return cv2.cvtColor(img, cv2.COLOR_BGR2RGB)


@pytest.fixture(scope=SESSION_SCOPE)
def app_config():
    """
    Fixture to provide the application configuration.

    Loads the configuration from '/project_root_folder/config/config.yml' using OmegaConf.
    """
    config_path = TESTS_DIR.parents[1] / 'config' / 'config.yml'
    return OmegaConf.load(config_path)


@pytest.fixture(scope=SESSION_SCOPE)
def app_container(app_config):
    """
    Fixture to provide an AppContainer instance configured with the test configuration.

    Returns an instance of AppContainer with its configuration loaded from the test config.
    """
    container = AppContainer()
    container.config.from_dict(app_config)
    return container


@pytest.fixture(scope=SESSION_SCOPE)
def wired_app_container(app_config):
    """
    Fixture to provide a wired AppContainer instance.

    Configures and wires the AppContainer for use with FastAPI routes. Unwires the container after the test session.
    """
    container = AppContainer()
    container.config.from_dict(app_config)
    container.wire([barcode_routes])
    logger.info('Container wired for tests.')
    yield
    container.unwire()


@pytest.fixture(scope=SESSION_SCOPE)
def test_app(wired_app_container):
    """
    Fixture to provide the FastAPI application instance.

    Returns the FastAPI app instance by invoking the create_app factory.
    """
    return create_app()


@pytest.fixture(scope=SESSION_SCOPE)
def client(test_app):
    """
    Fixture to provide a test client for the FastAPI application.

    Returns a TestClient instance for making requests to the FastAPI app in tests.
    """
    return TestClient(test_app)
