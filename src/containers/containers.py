from dependency_injector import containers, providers

from src.services.barcode_model import BarcodePipeline


class AppContainer(containers.DeclarativeContainer):
    """
    Dependency Injection container for the application.

    This container provides and configures the services for BarcodePipeline.

    Attributes:
        config (providers.Configuration): Configuration settings for the services.
        barcode_pipeline (providers.Singleton): Provides the BarcodePipeline service.
    """

    config = providers.Configuration()

    barcode_pipeline = providers.Singleton(
        BarcodePipeline,
        config_path=config.services.config_path,
    )
