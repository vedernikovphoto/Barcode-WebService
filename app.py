import uvicorn
import argparse
from fastapi import FastAPI
from fastapi.responses import HTMLResponse
from omegaconf import OmegaConf

from src.containers.containers import AppContainer
from src.routes.routers import router as app_router
from src.routes import barcode as barcode_routes


DEFAULT_PORT = 5007


def root() -> HTMLResponse:
    """
    The root endpoint that returns an HTML response.

    This function generates a simple HTML page with a welcome message
    and a link to the API documentation.

    Returns:
        HTMLResponse: A response containing the generated HTML content.
    """
    html_content = [
        '<html>',
        '    <head>',
        '        <title>Welcome to My API</title>',
        '    </head>',
        '    <body>',
        '        <h1>Welcome to My API</h1>',
        '        <p>Visit <a href=\"/docs\">/docs</a> for the API documentation.</p>',
        '    </body>',
        '</html>',
    ]
    return HTMLResponse('\n'.join(html_content))


def create_app() -> FastAPI:
    """
    Creates and configures an instance of the FastAPI application.

    This function sets up the dependency injection container, loads the configuration,
    wires dependencies, and includes the API routes.

    Returns:
        FastAPI: The configured FastAPI application instance.
    """
    container = AppContainer()
    cfg = OmegaConf.load('config/config.yml')
    container.config.from_dict(cfg)
    container.wire([barcode_routes])

    app = FastAPI()

    app.get('/', response_class=HTMLResponse)(root)

    app.include_router(app_router, prefix='/barcode', tags=['barcode'])

    return app


def arg_parse():
    """
    Parses command-line arguments to configure the FastAPI application.

    Returns:
        argparse.Namespace: Parsed command-line arguments.
    """
    parser = argparse.ArgumentParser(description='Run the FastAPI application.')
    parser.add_argument('--host', type=str, default='0.0.0.0', help='Host address to run the application')
    parser.add_argument('--port', type=int, default=DEFAULT_PORT, help='Port number to run the application')
    parser.add_argument('--config', type=str, default='config/config.yml', help='Path to the configuration file')

    return parser.parse_args()


if __name__ == '__main__':
    args = arg_parse()
    app = create_app(config_path=args.config)
    uvicorn.run(app, host=args.host, port=args.port)
