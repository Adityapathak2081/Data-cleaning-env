import os
import sys
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

try:
    from openenv.core.env_server import create_fastapi_app
except ImportError:
    from openenv.core.env_server.http_server import create_app as create_fastapi_app

try:
    from .data_cleaning_environment import DataCleaningEnvironment
except ImportError:
    from data_cleaning_environment import DataCleaningEnvironment

app = create_fastapi_app(DataCleaningEnvironment)

def main(host: str = "0.0.0.0", port: int = 7860):
    import uvicorn
    uvicorn.run(app, host=host, port=port)

if __name__ == "__main__":
    main()