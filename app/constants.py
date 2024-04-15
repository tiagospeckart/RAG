from pathlib import Path

from settings.settings import settings

PROJECT_ROOT_PATH: Path = Path(__file__).parents[1]
DOCUMENTS_PATH: Path = PROJECT_ROOT_PATH / settings().data.local_data_folder / "docs"
DB_PATH: Path = PROJECT_ROOT_PATH / settings().data.local_data_folder / "db"
