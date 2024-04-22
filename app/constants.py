from pathlib import Path

PROJECT_ROOT_PATH: str = Path(__file__).parents[1]
DATA_PATH: str = PROJECT_ROOT_PATH / "data"
DOCUMENTS_PATH: str = DATA_PATH / "docs"
DB_PATH: str = DATA_PATH / "db"
