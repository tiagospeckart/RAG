from pathlib import Path

PROJECT_ROOT_PATH: Path = Path(__file__).parents[1]
DATA_PATH: Path = PROJECT_ROOT_PATH / "data"
DOCUMENTS_PATH: Path = DATA_PATH / "docs"
DB_PATH: Path = DATA_PATH / "db"
