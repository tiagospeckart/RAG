import logging
import os
from pathlib import Path
from typing import Any

from dotenv import load_dotenv

from app.constants import PROJECT_ROOT_PATH
from app.settings.yaml import load_yaml_with_envvars

logger = logging.getLogger(__name__)

_settings_folder = os.environ.get("SETTINGS_FOLDER", PROJECT_ROOT_PATH)

# Load environment variables from .env file
load_dotenv()


def load_settings() -> dict[str, Any]:
    settings_file_name = "settings.yaml"
    path = Path(_settings_folder) / settings_file_name
    with Path(path).open("r") as f:
        config = load_yaml_with_envvars(f)
    if not isinstance(config, dict):
        raise TypeError(f"Config file has no top-level mapping: {path}")
    return config
