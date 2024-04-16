# start a fastapi server with uvicorn

import uvicorn

from main import app
from settings.settings import unsafe_typed_settings

# Set log_config=None to do not use the uvicorn logging configuration
# https://github.com/tiangolo/fastapi/discussions/7457#discussioncomment-5141108
uvicorn.run(app="app.main:app", host="0.0.0.0", port=unsafe_typed_settings.server.port, log_config=None, reload=True)
