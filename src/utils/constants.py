import os
from pathlib import Path

# Path to the JSON config holding API keys and app settings.
# Keep that file OUTSIDE this repository and point to it with the
# APP_CONFIG_PATH environment variable, e.g.
#   export APP_CONFIG_PATH="$HOME/secrets/ai-agents-from-scratch.json"
APP_CONFIG_PATH = Path(
    os.getenv("APP_CONFIG_PATH", "~/.config/ai-agents-from-scratch/config.json")
).expanduser()
