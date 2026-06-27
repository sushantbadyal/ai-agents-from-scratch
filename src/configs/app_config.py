import json
from pathlib import Path

from pydantic import BaseModel

from utils.constants import APP_CONFIG_PATH

class Secrets(BaseModel):
    ANTHROPIC_API_KEY: str
    OPEN_AI_API_KEY: str


class LLMConfigs(BaseModel):
    pass


class AppConfig(BaseModel):
    SECRETS: Secrets
    LLM_CONFIGS: LLMConfigs

    @classmethod
    def load(cls, path: Path = APP_CONFIG_PATH) -> "AppConfig":
        with open(path) as f:
            data = json.load(f)
        return cls.model_validate(data)


config = AppConfig.load()
