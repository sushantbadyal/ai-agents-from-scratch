import json
from pathlib import Path

from pydantic import BaseModel

from utils.constants import APP_CONFIG_PATH

class Secrets(BaseModel):
    ANTHROPIC_API_KEY: str
    OPEN_AI_API_KEY: str
    HF_TOKEN: str


class LLMConfigs(BaseModel):
    pass


class SparseRetrievalConfigs(BaseModel):
    MODEL_ID: str
    WIKI_DATASET: str
    DEFAULT_K: int
    DEFAULT_MAX_NEW_TOKENS: int
    INDEX_ROOT: str
    WIKI_INDEX_NAME: str
    SYNTHETIC_INDEX_NAME: str
    LOG_LEVEL: str
    FORCE_REBUILD_INDEX: bool
    VIZ_OUTPUT_DIR: str


class AppConfig(BaseModel):
    SECRETS: Secrets
    LLM_CONFIGS: LLMConfigs
    sparse_retrieval_configs: SparseRetrievalConfigs

    @classmethod
    def load(cls, path: Path = APP_CONFIG_PATH) -> "AppConfig":
        with open(path) as f:
            data = json.load(f)
        return cls.model_validate(data)


config = AppConfig.load()
