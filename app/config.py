from pydantic_settings import BaseSettings
from pydantic import ConfigDict

class Settings(BaseSettings):
  MODEL_NAME: str
  EMBED_MODEL: str

  # Qdrant
  QDRANT_URL: str
  QDRANT_COLLECTION:str
  LAW_COLLECTION:str

  # Chunking
  CHUNK_SIZE: int
  CHUNK_OVERLAP: int

  # Law
  LAW_PATH: str

  REDIS_URL: str

  model_config = ConfigDict(env_file=".env")


settings = Settings()
