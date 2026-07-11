import os

from dotenv import load_dotenv
from pydantic_settings import BaseSettings, SettingsConfigDict

# pydantic-settings parses .env into this Settings object only — it does not
# populate os.environ. The Langfuse SDK's @observe() decorator reads
# credentials directly from os.environ, so without this, Langfuse silently
# runs disabled even though the keys are present in .env.
load_dotenv()


class Settings(BaseSettings):
    model_config = SettingsConfigDict(env_file=".env", extra="ignore")

    pinecone_api_key: str
    pinecone_index_name: str = "test-sample-data"
    pinecone_cloud: str = "aws"
    pinecone_region: str = "us-east-1"
    pinecone_embedding_dimension: int = 3072

    openai_api_key: str
    embedding_model: str = "text-embedding-3-large"
    llm_model: str = "gpt-4o-mini"

    langfuse_public_key: str = ""
    langfuse_secret_key: str = ""
    langfuse_host: str = "https://cloud.langfuse.com"

    chunk_size: int = 1000
    chunk_overlap: int = 150
    retrieval_top_k: int = 5

    circuit_breaker_failure_threshold: int = 3
    circuit_breaker_cooldown_seconds: int = 30

    ingestion_registry_path: str = ".ingestion_registry.db"


settings = Settings()
