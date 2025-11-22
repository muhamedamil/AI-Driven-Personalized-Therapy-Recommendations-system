from pydantic_settings import BaseSettings
from pydantic import Field, ConfigDict

class Settings(BaseSettings):
    model_config = ConfigDict(env_file=".env", extra="allow")

    RAG_GROQ_API_KEY: str
    DATABASE_URL: str
    PGVECTOR_URL: str
    VECTOR_COLLECTION_NAME: str
    HUGGINGFACE_MODEL_NAME: str = "sentence-transformers/all-MiniLM-L12-v2"
    RAG_GROQ_MODEL_NAME: str = "llama-3.3-70b-versatile"
    RAG_GROQ_TEMPERATURE: float = 0.0
    RAG_GROQ_MAX_TOKENS: int = 64

settings = Settings()
