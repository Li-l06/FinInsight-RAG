from pydantic_settings import BaseSettings, SettingsConfigDict

class Settings(BaseSettings):
    model_config = SettingsConfigDict(
        env_file=".env",
        env_file_encoding="utf-8",
        case_sensitive=False,
        extra="ignore",
    )

    # 应用
    host: str = "0.0.0.0"
    port: int = 9900
    debug: bool = False

    # DashScope
    dashscope_api_key: str = ""
    dashscope_model: str = "qwen-max"
    dashscope_embedding_model: str = "text-embedding-v4"

    # RAG
    chunk_max_size: int = 800
    chunk_overlap: int = 100
    rag_top_k: int = 5

    # ChromaDB
    chroma_persist_dir: str = "./chroma_db"
    chroma_collection_name: str = "fininsight_knowledge"


config = Settings()