"""
통합 RAG 시스템 설정
두 시스템(FAISS, ChromaDB)의 설정을 통합 관리
"""

import os
import torch
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, Optional

try:
    BASE_DIR = Path(__file__).resolve().parent.parent.parent
except NameError:
    BASE_DIR = Path(os.getcwd()).resolve()

# 경로 설정
DATA_DIR = BASE_DIR / "data"
RAW_DIR = DATA_DIR / "raw"
PROCESSED_DIR = DATA_DIR / "preprocess"
CACHE_DIR = DATA_DIR / "cache"

# 통합 캐시 디렉토리 (모든 시스템이 공유)
UNIFIED_CACHE_DIR = CACHE_DIR

# FAISS 시스템 캐시 경로 (통합 캐시 내부)
FAISS_CACHE_DIR = UNIFIED_CACHE_DIR / "faiss"
FAISS_CHUNKS_PATH = FAISS_CACHE_DIR / "chunks.pkl"
FAISS_EMBEDDINGS_PATH = FAISS_CACHE_DIR / "embeddings.npy"
FAISS_METADATA_PATH = FAISS_CACHE_DIR / "metadata.pkl"

# ChromaDB 시스템 캐시 경로 (통합 캐시 내부)
CHROMA_DB_DIR = BASE_DIR / "chroma_db"
CHROMA_CACHE_DIR = UNIFIED_CACHE_DIR / "chromadb"
RERANK_CACHE_DIR = CHROMA_CACHE_DIR / "rerank_embeddings"
CHROMA_JSON_CACHE_PATH = CHROMA_CACHE_DIR / "cached_json_docs.pkl"
CHROMA_CSV_CACHE_PATH = CHROMA_CACHE_DIR / "cached_csv_docs.pkl"
CHROMA_BM25_PATH = CHROMA_CACHE_DIR / "bm25_index.pkl"

# 메타데이터 경로
META_CSV_PATH = DATA_DIR / "data_list_cleaned.csv"
CHUNK_CSV_PATH = DATA_DIR / "merged_final.csv"

# 모델 설정
FAISS_EMBEDDER_MODEL = "BAAI/bge-m3"
CHROMA_EMBEDDER_MODEL = "nlpai-lab/KURE-v1"
CHROMA_RERANKER_MODEL = "BM-K/KoSimCSE-roberta"

@dataclass
class SystemConfig:
    """개별 시스템 설정"""
    name: str
    embedder_model: str
    vector_db_type: str
    cache_dir: Path
    persist_directory: Optional[Path] = None
    reranker_model: Optional[str] = None
    tokenizer_engine: str = "kiwi"
    rerank_max_length: int = 512
    rerank_cache_dir: Optional[Path] = None

@dataclass
class UnifiedConfig:
    """통합 설정 클래스"""
    
    # 기본 경로
    base_dir: Path = BASE_DIR
    data_dir: Path = DATA_DIR
    raw_dir: Path = RAW_DIR
    processed_dir: Path = PROCESSED_DIR
    cache_dir: Path = CACHE_DIR
    
    # 메타데이터 경로
    meta_csv_path: Path = META_CSV_PATH
    chunk_csv_path: Path = CHUNK_CSV_PATH
    
    # 시스템 설정
    use_gpu: bool = torch.cuda.is_available()
    device: str = "cuda" if torch.cuda.is_available() else "cpu"
    
    # OpenAI API 키
    openai_api_key: str = os.getenv("OPENAI_API_KEY")
    
    # 모델 설정
    faiss_model_name: str = os.getenv("MODEL_NAME", "gpt-4o-mini")
    chroma_model_name: str = os.getenv("MODEL_NAME", "gpt-4o-mini")
    
    # 하이퍼파라미터
    chunk_size: int = 1000
    chunk_overlap: int = 200
    top_k: int = 5
    batch_size: int = 32
    learning_rate: float = 1e-4
    optimizer: str = "adam"
    
    # 기본 시스템 (사용자가 선택하지 않을 때)
    default_system: str = "faiss"  # "faiss" 또는 "chromadb"
    
    # 시스템별 설정
    systems: Dict[str, SystemConfig] = None
    
    def __post_init__(self):
        if self.systems is None:
            self.systems = {
                "faiss": SystemConfig(
                    name="FAISS System",
                    embedder_model=FAISS_EMBEDDER_MODEL,
                    vector_db_type="faiss",
                    cache_dir=FAISS_CACHE_DIR,
                    persist_directory=None,
                    reranker_model=None,
                    tokenizer_engine="kiwi"
                ),
                "chromadb": SystemConfig(
                    name="ChromaDB Hybrid System", 
                    embedder_model=CHROMA_EMBEDDER_MODEL,
                    vector_db_type="chromadb",
                    cache_dir=CHROMA_CACHE_DIR,
                    rerank_cache_dir=RERANK_CACHE_DIR,
                    persist_directory=CHROMA_DB_DIR,
                    reranker_model=CHROMA_RERANKER_MODEL,
                    tokenizer_engine="kiwi",
                    rerank_max_length=512
                )
            }
    
    def get_system_config(self, system_name: str) -> SystemConfig:
        """시스템별 설정 반환"""
        return self.systems.get(system_name)
    
    def get_available_systems(self) -> list:
        """사용 가능한 시스템 목록 반환"""
        return list(self.systems.keys())
    
    def get_system_info(self, system_name: str) -> dict:
        """시스템 정보 반환"""
        config = self.get_system_config(system_name)
        if not config:
            return {}
        
        return {
            "name": config.name,
            "embedder_model": config.embedder_model,
            "vector_db_type": config.vector_db_type,
            "has_reranker": config.reranker_model is not None
        }

# 전역 설정 인스턴스
config = UnifiedConfig()