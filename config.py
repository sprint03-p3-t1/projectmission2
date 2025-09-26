"""
설정 파일 - 기존 설정 100% 보존
"""
from dataclasses import dataclass
from pathlib import Path
import os
import torch

try:
    BASE_DIR = Path(__file__).resolve().parent
except NameError:
    BASE_DIR = Path(os.getcwd()).resolve()

# 경로 설정 (기존과 동일)
DATA_DIR = BASE_DIR / "data"
RAW_DIR = DATA_DIR / "raw"
JSON_DIR = DATA_DIR / "preprocess/json"
OUTPUT_DIR = DATA_DIR / "processed"
CACHE_DIR = DATA_DIR / "cache"
CHROMA_DB_DIR = BASE_DIR / "chroma_db"
META_CSV_PATH = DATA_DIR / "data_list_cleaned.csv"
CHUNK_CSV_PATH = DATA_DIR / "merged_final.csv"
RERANK_CACHE_DIR = CACHE_DIR / "rerank_embeddings"

@dataclass
class Config:
    """기존 Config 클래스 - 모든 설정 보존"""
    base_dir: Path = BASE_DIR
    data_dir: Path = DATA_DIR
    raw_dir: Path = RAW_DIR
    json_dir: Path = JSON_DIR
    output_dir: Path = OUTPUT_DIR
    meta_csv_path: Path = META_CSV_PATH
    chunk_csv_path: Path = CHUNK_CSV_PATH
    rerank_cache_dir: Path = RERANK_CACHE_DIR
    bm25_path: Path = CACHE_DIR / "bm25_index.pkl"
    
    # 시스템 설정 (기존과 동일)
    use_gpu: bool = torch.cuda.is_available()
    device: str = "cuda" if torch.cuda.is_available() else "cpu"

    # 학습 설정 (기존 유지)
    batch_size: int = 32
    learning_rate: float = 1e-4
    optimizer: str = "adam"

    # 모델 설정 (기존과 동일)
    tokenizer_engine: str = "kiwi"
    embedder_model: str = "nlpai-lab/KURE-v1"
    reranker_model: str = "BM-K/KoSimCSE-roberta"
    rerank_max_length: int = 512
    
    # 파일 경로 (기존과 동일)
    cached_json_path: str = str(CACHE_DIR / "chromadb/cached_json_docs.pkl")
    cached_csv_path: str = str(CACHE_DIR / "chromadb/cached_csv_docs.pkl")
    chroma_db_path: str = str(CHROMA_DB_DIR)
