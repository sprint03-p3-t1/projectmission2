"""
임베딩 캐시 관리자
임베딩 결과를 디스크에 저장하고 재사용
"""

import os
import pickle
import hashlib
import logging
from typing import List, Optional, Tuple
import numpy as np
from pathlib import Path

from data_processing import DocumentChunk

# 로깅 설정
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class EmbeddingCacheManager:
    """임베딩 캐시 관리자"""
    
    def __init__(self, cache_dir: str = "data/processed/embeddings_cache"):
        self.cache_dir = Path(cache_dir)
        self.cache_dir.mkdir(parents=True, exist_ok=True)
        
        # 캐시 파일 경로들
        self.chunks_file = self.cache_dir / "chunks.pkl"
        self.embeddings_file = self.cache_dir / "embeddings.npy"
        self.metadata_file = self.cache_dir / "metadata.pkl"
        
        logger.info(f"Embedding cache directory: {self.cache_dir}")
    
    def _compute_chunks_hash(self, chunks: List[DocumentChunk]) -> str:
        """청크들의 해시값 계산 (캐시 유효성 검증용)"""
        # 청크의 내용과 메타데이터를 기반으로 해시 생성
        content_str = ""
        for chunk in chunks:
            content_str += f"{chunk.chunk_id}|{chunk.content}|{str(chunk.metadata)}"
        
        return hashlib.md5(content_str.encode()).hexdigest()
    
    def save_embeddings(self, chunks: List[DocumentChunk], embeddings: np.ndarray, model_name: str):
        """임베딩과 청크들을 캐시에 저장"""
        try:
            # 메타데이터 저장
            metadata = {
                "model_name": model_name,
                "chunks_count": len(chunks),
                "embedding_dimension": embeddings.shape[1],
                "chunks_hash": self._compute_chunks_hash(chunks)
            }
            
            # 파일들 저장
            with open(self.chunks_file, 'wb') as f:
                pickle.dump(chunks, f)
            
            np.save(self.embeddings_file, embeddings)
            
            with open(self.metadata_file, 'wb') as f:
                pickle.dump(metadata, f)
            
            logger.info(f"Saved embeddings cache: {len(chunks)} chunks, model={model_name}")
            
        except Exception as e:
            logger.error(f"Failed to save embeddings cache: {e}")
    
    def load_embeddings(self, chunks: List[DocumentChunk], model_name: str) -> Optional[Tuple[List[DocumentChunk], np.ndarray]]:
        """캐시에서 임베딩 로드 (유효성 검증 포함)"""
        try:
            # 캐시 파일들이 존재하는지 확인
            if not all(f.exists() for f in [self.chunks_file, self.embeddings_file, self.metadata_file]):
                logger.info("Cache files not found")
                return None
            
            # 메타데이터 로드
            with open(self.metadata_file, 'rb') as f:
                metadata = pickle.load(f)
            
            # 캐시 유효성 검증
            current_hash = self._compute_chunks_hash(chunks)
            if metadata.get("chunks_hash") != current_hash:
                logger.info("Cache invalidated: chunks have changed")
                return None
            
            if metadata.get("model_name") != model_name:
                logger.info(f"Cache invalidated: model changed from {metadata.get('model_name')} to {model_name}")
                return None
            
            # 캐시된 데이터 로드
            with open(self.chunks_file, 'rb') as f:
                cached_chunks = pickle.load(f)
            
            cached_embeddings = np.load(self.embeddings_file)
            
            # 데이터 일관성 확인
            if len(cached_chunks) != len(chunks):
                logger.warning("Cache invalidated: chunks count mismatch")
                return None
            
            if cached_embeddings.shape[0] != len(chunks):
                logger.warning("Cache invalidated: embeddings count mismatch")
                return None
            
            logger.info(f"Loaded embeddings from cache: {len(cached_chunks)} chunks, model={model_name}")
            return cached_chunks, cached_embeddings
            
        except Exception as e:
            logger.error(f"Failed to load embeddings cache: {e}")
            return None
    
    def clear_cache(self):
        """캐시 삭제"""
        try:
            for file_path in [self.chunks_file, self.embeddings_file, self.metadata_file]:
                if file_path.exists():
                    file_path.unlink()
            logger.info("Embedding cache cleared")
        except Exception as e:
            logger.error(f"Failed to clear cache: {e}")
    
    def get_cache_info(self) -> dict:
        """캐시 정보 반환"""
        try:
            if not self.metadata_file.exists():
                return {"status": "no_cache"}
            
            with open(self.metadata_file, 'rb') as f:
                metadata = pickle.load(f)
            
            # 파일 크기 계산
            cache_size = sum(f.stat().st_size for f in [self.chunks_file, self.embeddings_file, self.metadata_file] if f.exists())
            
            return {
                "status": "cached",
                "model_name": metadata.get("model_name"),
                "chunks_count": metadata.get("chunks_count"),
                "embedding_dimension": metadata.get("embedding_dimension"),
                "cache_size_mb": cache_size / (1024 * 1024),
                "cache_files": {
                    "chunks": self.chunks_file.exists(),
                    "embeddings": self.embeddings_file.exists(),
                    "metadata": self.metadata_file.exists()
                }
            }
        except Exception as e:
            logger.error(f"Failed to get cache info: {e}")
            return {"status": "error", "error": str(e)}
