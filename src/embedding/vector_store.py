"""
벡터 저장소 모듈
FAISS 기반 벡터 검색
"""

import logging
from typing import List, Optional, Tuple
import numpy as np

# 벡터 DB
import faiss

import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from ..data_processing import DocumentChunk

# 로깅 설정
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class RFPVectorStore:
    """RFP 벡터 저장소"""
    
    def __init__(self, dimension: int):
        self.dimension = dimension
        self.index = faiss.IndexFlatIP(dimension)  # Inner Product (cosine similarity)
        self.chunks: List[DocumentChunk] = []
        self.embeddings: Optional[np.ndarray] = None
        self._is_ready = False
    
    def add_chunks(self, chunks: List[DocumentChunk], embeddings: np.ndarray):
        """청크와 임베딩을 벡터 저장소에 추가"""
        # 임베딩 정규화 (cosine similarity를 위해)
        embeddings_normalized = embeddings / np.linalg.norm(embeddings, axis=1, keepdims=True)
        
        self.index.add(embeddings_normalized)
        self.chunks.extend(chunks)
        
        if self.embeddings is None:
            self.embeddings = embeddings_normalized
        else:
            self.embeddings = np.vstack([self.embeddings, embeddings_normalized])
        
        self._is_ready = True
        logger.info(f"Added {len(chunks)} chunks to vector store. Total: {len(self.chunks)}")
    
    def search(self, query_embedding: np.ndarray, k: int = 10) -> List[Tuple[DocumentChunk, float]]:
        """유사한 청크 검색"""
        if not self._is_ready:
            raise RuntimeError("Vector store is not ready. Add chunks first.")
        
        query_normalized = query_embedding / np.linalg.norm(query_embedding)
        query_normalized = query_normalized.reshape(1, -1)
        
        scores, indices = self.index.search(query_normalized, k)
        
        results = []
        for score, idx in zip(scores[0], indices[0]):
            if idx < len(self.chunks):
                results.append((self.chunks[idx], float(score)))
        
        return results
    
    def is_ready(self) -> bool:
        """벡터 저장소 준비 상태 확인"""
        return self._is_ready and len(self.chunks) > 0
