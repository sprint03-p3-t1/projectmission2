"""
임베딩 생성 모듈
문서 임베딩 및 벡터화 처리
"""

import os
import logging
from typing import List
import numpy as np

# 임베딩 모델
from sentence_transformers import SentenceTransformer

import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from ..data_processing import DocumentChunk

# 로깅 설정
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class RFPEmbedder:
    """RFP 문서 임베딩 생성기"""
    
    def __init__(self, model_name: str = "BAAI/bge-m3"):
        self.model_name = model_name
        self.model = None
        self.dimension = None
        self._is_ready = False
    
    def initialize(self):
        """임베딩 모델 초기화"""
        logger.info(f"Loading embedding model: {self.model_name}")
        
        # GPU 사용 가능 여부 확인
        import torch
        device = 'cuda' if torch.cuda.is_available() else 'cpu'
        logger.info(f"Using device: {device}")
        
        self.model = SentenceTransformer(self.model_name, device=device)
        self.dimension = self.model.get_sentence_embedding_dimension()
        self._is_ready = True
        logger.info(f"Initialized embedding model: {self.model_name} (dimension: {self.dimension}) on {device}")
    
    def is_ready(self) -> bool:
        """임베딩 모델 준비 상태 확인"""
        return self._is_ready and self.model is not None
    
    def embed_chunks(self, chunks: List[DocumentChunk]) -> np.ndarray:
        """청크들을 임베딩으로 변환"""
        if not self.is_ready():
            raise RuntimeError("Embedder is not initialized. Call initialize() first.")
        
        texts = [chunk.content for chunk in chunks]
        embeddings = self.model.encode(texts, convert_to_numpy=True)
        logger.info(f"Generated embeddings for {len(chunks)} chunks")
        return embeddings
    
    def embed_query(self, query: str) -> np.ndarray:
        """쿼리를 임베딩으로 변환"""
        if not self.is_ready():
            raise RuntimeError("Embedder is not initialized. Call initialize() first.")
        
        return self.model.encode([query], convert_to_numpy=True)[0]
