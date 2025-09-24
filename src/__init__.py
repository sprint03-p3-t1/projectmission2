"""
RFP RAG 시스템
복잡한 RFP 문서를 분석하고 질의응답하는 AI 시스템
"""

from .rfp_rag_main import RFPRAGSystem

# 각 모듈별 주요 클래스들
from .data_processing import RFPDocument, DocumentChunk, RetrievalResult, RAGResponse, RFPDataLoader
from .retrieval import RFPRetriever, RFPChunker
from .generation import RFPGenerator
from .embedding import RFPEmbedder, RFPVectorStore

__version__ = "1.0.0"

__all__ = [
    # 메인 시스템
    'RFPRAGSystem',
    
    # 데이터 모델
    'RFPDocument',
    'DocumentChunk', 
    'RetrievalResult',
    'RAGResponse',
    
    # 각 모듈 주요 클래스
    'RFPDataLoader',
    'RFPRetriever',
    'RFPChunker',
    'RFPGenerator',
    'RFPEmbedder',
    'RFPVectorStore'
]
