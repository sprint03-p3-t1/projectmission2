"""
RFP RAG 시스템 - 데이터 모델 정의
공통으로 사용되는 데이터 클래스들
"""

from dataclasses import dataclass
from typing import List, Dict, Any, Optional
from pydantic import BaseModel

@dataclass
class RFPDocument:
    """RFP 문서 데이터 클래스"""
    doc_id: str
    공고번호: str
    사업명: str
    발주기관: str
    사업금액: int
    공개일자: str
    입찰시작일: str
    입찰마감일: str
    사업요약: str
    파일명: str
    pdf_pages: List[Dict[str, Any]]
    missing_info: Dict[str, bool]

@dataclass
class DocumentChunk:
    """문서 청크 데이터 클래스"""
    chunk_id: str
    doc_id: str
    content: str
    chunk_type: str  # 'metadata', 'summary', 'page_text', 'table'
    page_number: Optional[int] = None
    metadata: Optional[Dict[str, Any]] = None

@dataclass
class RetrievalResult:
    """검색 결과 데이터 클래스"""
    chunk: DocumentChunk
    score: float
    rank: int

class RAGResponse(BaseModel):
    """RAG 시스템 응답 데이터 클래스"""
    question: str
    answer: str
    retrieved_chunks: List[RetrievalResult]  # RetrievalResult 객체 사용
    generation_metadata: Dict[str, Any]  # 토큰 사용량, 응답 시간 등
    
class RAGSystemInterface:
    """RAG 시스템 컴포넌트들의 인터페이스 정의"""
    
    def initialize(self):
        """컴포넌트 초기화"""
        raise NotImplementedError
    
    def is_ready(self) -> bool:
        """컴포넌트 준비 상태 확인"""
        raise NotImplementedError
