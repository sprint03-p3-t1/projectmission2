"""
RFP RAG 시스템 - 데이터 로더
JSON 파일들을 로드하여 RFPDocument 객체로 변환
"""

import os
import json
import logging
from typing import List
from pathlib import Path

from .data_models import RFPDocument, RAGSystemInterface

# 로깅 설정
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class RFPDataLoader(RAGSystemInterface):
    """RFP JSON 데이터 로더"""
    
    def __init__(self, json_dir: str):
        self.json_dir = Path(json_dir)
        self.documents: List[RFPDocument] = []
        self._is_ready = False
    
    def initialize(self):
        """데이터 로더 초기화"""
        self.documents = self.load_documents()
        self._is_ready = True
    
    def is_ready(self) -> bool:
        """데이터 로더 준비 상태 확인"""
        return self._is_ready and len(self.documents) > 0
    
    def load_documents(self) -> List[RFPDocument]:
        """JSON 파일들을 로드하여 RFPDocument 객체로 변환"""
        logger.info(f"Loading RFP documents from {self.json_dir}")
        
        if not self.json_dir.exists():
            logger.error(f"JSON directory does not exist: {self.json_dir}")
            return []
        
        json_files = list(self.json_dir.glob("*.json"))
        logger.info(f"Found {len(json_files)} JSON files")
        
        documents = []
        for json_file in json_files:
            try:
                with open(json_file, 'r', encoding='utf-8') as f:
                    data = json.load(f)
                
                # JSON 구조 파싱
                csv_meta = data.get('csv_metadata', {})
                missing_values = data.get('missing_values', {})
                pdf_data = data.get('pdf_data', [])
                
                # RFPDocument 객체 생성
                doc = RFPDocument(
                    doc_id=json_file.stem,
                    공고번호=csv_meta.get('공고 번호', ''),
                    사업명=csv_meta.get('사업명', ''),
                    발주기관=csv_meta.get('발주 기관', ''),
                    사업금액=csv_meta.get('사업 금액', 0),
                    공개일자=csv_meta.get('공개 일자', ''),
                    입찰시작일=csv_meta.get('입찰 참여 시작일', ''),
                    입찰마감일=csv_meta.get('입찰 참여 마감일', ''),
                    사업요약=csv_meta.get('사업 요약', ''),
                    파일명=csv_meta.get('파일명', ''),
                    pdf_pages=pdf_data,
                    missing_info=missing_values
                )
                
                documents.append(doc)
                
            except Exception as e:
                logger.error(f"Error loading {json_file}: {e}")
        
        self.documents = documents
        self._is_ready = True
        logger.info(f"Successfully loaded {len(documents)} RFP documents")
        return documents
    
    def get_documents(self) -> List[RFPDocument]:
        """로드된 문서들 반환"""
        return self.documents
    
    def get_document_by_id(self, doc_id: str) -> RFPDocument:
        """문서 ID로 특정 문서 검색"""
        for doc in self.documents:
            if doc.doc_id == doc_id:
                return doc
        return None
    
    def search_documents_by_metadata(self, **filters) -> List[RFPDocument]:
        """메타데이터 기반 문서 검색"""
        results = []
        
        for doc in self.documents:
            match = True
            
            if '발주기관' in filters and doc.발주기관 != filters['발주기관']:
                match = False
            
            # 사업금액 비교 (정수형이면 그대로 사용, 문자열이면 변환)
            if '최소금액' in filters:
                try:
                    if isinstance(doc.사업금액, int):
                        doc_amount = doc.사업금액
                    else:
                        doc_amount = int(str(doc.사업금액).replace(',', '').replace('원', '').strip())
                    if doc_amount < filters['최소금액']:
                        match = False
                except (ValueError, AttributeError):
                    # 사업금액이 숫자로 변환되지 않는 경우 해당 문서 제외
                    match = False
                    
            if '최대금액' in filters:
                try:
                    if isinstance(doc.사업금액, int):
                        doc_amount = doc.사업금액
                    else:
                        doc_amount = int(str(doc.사업금액).replace(',', '').replace('원', '').strip())
                    if doc_amount > filters['최대금액']:
                        match = False
                except (ValueError, AttributeError):
                    # 사업금액이 숫자로 변환되지 않는 경우 해당 문서 제외
                    match = False
                    
            if '키워드' in filters and filters['키워드'].lower() not in doc.사업명.lower():
                match = False
            
            if match:
                results.append(doc)
        
        return results
    
    def get_summary_statistics(self) -> dict:
        """문서 통계 정보 반환"""
        if not self.documents:
            return {"message": "로드된 문서가 없습니다."}
        
        # 발주기관별 통계
        agencies = [doc.발주기관 for doc in self.documents if doc.발주기관]
        agency_stats = {}
        for agency in set(agencies):
            agency_stats[agency] = agencies.count(agency)
        
        # 사업금액 통계
        amounts = [doc.사업금액 for doc in self.documents if isinstance(doc.사업금액, (int, float)) and doc.사업금액 > 0]
        amount_stats = {}
        if amounts:
            import numpy as np
            amount_stats = {
                "평균": int(np.mean(amounts)),
                "최대": max(amounts),
                "최소": min(amounts),
                "총합": sum(amounts)
            }
        
        # 최근 공고
        sorted_docs = sorted(self.documents, key=lambda x: x.공개일자, reverse=True)
        recent_docs = []
        for doc in sorted_docs[:5]:
            recent_docs.append({
                "공고번호": doc.공고번호,
                "사업명": doc.사업명,
                "발주기관": doc.발주기관,
                "공개일자": doc.공개일자,
                "사업금액": doc.사업금액
            })
        
        return {
            "총_문서_수": len(self.documents),
            "발주기관별_문서_수": agency_stats,
            "사업금액_통계": amount_stats,
            "최근_공고": recent_docs
        }
