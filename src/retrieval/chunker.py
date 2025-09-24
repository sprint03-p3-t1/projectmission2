"""
문서 청킹 모듈
RFP 문서를 의미 있는 청크로 분할
"""

import logging
from typing import List, Optional
import tiktoken
import re

import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from data_processing import RFPDocument, DocumentChunk

# 로깅 설정
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class RFPChunker:
    """RFP 문서를 의미 있는 청크로 분할"""
    
    def __init__(self, chunk_size: int = 1000, overlap: int = 200):
        self.chunk_size = chunk_size
        self.overlap = overlap
        self.encoding = tiktoken.encoding_for_model("gpt-3.5-turbo")
    
    def chunk_document(self, doc: RFPDocument) -> List[DocumentChunk]:
        """RFP 문서를 여러 청크로 분할"""
        chunks = []
        
        # 1. 메타데이터 청크
        metadata_content = self._create_metadata_content(doc)
        chunks.append(DocumentChunk(
            chunk_id=f"{doc.doc_id}_metadata",
            doc_id=doc.doc_id,
            content=metadata_content,
            chunk_type="metadata",
            metadata={
                "공고번호": doc.공고번호,
                "사업명": doc.사업명,
                "발주기관": doc.발주기관,
                "사업금액": doc.사업금액
            }
        ))
        
        # 2. 사업 요약 청크
        if doc.사업요약:
            chunks.append(DocumentChunk(
                chunk_id=f"{doc.doc_id}_summary",
                doc_id=doc.doc_id,
                content=f"사업 요약:\n{doc.사업요약}",
                chunk_type="summary",
                metadata={"공고번호": doc.공고번호, "사업명": doc.사업명}
            ))
        
        # 3. PDF 페이지별 텍스트 청크
        for page in doc.pdf_pages:
            page_num = page.get('page', 0)
            page_text = page.get('text', '')
            
            if page_text.strip():
                # 긴 텍스트는 더 작은 청크로 분할
                text_chunks = self._split_text(page_text, self.chunk_size, self.overlap)
                
                for i, chunk_text in enumerate(text_chunks):
                    chunks.append(DocumentChunk(
                        chunk_id=f"{doc.doc_id}_page_{page_num}_{i}",
                        doc_id=doc.doc_id,
                        content=chunk_text,
                        chunk_type="page_text",
                        page_number=page_num,
                        metadata={"공고번호": doc.공고번호, "사업명": doc.사업명}
                    ))
            
            # 4. 테이블 청크
            tables = page.get('tables', [])
            for j, table in enumerate(tables):
                table_text = self._table_to_text(table)
                if table_text.strip():
                    chunks.append(DocumentChunk(
                        chunk_id=f"{doc.doc_id}_page_{page_num}_table_{j}",
                        doc_id=doc.doc_id,
                        content=f"테이블 (페이지 {page_num}):\n{table_text}",
                        chunk_type="table",
                        page_number=page_num,
                        metadata={"공고번호": doc.공고번호, "사업명": doc.사업명}
                    ))
        
        return chunks
    
    def _create_metadata_content(self, doc: RFPDocument) -> str:
        """메타데이터를 텍스트로 변환"""
        # 사업금액 포맷팅
        if isinstance(doc.사업금액, (int, float)) and doc.사업금액 > 0:
            amount_str = f"{int(doc.사업금액):,}원"
        else:
            amount_str = str(doc.사업금액)
        
        return f"""
공고번호: {doc.공고번호}
사업명: {doc.사업명}
발주기관: {doc.발주기관}
사업금액: {amount_str}
공개일자: {doc.공개일자}
입찰 참여 시작일: {doc.입찰시작일}
입찰 참여 마감일: {doc.입찰마감일}
파일명: {doc.파일명}
        """.strip()
    
    def _split_text(self, text: str, chunk_size: int, overlap: int) -> List[str]:
        """텍스트를 토큰 기준으로 분할"""
        tokens = self.encoding.encode(text)
        chunks = []
        
        start = 0
        while start < len(tokens):
            end = start + chunk_size
            chunk_tokens = tokens[start:end]
            chunk_text = self.encoding.decode(chunk_tokens)
            chunks.append(chunk_text)
            
            if end >= len(tokens):
                break
            
            start = end - overlap
        
        return chunks
    
    def _table_to_text(self, table: List[List[str]]) -> str:
        """테이블을 텍스트로 변환"""
        if not table:
            return ""
        
        text_lines = []
        for row in table:
            row_text = " | ".join(str(cell).strip() for cell in row if cell)
            if row_text:
                text_lines.append(row_text)
        
        return "\n".join(text_lines)