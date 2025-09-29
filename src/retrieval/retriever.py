"""
검색 모듈
문서 청킹, 임베딩, 벡터 검색을 통합하는 리트리버
"""

import os
import logging
from typing import List, Dict, Any, Optional
import numpy as np

import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from ..data_processing import RFPDocument, DocumentChunk, RetrievalResult, RAGSystemInterface
from ..embedding import RFPEmbedder, RFPVectorStore, EmbeddingCacheManager
from .chunker import RFPChunker

# 로깅 설정
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class RFPRetriever(RAGSystemInterface):
    """RFP 특화 검색기"""
    
    def __init__(self, embedding_model: str = None, chunk_size: int = 1000, overlap: int = 200, enable_cache: bool = True, cache_dir: str = None):
        # 환경변수에서 임베딩 모델 가져오기
        if embedding_model is None:
            embedding_model = os.getenv("EMBEDDING_MODEL", "BAAI/bge-m3")
        
        self.embedding_model = embedding_model
        self.chunker = RFPChunker(chunk_size, overlap)
        self.embedder = RFPEmbedder(embedding_model)
        self.vector_store = None
        self.chunks: List[DocumentChunk] = []
        self.cache_manager = EmbeddingCacheManager(cache_dir) if enable_cache else None
        self._is_ready = False
    
    def initialize(self, documents: List[RFPDocument]):
        """리트리버 초기화 - 문서 청킹, 임베딩, 인덱싱"""
        logger.info("Initializing RFP Retriever...")
        
        # 1. 임베딩 모델 초기화
        self.embedder.initialize()
        
        # 2. 벡터 저장소 초기화
        self.vector_store = RFPVectorStore(self.embedder.dimension)
        
        # 3. 문서 청킹
        logger.info("Chunking documents...")
        for doc in documents:
            chunks = self.chunker.chunk_document(doc)
            self.chunks.extend(chunks)
        
        logger.info(f"Created {len(self.chunks)} chunks from {len(documents)} documents")
        
        # 4. 임베딩 생성 (캐시 활용)
        embeddings = None
        
        # 캐시에서 로드 시도
        if self.cache_manager:
            logger.info("Checking embedding cache...")
            cached_data = self.cache_manager.load_embeddings(self.chunks, self.embedding_model)
            if cached_data:
                cached_chunks, embeddings = cached_data
                logger.info("✅ Using cached embeddings - skipping embedding generation!")
        
        # 캐시가 없으면 새로 생성
        if embeddings is None:
            logger.info("Generating new embeddings...")
            batch_size = 32
            all_embeddings = []
            
            for i in range(0, len(self.chunks), batch_size):
                batch_chunks = self.chunks[i:i+batch_size]
                batch_embeddings = self.embedder.embed_chunks(batch_chunks)
                all_embeddings.append(batch_embeddings)
            
            embeddings = np.vstack(all_embeddings)
            
            # 캐시에 저장
            if self.cache_manager:
                logger.info("Saving embeddings to cache...")
                self.cache_manager.save_embeddings(self.chunks, embeddings, self.embedding_model)
        
        # 5. 벡터 저장소에 추가
        self.vector_store.add_chunks(self.chunks, embeddings)
        
        self._is_ready = True
        logger.info("RFP Retriever initialization completed!")
    
    def is_ready(self) -> bool:
        """리트리버 준비 상태 확인"""
        return (self._is_ready and 
                self.embedder.is_ready() and 
                self.vector_store is not None and 
                self.vector_store.is_ready())
    
    def retrieve(self, query: str, k: int = 10, filters: Optional[Dict[str, Any]] = None) -> List[RetrievalResult]:
        """쿼리에 대한 관련 청크 검색"""
        if not self.is_ready():
            raise RuntimeError("Retriever is not initialized. Call initialize() first.")
        
        # 쿼리 임베딩 생성
        query_embedding = self.embedder.embed_query(query)
        
        # 벡터 검색
        results = self.vector_store.search(query_embedding, k * 2)  # 필터링을 위해 더 많이 검색
        
        # 메타데이터 필터링
        if filters:
            filtered_results = []
            for chunk, score in results:
                if self._apply_filters(chunk, filters):
                    filtered_results.append((chunk, score))
            results = filtered_results[:k]
        else:
            results = results[:k]
        
        # RetrievalResult 객체로 변환
        retrieval_results = []
        for rank, (chunk, score) in enumerate(results):
            retrieval_results.append(RetrievalResult(
                chunk=chunk,
                score=score,
                rank=rank + 1
            ))
        
        return retrieval_results
    
    def _apply_filters(self, chunk: DocumentChunk, filters: Dict[str, Any]) -> bool:
        """메타데이터 필터 적용"""
        for key, value in filters.items():
            if chunk.metadata and key in chunk.metadata:
                if chunk.metadata[key] != value:
                    return False
        return True
    
    def get_chunks_by_doc_id(self, doc_id: str) -> List[DocumentChunk]:
        """특정 문서의 모든 청크 반환"""
        return [chunk for chunk in self.chunks if chunk.doc_id == doc_id]
    
    def get_chunk_statistics(self) -> Dict[str, Any]:
        """청크 통계 정보 반환"""
        chunk_types = {}
        for chunk in self.chunks:
            chunk_type = chunk.chunk_type
            chunk_types[chunk_type] = chunk_types.get(chunk_type, 0) + 1
        
        doc_ids = set(chunk.doc_id for chunk in self.chunks)
        
        stats = {
            "총_청크_수": len(self.chunks),
            "문서_수": len(doc_ids),
            "청크_타입별_분포": chunk_types,
            "평균_청크_길이": sum(len(chunk.content) for chunk in self.chunks) / len(self.chunks) if self.chunks else 0
        }
        
        # 캐시 정보 추가
        if self.cache_manager:
            cache_info = self.cache_manager.get_cache_info()
            stats["캐시_정보"] = cache_info
        
        return stats
    
    def clear_embedding_cache(self):
        """임베딩 캐시 삭제"""
        if self.cache_manager:
            self.cache_manager.clear_cache()
            logger.info("Embedding cache cleared")
        else:
            logger.warning("Cache manager not available")
    
    def get_cache_info(self) -> dict:
        """캐시 정보 반환"""
        if self.cache_manager:
            return self.cache_manager.get_cache_info()
        return {"status": "cache_disabled"}

# 사용 예시 및 테스트 함수들
def test_retriever_standalone():
    """리트리버 단독 테스트 함수"""
    from data_processing import RFPDataLoader
    
    # 데이터 로드
    json_dir = "/Users/leeyoungho/develop/ai_study/project/projectmission2/data/preprocess/json"
    data_loader = RFPDataLoader(json_dir)
    data_loader.initialize()
    documents = data_loader.get_documents()
    
    # 리트리버 초기화
    retriever = RFPRetriever()
    retriever.initialize(documents)
    
    # 테스트 쿼리
    test_queries = [
        "한국사학진흥재단 사업 요구사항",
        "입찰 참가 자격",
        "시스템 구축 사업"
    ]
    
    for query in test_queries:
        print(f"\n쿼리: {query}")
        results = retriever.retrieve(query, k=5)
        
        for result in results:
            print(f"  순위: {result.rank}, 점수: {result.score:.3f}")
            print(f"  문서: {result.chunk.doc_id}")
            print(f"  타입: {result.chunk.chunk_type}")
            print(f"  내용: {result.chunk.content[:100]}...")
            print("-" * 50)

if __name__ == "__main__":
    test_retriever_standalone()
