"""
RAG 시스템 선택기
두 시스템(FAISS, ChromaDB) 중 하나를 선택하여 사용할 수 있는 통합 인터페이스
"""

import logging
import sys
from typing import Dict, Any, List, Optional
from pathlib import Path

# 프로젝트 루트를 Python 경로에 추가
project_root = Path(__file__).parent.parent.parent
sys.path.insert(0, str(project_root))

from src.config.unified_config import UnifiedConfig
from src.rfp_rag_main import RFPRAGSystem  # 기존 시스템
from src.retrieval.hybrid_retriever import Retriever # 리트리버 
from src.retrieval.rerank import RerankModel # 리랭크 모델
from src.generation.generator import RFPGenerator

logger = logging.getLogger(__name__)

class SystemSelector:
    """RAG 시스템 선택 및 관리 클래스"""
    
    def __init__(self, config: UnifiedConfig):
        self.config = config
        self._systems: Dict[str, Any] = {}
        self._current_system: Optional[str] = None
        
    def initialize_system(self, system_name: str, force_rebuild: bool = False) -> Any:
        """
        특정 시스템 초기화
        
        Args:
            system_name: "faiss" 또는 "chromadb"
            force_rebuild: 강제 재구축 여부
            
        Returns:
            초기화된 시스템 인스턴스
        """
        if system_name in self._systems and not force_rebuild:
            logger.info(f"✅ {system_name} 시스템이 이미 초기화됨")
            return self._systems[system_name]
            
        logger.info(f"🔄 {system_name} 시스템 초기화 중...")
        
        if system_name == "faiss":
            system = self._initialize_faiss_system()
        elif system_name == "chromadb":
            system = self._initialize_chromadb_system()
        else:
            raise ValueError(f"지원하지 않는 시스템: {system_name}")
            
        self._systems[system_name] = system
        self._current_system = system_name
        logger.info(f"✅ {system_name} 시스템 초기화 완료")
        
        return system
    
    def _initialize_faiss_system(self) -> RFPRAGSystem:
        """기존 FAISS 시스템 초기화"""
        try:
            # 기존 시스템 초기화
            system = RFPRAGSystem(
                json_dir=str(self.config.processed_dir / "json"),
                embedding_model=self.config.get_system_config("faiss").embedder_model,
                chunk_size=1000,
                overlap=150,
                cache_dir=str(self.config.get_system_config("faiss").cache_dir)
            )
            
            # 시스템 초기화 실행
            logger.info("🔧 FAISS 시스템 초기화 중...")
            system.initialize()
            logger.info("✅ FAISS 시스템 초기화 완료")
            
            return system
        except Exception as e:
            logger.error(f"❌ FAISS 시스템 초기화 실패: {e}")
            raise
    
    def _initialize_chromadb_system(self) -> Retriever:
        """팀원 ChromaDB 시스템 초기화"""
        try:
            import pandas as pd
            from langchain_huggingface import HuggingFaceEmbeddings
            from sentence_transformers import CrossEncoder
            from src.retrieval.tokenizer_wrapper import TokenizerWrapper
            
            # 메타데이터 로드
            meta_df = pd.read_csv(self.config.meta_csv_path)
            
            # 시스템 설정
            system_config = self.config.get_system_config("chromadb")
            
            # 모델 초기화 (GPU 사용)
            logger.info(f"🔧 ChromaDB 임베딩 모델 초기화: {system_config.embedder_model} on {self.config.device}")
            embedder = HuggingFaceEmbeddings(
                model_name=system_config.embedder_model,
                model_kwargs={"device": self.config.device},
                encode_kwargs={"device": self.config.device}
            )
            
            reranker = RerankModel(
                model_name=system_config.reranker_model,
                cache_dir=system_config.rerank_cache_dir,
                device=self.config.device,
            )
            
            tokenizer = TokenizerWrapper(system_config.tokenizer_engine)
            
            # Retriever 초기화
            persist_dir = str(system_config.persist_directory) if system_config.persist_directory else None
            logger.info(f"🔧 ChromaDB persist_directory: {persist_dir}")
            logger.info(f"🔧 system_config.persist_directory: {system_config.persist_directory}")
            retriever = Retriever(
                meta_df=meta_df,
                embedder=embedder,
                reranker=reranker,
                tokenizer=tokenizer,
                persist_directory=persist_dir,
                rerank_max_length=system_config.rerank_max_length,
                bm25_path=str(system_config.cache_dir / "bm25_index.pkl"),
                debug_mode=True
            )
            
            # 문서 로딩 및 벡터 DB 구축 (중요!)
            logger.info("📚 문서 로딩 중...")
            import asyncio
            json_dir = self.config.processed_dir / "json"
            logger.info(f"📂 JSON 디렉토리: {json_dir}")
            docs = asyncio.run(retriever.load_or_cache_json_docs(
                str(json_dir), 
                cache_path=str(system_config.cache_dir / "cached_json_docs.pkl")
            ))
            
            logger.info("🔧 벡터 DB 구축 중...")
            logger.info(f"📊 로드된 문서 수: {len(docs) if docs else 0}")
            retriever.set_weights(bm25_weight=0.3, rerank_weight=0.7)
            retriever.load_or_build_vector_db(docs)
            
            logger.info("✅ ChromaDB 시스템 초기화 완료")
            return retriever
            
        except Exception as e:
            logger.error(f"❌ ChromaDB 시스템 초기화 실패: {e}")
            raise
    
    def get_system(self, system_name: str = None) -> Any:
        """시스템 인스턴스 반환"""
        if system_name is None:
            system_name = self.config.default_system
            
        if system_name not in self._systems:
            return self.initialize_system(system_name)
            
        return self._systems[system_name]
    
    def switch_system(self, system_name: str) -> Any:
        """시스템 전환"""
        if system_name not in self.config.get_available_systems():
            raise ValueError(f"지원하지 않는 시스템: {system_name}")
            
        return self.get_system(system_name)
    
    def get_system_status(self) -> Dict[str, Any]:
        """모든 시스템의 상태 반환"""
        status = {}
        
        for system_name in self.config.get_available_systems():
            system_info = self.config.get_system_info(system_name)
            is_initialized = system_name in self._systems
            
            status[system_name] = {
                **system_info,
                "initialized": is_initialized,
                "is_current": system_name == self._current_system
            }
            
        return status
    
    def clear_cache(self, system_name: str = None):
        """시스템 캐시 정리"""
        if system_name is None:
            # 모든 시스템 캐시 정리
            for sys_name in self.config.get_available_systems():
                self._clear_system_cache(sys_name)
        else:
            self._clear_system_cache(system_name)
    
    def _clear_system_cache(self, system_name: str):
        """특정 시스템 캐시 정리"""
        system_config = self.config.get_system_config(system_name)
        
        if system_name == "faiss":
            # FAISS 캐시 정리
            cache_files = [
                system_config.cache_dir / "chunks.pkl",
                system_config.cache_dir / "embeddings.npy", 
                system_config.cache_dir / "metadata.pkl"
            ]
            
        elif system_name == "chromadb":
            # ChromaDB 캐시 정리
            cache_files = [
                system_config.cache_dir / "cached_json_docs.pkl",
                system_config.cache_dir / "cached_csv_docs.pkl",
                system_config.cache_dir / "bm25_index.pkl"
            ]
            
            # ChromaDB 디렉토리도 정리
            if system_config.persist_directory and system_config.persist_directory.exists():
                import shutil
                shutil.rmtree(system_config.persist_directory)
                logger.info(f"🗑️ ChromaDB 디렉토리 정리: {system_config.persist_directory}")
            
            # Rerank 캐시 디렉토리도 정리   
            if system_config.rerank_cache_dir and system_config.rerank_cache_dir.exists():
                shutil.rmtree(system_config.rerank_cache_dir)
                logger.info(f"🗑️ Rerank 캐시 디렉토리 정리: {system_config.rerank_cache_dir}")
        
        # 캐시 파일 정리
        for cache_file in cache_files:
            if cache_file.exists():
                cache_file.unlink()
                logger.info(f"🗑️ 캐시 파일 삭제: {cache_file}")
        
        # 메모리에서 시스템 제거
        if system_name in self._systems:
            del self._systems[system_name]
            if self._current_system == system_name:
                self._current_system = None
                
        logger.info(f"✅ {system_name} 시스템 캐시 정리 완료")
    
    def ask(self, question: str, system_name: str = None) -> Dict[str, Any]:
        """선택된 시스템으로 질문 처리"""
        if system_name is None:
            system_name = self._current_system
            
        if not system_name:
            return {"answer": "시스템이 선택되지 않았습니다.", "sources": []}
            
        # 시스템 초기화 확인
        if system_name not in self._systems:
            return {"answer": f"{system_name} 시스템이 초기화되지 않았습니다.", "sources": []}
            
        system = self._systems[system_name]
        
        try:
            if system_name == "faiss":
                # FAISS 시스템 질문 처리
                response = system.ask_detailed(question)
                return {
                    "answer": response.answer,
                    "sources": [
                        {
                            "content": chunk.content,
                            "source_file": chunk.metadata.get("source_file", "N/A"),
                            "page": chunk.metadata.get("page", "N/A"),
                            "score": chunk.score
                        }
                        for chunk in response.retrieved_chunks
                    ],
                    "total_documents": system.retriever.get_total_documents(),
                    "total_chunks": system.retriever.get_total_chunks()
                }
            elif system_name == "chromadb":
                # ChromaDB 시스템 질문 처리
                results = system.smart_search(question, top_k=5)
                
                # LLM을 사용하여 답변 생성
                from src.generation.generator import RFPGenerator
                generator = RFPGenerator(
                    model_name=self.config.get_system_config("faiss").llm_model,
                    api_key=self.config.openai_api_key
                )
                llm_response = generator.generate_response(question, results)
                
                return {
                    "answer": llm_response.answer,
                    "sources": [
                        {
                            "content": doc.page_content,
                            "source_file": doc.metadata.get("source_file", "N/A"),
                            "page": doc.metadata.get("page", "N/A"),
                            "score": system.last_scores.get(system.get_doc_key(doc), {}).get("combined", 0.0)
                        }
                        for doc in results
                    ],
                    "total_documents": len(system.documents),
                    "total_chunks": len(system.documents)
                }
            else:
                return {"answer": "지원하지 않는 시스템입니다.", "sources": []}
                
        except Exception as e:
            logger.error(f"❌ {system_name} 시스템 질문 처리 실패: {e}")
            return {"answer": f"검색 중 오류 발생: {str(e)}", "sources": []}
