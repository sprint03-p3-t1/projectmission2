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
        """ChromaDB 시스템 초기화"""
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
        """선택된 시스템으로 질문 처리 (질문 분류 단계 포함)"""
        if system_name is None:
            system_name = self._current_system
            
        if not system_name:
            return {"answer": "시스템이 선택되지 않았습니다.", "sources": []}
            
        # 시스템 초기화 확인
        if system_name not in self._systems:
            return {"answer": f"{system_name} 시스템이 초기화되지 않았습니다.", "sources": []}
            
        system = self._systems[system_name]
        
        # 1. 질문 분류 단계 추가
        logger.info(f"🔍 질문 분류 시작: {question[:50]}...")
        try:
            from src.classification.question_classifier import get_question_classifier
            classifier = get_question_classifier()
            classification_result = classifier.classify_question(question)
            
            logger.info(f"✅ 질문 분류 완료: {classification_result.question_type.value} (신뢰도: {classification_result.confidence:.3f})")
            logger.info(f"📝 분류 근거: {classification_result.reasoning}")
            logger.info(f"🎯 제안 프롬프트 타입: {classification_result.suggested_prompt_type}")

            # 일상 질문인 경우 RFP 문서 검색 없이 간단히 응답
            if classification_result.question_type.value == "일상":
                logger.info("💬 일상 질문으로 분류됨 - RFP 문서 검색 생략")
                return {
                    "answer": "안녕하세요! RFP 문서 분석 도구에 오신 것을 환영합니다. 궁금한 사업 정보나 입찰 관련 질문이 있으시면 언제든지 말씀해 주세요.",
                    "sources": [],
                    "total_documents": 0,
                    "total_chunks": 0,
                    "question_classification": {
                        "type": classification_result.question_type.value,
                        "confidence": classification_result.confidence,
                        "reasoning": classification_result.reasoning,
                        "prompt_type": classification_result.suggested_prompt_type
                    }
                }

        except Exception as e:
            logger.error(f"❌ 질문 분류 실패: {e}")
            # 분류 실패 시 기본값 사용
            classification_result = None
        
        try:
            if system_name == "faiss":
                # FAISS 시스템 질문 처리 (질문 분류 결과 적용)
                logger.info(f"🔍 FAISS 시스템 질문 처리 시작: {question[:50]}...")
                
                # 질문 분류 결과를 FAISS 시스템에 전달
                if classification_result:
                    # RFPGenerator에 질문 유형 설정
                    if hasattr(system, 'generator') and hasattr(system.generator, 'question_type'):
                        system.generator.question_type = classification_result.suggested_prompt_type
                        logger.info(f"🎯 FAISS Generator에 질문 유형 적용: {classification_result.suggested_prompt_type}")
                
                response = system.ask_detailed(question)
                logger.info(f"✅ FAISS 답변 생성 완료: {response.answer[:100]}...")
                
                return {
                    "answer": response.answer,
                    "sources": [
                        {
                            "content": chunk.get("content", "N/A"),
                            "source_file": chunk.get("source_file", "N/A"),
                            "page": chunk.get("page", "N/A"),
                            "score": chunk.get("score", 0.0)
                        }
                        for chunk in response.retrieved_chunks
                    ],
                    "total_documents": len(system.documents),
                    "total_chunks": len(system.retriever.vector_store.chunks) if hasattr(system.retriever, 'vector_store') else 0,
                    "question_classification": {
                        "type": classification_result.question_type.value if classification_result else "unknown",
                        "confidence": classification_result.confidence if classification_result else 0.0,
                        "reasoning": classification_result.reasoning if classification_result else "분류 실패",
                        "prompt_type": classification_result.suggested_prompt_type if classification_result else "general"
                    }
                }
            elif system_name == "chromadb":
                # ChromaDB 시스템 질문 처리
                logger.info(f"🔍 ChromaDB 시스템 질문 처리 시작: {question[:50]}...")
                results = system.smart_search(question, top_k=5)
                logger.info(f"✅ ChromaDB 검색 완료: {len(results)}개 결과")
                
                # LLM을 사용하여 답변 생성 (프롬프트 매니저 적용)
                logger.info("🤖 LLM 답변 생성 시작...")
                from src.generation.generator import RFPGenerator
                generator = RFPGenerator()  # rag_config.yaml에서 자동으로 설정 로드
                
                # Generator 초기화
                logger.info("🔧 RFPGenerator 초기화 중...")
                generator.initialize()
                logger.info("✅ RFPGenerator 초기화 완료")
                
                # 프롬프트 매니저 초기화
                logger.info("📝 프롬프트 매니저 초기화 중...")
                from src.prompts.prompt_manager import get_prompt_manager
                prompt_manager = get_prompt_manager()
                generator.prompt_manager = prompt_manager
                logger.info(f"✅ 프롬프트 매니저 초기화 완료: {prompt_manager.current_version}")
                
                # 질문 분류 결과를 프롬프트 매니저에 전달
                if classification_result:
                    generator.question_type = classification_result.suggested_prompt_type
                    logger.info(f"🎯 질문 유형 기반 프롬프트 적용: {classification_result.suggested_prompt_type}")
                
                # 검색 결과를 RetrievalResult 형태로 변환
                from src.data_processing.data_models import RetrievalResult, DocumentChunk
                retrieval_results = []
                for i, doc in enumerate(results):
                    chunk = DocumentChunk(
                        chunk_id=f"chromadb_{i}",
                        doc_id=doc.metadata.get("source_file", "unknown"),
                        content=doc.page_content,
                        chunk_type="text",
                        metadata=doc.metadata
                    )
                    score = system.last_scores.get(system.get_doc_key(doc), {}).get("combined", 0.0)
                    retrieval_results.append(RetrievalResult(chunk=chunk, score=score, rank=i+1))
                
                logger.info(f"🔄 {len(retrieval_results)}개 검색 결과를 RetrievalResult로 변환 완료")
                llm_response = generator.generate_response(question, retrieval_results)
                logger.info(f"✅ LLM 답변 생성 완료: {llm_response.answer[:100]}...")
                
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
                    "total_chunks": len(system.documents),
                    "question_classification": {
                        "type": classification_result.question_type.value if classification_result else "unknown",
                        "confidence": classification_result.confidence if classification_result else 0.0,
                        "reasoning": classification_result.reasoning if classification_result else "분류 실패",
                        "prompt_type": classification_result.suggested_prompt_type if classification_result else "general"
                    }
                }
            else:
                return {"answer": "지원하지 않는 시스템입니다.", "sources": []}
                
        except Exception as e:
            logger.error(f"❌ {system_name} 시스템 질문 처리 실패: {e}")
            return {"answer": f"검색 중 오류 발생: {str(e)}", "sources": []}
