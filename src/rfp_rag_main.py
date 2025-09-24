"""
RFP RAG 시스템 - 메인 통합 클래스
모든 모듈을 통합하여 완전한 RAG 시스템을 구성
"""

import os
import logging
from typing import List, Dict, Any, Optional
from dotenv import load_dotenv

from .data_processing import RFPDocument, RetrievalResult, RAGResponse, RFPDataLoader
from .retrieval import RFPRetriever
from .generation import RFPGenerator

# 로깅 설정
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class RFPRAGSystem:
    """RFP RAG 시스템 메인 클래스 - 모든 모듈 통합"""
    
    def __init__(self, 
                 json_dir: str,
                 openai_api_key: str = None,
                 embedding_model: str = None,
                 llm_model: str = None,
                 chunk_size: int = 1000,
                 overlap: int = 200,
                 cache_dir: str = None):
        """
        RFP RAG 시스템 초기화
        
        Args:
            json_dir: JSON 파일들이 있는 디렉토리 경로
            openai_api_key: OpenAI API 키
            embedding_model: 임베딩 모델명
            llm_model: LLM 모델명
            chunk_size: 청크 크기
            overlap: 청크 중첩 크기
        """
        # .env 파일 로드
        load_dotenv()
        
        # 환경변수에서 설정값 가져오기
        self.json_dir = json_dir
        self.openai_api_key = openai_api_key or os.getenv("OPENAI_API_KEY")
        self.embedding_model = embedding_model or os.getenv("EMBEDDING_MODEL", "BAAI/bge-m3")
        self.llm_model = llm_model or os.getenv("MODEL_NAME", "gpt-3.5-turbo")
        
        # 컴포넌트 초기화
        self.data_loader = RFPDataLoader(json_dir)
        self.retriever = RFPRetriever(
            embedding_model=self.embedding_model,
            chunk_size=chunk_size,
            overlap=overlap,
            cache_dir=cache_dir
        )
        self.generator = RFPGenerator(
            api_key=self.openai_api_key,
            model=self.llm_model
        )
        
        # 상태 관리
        self.documents: List[RFPDocument] = []
        self.is_initialized = False
    
    def initialize(self):
        """시스템 전체 초기화"""
        logger.info("Initializing RFP RAG System...")
        
        try:
            # 1. 데이터 로더 초기화
            logger.info("Initializing Data Loader...")
            self.data_loader.initialize()
            if not self.data_loader.is_ready():
                raise RuntimeError("Data Loader initialization failed")
            
            self.documents = self.data_loader.get_documents()
            logger.info(f"Loaded {len(self.documents)} documents")
            
            # 2. 리트리버 초기화
            logger.info("Initializing Retriever...")
            self.retriever.initialize(self.documents)
            if not self.retriever.is_ready():
                raise RuntimeError("Retriever initialization failed")
            
            # 3. 제네레이터 초기화
            logger.info("Initializing Generator...")
            self.generator.initialize()
            if not self.generator.is_ready():
                raise RuntimeError("Generator initialization failed")
            
            self.is_initialized = True
            logger.info("RFP RAG System initialization completed successfully!")
            
        except Exception as e:
            logger.error(f"System initialization failed: {e}")
            raise
    
    def ask(self, 
            question: str, 
            k: int = 10, 
            filters: Optional[Dict[str, Any]] = None,
            use_conversation_history: bool = True) -> str:
        """
        질문에 대한 답변 생성
        
        Args:
            question: 사용자 질문
            k: 검색할 청크 수
            filters: 메타데이터 필터
            use_conversation_history: 대화 히스토리 사용 여부
        
        Returns:
            생성된 답변
        """
        if not self.is_initialized:
            return "시스템이 초기화되지 않았습니다. initialize() 메서드를 먼저 실행해주세요."
        
        try:
            # 1. 관련 청크 검색
            retrieved_results = self.retriever.retrieve(question, k, filters)
            
            if not retrieved_results:
                return "관련된 문서를 찾을 수 없습니다."
            
            # 2. 응답 생성
            response = self.generator.generate_response(
                question, 
                retrieved_results, 
                use_conversation_history
            )
            
            return response.answer
            
        except Exception as e:
            logger.error(f"Error processing question: {e}")
            return "질문 처리 중 오류가 발생했습니다."
    
    def ask_detailed(self, 
                    question: str, 
                    k: int = 10, 
                    filters: Optional[Dict[str, Any]] = None,
                    use_conversation_history: bool = True) -> RAGResponse:
        """
        질문에 대한 상세 응답 생성 (메타데이터 포함)
        
        Returns:
            RAGResponse 객체 (답변, 검색 결과, 메타데이터 포함)
        """
        if not self.is_initialized:
            raise RuntimeError("시스템이 초기화되지 않았습니다.")
        
        # 1. 관련 청크 검색
        retrieved_results = self.retriever.retrieve(question, k, filters)
        
        # 2. 응답 생성
        response = self.generator.generate_response(
            question, 
            retrieved_results, 
            use_conversation_history
        )
        
        return response
    
    def search_documents(self, **filters) -> List[Dict[str, Any]]:
        """문서 검색 (메타데이터 기반)"""
        if not self.is_initialized:
            return []
        
        documents = self.data_loader.search_documents_by_metadata(**filters)
        
        results = []
        for doc in documents:
            results.append({
                "공고번호": doc.공고번호,
                "사업명": doc.사업명,
                "발주기관": doc.발주기관,
                "사업금액": doc.사업금액,
                "공개일자": doc.공개일자,
                "입찰마감일": doc.입찰마감일
            })
        
        return results
    
    def get_document_summary(self) -> Dict[str, Any]:
        """로드된 문서들의 요약 정보 반환"""
        if not self.is_initialized:
            return {"message": "시스템이 초기화되지 않았습니다."}
        
        # 데이터 로더에서 기본 통계 가져오기
        summary = self.data_loader.get_summary_statistics()
        
        # 리트리버에서 청크 통계 추가
        if self.retriever.is_ready():
            chunk_stats = self.retriever.get_chunk_statistics()
            summary.update(chunk_stats)
        
        return summary
    
    def clear_conversation_history(self):
        """대화 히스토리 초기화"""
        if self.is_initialized and self.generator.is_ready():
            self.generator.clear_conversation_history()
    
    def get_conversation_history(self) -> List[Dict[str, str]]:
        """대화 히스토리 반환"""
        if self.is_initialized and self.generator.is_ready():
            return self.generator.get_conversation_history()
        return []
    
    def update_generation_parameters(self, temperature: float = None, max_tokens: int = None):
        """생성 파라미터 업데이트"""
        if self.is_initialized and self.generator.is_ready():
            self.generator.set_generation_parameters(temperature, max_tokens)
    
    def generate_document_summary(self, 
                                doc_id: str = None, 
                                query: str = None, 
                                summary_type: str = "general") -> str:
        """특정 문서 또는 검색 결과의 요약 생성"""
        if not self.is_initialized:
            return "시스템이 초기화되지 않았습니다."
        
        if doc_id:
            # 특정 문서의 모든 청크 가져오기
            chunks = self.retriever.get_chunks_by_doc_id(doc_id)
            if not chunks:
                return f"문서 ID '{doc_id}'를 찾을 수 없습니다."
            
            # 청크를 RetrievalResult로 변환
            results = [RetrievalResult(chunk=chunk, score=1.0, rank=i+1) 
                      for i, chunk in enumerate(chunks[:10])]  # 상위 10개만
        
        elif query:
            # 쿼리 기반 검색 결과
            results = self.retriever.retrieve(query, k=10)
            if not results:
                return "관련된 문서를 찾을 수 없습니다."
        
        else:
            return "문서 ID 또는 검색 쿼리를 제공해주세요."
        
        # 요약 생성
        return self.generator.generate_summary(results, summary_type)
    
    def compare_documents(self, 
                         queries: List[str], 
                         comparison_aspects: List[str] = None) -> str:
        """여러 쿼리 결과 비교 분석"""
        if not self.is_initialized:
            return "시스템이 초기화되지 않았습니다."
        
        if len(queries) < 2:
            return "비교를 위해 최소 2개의 쿼리가 필요합니다."
        
        # 각 쿼리에 대한 검색 결과 수집
        results_list = []
        for query in queries:
            results = self.retriever.retrieve(query, k=5)
            if results:
                results_list.append(results)
        
        if len(results_list) < 2:
            return "비교할 충분한 검색 결과를 찾을 수 없습니다."
        
        # 기본 비교 측면
        if not comparison_aspects:
            comparison_aspects = ["사업 규모", "요구사항", "기간", "조건"]
        
        return self.generator.generate_comparison(results_list, comparison_aspects)
    
    def get_system_status(self) -> Dict[str, Any]:
        """시스템 상태 정보 반환"""
        status = {
            "is_initialized": self.is_initialized,
            "data_loader_ready": self.data_loader.is_ready() if hasattr(self.data_loader, 'is_ready') else False,
            "retriever_ready": self.retriever.is_ready(),
            "generator_ready": self.generator.is_ready(),
            "document_count": len(self.documents),
            "embedding_model": self.embedding_model,
            "llm_model": self.llm_model,
            "conversation_history_length": len(self.get_conversation_history())
        }
        
        # 캐시 정보 추가
        if self.is_initialized and hasattr(self.retriever, 'get_cache_info'):
            status["cache_info"] = self.retriever.get_cache_info()
        
        return status
    
    def clear_embedding_cache(self):
        """임베딩 캐시 삭제"""
        if self.is_initialized and hasattr(self.retriever, 'clear_embedding_cache'):
            self.retriever.clear_embedding_cache()
            logger.info("Embedding cache cleared from main system")
        else:
            logger.warning("Cannot clear cache: system not initialized or cache not available")

# 사용 예시 및 테스트
def main():
    """메인 실행 함수"""
    import os
    from dotenv import load_dotenv
    
    # .env 파일 로드
    load_dotenv()
    
    # 환경변수에서 API 키 확인
    api_key = os.getenv("OPENAI_API_KEY")
    if not api_key:
        print("OPENAI_API_KEY 환경변수를 설정해주세요.")
        return
    
    # RFP RAG 시스템 초기화
    json_dir = "/Users/leeyoungho/develop/ai_study/project/projectmission2/data/preprocess/json"
    rag_system = RFPRAGSystem(json_dir, api_key)
    
    try:
        # 시스템 초기화
        rag_system.initialize()
        
        # 시스템 상태 확인
        status = rag_system.get_system_status()
        print("=== 시스템 상태 ===")
        for key, value in status.items():
            print(f"{key}: {value}")
        
        # 문서 요약 정보 출력
        summary = rag_system.get_document_summary()
        print("\n=== 문서 요약 ===")
        print(f"총 문서 수: {summary.get('총_문서_수', 0)}")
        print(f"총 청크 수: {summary.get('총_청크_수', 0)}")
        
        # 질문 예시
        questions = [
            "한국사학진흥재단이 발주한 사업의 요구사항을 알려주세요.",
            "대학재정정보시스템 고도화 사업의 사업금액과 기간은 얼마인가요?",
            "입찰 참가 자격 요건이 있는 사업들을 알려주세요."
        ]
        
        print("\n=== 질의응답 예시 ===")
        for question in questions:
            print(f"\n질문: {question}")
            answer = rag_system.ask(question)
            print(f"답변: {answer}")
            print("-" * 80)
    
    except Exception as e:
        print(f"오류 발생: {e}")

if __name__ == "__main__":
    main()
