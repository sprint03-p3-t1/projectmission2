"""
RFP RAG 시스템 - 제네레이션 모듈
검색된 청크를 바탕으로 답변을 생성하는 모듈
본인 작업용 모듈
"""

import os
import logging
import time
from typing import List, Dict, Any, Optional
from datetime import datetime

# LLM
from openai import OpenAI

import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from data_processing import RetrievalResult, RAGResponse, RAGSystemInterface

# 로깅 설정
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class RFPGenerator(RAGSystemInterface):
    """RFP 응답 생성기 - 본인 작업 영역"""
    
    def __init__(self, api_key: str = None, model: str = None, temperature: float = None, max_tokens: int = None):
        # 환경변수에서 설정값 가져오기
        self.api_key = api_key or os.getenv("OPENAI_API_KEY")
        self.model = model or os.getenv("MODEL_NAME", "gpt-3.5-turbo")
        self.temperature = temperature or float(os.getenv("TEMPERATURE", "0.1"))
        self.max_tokens = max_tokens or int(os.getenv("MAX_TOKENS", "2000"))
        
        self.client = None
        self.conversation_history: List[Dict[str, str]] = []
        self._is_ready = False
    
    def initialize(self):
        """제네레이터 초기화"""
        if not self.api_key:
            raise ValueError("OpenAI API key is required")
        
        self.client = OpenAI(api_key=self.api_key)
        self._is_ready = True
        logger.info(f"Initialized RFP Generator with model: {self.model}")
    
    def is_ready(self) -> bool:
        """제네레이터 준비 상태 확인"""
        return self._is_ready and self.client is not None
    
    def generate_response(self, 
                         question: str, 
                         retrieved_results: List[RetrievalResult],
                         use_conversation_history: bool = True) -> RAGResponse:
        """검색된 청크를 바탕으로 응답 생성"""
        if not self.is_ready():
            raise RuntimeError("Generator is not initialized. Call initialize() first.")
        
        start_time = time.time()
        
        # 컨텍스트 구성
        context = self._build_context(retrieved_results)
        
        # 시스템 프롬프트
        system_prompt = self._get_system_prompt()
        
        # 대화 히스토리 구성
        messages = [{"role": "system", "content": system_prompt}]
        
        if use_conversation_history and self.conversation_history:
            # 최근 6턴의 대화만 유지 (메모리 관리)
            messages.extend(self.conversation_history[-6:])
        
        # 사용자 쿼리와 컨텍스트
        user_message = self._create_user_message(question, context)
        messages.append({"role": "user", "content": user_message})
        
        try:
            # OpenAI API 호출 - Pydantic 에러 우회
            try:
                response = self.client.chat.completions.create(
                    model=self.model,
                    messages=messages,
                    temperature=self.temperature,
                    max_tokens=self.max_tokens
                )
                answer = response.choices[0].message.content
            except Exception as pydantic_error:
                # Pydantic 에러 발생 시 간단한 응답 생성
                logger.warning(f"Pydantic error occurred, using fallback: {pydantic_error}")
                answer = f"검색된 문서를 바탕으로 답변드리겠습니다.\n\n{context[:1000]}..."
            
            # 생성 메타데이터
            generation_time = time.time() - start_time
            generation_metadata = {
                "model": self.model,
                "temperature": self.temperature,
                "max_tokens": self.max_tokens,
                "prompt_tokens": response.usage.prompt_tokens,
                "completion_tokens": response.usage.completion_tokens,
                "total_tokens": response.usage.total_tokens,
                "generation_time": generation_time,
                "timestamp": datetime.now().isoformat()
            }
            
            # 대화 히스토리 업데이트
            if use_conversation_history:
                self.conversation_history.append({"role": "user", "content": question})
                self.conversation_history.append({"role": "assistant", "content": answer})
            
            # RetrievalResult 객체들을 딕셔너리로 변환
            chunks_dict = []
            for result in retrieved_results:
                chunk_dict = {
                    "chunk": {
                        "chunk_id": result.chunk.chunk_id,
                        "doc_id": result.chunk.doc_id,
                        "content": result.chunk.content,
                        "chunk_type": result.chunk.chunk_type,
                        "page_number": result.chunk.page_number,
                        "metadata": result.chunk.metadata
                    },
                    "score": result.score,
                    "rank": result.rank
                }
                chunks_dict.append(chunk_dict)
            
            return RAGResponse(
                question=question,
                answer=answer,
                retrieved_chunks=chunks_dict,
                generation_metadata=generation_metadata
            )
        
        except Exception as e:
            logger.error(f"Error generating response: {e}")
            # 에러 시에도 retrieved_chunks를 Dict로 변환
            chunks_dict = []
            for result in retrieved_results:
                chunks_dict.append({
                    "chunk_id": result.chunk.chunk_id,
                    "content": result.chunk.content[:200] + "..." if len(result.chunk.content) > 200 else result.chunk.content,
                    "score": result.score,
                    "rank": result.rank
                })
            
            return RAGResponse(
                question=question,
                answer="죄송합니다. 응답 생성 중 오류가 발생했습니다.",
                retrieved_chunks=retrieved_results,
                generation_metadata={"error": str(e), "generation_time": time.time() - start_time}
            )
    
    def _build_context(self, retrieved_results: List[RetrievalResult]) -> str:
        """검색된 청크들을 컨텍스트로 구성"""
        if not retrieved_results:
            return "관련된 문서 정보를 찾을 수 없습니다."
        
        context_parts = []
        
        for result in retrieved_results:
            chunk = result.chunk
            context_part = f"""
[문서 {result.rank}] (유사도: {result.score:.3f})
공고번호: {chunk.metadata.get('공고번호', 'N/A')}
사업명: {chunk.metadata.get('사업명', 'N/A')}
청크 유형: {chunk.chunk_type}
{f"페이지: {chunk.page_number}" if chunk.page_number else ""}

내용:
{chunk.content}
"""
            context_parts.append(context_part)
        
        return "\n" + "="*80 + "\n".join(context_parts)
    
    def _create_user_message(self, question: str, context: str) -> str:
        """사용자 메시지 생성"""
        return f"""
질문: {question}

관련 RFP 문서 정보:
{context}

위 정보를 바탕으로 질문에 답변해 주세요.
문서에 정보가 있으면 사업명, 발주기관, 사업금액, 기간 등 핵심 정보를 포함하여 상세하게 답변하세요.
문서에 없는 내용에 대해서는 "문서에서 해당 정보를 찾을 수 없습니다"라고 명확히 말씀해 주세요.
"""
    
    def _get_system_prompt(self) -> str:
        """시스템 프롬프트 반환 - 커스터마이징 가능"""
        return """
당신은 RFP(제안요청서) 분석 전문가입니다. 
정부기관과 기업의 입찰 공고 문서를 분석하여 컨설턴트들이 필요한 정보를 빠르게 파악할 수 있도록 도와주는 역할을 합니다.

다음 원칙을 지켜주세요:
1. 제공된 문서 정보만을 바탕으로 정확하게 답변하세요.
2. 문서에 없는 내용에 대해서는 "문서에서 해당 정보를 찾을 수 없습니다"라고 간단하고 명확히 말하세요.
3. 문서에 정보가 있으면 사업명, 발주기관, 사업금액, 기간 등 핵심 정보를 포함하여 답변하세요.
4. 표나 목록이 있는 경우 구조화하여 보기 쉽게 정리하세요.
5. 입찰 참가 자격, 평가 기준, 제출 서류 등 중요한 요구사항은 놓치지 말고 포함하세요.
6. 답변은 한국어로 작성하고, 간결하고 명확하게 설명하세요.
7. 문서에 없는 내용에 대해서는 추측하거나 관련 없는 정보를 제공하지 마세요.
"""
    
    def update_system_prompt(self, new_prompt: str):
        """시스템 프롬프트 업데이트"""
        self._system_prompt = new_prompt
        logger.info("System prompt updated")
    
    def clear_conversation_history(self):
        """대화 히스토리 초기화"""
        self.conversation_history = []
        logger.info("Conversation history cleared")
    
    def get_conversation_history(self) -> List[Dict[str, str]]:
        """대화 히스토리 반환"""
        return self.conversation_history.copy()
    
    def set_generation_parameters(self, temperature: float = None, max_tokens: int = None):
        """생성 파라미터 업데이트"""
        if temperature is not None:
            self.temperature = temperature
        if max_tokens is not None:
            self.max_tokens = max_tokens
        logger.info(f"Generation parameters updated: temperature={self.temperature}, max_tokens={self.max_tokens}")
    
    def generate_summary(self, retrieved_results: List[RetrievalResult], summary_type: str = "general") -> str:
        """검색된 문서들의 요약 생성"""
        if not retrieved_results:
            return "요약할 문서가 없습니다."
        
        context = self._build_context(retrieved_results)
        
        summary_prompts = {
            "general": "위 문서들의 주요 내용을 요약해주세요.",
            "requirements": "위 문서들에서 입찰 요구사항과 조건들을 정리해주세요.",
            "evaluation": "위 문서들의 평가 기준과 방법을 정리해주세요.",
            "timeline": "위 문서들의 일정과 마감일을 정리해주세요."
        }
        
        prompt = summary_prompts.get(summary_type, summary_prompts["general"])
        
        messages = [
            {"role": "system", "content": self._get_system_prompt()},
            {"role": "user", "content": f"{context}\n\n{prompt}"}
        ]
        
        try:
            response = self.client.chat.completions.create(
                model=self.model,
                messages=messages,
                temperature=self.temperature,
                max_tokens=self.max_tokens
            )
            
            return response.choices[0].message.content
        
        except Exception as e:
            logger.error(f"Error generating summary: {e}")
            return "요약 생성 중 오류가 발생했습니다."
    
    def generate_comparison(self, results_list: List[List[RetrievalResult]], comparison_aspects: List[str]) -> str:
        """여러 문서 세트 비교 분석"""
        if not results_list or len(results_list) < 2:
            return "비교할 문서가 충분하지 않습니다."
        
        contexts = []
        for i, results in enumerate(results_list):
            context = self._build_context(results)
            contexts.append(f"[문서 그룹 {i+1}]\n{context}")
        
        comparison_context = "\n\n".join(contexts)
        aspects_str = ", ".join(comparison_aspects)
        
        messages = [
            {"role": "system", "content": self._get_system_prompt()},
            {"role": "user", "content": f"""
다음 문서 그룹들을 {aspects_str} 측면에서 비교 분석해주세요:

{comparison_context}

각 그룹의 특징과 차이점을 명확히 정리해주세요.
"""}
        ]
        
        try:
            response = self.client.chat.completions.create(
                model=self.model,
                messages=messages,
                temperature=self.temperature,
                max_tokens=self.max_tokens
            )
            
            return response.choices[0].message.content
        
        except Exception as e:
            logger.error(f"Error generating comparison: {e}")
            return "비교 분석 생성 중 오류가 발생했습니다."

# 제네레이션 모듈 단독 테스트 함수
def test_generator_standalone():
    """제네레이션 모듈 단독 테스트"""
    from data_processing import DocumentChunk, RetrievalResult
    
    # 환경변수에서 API 키 가져오기
    api_key = os.getenv("OPENAI_API_KEY")
    if not api_key:
        print("OPENAI_API_KEY 환경변수를 설정해주세요.")
        return
    
    # 제네레이터 초기화
    generator = RFPGenerator(api_key)
    generator.initialize()
    
    # 테스트용 가짜 검색 결과 생성
    test_chunk = DocumentChunk(
        chunk_id="test_chunk",
        doc_id="test_doc",
        content="한국사학진흥재단에서 대학재정정보시스템 고도화 사업을 발주했습니다. 사업금액은 2억 1천만원이며, 계약기간은 6개월입니다.",
        chunk_type="metadata",
        metadata={
            "공고번호": "20240815487",
            "사업명": "대학재정정보시스템 고도화",
            "발주기관": "한국사학진흥재단",
            "사업금액": 211000000
        }
    )
    
    test_results = [
        RetrievalResult(chunk=test_chunk, score=0.95, rank=1)
    ]
    
    # 테스트 질문
    test_question = "한국사학진흥재단 사업의 주요 내용을 알려주세요."
    
    # 응답 생성
    response = generator.generate_response(test_question, test_results)
    
    print(f"질문: {response.question}")
    print(f"답변: {response.answer}")
    print(f"메타데이터: {response.generation_metadata}")

if __name__ == "__main__":
    test_generator_standalone()
