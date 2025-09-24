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
from ops import get_quality_metrics, get_quality_monitor

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
        
        # MLOps 구성 요소 초기화
        self.quality_metrics = get_quality_metrics()
        self.quality_monitor = get_quality_monitor()
        self.enable_quality_evaluation = True
    
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
            
            # 품질 평가 수행 (옵션)
            quality_evaluation = None
            if self.enable_quality_evaluation:
                try:
                    quality_evaluation = self.evaluate_response_quality(question, answer, context)
                    
                    # 품질 평가 결과 저장
                    evaluation_id = self.quality_metrics.store_evaluation(
                        question=question,
                        answer=answer,
                        context=context,
                        scores=quality_evaluation["scores"],
                        overall_score=quality_evaluation["overall_score"],
                        suggestions=quality_evaluation["suggestions"],
                        evaluation_text=quality_evaluation["evaluation_text"],
                        model_name=self.model,
                        user_id=None,  # TODO: 사용자 ID 추가
                        session_id=None  # TODO: 세션 ID 추가
                    )
                    
                    # 생성 메타데이터에 품질 평가 정보 추가
                    generation_metadata["quality_evaluation"] = quality_evaluation
                    generation_metadata["evaluation_id"] = evaluation_id
                    
                    logger.info(f"Quality evaluation completed: {quality_evaluation['overall_score']:.3f}")
                    
                except Exception as e:
                    logger.error(f"Quality evaluation failed: {e}")
                    quality_evaluation = {"error": str(e)}
            
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
    
    def evaluate_response_quality(self, question: str, answer: str, context: str) -> Dict[str, Any]:
        """LLM 기반 답변 품질 평가"""
        if not self.is_ready():
            raise RuntimeError("Generator is not initialized. Call initialize() first.")
        
        evaluation_prompt = f"""
다음 질문과 답변을 평가해주세요. 각 항목을 0-1 점수로 평가하고, 개선 제안을 해주세요.

질문: {question}

답변: {answer}

참고 문서: {context[:2000]}...

평가 기준:
1. 관련성 (Relevance): 답변이 질문에 얼마나 관련있는가? (0-1)
2. 완성도 (Completeness): 질문에 대한 답변이 얼마나 완전한가? (0-1)
3. 정확성 (Accuracy): 답변 내용이 얼마나 정확한가? (0-1)
4. 명확성 (Clarity): 답변이 얼마나 이해하기 쉬운가? (0-1)
5. 구조화 (Structure): 답변이 얼마나 체계적으로 구성되었는가? (0-1)

응답 형식:
관련성: 0.85
완성도: 0.78
정확성: 0.92
명확성: 0.80
구조화: 0.75
종합점수: 0.82
개선제안: [구체적인 개선 제안 3가지]
"""
        
        try:
            response = self.client.chat.completions.create(
                model=self.model,
                messages=[
                    {"role": "system", "content": "당신은 답변 품질 평가 전문가입니다. 객관적이고 정확한 평가를 해주세요."},
                    {"role": "user", "content": evaluation_prompt}
                ],
                temperature=0.1,  # 일관된 평가를 위해 낮은 temperature
                max_tokens=500
            )
            
            evaluation_text = response.choices[0].message.content
            
            # 평가 결과 파싱
            scores = self._parse_evaluation_scores(evaluation_text)
            suggestions = self._parse_improvement_suggestions(evaluation_text)
            
            # 종합 점수 계산
            overall_score = sum(scores.values()) / len(scores) if scores else 0.0
            
            return {
                "scores": scores,
                "overall_score": overall_score,
                "suggestions": suggestions,
                "evaluation_text": evaluation_text,
                "timestamp": datetime.now().isoformat()
            }
            
        except Exception as e:
            logger.error(f"Error evaluating response quality: {e}")
            return {
                "scores": {"relevance": 0.0, "completeness": 0.0, "accuracy": 0.0, "clarity": 0.0, "structure": 0.0},
                "overall_score": 0.0,
                "suggestions": ["평가 중 오류가 발생했습니다."],
                "evaluation_text": "평가 실패",
                "timestamp": datetime.now().isoformat(),
                "error": str(e)
            }
    
    def _parse_evaluation_scores(self, evaluation_text: str) -> Dict[str, float]:
        """평가 텍스트에서 점수 추출"""
        scores = {}
        lines = evaluation_text.split('\n')
        
        for line in lines:
            line = line.strip()
            if ':' in line and any(keyword in line.lower() for keyword in ['관련성', '완성도', '정확성', '명확성', '구조화']):
                try:
                    key_value = line.split(':')
                    if len(key_value) == 2:
                        key = key_value[0].strip()
                        value = float(key_value[1].strip())
                        scores[key] = value
                except ValueError:
                    continue
        
        return scores
    
    def _parse_improvement_suggestions(self, evaluation_text: str) -> List[str]:
        """평가 텍스트에서 개선 제안 추출"""
        suggestions = []
        lines = evaluation_text.split('\n')
        
        in_suggestions = False
        for line in lines:
            line = line.strip()
            if '개선제안' in line or '개선 제안' in line:
                in_suggestions = True
                continue
            if in_suggestions and line:
                if line.startswith('-') or line.startswith('•') or line.startswith('*'):
                    suggestions.append(line[1:].strip())
                elif line and not any(keyword in line.lower() for keyword in ['관련성', '완성도', '정확성', '명확성', '구조화', '종합점수']):
                    suggestions.append(line)
        
        return suggestions[:3] if suggestions else ["구체적인 개선 제안을 제공할 수 없습니다."]
    
    def enable_quality_evaluation(self, enable: bool = True):
        """품질 평가 활성화/비활성화"""
        self.enable_quality_evaluation = enable
        logger.info(f"Quality evaluation {'enabled' if enable else 'disabled'}")
    
    def get_quality_statistics(self, days: int = 7) -> Dict[str, Any]:
        """품질 통계 조회"""
        return self.quality_metrics.get_quality_statistics(days)
    
    def get_quality_trends(self, days: int = 30) -> Dict[str, Any]:
        """품질 트렌드 조회"""
        trends_df = self.quality_metrics.get_quality_trends(days)
        return trends_df.to_dict('records') if not trends_df.empty else []
    
    def get_quality_insights(self) -> Dict[str, Any]:
        """품질 인사이트 조회"""
        return self.quality_monitor.get_quality_insights()
    
    def start_quality_monitoring(self):
        """품질 모니터링 시작"""
        self.quality_monitor.start_monitoring()
        logger.info("Quality monitoring started")
    
    def stop_quality_monitoring(self):
        """품질 모니터링 중지"""
        self.quality_monitor.stop_monitoring()
        logger.info("Quality monitoring stopped")
    
    def get_monitoring_status(self) -> Dict[str, Any]:
        """모니터링 상태 조회"""
        return self.quality_monitor.get_monitoring_status()

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
    
    # 품질 평가 결과 확인
    if "quality_evaluation" in response.generation_metadata:
        quality_eval = response.generation_metadata["quality_evaluation"]
        print(f"\n=== 품질 평가 결과 ===")
        print(f"종합 점수: {quality_eval['overall_score']:.3f}")
        print(f"세부 점수: {quality_eval['scores']}")
        print(f"개선 제안: {quality_eval['suggestions']}")
    
    # 품질 통계 조회
    print(f"\n=== 품질 통계 (최근 7일) ===")
    stats = generator.get_quality_statistics(days=7)
    print(f"평균 품질 점수: {stats['avg_overall_score']:.3f}")
    print(f"총 평가 수: {stats['total_evaluations']}")
    
    # 품질 인사이트 조회
    print(f"\n=== 품질 인사이트 ===")
    insights = generator.get_quality_insights()
    print(f"인사이트: {insights.get('insights', [])}")

if __name__ == "__main__":
    test_generator_standalone()
