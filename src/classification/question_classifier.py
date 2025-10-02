"""
질문 분류기 모듈
사용자 질문을 분석하여 적절한 질문 유형을 파악하고, 해당 유형에 맞는 프롬프트를 제공
"""

import logging
import json
from typing import Dict, Any, Optional, List
from dataclasses import dataclass
from enum import Enum

logger = logging.getLogger(__name__)

class QuestionType(Enum):
    """질문 유형 열거형"""
    EVERYDAY = "일상"           # 일반적인 대화나 상식 질문
    STATISTICS = "통계"         # 수치, 데이터, 통계 관련 질문
    ANALYSIS = "분석"           # 심층 분석, 비교, 평가 요청
    SUMMARY = "요약"            # 문서나 내용 요약 요청
    SEARCH = "검색"             # 특정 정보 검색 요청
    COMPARISON = "비교"         # 여러 항목 비교 요청
    EXPLANATION = "설명"        # 개념이나 과정 설명 요청
    RECOMMENDATION = "추천"     # 추천이나 제안 요청

@dataclass
class ClassificationResult:
    """질문 분류 결과"""
    question_type: QuestionType
    confidence: float
    reasoning: str
    suggested_prompt_type: str

class QuestionClassifier:
    """질문 분류기 클래스"""
    
    def __init__(self, model_name: str = None, api_key: Optional[str] = None):
        self.model_name = model_name
        self.api_key = api_key
        self.client = None
        self._is_ready = False
        
        # 질문 유형별 키워드 매핑
        self.type_keywords = {
            QuestionType.EVERYDAY: ["안녕", "어떻게", "뭐야", "왜", "언제", "어디", "누구", "일반적", "상식", "날씨", "데이트"],
            QuestionType.STATISTICS: ["몇", "얼마", "수치", "통계", "비율", "퍼센트", "개수", "금액", "예산", "비용"],
            QuestionType.ANALYSIS: ["분석", "비교", "평가", "검토", "검증", "고려사항", "장단점", "특징"],
            QuestionType.SUMMARY: ["요약", "정리", "핵심", "개요", "줄거리", "간단히"],
            QuestionType.SEARCH: ["찾아", "검색", "어디에", "무엇이", "무엇을", "정보"],
            QuestionType.COMPARISON: ["비교", "차이", "vs", "대비", "상대적", "어느쪽"],
            QuestionType.EXPLANATION: ["설명", "이해", "의미", "정의", "과정", "방법", "어떻게"],
            QuestionType.RECOMMENDATION: ["추천", "제안", "권장", "어떤", "선택", "결정"]
        }
        
        # 질문 유형별 프롬프트 타입 매핑
        self.prompt_type_mapping = {
            QuestionType.EVERYDAY: "general",
            QuestionType.STATISTICS: "statistical",
            QuestionType.ANALYSIS: "analytical",
            QuestionType.SUMMARY: "summarization",
            QuestionType.SEARCH: "search",
            QuestionType.COMPARISON: "comparison",
            QuestionType.EXPLANATION: "explanatory",
            QuestionType.RECOMMENDATION: "recommendation"
        }
    
    def initialize(self):
        """분류기 초기화"""
        try:
            from openai import OpenAI
            import yaml
            import os
            from pathlib import Path
            
            # API 키 설정
            api_key = self.api_key
            
            # 설정 파일에서 모델명과 API 키 정보 로드
            try:
                # .env 파일 로드 시도
                try:
                    from dotenv import load_dotenv
                    load_dotenv()
                    logger.info("✅ .env 파일 로드 완료")
                except ImportError:
                    logger.warning("⚠️ python-dotenv가 설치되지 않았습니다")
                except Exception as dotenv_error:
                    logger.warning(f"⚠️ .env 파일 로드 실패: {dotenv_error}")
                
                config_path = Path("config/rag_config.yaml")
                if config_path.exists():
                    with open(config_path, 'r', encoding='utf-8') as f:
                        config = yaml.safe_load(f)
                    
                    # 모델명 설정 (기본값: gpt-4.1-mini)
                    if not self.model_name:
                        self.model_name = config.get('llm', {}).get('model', 'gpt-4.1-mini')
                    
                    # API 키 설정
                    if not api_key:
                        api_key_env = config.get('llm', {}).get('api_key_env', 'OPENAI_API_KEY')
                        api_key = os.getenv(api_key_env)
                        
                        if api_key:
                            logger.info(f"✅ 환경변수에서 API 키 로드: {api_key_env}")
                        else:
                            logger.warning(f"⚠️ 환경변수 {api_key_env}에서 API 키를 찾을 수 없습니다")
            except Exception as config_error:
                logger.warning(f"⚠️ 설정 파일 로드 실패: {config_error}")
                if not self.model_name:
                    self.model_name = "gpt-3.5-turbo"  # 폴백 모델
            
            # OpenAI 클라이언트 초기화
            if api_key:
                self.client = OpenAI(api_key=api_key)
            else:
                # API 키가 없어도 클라이언트 생성 (환경변수에서 자동으로 찾음)
                self.client = OpenAI()
            
            self._is_ready = True
            logger.info(f"✅ 질문 분류기 초기화 완료: {self.model_name}")
            
        except Exception as e:
            logger.error(f"❌ 질문 분류기 초기화 실패: {e}")
            raise
    
    def is_ready(self) -> bool:
        """초기화 상태 확인"""
        return self._is_ready and self.client is not None
    
    def classify_question(self, question: str) -> ClassificationResult:
        """
        질문을 분류하여 적절한 질문 유형을 결정
        
        Args:
            question: 사용자 질문
            
        Returns:
            ClassificationResult: 분류 결과
        """
        if not self.is_ready():
            raise RuntimeError("Classifier is not initialized. Call initialize() first.")
        
        try:
            # 1. 키워드 기반 빠른 분류 시도
            quick_result = self._quick_classify(question)
            if quick_result and quick_result.confidence > 0.8:
                logger.info(f"🔍 키워드 기반 빠른 분류: {quick_result.question_type.value}")
                return quick_result
            
            # 2. LLM 기반 정밀 분류
            llm_result = self._llm_classify(question)
            logger.info(f"🤖 LLM 기반 분류: {llm_result.question_type.value} (신뢰도: {llm_result.confidence:.3f})")
            return llm_result
            
        except Exception as e:
            logger.error(f"❌ 질문 분류 실패: {e}")
            # 기본값 반환
            return ClassificationResult(
                question_type=QuestionType.SEARCH,
                confidence=0.5,
                reasoning="분류 실패로 인한 기본값 설정",
                suggested_prompt_type="search"
            )
    
    def _quick_classify(self, question: str) -> Optional[ClassificationResult]:
        """키워드 기반 빠른 분류"""
        question_lower = question.lower()
        
        # 특정 키워드에 대한 강제 분류 (높은 우선순위)
        high_priority_keywords = {
            "안녕": QuestionType.EVERYDAY,
            "날씨": QuestionType.EVERYDAY,
            "데이트": QuestionType.EVERYDAY,
            "상식": QuestionType.EVERYDAY
        }
        
        # 고우선순위 키워드 확인
        for keyword, q_type in high_priority_keywords.items():
            if keyword in question_lower:
                return ClassificationResult(
                    question_type=q_type,
                    confidence=0.9,  # 높은 신뢰도로 설정
                    reasoning=f"고우선순위 키워드 매칭: {keyword}",
                    suggested_prompt_type=self.prompt_type_mapping[q_type]
                )
        
        # 각 유형별 키워드 매칭 점수 계산
        type_scores = {}
        for q_type, keywords in self.type_keywords.items():
            score = 0
            matched_keywords = []
            
            for keyword in keywords:
                if keyword in question_lower:
                    score += 1
                    matched_keywords.append(keyword)
            
            if score > 0:
                type_scores[q_type] = {
                    'score': score,
                    'matched_keywords': matched_keywords,
                    'confidence': min(score / len(keywords), 1.0)
                }
        
        if type_scores:
            # 가장 높은 점수의 유형 선택
            best_type = max(type_scores.keys(), key=lambda x: type_scores[x]['score'])
            best_score = type_scores[best_type]
            
            if best_score['confidence'] > 0.3:  # 최소 임계값
                return ClassificationResult(
                    question_type=best_type,
                    confidence=best_score['confidence'],
                    reasoning=f"키워드 매칭: {', '.join(best_score['matched_keywords'])}",
                    suggested_prompt_type=self.prompt_type_mapping[best_type]
                )
        
        return None
    
    def _llm_classify(self, question: str) -> ClassificationResult:
        """LLM을 사용한 정밀 분류"""
        
        # 분류 프롬프트 구성
        system_prompt = """당신은 사용자 질문을 분석하여 적절한 질문 유형을 분류하는 전문가입니다.

다음 질문 유형 중에서 가장 적합한 것을 선택해주세요:

1. 일상: 일반적인 대화나 상식 질문
2. 통계: 수치, 데이터, 통계 관련 질문  
3. 분석: 심층 분석, 비교, 평가 요청
4. 요약: 문서나 내용 요약 요청
5. 검색: 특정 정보 검색 요청
6. 비교: 여러 항목 비교 요청
7. 설명: 개념이나 과정 설명 요청
8. 추천: 추천이나 제안 요청

응답은 다음 JSON 형식으로만 제공해주세요:
{
    "question_type": "선택된_유형",
    "confidence": 0.0-1.0,
    "reasoning": "분류 근거"
}"""

        user_prompt = f"다음 질문을 분류해주세요: {question}"
        
        try:
            response = self.client.chat.completions.create(
                model=self.model_name,
                messages=[
                    {"role": "system", "content": system_prompt},
                    {"role": "user", "content": user_prompt}
                ],
                temperature=0.1,
                max_tokens=200
            )
            
            # JSON 응답 파싱
            content = response.choices[0].message.content.strip()
            logger.debug(f"분류기 LLM 응답: {content}")
            
            # JSON 파싱 시도
            try:
                result_data = json.loads(content)
            except json.JSONDecodeError:
                # JSON 파싱 실패 시 텍스트에서 추출 시도
                result_data = self._parse_non_json_response(content)
            
            # QuestionType 열거형으로 변환
            question_type_str = result_data.get("question_type", "검색")
            question_type = self._str_to_question_type(question_type_str)
            
            return ClassificationResult(
                question_type=question_type,
                confidence=float(result_data.get("confidence", 0.7)),
                reasoning=result_data.get("reasoning", "LLM 분류 결과"),
                suggested_prompt_type=self.prompt_type_mapping[question_type]
            )
            
        except Exception as e:
            logger.error(f"LLM 분류 중 오류: {e}")
            raise
    
    def _parse_non_json_response(self, content: str) -> Dict[str, Any]:
        """JSON이 아닌 응답에서 정보 추출"""
        import re
        
        # 질문 유형 추출
        type_match = re.search(r'(일상|통계|분석|요약|검색|비교|설명|추천)', content)
        question_type = type_match.group(1) if type_match else "검색"
        
        # 신뢰도 추출 (0.0-1.0 범위의 숫자)
        confidence_match = re.search(r'(\d+\.?\d*)', content)
        confidence = float(confidence_match.group(1)) if confidence_match else 0.7
        if confidence > 1.0:
            confidence = confidence / 100.0
        
        return {
            "question_type": question_type,
            "confidence": confidence,
            "reasoning": content
        }
    
    def _str_to_question_type(self, type_str: str) -> QuestionType:
        """문자열을 QuestionType 열거형으로 변환"""
        mapping = {
            "일상": QuestionType.EVERYDAY,
            "통계": QuestionType.STATISTICS,
            "분석": QuestionType.ANALYSIS,
            "요약": QuestionType.SUMMARY,
            "검색": QuestionType.SEARCH,
            "비교": QuestionType.COMPARISON,
            "설명": QuestionType.EXPLANATION,
            "추천": QuestionType.RECOMMENDATION
        }
        return mapping.get(type_str, QuestionType.SEARCH)
    
    def get_prompt_type_for_question(self, question: str) -> str:
        """질문에 대한 프롬프트 타입 반환 (간편 메서드)"""
        result = self.classify_question(question)
        return result.suggested_prompt_type
    
    def get_available_question_types(self) -> List[str]:
        """사용 가능한 질문 유형 목록 반환"""
        return [q_type.value for q_type in QuestionType]
    
    def get_type_keywords(self, question_type: QuestionType) -> List[str]:
        """특정 질문 유형의 키워드 목록 반환"""
        return self.type_keywords.get(question_type, [])

# 전역 인스턴스
_classifier_instance = None

def get_question_classifier() -> QuestionClassifier:
    """전역 질문 분류기 인스턴스 반환"""
    global _classifier_instance
    if _classifier_instance is None:
        _classifier_instance = QuestionClassifier()
        _classifier_instance.initialize()
    return _classifier_instance
