#!/usr/bin/env python3
"""
질문 분류기 테스트 스크립트
다양한 질문 유형에 대해 분류기를 테스트합니다.
"""

import sys
from pathlib import Path

# 프로젝트 루트를 Python 경로에 추가
project_root = Path(__file__).parent
sys.path.insert(0, str(project_root))

from src.classification.question_classifier import get_question_classifier, QuestionType

def test_question_classifier():
    """질문 분류기 테스트"""
    print("🧪 질문 분류기 테스트 시작")
    print("=" * 60)
    
    # 테스트 질문들
    test_questions = [
        # 일상 질문
        ("안녕하세요, 어떻게 지내세요?", QuestionType.EVERYDAY),
        
        # 통계 질문
        ("이 사업의 예산은 얼마인가요?", QuestionType.STATISTICS),
        ("프로젝트 기간은 몇 개월인가요?", QuestionType.STATISTICS),
        ("참여 업체 수는 몇 개인가요?", QuestionType.STATISTICS),
        
        # 분석 질문
        ("이 사업의 장단점을 분석해주세요", QuestionType.ANALYSIS),
        ("요구사항을 평가해주세요", QuestionType.ANALYSIS),
        
        # 요약 질문
        ("사업 개요를 요약해주세요", QuestionType.SUMMARY),
        ("핵심 내용을 정리해주세요", QuestionType.SUMMARY),
        
        # 검색 질문
        ("발주기관이 어디인가요?", QuestionType.SEARCH),
        ("사업명을 찾아주세요", QuestionType.SEARCH),
        
        # 비교 질문
        ("A안과 B안을 비교해주세요", QuestionType.COMPARISON),
        ("차이점은 무엇인가요?", QuestionType.COMPARISON),
        
        # 설명 질문
        ("이 과정을 설명해주세요", QuestionType.EXPLANATION),
        ("어떻게 진행되나요?", QuestionType.EXPLANATION),
        
        # 추천 질문
        ("어떤 방법을 추천하시나요?", QuestionType.RECOMMENDATION),
        ("제안사항이 있나요?", QuestionType.RECOMMENDATION),
    ]
    
    try:
        # 분류기 초기화
        classifier = get_question_classifier()
        print("✅ 질문 분류기 초기화 완료\n")
        
        correct_predictions = 0
        total_questions = len(test_questions)
        
        for i, (question, expected_type) in enumerate(test_questions, 1):
            print(f"📝 테스트 {i}/{total_questions}: {question}")
            
            try:
                # 질문 분류
                result = classifier.classify_question(question)
                
                # 결과 출력
                print(f"   예상: {expected_type.value}")
                print(f"   예측: {result.question_type.value}")
                print(f"   신뢰도: {result.confidence:.3f}")
                print(f"   근거: {result.reasoning}")
                print(f"   프롬프트: {result.suggested_prompt_type}")
                
                # 정확도 계산
                if result.question_type == expected_type:
                    correct_predictions += 1
                    print("   ✅ 정확")
                else:
                    print("   ❌ 오류")
                
                print("-" * 40)
                
            except Exception as e:
                print(f"   ❌ 오류 발생: {e}")
                print("-" * 40)
        
        # 전체 정확도 출력
        accuracy = correct_predictions / total_questions
        print(f"\n📊 테스트 결과:")
        print(f"   전체 질문 수: {total_questions}")
        print(f"   정확한 예측: {correct_predictions}")
        print(f"   정확도: {accuracy:.3f} ({accuracy*100:.1f}%)")
        
        if accuracy >= 0.8:
            print("🎉 우수한 성능!")
        elif accuracy >= 0.6:
            print("👍 양호한 성능")
        else:
            print("⚠️ 개선이 필요합니다")
            
    except Exception as e:
        print(f"❌ 테스트 실패: {e}")

def test_prompt_templates():
    """프롬프트 템플릿 테스트"""
    print("\n🧪 프롬프트 템플릿 테스트")
    print("=" * 60)
    
    try:
        from src.prompts.prompt_manager import get_prompt_manager
        prompt_manager = get_prompt_manager()
        
        # 각 질문 유형별 템플릿 테스트
        question_types = ["general", "statistical", "analytical", "summarization", 
                         "search", "comparison", "explanatory", "recommendation"]
        
        for question_type in question_types:
            print(f"📝 {question_type} 템플릿:")
            template = prompt_manager.get_user_template_by_type(question_type)
            
            if template:
                # 템플릿에 샘플 데이터 적용
                sample_template = template.format(
                    question="테스트 질문입니다",
                    context="테스트 컨텍스트입니다"
                )
                print(f"   길이: {len(template)} 문자")
                print(f"   미리보기: {sample_template[:100]}...")
            else:
                print("   ❌ 템플릿을 찾을 수 없음")
            print("-" * 40)
            
    except Exception as e:
        print(f"❌ 프롬프트 템플릿 테스트 실패: {e}")

if __name__ == "__main__":
    print("🚀 질문 분류 시스템 테스트")
    print("=" * 60)
    
    # 질문 분류기 테스트
    test_question_classifier()
    
    # 프롬프트 템플릿 테스트
    test_prompt_templates()
    
    print("\n✅ 모든 테스트 완료!")
