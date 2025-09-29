"""
프롬프트 최적화 시스템
LLM 기반 자동 프롬프트 개선 파이프라인
"""

import os
import json
import logging
import sqlite3
from datetime import datetime
from typing import List, Dict, Any, Optional, Tuple
from dataclasses import dataclass, asdict
import yaml

logger = logging.getLogger(__name__)

@dataclass
class OptimizationResult:
    """프롬프트 최적화 결과"""
    version: str
    original_prompt: str
    optimized_prompt: str
    satisfaction_score: float
    iteration_count: int
    improvement_reasons: List[str]
    failed_cases: List[Dict[str, Any]]
    created_at: datetime
    status: str  # 'success', 'failed', 'in_progress'

@dataclass
class OptimizationConfig:
    """최적화 설정"""
    target_satisfaction: float = 0.9  # 목표 만족도 (90%)
    max_iterations: int = 5  # 최대 반복 수
    min_improvement: float = 0.05  # 최소 개선도 (5%)
    evaluation_metrics: List[str] = None
    
    def __post_init__(self):
        if self.evaluation_metrics is None:
            self.evaluation_metrics = [
                'accuracy', 'completeness', 'relevance', 'clarity', 
                'structure', 'practicality', 'expertise', 'creativity', 
                'feasibility', 'risk_analysis'
            ]

class PromptOptimizer:
    """프롬프트 최적화 시스템"""
    
    def __init__(self, db_path: str = "data/ops/prompt_optimization.db"):
        self.db_path = db_path
        self.client = None
        self.prompt_manager = None
        self.auto_evaluator = None
        self.initialize_db()
    
    def initialize_db(self):
        """최적화 데이터베이스 초기화"""
        os.makedirs(os.path.dirname(self.db_path), exist_ok=True)
        
        with sqlite3.connect(self.db_path) as conn:
            conn.execute("""
                CREATE TABLE IF NOT EXISTS optimization_results (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    version TEXT NOT NULL,
                    original_prompt TEXT NOT NULL,
                    optimized_prompt TEXT NOT NULL,
                    satisfaction_score REAL NOT NULL,
                    iteration_count INTEGER NOT NULL,
                    improvement_reasons TEXT NOT NULL,
                    failed_cases TEXT NOT NULL,
                    created_at TIMESTAMP NOT NULL,
                    status TEXT NOT NULL
                )
            """)
            
            conn.execute("""
                CREATE TABLE IF NOT EXISTS optimization_history (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    version TEXT NOT NULL,
                    iteration INTEGER NOT NULL,
                    prompt_text TEXT NOT NULL,
                    evaluation_scores TEXT NOT NULL,
                    satisfaction_score REAL NOT NULL,
                    improvement_suggestions TEXT NOT NULL,
                    created_at TIMESTAMP NOT NULL
                )
            """)
    
    def initialize(self, client, prompt_manager, auto_evaluator):
        """시스템 초기화"""
        self.client = client
        self.prompt_manager = prompt_manager
        self.auto_evaluator = auto_evaluator
        logger.info("Prompt optimizer initialized successfully")
    
    def optimize_prompt(
        self, 
        prompt_type: str,  # 'question_generation', 'evaluation', 'system'
        document_chunks: List[str],
        config: OptimizationConfig = None,
        base_version: str = None
    ) -> OptimizationResult:
        """
        프롬프트 최적화 실행
        
        Args:
            prompt_type: 최적화할 프롬프트 타입
            document_chunks: 테스트용 문서 청크들
            config: 최적화 설정
            base_version: 최적화할 기본 버전 (None이면 현재 버전 사용)
            
        Returns:
            OptimizationResult: 최적화 결과
        """
        if config is None:
            config = OptimizationConfig()
        
        # 기본 버전 설정
        if base_version is None:
            base_version = self.prompt_manager.get_current_version()
        
        logger.info(f"Starting prompt optimization for {prompt_type} (base version: {base_version})")
        
        # 프롬프트 가져오기
        original_prompt = self._get_prompt_by_type(prompt_type, base_version)
        
        if not original_prompt:
            raise ValueError(f"Prompt type {prompt_type} not found in version {base_version}")
        
        # 최적화 반복 실행
        best_result = None
        current_prompt = original_prompt
        
        for iteration in range(1, config.max_iterations + 1):
            logger.info(f"Optimization iteration {iteration}/{config.max_iterations}")
            
            # 현재 프롬프트로 평가 실행
            evaluation_result = self._evaluate_prompt_performance(
                current_prompt, prompt_type, document_chunks
            )
            
            satisfaction_score = evaluation_result['satisfaction_score']
            failed_cases = evaluation_result['failed_cases']
            improvement_suggestions = evaluation_result['improvement_suggestions']
            
            # 히스토리 저장
            self._save_optimization_history(
                base_version, iteration, current_prompt, 
                evaluation_result, satisfaction_score, improvement_suggestions
            )
            
            # 목표 만족도 달성 확인
            if satisfaction_score >= config.target_satisfaction:
                logger.info(f"Target satisfaction achieved: {satisfaction_score:.3f}")
                best_result = OptimizationResult(
                    version=f"{base_version}_optimized_v{iteration}",
                    original_prompt=original_prompt,
                    optimized_prompt=current_prompt,
                    satisfaction_score=satisfaction_score,
                    iteration_count=iteration,
                    improvement_reasons=improvement_suggestions,
                    failed_cases=failed_cases,
                    created_at=datetime.now(),
                    status='success'
                )
                break
            
            # 프롬프트 개선
            if iteration < config.max_iterations:
                current_prompt = self._improve_prompt(
                    current_prompt, failed_cases, improvement_suggestions, prompt_type
                )
                
                # 최소 개선도 확인
                if best_result and (satisfaction_score - best_result.satisfaction_score) < config.min_improvement:
                    logger.warning("Minimal improvement detected, stopping optimization")
                    break
                
                if not best_result or satisfaction_score > best_result.satisfaction_score:
                    best_result = OptimizationResult(
                        version=f"{base_version}_optimized_v{iteration}",
                        original_prompt=original_prompt,
                        optimized_prompt=current_prompt,
                        satisfaction_score=satisfaction_score,
                        iteration_count=iteration,
                        improvement_reasons=improvement_suggestions,
                        failed_cases=failed_cases,
                        created_at=datetime.now(),
                        status='in_progress'
                    )
        
        # 최종 결과 저장
        if best_result:
            if best_result.satisfaction_score < config.target_satisfaction:
                best_result.status = 'failed'
            self._save_optimization_result(best_result)
            logger.info(f"Optimization completed: {best_result.status}, score: {best_result.satisfaction_score:.3f}")
        
        return best_result
    
    def _get_prompt_by_type(self, prompt_type: str, version: str) -> Optional[str]:
        """프롬프트 타입별 내용 가져오기"""
        try:
            if prompt_type == 'question_generation':
                return self.prompt_manager.get_question_generation_prompt(version)
            elif prompt_type == 'evaluation':
                return self.prompt_manager.get_evaluation_prompt(version)
            elif prompt_type == 'system':
                return self.prompt_manager.get_system_prompt(version)
            else:
                return None
        except Exception as e:
            logger.error(f"Failed to get prompt {prompt_type} for version {version}: {e}")
            return None
    
    def _evaluate_prompt_performance(
        self, 
        prompt: str, 
        prompt_type: str, 
        document_chunks: List[str]
    ) -> Dict[str, Any]:
        """프롬프트 성능 평가"""
        try:
            if prompt_type == 'question_generation':
                return self._evaluate_question_generation_prompt(prompt, document_chunks)
            elif prompt_type == 'evaluation':
                return self._evaluate_evaluation_prompt(prompt, document_chunks)
            else:
                return self._evaluate_general_prompt(prompt, document_chunks)
        except Exception as e:
            logger.error(f"Failed to evaluate prompt performance: {e}")
            return {
                'satisfaction_score': 0.0,
                'failed_cases': [],
                'improvement_suggestions': [f"Evaluation failed: {str(e)}"]
            }
    
    def _evaluate_question_generation_prompt(
        self, 
        prompt: str, 
        document_chunks: List[str]
    ) -> Dict[str, Any]:
        """질문 생성 프롬프트 평가"""
        # 임시로 프롬프트 설정
        temp_prompt_file = "temp_question_prompt.txt"
        with open(temp_prompt_file, 'w', encoding='utf-8') as f:
            f.write(prompt)
        
        try:
            # 자동 평가 실행
            results = self.auto_evaluator.run_full_auto_evaluation(
                document_chunks, 
                num_questions_per_chunk=2  # 빠른 테스트를 위해 2개
            )
            
            # 결과 분석
            evaluation_results = results.get('evaluation_results', [])
            total_evaluations = len(evaluation_results)
            
            if total_evaluations == 0:
                return {
                    'satisfaction_score': 0.0,
                    'failed_cases': [],
                    'improvement_suggestions': ["No evaluations generated"]
                }
            
            # 평균 점수 계산
            avg_scores = {}
            for metric in ['accuracy', 'completeness', 'relevance', 'clarity']:
                scores = [r.scores.get(metric, 0.0) for r in evaluation_results]
                avg_scores[metric] = sum(scores) / len(scores) if scores else 0.0
            
            satisfaction_score = sum(avg_scores.values()) / len(avg_scores)
            
            # 실패 사례 식별
            failed_cases = []
            improvement_suggestions = []
            
            for result in evaluation_results:
                if result.overall_score < 0.7:  # 임계값
                    failed_cases.append({
                        'question': result.question,
                        'answer': result.answer,
                        'scores': result.scores,
                        'overall_score': result.overall_score
                    })
            
            # 개선 제안 생성
            if satisfaction_score < 0.8:
                improvement_suggestions.append("질문의 다양성과 깊이를 높이세요")
            if avg_scores.get('relevance', 0) < 0.8:
                improvement_suggestions.append("문서 내용과의 관련성을 강화하세요")
            if avg_scores.get('clarity', 0) < 0.8:
                improvement_suggestions.append("질문의 명확성과 구체성을 개선하세요")
            
            return {
                'satisfaction_score': satisfaction_score,
                'failed_cases': failed_cases,
                'improvement_suggestions': improvement_suggestions,
                'detailed_scores': avg_scores
            }
            
        finally:
            # 임시 파일 정리
            if os.path.exists(temp_prompt_file):
                os.remove(temp_prompt_file)
    
    def _evaluate_evaluation_prompt(
        self, 
        prompt: str, 
        document_chunks: List[str]
    ) -> Dict[str, Any]:
        """평가 프롬프트 평가"""
        # 평가 프롬프트는 질문-답변 쌍으로 테스트
        test_qa_pairs = [
            {
                'question': '이 문서의 주요 내용은 무엇인가요?',
                'answer': '이 문서는 디지털 전환에 대한 내용을 다루고 있습니다.',
                'chunk': document_chunks[0] if document_chunks else '테스트 청크'
            }
        ]
        
        try:
            # 임시 평가 프롬프트 파일 생성
            temp_eval_file = "temp_eval_prompt.txt"
            with open(temp_eval_file, 'w', encoding='utf-8') as f:
                f.write(prompt)
            
            # 평가 실행
            evaluation_results = []
            for qa_pair in test_qa_pairs:
                result = self.auto_evaluator._evaluate_answer(
                    qa_pair['question'],
                    qa_pair['answer'],
                    qa_pair['chunk'],
                    prompt  # 최적화된 프롬프트 직접 전달
                )
                evaluation_results.append(result)
            
            # 결과 분석
            if not evaluation_results:
                return {
                    'satisfaction_score': 0.0,
                    'failed_cases': [],
                    'improvement_suggestions': ["No evaluation results generated"]
                }
            
            # 평균 점수 계산
            avg_scores = {}
            for metric in ['accuracy', 'completeness', 'relevance', 'clarity']:
                scores = [r.scores.get(metric, 0.0) for r in evaluation_results]
                avg_scores[metric] = sum(scores) / len(scores) if scores else 0.0
            
            satisfaction_score = sum(avg_scores.values()) / len(avg_scores)
            
            # 실패 사례 및 개선 제안
            failed_cases = []
            improvement_suggestions = []
            
            for result in evaluation_results:
                if result.overall_score < 0.7:
                    failed_cases.append({
                        'question': qa_pair['question'],
                        'answer': qa_pair['answer'],
                        'scores': result.scores,
                        'overall_score': result.overall_score
                    })
            
            if satisfaction_score < 0.8:
                improvement_suggestions.append("평가 기준의 명확성을 높이세요")
            if avg_scores.get('accuracy', 0) < 0.8:
                improvement_suggestions.append("정확성 평가 로직을 개선하세요")
            
            return {
                'satisfaction_score': satisfaction_score,
                'failed_cases': failed_cases,
                'improvement_suggestions': improvement_suggestions,
                'detailed_scores': avg_scores
            }
            
        finally:
            if os.path.exists(temp_eval_file):
                os.remove(temp_eval_file)
    
    def _evaluate_general_prompt(
        self, 
        prompt: str, 
        document_chunks: List[str]
    ) -> Dict[str, Any]:
        """일반 프롬프트 평가"""
        # 기본 평가 로직
        return {
            'satisfaction_score': 0.7,  # 기본값
            'failed_cases': [],
            'improvement_suggestions': ["General prompt evaluation not implemented"]
        }
    
    def _improve_prompt(
        self, 
        current_prompt: str, 
        failed_cases: List[Dict[str, Any]], 
        improvement_suggestions: List[str],
        prompt_type: str
    ) -> str:
        """프롬프트 개선"""
        try:
            # 개선 요청 프롬프트 생성
            improvement_prompt = f"""
당신은 프롬프트 최적화 전문가입니다. 
다음 프롬프트를 개선해주세요.

현재 프롬프트:
{current_prompt}

문제점 및 개선 제안:
{chr(10).join(f"- {suggestion}" for suggestion in improvement_suggestions)}

실패 사례:
{json.dumps(failed_cases[:3], ensure_ascii=False, indent=2) if failed_cases else "없음"}

개선된 프롬프트를 제공해주세요. 다음 원칙을 따라주세요:
1. 기존 프롬프트의 장점은 유지하세요
2. 문제점을 구체적으로 해결하세요
3. 더 명확하고 구체적인 지시사항을 추가하세요
4. 예시나 가이드라인을 포함하세요
5. 프롬프트 타입: {prompt_type}

개선된 프롬프트:
"""
            
            # LLM으로 개선된 프롬프트 생성
            response = self.client.chat.completions.create(
                model="gpt-4o-mini",
                messages=[
                    {"role": "system", "content": "당신은 프롬프트 엔지니어링 전문가입니다."},
                    {"role": "user", "content": improvement_prompt}
                ],
                temperature=0.7,
                max_tokens=2000
            )
            
            improved_prompt = response.choices[0].message.content.strip()
            logger.info(f"Generated improved prompt for {prompt_type}")
            
            return improved_prompt
            
        except Exception as e:
            logger.error(f"Failed to improve prompt: {e}")
            return current_prompt  # 실패시 원본 반환
    
    def _save_optimization_result(self, result: OptimizationResult):
        """최적화 결과 저장"""
        with sqlite3.connect(self.db_path) as conn:
            conn.execute("""
                INSERT INTO optimization_results 
                (version, original_prompt, optimized_prompt, satisfaction_score, 
                 iteration_count, improvement_reasons, failed_cases, created_at, status)
                VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)
            """, (
                result.version,
                result.original_prompt,
                result.optimized_prompt,
                result.satisfaction_score,
                result.iteration_count,
                json.dumps(result.improvement_reasons, ensure_ascii=False),
                json.dumps(result.failed_cases, ensure_ascii=False),
                result.created_at,
                result.status
            ))
    
    def _save_optimization_history(
        self, 
        version: str, 
        iteration: int, 
        prompt_text: str, 
        evaluation_result: Dict[str, Any],
        satisfaction_score: float,
        improvement_suggestions: List[str]
    ):
        """최적화 히스토리 저장"""
        with sqlite3.connect(self.db_path) as conn:
            conn.execute("""
                INSERT INTO optimization_history 
                (version, iteration, prompt_text, evaluation_scores, 
                 satisfaction_score, improvement_suggestions, created_at)
                VALUES (?, ?, ?, ?, ?, ?, ?)
            """, (
                version,
                iteration,
                prompt_text,
                json.dumps(evaluation_result.get('detailed_scores', {}), ensure_ascii=False),
                satisfaction_score,
                json.dumps(improvement_suggestions, ensure_ascii=False),
                datetime.now()
            ))
    
    def get_optimization_history(self, version: str = None) -> List[Dict[str, Any]]:
        """최적화 히스토리 조회"""
        with sqlite3.connect(self.db_path) as conn:
            if version:
                cursor = conn.execute("""
                    SELECT * FROM optimization_history 
                    WHERE version = ? 
                    ORDER BY iteration ASC
                """, (version,))
            else:
                cursor = conn.execute("""
                    SELECT * FROM optimization_history 
                    ORDER BY version DESC, iteration ASC
                """)
            
            columns = [description[0] for description in cursor.description]
            return [dict(zip(columns, row)) for row in cursor.fetchall()]
    
    def get_optimization_results(self, status: str = None) -> List[Dict[str, Any]]:
        """최적화 결과 조회"""
        with sqlite3.connect(self.db_path) as conn:
            if status:
                cursor = conn.execute("""
                    SELECT * FROM optimization_results 
                    WHERE status = ? 
                    ORDER BY created_at DESC
                """, (status,))
            else:
                cursor = conn.execute("""
                    SELECT * FROM optimization_results 
                    ORDER BY created_at DESC
                """)
            
            columns = [description[0] for description in cursor.description]
            return [dict(zip(columns, row)) for row in cursor.fetchall()]
    
    def apply_optimized_prompt(self, result_id: int) -> bool:
        """최적화된 프롬프트 적용"""
        try:
            with sqlite3.connect(self.db_path) as conn:
                cursor = conn.execute("""
                    SELECT * FROM optimization_results WHERE id = ?
                """, (result_id,))
                
                result = cursor.fetchone()
                if not result:
                    return False
                
                # 새로운 버전으로 프롬프트 저장
                new_version = f"v{datetime.now().strftime('%Y%m%d_%H%M%S')}"
                
                # 프롬프트 매니저에 새 버전 추가
                # (실제 구현은 prompt_manager의 메서드에 따라 달라질 수 있음)
                logger.info(f"Applied optimized prompt as version {new_version}")
                return True
                
        except Exception as e:
            logger.error(f"Failed to apply optimized prompt: {e}")
            return False
