"""
자동 평가 시스템 - LLM 기반 질문 생성, 답변 생성, 평가 수행
"""

import logging
import json
import re
from typing import Dict, List, Any, Optional, Tuple
from datetime import datetime
import sqlite3
from dataclasses import dataclass

from src.prompts.prompt_manager import PromptManager

logger = logging.getLogger(__name__)

@dataclass
class GeneratedQuestion:
    """생성된 질문 정보"""
    question: str
    question_type: str  # 사업정보, 일정, 요구사항, 평가기준, 전략적
    difficulty: str     # Easy, Medium, Hard
    keywords: List[str]
    expected_answer: str
    source_chunk: str

@dataclass
class EvaluationResult:
    """평가 결과"""
    question: str
    answer: str
    scores: Dict[str, float]
    overall_score: float
    strengths: List[str]
    weaknesses: List[str]
    improvement_suggestions: List[str]
    evaluation_notes: str
    timestamp: str

class AutoEvaluator:
    """자동 평가 시스템"""
    
    def __init__(self, config_path: str = "config/rag_config.yaml"):
        self.config_path = config_path
        self.prompt_manager = None
        self.generator = None
        self.db_path = "data/ops/auto_evaluation.db"
        self._initialize_db()
        
    def _initialize_db(self):
        """자동 평가 데이터베이스 초기화"""
        try:
            conn = sqlite3.connect(self.db_path)
            cursor = conn.cursor()
            
            # 생성된 질문 테이블
            cursor.execute('''
                CREATE TABLE IF NOT EXISTS generated_questions (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    question TEXT NOT NULL,
                    question_type TEXT NOT NULL,
                    difficulty TEXT NOT NULL,
                    keywords TEXT,
                    expected_answer TEXT,
                    source_chunk TEXT,
                    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
                )
            ''')
            
            # 자동 평가 결과 테이블
            cursor.execute('''
                CREATE TABLE IF NOT EXISTS auto_evaluation_results (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    question_id INTEGER,
                    question TEXT NOT NULL,
                    answer TEXT NOT NULL,
                    accuracy_score REAL,
                    completeness_score REAL,
                    relevance_score REAL,
                    clarity_score REAL,
                    structure_score REAL,
                    practicality_score REAL,
                    expertise_score REAL,
                    creativity_score REAL,
                    feasibility_score REAL,
                    risk_analysis_score REAL,
                    overall_score REAL,
                    strengths TEXT,
                    weaknesses TEXT,
                    improvement_suggestions TEXT,
                    evaluation_notes TEXT,
                    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                    FOREIGN KEY (question_id) REFERENCES generated_questions (id)
                )
            ''')
            
            conn.commit()
            conn.close()
            logger.info(f"Auto evaluation database initialized: {self.db_path}")
            
        except Exception as e:
            logger.error(f"Failed to initialize auto evaluation database: {e}")
            raise
    
    def initialize(self):
        """시스템 초기화"""
        try:
            # 프롬프트 매니저 초기화
            self.prompt_manager = PromptManager()
            
            # RFP Generator 초기화 (순환 import 방지를 위해 여기서 import)
            from src.generation.generator import RFPGenerator
            self.generator = RFPGenerator()
            self.generator.initialize()
            
            logger.info("Auto evaluator initialized successfully")
            
        except Exception as e:
            logger.error(f"Failed to initialize auto evaluator: {e}")
            raise
    
    def generate_questions_from_chunks(self, chunks: List[str], num_questions_per_chunk: int = 3) -> List[GeneratedQuestion]:
        """문서 청크에서 질문 생성"""
        if not self.prompt_manager or not self.generator:
            raise RuntimeError("Auto evaluator not initialized")
        
        generated_questions = []
        
        try:
            # v3 질문 생성 프롬프트 로드
            question_prompt = self.prompt_manager.load_prompt_file("v3_question_generation_prompt.txt")
            
            for chunk in chunks:
                if not chunk.strip():
                    continue
                    
                # 질문 생성 프롬프트 구성
                prompt = f"""
{question_prompt}

문서 청크:
{chunk[:2000]}...

위 문서 청크를 바탕으로 {num_questions_per_chunk}개의 질문을 생성해주세요.
"""
                
                # LLM으로 질문 생성
                response = self.generator.client.chat.completions.create(
                    model=self.generator.model,
                    messages=[
                        {"role": "system", "content": "당신은 RFP 문서 분석 전문가입니다."},
                        {"role": "user", "content": prompt}
                    ],
                    temperature=0.7,
                    max_tokens=1000
                )
                
                # 생성된 질문 파싱
                questions_text = response.choices[0].message.content
                parsed_questions = self._parse_generated_questions(questions_text, chunk)
                generated_questions.extend(parsed_questions)
                
                # 데이터베이스에 저장
                self._save_generated_questions(parsed_questions)
                
        except Exception as e:
            logger.error(f"Failed to generate questions: {e}")
            raise
        
        logger.info(f"Generated {len(generated_questions)} questions from {len(chunks)} chunks")
        return generated_questions
    
    def _parse_generated_questions(self, questions_text: str, source_chunk: str) -> List[GeneratedQuestion]:
        """생성된 질문 텍스트 파싱"""
        questions = []
        
        try:
            # 질문 블록 분리
            question_blocks = re.split(r'질문:', questions_text)
            
            for block in question_blocks[1:]:  # 첫 번째는 빈 블록
                if not block.strip():
                    continue
                    
                lines = block.strip().split('\n')
                if len(lines) < 2:
                    continue
                
                question = lines[0].strip()
                
                # 메타데이터 추출
                question_type = "사업정보"
                difficulty = "Medium"
                keywords = []
                expected_answer = ""
                
                for line in lines[1:]:
                    line = line.strip()
                    if line.startswith('유형:'):
                        question_type = line.replace('유형:', '').strip()
                    elif line.startswith('난이도:'):
                        difficulty = line.replace('난이도:', '').strip()
                    elif line.startswith('키워드:'):
                        keywords_text = line.replace('키워드:', '').strip()
                        keywords = [k.strip() for k in keywords_text.split(',') if k.strip()]
                    elif line.startswith('예상답변:'):
                        expected_answer = line.replace('예상답변:', '').strip()
                
                if question:
                    questions.append(GeneratedQuestion(
                        question=question,
                        question_type=question_type,
                        difficulty=difficulty,
                        keywords=keywords,
                        expected_answer=expected_answer,
                        source_chunk=source_chunk[:500]  # 청크는 500자로 제한
                    ))
                    
        except Exception as e:
            logger.error(f"Failed to parse generated questions: {e}")
        
        return questions
    
    def _save_generated_questions(self, questions: List[GeneratedQuestion]):
        """생성된 질문을 데이터베이스에 저장"""
        try:
            conn = sqlite3.connect(self.db_path)
            cursor = conn.cursor()
            
            for question in questions:
                cursor.execute('''
                    INSERT INTO generated_questions 
                    (question, question_type, difficulty, keywords, expected_answer, source_chunk)
                    VALUES (?, ?, ?, ?, ?, ?)
                ''', (
                    question.question,
                    question.question_type,
                    question.difficulty,
                    json.dumps(question.keywords, ensure_ascii=False),
                    question.expected_answer,
                    question.source_chunk
                ))
            
            conn.commit()
            conn.close()
            
        except Exception as e:
            logger.error(f"Failed to save generated questions: {e}")
    
    def run_auto_evaluation(self, questions: List[GeneratedQuestion]) -> List[EvaluationResult]:
        """자동 평가 실행"""
        if not self.generator or not self.prompt_manager:
            raise RuntimeError("Auto evaluator not initialized")
        
        evaluation_results = []
        
        try:
            # v3 고도화된 평가 프롬프트 로드
            evaluation_prompt = self.prompt_manager.load_prompt_file("v3_advanced_evaluation_prompt.txt")
            
            for question in questions:
                try:
                    # 간단한 답변 생성 (검색 없이)
                    # LLM에게 직접 질문에 답변하도록 요청
                    prompt = f"""
다음 질문에 대해 RFP 전문가 관점에서 답변해주세요:

질문: {question.question}

참고 문서: {question.source_chunk}

위 정보를 바탕으로 전문적이고 구체적인 답변을 제공해주세요.
"""
                    
                    response = self.generator.client.chat.completions.create(
                        model=self.generator.model,
                        messages=[
                            {"role": "system", "content": "당신은 RFP 문서 분석 전문가입니다."},
                            {"role": "user", "content": prompt}
                        ],
                        temperature=0.1,
                        max_tokens=1000
                    )
                    
                    answer = response.choices[0].message.content
                    
                    # 답변 품질 평가
                    evaluation_result = self._evaluate_answer(
                        question.question, 
                        answer, 
                        question.source_chunk,
                        evaluation_prompt
                    )
                    
                    evaluation_results.append(evaluation_result)
                    
                    # 데이터베이스에 저장
                    self._save_evaluation_result(question, evaluation_result)
                    
                except Exception as e:
                    logger.error(f"Failed to evaluate question '{question.question}': {e}")
                    continue
                    
        except Exception as e:
            logger.error(f"Failed to run auto evaluation: {e}")
            raise
        
        logger.info(f"Completed auto evaluation for {len(evaluation_results)} questions")
        return evaluation_results
    
    def _evaluate_answer(self, question: str, answer: str, context: str, evaluation_prompt: str) -> EvaluationResult:
        """답변 평가"""
        try:
            # 평가 프롬프트 구성
            prompt = f"""
{evaluation_prompt}

질문: {question}

답변: {answer}

참고 문서: {context[:2000]}...

위 질문과 답변을 10개 기준으로 평가해주세요.
"""
            
            # LLM으로 평가 수행
            response = self.generator.client.chat.completions.create(
                model=self.generator.model,
                messages=[
                    {"role": "system", "content": "당신은 RFP 컨설팅 전문가이자 품질 평가 전문가입니다."},
                    {"role": "user", "content": prompt}
                ],
                temperature=0.1,
                max_tokens=1000
            )
            
            # 평가 결과 파싱
            evaluation_text = response.choices[0].message.content
            return self._parse_evaluation_result(question, answer, evaluation_text)
            
        except Exception as e:
            logger.error(f"Failed to evaluate answer: {e}")
            raise
    
    def _parse_evaluation_result(self, question: str, answer: str, evaluation_text: str) -> EvaluationResult:
        """평가 결과 파싱"""
        try:
            # 점수 추출
            scores = {}
            score_patterns = {
                'accuracy': r'정확성:\s*([0-9.]+)',
                'completeness': r'완성도:\s*([0-9.]+)',
                'relevance': r'관련성:\s*([0-9.]+)',
                'clarity': r'명확성:\s*([0-9.]+)',
                'structure': r'구조화:\s*([0-9.]+)',
                'practicality': r'실용성:\s*([0-9.]+)',
                'expertise': r'전문성:\s*([0-9.]+)',
                'creativity': r'창의성:\s*([0-9.]+)',
                'feasibility': r'실행가능성:\s*([0-9.]+)',
                'risk_analysis': r'리스크분석:\s*([0-9.]+)'
            }
            
            for key, pattern in score_patterns.items():
                match = re.search(pattern, evaluation_text)
                if match:
                    scores[key] = float(match.group(1))
                else:
                    scores[key] = 0.0
            
            # 종합 점수 추출
            overall_match = re.search(r'종합점수:\s*([0-9.]+)', evaluation_text)
            overall_score = float(overall_match.group(1)) if overall_match else sum(scores.values()) / len(scores)
            
            # 강점, 약점, 개선제안 추출
            strengths = self._extract_list_section(evaluation_text, '강점:')
            weaknesses = self._extract_list_section(evaluation_text, '약점:')
            improvement_suggestions = self._extract_list_section(evaluation_text, '개선제안:')
            
            # 전반 평가 추출
            evaluation_notes = self._extract_section(evaluation_text, '전반평가:')
            
            return EvaluationResult(
                question=question,
                answer=answer,
                scores=scores,
                overall_score=overall_score,
                strengths=strengths,
                weaknesses=weaknesses,
                improvement_suggestions=improvement_suggestions,
                evaluation_notes=evaluation_notes,
                timestamp=datetime.now().isoformat()
            )
            
        except Exception as e:
            logger.error(f"Failed to parse evaluation result: {e}")
            raise
    
    def _extract_list_section(self, text: str, section_name: str) -> List[str]:
        """리스트 섹션 추출"""
        try:
            lines = text.split('\n')
            in_section = False
            items = []
            
            for line in lines:
                line = line.strip()
                if line.startswith(section_name):
                    in_section = True
                    continue
                elif in_section:
                    if line.startswith('-') or line.startswith('•') or line.startswith('*'):
                        items.append(line[1:].strip())
                    elif line and not any(keyword in line for keyword in ['강점:', '약점:', '개선제안:', '전반평가:']):
                        items.append(line)
                    elif line.startswith(('강점:', '약점:', '개선제안:', '전반평가:')):
                        break
            
            return items[:3] if items else ["구체적인 내용을 제공할 수 없습니다."]
            
        except Exception as e:
            logger.error(f"Failed to extract list section '{section_name}': {e}")
            return ["구체적인 내용을 제공할 수 없습니다."]
    
    def _extract_section(self, text: str, section_name: str) -> str:
        """섹션 추출"""
        try:
            lines = text.split('\n')
            in_section = False
            content = []
            
            for line in lines:
                line = line.strip()
                if line.startswith(section_name):
                    in_section = True
                    content.append(line.replace(section_name, '').strip())
                    continue
                elif in_section:
                    if line and not any(keyword in line for keyword in ['강점:', '약점:', '개선제안:', '전반평가:']):
                        content.append(line)
                    elif line.startswith(('강점:', '약점:', '개선제안:', '전반평가:')):
                        break
            
            return ' '.join(content).strip() if content else "전반적인 평가를 제공할 수 없습니다."
            
        except Exception as e:
            logger.error(f"Failed to extract section '{section_name}': {e}")
            return "전반적인 평가를 제공할 수 없습니다."
    
    def _save_evaluation_result(self, question: GeneratedQuestion, result: EvaluationResult):
        """평가 결과를 데이터베이스에 저장"""
        try:
            conn = sqlite3.connect(self.db_path)
            cursor = conn.cursor()
            
            # 질문 ID 조회
            cursor.execute('SELECT id FROM generated_questions WHERE question = ?', (question.question,))
            question_row = cursor.fetchone()
            question_id = question_row[0] if question_row else None
            
            cursor.execute('''
                INSERT INTO auto_evaluation_results 
                (question_id, question, answer, accuracy_score, completeness_score, relevance_score,
                 clarity_score, structure_score, practicality_score, expertise_score, creativity_score,
                 feasibility_score, risk_analysis_score, overall_score, strengths, weaknesses,
                 improvement_suggestions, evaluation_notes)
                VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
            ''', (
                question_id,
                result.question,
                result.answer,
                result.scores.get('accuracy', 0.0),
                result.scores.get('completeness', 0.0),
                result.scores.get('relevance', 0.0),
                result.scores.get('clarity', 0.0),
                result.scores.get('structure', 0.0),
                result.scores.get('practicality', 0.0),
                result.scores.get('expertise', 0.0),
                result.scores.get('creativity', 0.0),
                result.scores.get('feasibility', 0.0),
                result.scores.get('risk_analysis', 0.0),
                result.overall_score,
                json.dumps(result.strengths, ensure_ascii=False),
                json.dumps(result.weaknesses, ensure_ascii=False),
                json.dumps(result.improvement_suggestions, ensure_ascii=False),
                result.evaluation_notes
            ))
            
            conn.commit()
            conn.close()
            
        except Exception as e:
            logger.error(f"Failed to save evaluation result: {e}")
    
    def get_evaluation_statistics(self) -> Dict[str, Any]:
        """평가 통계 조회"""
        try:
            conn = sqlite3.connect(self.db_path)
            cursor = conn.cursor()
            
            # 전체 통계
            cursor.execute('SELECT COUNT(*) FROM auto_evaluation_results')
            total_evaluations = cursor.fetchone()[0]
            
            # 평균 점수
            cursor.execute('''
                SELECT 
                    AVG(overall_score),
                    AVG(accuracy_score),
                    AVG(completeness_score),
                    AVG(relevance_score),
                    AVG(clarity_score),
                    AVG(structure_score),
                    AVG(practicality_score),
                    AVG(expertise_score),
                    AVG(creativity_score),
                    AVG(feasibility_score),
                    AVG(risk_analysis_score)
                FROM auto_evaluation_results
            ''')
            
            avg_scores = cursor.fetchone()
            
            # 질문 유형별 통계
            cursor.execute('''
                SELECT 
                    gq.question_type,
                    COUNT(*) as count,
                    AVG(aer.overall_score) as avg_score
                FROM generated_questions gq
                LEFT JOIN auto_evaluation_results aer ON gq.id = aer.question_id
                GROUP BY gq.question_type
            ''')
            
            type_stats = cursor.fetchall()
            
            # 난이도별 통계
            cursor.execute('''
                SELECT 
                    gq.difficulty,
                    COUNT(*) as count,
                    AVG(aer.overall_score) as avg_score
                FROM generated_questions gq
                LEFT JOIN auto_evaluation_results aer ON gq.id = aer.question_id
                GROUP BY gq.difficulty
            ''')
            
            difficulty_stats = cursor.fetchall()
            
            conn.close()
            
            return {
                'total_evaluations': total_evaluations,
                'average_scores': {
                    'overall': avg_scores[0] if avg_scores[0] else 0.0,
                    'accuracy': avg_scores[1] if avg_scores[1] else 0.0,
                    'completeness': avg_scores[2] if avg_scores[2] else 0.0,
                    'relevance': avg_scores[3] if avg_scores[3] else 0.0,
                    'clarity': avg_scores[4] if avg_scores[4] else 0.0,
                    'structure': avg_scores[5] if avg_scores[5] else 0.0,
                    'practicality': avg_scores[6] if avg_scores[6] else 0.0,
                    'expertise': avg_scores[7] if avg_scores[7] else 0.0,
                    'creativity': avg_scores[8] if avg_scores[8] else 0.0,
                    'feasibility': avg_scores[9] if avg_scores[9] else 0.0,
                    'risk_analysis': avg_scores[10] if avg_scores[10] else 0.0
                },
                'type_statistics': [
                    {'type': row[0], 'count': row[1], 'avg_score': row[2] if row[2] else 0.0}
                    for row in type_stats
                ],
                'difficulty_statistics': [
                    {'difficulty': row[0], 'count': row[1], 'avg_score': row[2] if row[2] else 0.0}
                    for row in difficulty_stats
                ]
            }
            
        except Exception as e:
            logger.error(f"Failed to get evaluation statistics: {e}")
            return {}
    
    def run_full_auto_evaluation(self, document_chunks: List[str], num_questions_per_chunk: int = 3) -> Dict[str, Any]:
        """전체 자동 평가 파이프라인 실행"""
        try:
            logger.info("Starting full auto evaluation pipeline")
            
            # 1. 질문 생성
            logger.info("Step 1: Generating questions from document chunks")
            questions = self.generate_questions_from_chunks(document_chunks, num_questions_per_chunk)
            
            # 2. 자동 평가 실행
            logger.info("Step 2: Running auto evaluation")
            evaluation_results = self.run_auto_evaluation(questions)
            
            # 3. 통계 생성
            logger.info("Step 3: Generating statistics")
            statistics = self.get_evaluation_statistics()
            
            return {
                'questions_generated': len(questions),
                'evaluations_completed': len(evaluation_results),
                'statistics': statistics,
                'questions': questions,
                'evaluation_results': evaluation_results
            }
            
        except Exception as e:
            logger.error(f"Failed to run full auto evaluation: {e}")
            raise
