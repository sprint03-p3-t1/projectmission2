"""
RFP RAG 시스템 - 품질 평가 메트릭스 관리 모듈
MLOps 파이프라인의 핵심 구성 요소
"""

import os
import json
import sqlite3
import logging
from typing import Dict, List, Any, Optional
from datetime import datetime, timedelta
from pathlib import Path
import pandas as pd

# 로깅 설정
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class QualityMetrics:
    """품질 평가 메트릭스 관리 클래스"""
    
    def __init__(self, db_path: str = None):
        """품질 메트릭스 초기화"""
        if db_path is None:
            # 기본 데이터베이스 경로 설정
            project_root = Path(__file__).resolve().parent.parent.parent
            db_path = str(project_root / "data" / "quality_metrics.db")
        
        self.db_path = db_path
        self._init_database()
    
    def _init_database(self):
        """데이터베이스 초기화"""
        # 데이터베이스 디렉토리 생성
        os.makedirs(os.path.dirname(self.db_path), exist_ok=True)
        
        with sqlite3.connect(self.db_path) as conn:
            cursor = conn.cursor()
            
            # 품질 평가 결과 테이블
            cursor.execute("""
                CREATE TABLE IF NOT EXISTS quality_evaluations (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    timestamp TEXT NOT NULL,
                    question TEXT NOT NULL,
                    answer TEXT NOT NULL,
                    context TEXT,
                    relevance_score REAL,
                    completeness_score REAL,
                    accuracy_score REAL,
                    clarity_score REAL,
                    structure_score REAL,
                    overall_score REAL,
                    suggestions TEXT,
                    evaluation_text TEXT,
                    model_name TEXT,
                    user_id TEXT,
                    session_id TEXT
                )
            """)
            
            # 품질 트렌드 테이블 (일별 집계)
            cursor.execute("""
                CREATE TABLE IF NOT EXISTS quality_trends (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    date TEXT NOT NULL,
                    avg_overall_score REAL,
                    avg_relevance_score REAL,
                    avg_completeness_score REAL,
                    avg_accuracy_score REAL,
                    avg_clarity_score REAL,
                    avg_structure_score REAL,
                    total_evaluations INTEGER,
                    high_quality_count INTEGER,
                    medium_quality_count INTEGER,
                    low_quality_count INTEGER
                )
            """)
            
            # 사용자 피드백 테이블
            cursor.execute("""
                CREATE TABLE IF NOT EXISTS user_feedback (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    timestamp TEXT NOT NULL,
                    evaluation_id INTEGER,
                    user_rating INTEGER,
                    user_comment TEXT,
                    feedback_type TEXT,
                    FOREIGN KEY (evaluation_id) REFERENCES quality_evaluations (id)
                )
            """)
            
            conn.commit()
            logger.info(f"Quality metrics database initialized: {self.db_path}")
    
    def store_evaluation(self, 
                        question: str, 
                        answer: str, 
                        context: str,
                        scores: Dict[str, float],
                        overall_score: float,
                        suggestions: List[str],
                        evaluation_text: str,
                        model_name: str = None,
                        user_id: str = None,
                        session_id: str = None) -> int:
        """평가 결과 저장"""
        
        with sqlite3.connect(self.db_path) as conn:
            cursor = conn.cursor()
            
            cursor.execute("""
                INSERT INTO quality_evaluations (
                    timestamp, question, answer, context,
                    relevance_score, completeness_score, accuracy_score,
                    clarity_score, structure_score, overall_score,
                    suggestions, evaluation_text, model_name, user_id, session_id
                ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
            """, (
                datetime.now().isoformat(),
                question,
                answer,
                context,
                scores.get('relevance', 0.0),
                scores.get('completeness', 0.0),
                scores.get('accuracy', 0.0),
                scores.get('clarity', 0.0),
                scores.get('structure', 0.0),
                overall_score,
                json.dumps(suggestions, ensure_ascii=False),
                evaluation_text,
                model_name,
                user_id,
                session_id
            ))
            
            evaluation_id = cursor.lastrowid
            conn.commit()
            
            logger.info(f"Quality evaluation stored with ID: {evaluation_id}")
            return evaluation_id
    
    def get_quality_trends(self, days: int = 30) -> pd.DataFrame:
        """품질 트렌드 조회"""
        
        with sqlite3.connect(self.db_path) as conn:
            query = """
                SELECT 
                    DATE(timestamp) as date,
                    AVG(overall_score) as avg_overall_score,
                    AVG(relevance_score) as avg_relevance_score,
                    AVG(completeness_score) as avg_completeness_score,
                    AVG(accuracy_score) as avg_accuracy_score,
                    AVG(clarity_score) as avg_clarity_score,
                    AVG(structure_score) as avg_structure_score,
                    COUNT(*) as total_evaluations,
                    SUM(CASE WHEN overall_score >= 0.8 THEN 1 ELSE 0 END) as high_quality_count,
                    SUM(CASE WHEN overall_score >= 0.6 AND overall_score < 0.8 THEN 1 ELSE 0 END) as medium_quality_count,
                    SUM(CASE WHEN overall_score < 0.6 THEN 1 ELSE 0 END) as low_quality_count
                FROM quality_evaluations
                WHERE timestamp >= datetime('now', '-{} days')
                GROUP BY DATE(timestamp)
                ORDER BY date
            """.format(days)
            
            df = pd.read_sql_query(query, conn)
            return df
    
    def get_recent_evaluations(self, limit: int = 100) -> pd.DataFrame:
        """최근 평가 결과 조회"""
        
        with sqlite3.connect(self.db_path) as conn:
            query = """
                SELECT 
                    id, timestamp, question, answer,
                    relevance_score, completeness_score, accuracy_score,
                    clarity_score, structure_score, overall_score,
                    suggestions, model_name, user_id, session_id
                FROM quality_evaluations
                ORDER BY timestamp DESC
                LIMIT ?
            """
            
            df = pd.read_sql_query(query, conn, params=(limit,))
            return df
    
    def get_quality_statistics(self, days: int = 7) -> Dict[str, Any]:
        """품질 통계 조회"""
        
        with sqlite3.connect(self.db_path) as conn:
            cursor = conn.cursor()
            
            # 기본 통계
            cursor.execute("""
                SELECT 
                    COUNT(*) as total_evaluations,
                    AVG(overall_score) as avg_overall_score,
                    AVG(relevance_score) as avg_relevance_score,
                    AVG(completeness_score) as avg_completeness_score,
                    AVG(accuracy_score) as avg_accuracy_score,
                    AVG(clarity_score) as avg_clarity_score,
                    AVG(structure_score) as avg_structure_score,
                    MIN(overall_score) as min_overall_score,
                    MAX(overall_score) as max_overall_score,
                    STDDEV(overall_score) as std_overall_score
                FROM quality_evaluations
                WHERE timestamp >= datetime('now', '-{} days')
            """.format(days))
            
            stats = cursor.fetchone()
            
            # 품질 분포
            cursor.execute("""
                SELECT 
                    SUM(CASE WHEN overall_score >= 0.8 THEN 1 ELSE 0 END) as high_quality,
                    SUM(CASE WHEN overall_score >= 0.6 AND overall_score < 0.8 THEN 1 ELSE 0 END) as medium_quality,
                    SUM(CASE WHEN overall_score < 0.6 THEN 1 ELSE 0 END) as low_quality
                FROM quality_evaluations
                WHERE timestamp >= datetime('now', '-{} days')
            """.format(days))
            
            distribution = cursor.fetchone()
            
            return {
                "total_evaluations": stats[0] or 0,
                "avg_overall_score": round(stats[1] or 0, 3),
                "avg_relevance_score": round(stats[2] or 0, 3),
                "avg_completeness_score": round(stats[3] or 0, 3),
                "avg_accuracy_score": round(stats[4] or 0, 3),
                "avg_clarity_score": round(stats[5] or 0, 3),
                "avg_structure_score": round(stats[6] or 0, 3),
                "min_overall_score": round(stats[7] or 0, 3),
                "max_overall_score": round(stats[8] or 0, 3),
                "std_overall_score": round(stats[9] or 0, 3),
                "quality_distribution": {
                    "high_quality": distribution[0] or 0,
                    "medium_quality": distribution[1] or 0,
                    "low_quality": distribution[2] or 0
                }
            }
    
    def store_user_feedback(self, 
                           evaluation_id: int,
                           user_rating: int,
                           user_comment: str = None,
                           feedback_type: str = "rating") -> int:
        """사용자 피드백 저장"""
        
        with sqlite3.connect(self.db_path) as conn:
            cursor = conn.cursor()
            
            cursor.execute("""
                INSERT INTO user_feedback (
                    timestamp, evaluation_id, user_rating, user_comment, feedback_type
                ) VALUES (?, ?, ?, ?, ?)
            """, (
                datetime.now().isoformat(),
                evaluation_id,
                user_rating,
                user_comment,
                feedback_type
            ))
            
            feedback_id = cursor.lastrowid
            conn.commit()
            
            logger.info(f"User feedback stored with ID: {feedback_id}")
            return feedback_id
    
    def get_improvement_suggestions(self, days: int = 7) -> List[str]:
        """개선 제안 분석"""
        
        with sqlite3.connect(self.db_path) as conn:
            cursor = conn.cursor()
            
            cursor.execute("""
                SELECT suggestions
                FROM quality_evaluations
                WHERE timestamp >= datetime('now', '-{} days')
                AND overall_score < 0.7
                ORDER BY overall_score ASC
                LIMIT 50
            """.format(days))
            
            suggestions_data = cursor.fetchall()
            
            # 개선 제안 빈도 분석
            all_suggestions = []
            for row in suggestions_data:
                try:
                    suggestions = json.loads(row[0])
                    all_suggestions.extend(suggestions)
                except:
                    continue
            
            # 빈도 기반 상위 제안 추출
            from collections import Counter
            suggestion_counts = Counter(all_suggestions)
            
            return [suggestion for suggestion, count in suggestion_counts.most_common(5)]
    
    def export_data(self, output_path: str, days: int = 30):
        """데이터 내보내기"""
        
        with sqlite3.connect(self.db_path) as conn:
            # 평가 데이터 내보내기
            evaluations_df = pd.read_sql_query("""
                SELECT * FROM quality_evaluations
                WHERE timestamp >= datetime('now', '-{} days')
            """.format(days), conn)
            
            # 트렌드 데이터 내보내기
            trends_df = self.get_quality_trends(days)
            
            # Excel 파일로 내보내기
            with pd.ExcelWriter(output_path, engine='openpyxl') as writer:
                evaluations_df.to_excel(writer, sheet_name='Evaluations', index=False)
                trends_df.to_excel(writer, sheet_name='Trends', index=False)
            
            logger.info(f"Quality data exported to: {output_path}")

# 품질 메트릭스 관리자 싱글톤
_quality_metrics_instance = None

def get_quality_metrics() -> QualityMetrics:
    """품질 메트릭스 인스턴스 반환 (싱글톤)"""
    global _quality_metrics_instance
    if _quality_metrics_instance is None:
        _quality_metrics_instance = QualityMetrics()
    return _quality_metrics_instance
