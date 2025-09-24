"""
RFP RAG 시스템 - 대화 추적 및 로깅 시스템
질문부터 답변까지의 전체 루트를 추적하고 분석 가능한 형태로 저장
"""

import sqlite3
import json
import logging
from datetime import datetime
from typing import Dict, List, Any, Optional
from pathlib import Path
import uuid

logger = logging.getLogger(__name__)

class ConversationTracker:
    """대화 추적 및 로깅 시스템"""
    
    def __init__(self, db_path: str = "data/conversation_logs.db"):
        """대화 추적기 초기화"""
        self.db_path = Path(db_path)
        self.db_path.parent.mkdir(parents=True, exist_ok=True)
        self._init_database()
        logger.info(f"Conversation tracker initialized: {self.db_path}")
    
    def _init_database(self):
        """데이터베이스 초기화"""
        with sqlite3.connect(self.db_path) as conn:
            cursor = conn.cursor()
            
            # 대화 세션 테이블
            cursor.execute("""
                CREATE TABLE IF NOT EXISTS conversation_sessions (
                    session_id TEXT PRIMARY KEY,
                    user_id TEXT,
                    start_time TIMESTAMP,
                    end_time TIMESTAMP,
                    total_questions INTEGER DEFAULT 0,
                    session_metadata TEXT,
                    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
                )
            """)
            
            # 질문-답변 루트 테이블
            cursor.execute("""
                CREATE TABLE IF NOT EXISTS conversation_logs (
                    log_id TEXT PRIMARY KEY,
                    session_id TEXT,
                    question TEXT NOT NULL,
                    answer TEXT NOT NULL,
                    question_timestamp TIMESTAMP,
                    answer_timestamp TIMESTAMP,
                    
                    -- 시스템 정보
                    system_type TEXT,  -- 'faiss' or 'chromadb'
                    model_name TEXT,
                    temperature REAL,
                    max_tokens INTEGER,
                    
                    -- 검색 정보
                    search_method TEXT,  -- 'vector', 'hybrid', 'bm25'
                    retrieved_chunks_count INTEGER,
                    search_time_ms INTEGER,
                    
                    -- 생성 정보
                    generation_time_ms INTEGER,
                    prompt_tokens INTEGER,
                    completion_tokens INTEGER,
                    total_tokens INTEGER,
                    
                    -- 품질 평가
                    quality_scores TEXT,  -- JSON
                    overall_quality_score REAL,
                    quality_evaluation_id TEXT,
                    
                    -- 메타데이터
                    conversation_history TEXT,  -- JSON
                    retrieved_chunks TEXT,  -- JSON
                    generation_metadata TEXT,  -- JSON
                    error_log TEXT,
                    
                    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                    FOREIGN KEY (session_id) REFERENCES conversation_sessions (session_id)
                )
            """)
            
            # 검색 단계별 상세 로그
            cursor.execute("""
                CREATE TABLE IF NOT EXISTS search_step_logs (
                    step_id TEXT PRIMARY KEY,
                    log_id TEXT,
                    step_type TEXT,  -- 'embedding', 'vector_search', 'bm25_search', 'reranking'
                    step_order INTEGER,
                    input_data TEXT,  -- JSON
                    output_data TEXT,  -- JSON
                    execution_time_ms INTEGER,
                    metadata TEXT,  -- JSON
                    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                    FOREIGN KEY (log_id) REFERENCES conversation_logs (log_id)
                )
            """)
            
            # 인덱스 생성
            cursor.execute("CREATE INDEX IF NOT EXISTS idx_conversation_sessions_user_id ON conversation_sessions (user_id)")
            cursor.execute("CREATE INDEX IF NOT EXISTS idx_conversation_sessions_start_time ON conversation_sessions (start_time)")
            cursor.execute("CREATE INDEX IF NOT EXISTS idx_conversation_logs_session_id ON conversation_logs (session_id)")
            cursor.execute("CREATE INDEX IF NOT EXISTS idx_conversation_logs_question_timestamp ON conversation_logs (question_timestamp)")
            cursor.execute("CREATE INDEX IF NOT EXISTS idx_conversation_logs_system_type ON conversation_logs (system_type)")
            cursor.execute("CREATE INDEX IF NOT EXISTS idx_conversation_logs_quality_score ON conversation_logs (overall_quality_score)")
            cursor.execute("CREATE INDEX IF NOT EXISTS idx_search_step_logs_log_id ON search_step_logs (log_id)")
            
            conn.commit()
    
    def start_session(self, user_id: str = None, session_metadata: Dict[str, Any] = None) -> str:
        """새 대화 세션 시작"""
        session_id = str(uuid.uuid4())
        
        with sqlite3.connect(self.db_path) as conn:
            cursor = conn.cursor()
            cursor.execute("""
                INSERT INTO conversation_sessions 
                (session_id, user_id, start_time, session_metadata)
                VALUES (?, ?, ?, ?)
            """, (
                session_id,
                user_id,
                datetime.now().isoformat(),
                json.dumps(session_metadata or {})
            ))
            conn.commit()
        
        logger.info(f"New conversation session started: {session_id}")
        return session_id
    
    def end_session(self, session_id: str, end_metadata: Dict[str, Any] = None):
        """대화 세션 종료"""
        with sqlite3.connect(self.db_path) as conn:
            cursor = conn.cursor()
            
            # 총 질문 수 계산
            cursor.execute("SELECT COUNT(*) FROM conversation_logs WHERE session_id = ?", (session_id,))
            total_questions = cursor.fetchone()[0]
            
            # 세션 종료
            cursor.execute("""
                UPDATE conversation_sessions 
                SET end_time = ?, total_questions = ?, session_metadata = ?
                WHERE session_id = ?
            """, (
                datetime.now().isoformat(),
                total_questions,
                json.dumps(end_metadata or {}),
                session_id
            ))
            conn.commit()
        
        logger.info(f"Conversation session ended: {session_id}, total questions: {total_questions}")
    
    def log_conversation(
        self,
        session_id: str,
        question: str,
        answer: str,
        system_type: str,
        model_name: str,
        temperature: float,
        max_tokens: int,
        search_method: str,
        retrieved_chunks: List[Dict[str, Any]],
        generation_metadata: Dict[str, Any],
        quality_evaluation: Dict[str, Any] = None,
        conversation_history: List[Dict[str, str]] = None,
        search_steps: List[Dict[str, Any]] = None,
        error_log: str = None
    ) -> str:
        """질문-답변 루트 로깅"""
        log_id = str(uuid.uuid4())
        question_time = datetime.now().isoformat()
        
        # 검색 시간 계산
        search_time_ms = 0
        if search_steps:
            search_time_ms = sum(step.get('execution_time_ms', 0) for step in search_steps)
        
        # 품질 평가 정보 추출
        quality_scores = None
        overall_quality_score = None
        quality_evaluation_id = None
        
        if quality_evaluation:
            quality_scores = json.dumps(quality_evaluation.get('scores', {}))
            overall_quality_score = quality_evaluation.get('overall_score')
            quality_evaluation_id = quality_evaluation.get('evaluation_id')
        
        with sqlite3.connect(self.db_path) as conn:
            cursor = conn.cursor()
            
            # 메인 대화 로그 저장
            cursor.execute("""
                INSERT INTO conversation_logs (
                    log_id, session_id, question, answer, question_timestamp,
                    system_type, model_name, temperature, max_tokens,
                    search_method, retrieved_chunks_count, search_time_ms,
                    generation_time_ms, prompt_tokens, completion_tokens, total_tokens,
                    quality_scores, overall_quality_score, quality_evaluation_id,
                    conversation_history, retrieved_chunks, generation_metadata, error_log
                ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
            """, (
                log_id, session_id, question, answer, question_time,
                system_type, model_name, temperature, max_tokens,
                search_method, len(retrieved_chunks), search_time_ms,
                generation_metadata.get('generation_time', 0) * 1000,  # 초를 밀리초로 변환
                generation_metadata.get('prompt_tokens', 0),
                generation_metadata.get('completion_tokens', 0),
                generation_metadata.get('total_tokens', 0),
                quality_scores, overall_quality_score, quality_evaluation_id,
                json.dumps(conversation_history or []),
                json.dumps(retrieved_chunks),
                json.dumps(generation_metadata),
                error_log
            ))
            
            # 검색 단계별 상세 로그 저장
            if search_steps:
                for i, step in enumerate(search_steps):
                    step_id = str(uuid.uuid4())
                    cursor.execute("""
                        INSERT INTO search_step_logs (
                            step_id, log_id, step_type, step_order,
                            input_data, output_data, execution_time_ms, metadata
                        ) VALUES (?, ?, ?, ?, ?, ?, ?, ?)
                    """, (
                        step_id, log_id, step.get('type', 'unknown'), i,
                        json.dumps(step.get('input', {})),
                        json.dumps(step.get('output', {})),
                        step.get('execution_time_ms', 0),
                        json.dumps(step.get('metadata', {}))
                    ))
            
            conn.commit()
        
        logger.info(f"Conversation logged: {log_id}, session: {session_id}")
        return log_id
    
    def get_session_logs(self, session_id: str) -> List[Dict[str, Any]]:
        """세션별 대화 로그 조회"""
        with sqlite3.connect(self.db_path) as conn:
            conn.row_factory = sqlite3.Row
            cursor = conn.cursor()
            
            cursor.execute("""
                SELECT * FROM conversation_logs 
                WHERE session_id = ? 
                ORDER BY question_timestamp
            """, (session_id,))
            
            logs = []
            for row in cursor.fetchall():
                log_dict = dict(row)
                # JSON 필드 파싱
                log_dict['conversation_history'] = json.loads(log_dict['conversation_history'] or '[]')
                log_dict['retrieved_chunks'] = json.loads(log_dict['retrieved_chunks'] or '[]')
                log_dict['generation_metadata'] = json.loads(log_dict['generation_metadata'] or '{}')
                log_dict['quality_scores'] = json.loads(log_dict['quality_scores'] or '{}')
                logs.append(log_dict)
            
            return logs
    
    def get_conversation_analytics(self, days: int = 7) -> Dict[str, Any]:
        """대화 분석 통계"""
        with sqlite3.connect(self.db_path) as conn:
            cursor = conn.cursor()
            
            # 기본 통계
            cursor.execute("""
                SELECT 
                    COUNT(*) as total_conversations,
                    COUNT(DISTINCT session_id) as total_sessions,
                    AVG(overall_quality_score) as avg_quality_score,
                    AVG(generation_time_ms) as avg_generation_time,
                    AVG(search_time_ms) as avg_search_time,
                    SUM(total_tokens) as total_tokens_used
                FROM conversation_logs 
                WHERE question_timestamp >= datetime('now', '-{} days')
            """.format(days))
            
            basic_stats = dict(zip([desc[0] for desc in cursor.description], cursor.fetchone()))
            
            # 시스템별 통계
            cursor.execute("""
                SELECT 
                    system_type,
                    COUNT(*) as count,
                    AVG(overall_quality_score) as avg_quality,
                    AVG(generation_time_ms) as avg_generation_time
                FROM conversation_logs 
                WHERE question_timestamp >= datetime('now', '-{} days')
                GROUP BY system_type
            """.format(days))
            
            system_stats = {}
            for row in cursor.fetchall():
                system_stats[row[0]] = {
                    'count': row[1],
                    'avg_quality': row[2],
                    'avg_generation_time': row[3]
                }
            
            # 시간대별 통계
            cursor.execute("""
                SELECT 
                    strftime('%H', question_timestamp) as hour,
                    COUNT(*) as count,
                    AVG(overall_quality_score) as avg_quality
                FROM conversation_logs 
                WHERE question_timestamp >= datetime('now', '-{} days')
                GROUP BY strftime('%H', question_timestamp)
                ORDER BY hour
            """.format(days))
            
            hourly_stats = {}
            for row in cursor.fetchall():
                hourly_stats[row[0]] = {
                    'count': row[1],
                    'avg_quality': row[2]
                }
            
            return {
                'basic_stats': basic_stats,
                'system_stats': system_stats,
                'hourly_stats': hourly_stats,
                'analysis_period_days': days
            }
    
    def search_conversations(
        self,
        query: str = None,
        system_type: str = None,
        min_quality_score: float = None,
        date_from: str = None,
        date_to: str = None,
        limit: int = 100
    ) -> List[Dict[str, Any]]:
        """대화 로그 검색"""
        with sqlite3.connect(self.db_path) as conn:
            conn.row_factory = sqlite3.Row
            cursor = conn.cursor()
            
            # 동적 쿼리 구성
            where_conditions = []
            params = []
            
            if query:
                where_conditions.append("(question LIKE ? OR answer LIKE ?)")
                params.extend([f"%{query}%", f"%{query}%"])
            
            if system_type:
                where_conditions.append("system_type = ?")
                params.append(system_type)
            
            if min_quality_score is not None:
                where_conditions.append("overall_quality_score >= ?")
                params.append(min_quality_score)
            
            if date_from:
                where_conditions.append("question_timestamp >= ?")
                params.append(date_from)
            
            if date_to:
                where_conditions.append("question_timestamp <= ?")
                params.append(date_to)
            
            where_clause = " AND ".join(where_conditions) if where_conditions else "1=1"
            
            cursor.execute(f"""
                SELECT * FROM conversation_logs 
                WHERE {where_clause}
                ORDER BY question_timestamp DESC
                LIMIT ?
            """, params + [limit])
            
            logs = []
            for row in cursor.fetchall():
                log_dict = dict(row)
                # JSON 필드 파싱
                log_dict['conversation_history'] = json.loads(log_dict['conversation_history'] or '[]')
                log_dict['retrieved_chunks'] = json.loads(log_dict['retrieved_chunks'] or '[]')
                log_dict['generation_metadata'] = json.loads(log_dict['generation_metadata'] or '{}')
                log_dict['quality_scores'] = json.loads(log_dict['quality_scores'] or '{}')
                logs.append(log_dict)
            
            return logs
    
    def get_search_step_details(self, log_id: str) -> List[Dict[str, Any]]:
        """특정 대화의 검색 단계별 상세 정보"""
        with sqlite3.connect(self.db_path) as conn:
            conn.row_factory = sqlite3.Row
            cursor = conn.cursor()
            
            cursor.execute("""
                SELECT * FROM search_step_logs 
                WHERE log_id = ? 
                ORDER BY step_order
            """, (log_id,))
            
            steps = []
            for row in cursor.fetchall():
                step_dict = dict(row)
                # JSON 필드 파싱
                step_dict['input_data'] = json.loads(step_dict['input_data'] or '{}')
                step_dict['output_data'] = json.loads(step_dict['output_data'] or '{}')
                step_dict['metadata'] = json.loads(step_dict['metadata'] or '{}')
                steps.append(step_dict)
            
            return steps


def get_conversation_tracker() -> ConversationTracker:
    """대화 추적기 싱글톤 인스턴스 반환"""
    if not hasattr(get_conversation_tracker, '_instance'):
        get_conversation_tracker._instance = ConversationTracker()
    return get_conversation_tracker._instance
