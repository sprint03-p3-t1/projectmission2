"""
YAML 설정 파일 로더
RFP RAG 시스템의 모든 설정을 YAML 파일에서 관리
"""

import yaml
import os
from pathlib import Path
from typing import Dict, Any, Optional

class YAMLConfig:
    """YAML 설정 파일 관리 클래스"""
    
    def __init__(self, config_path: str = "config/rag_config.yaml"):
        """YAML 설정 초기화"""
        self.config_path = Path(config_path)
        self.config = self._load_config()
    
    def _load_config(self) -> Dict[str, Any]:
        """YAML 설정 파일 로드"""
        try:
            with open(self.config_path, 'r', encoding='utf-8') as file:
                config = yaml.safe_load(file)
                return config or {}
        except FileNotFoundError:
            print(f"⚠️ 설정 파일을 찾을 수 없습니다: {self.config_path}")
            return {}
        except yaml.YAMLError as e:
            print(f"⚠️ YAML 파일 파싱 오류: {e}")
            return {}
    
    def get(self, key: str, default: Any = None) -> Any:
        """설정 값 가져오기 (점 표기법 지원)"""
        keys = key.split('.')
        value = self.config
        
        for k in keys:
            if isinstance(value, dict) and k in value:
                value = value[k]
            else:
                return default
        
        return value
    
    def get_data_path(self, sub_path: str = "") -> str:
        """데이터 경로 가져오기"""
        base_path = self.get('data.ops_dir', 'data/ops')
        if sub_path:
            return os.path.join(base_path, sub_path)
        return base_path
    
    def get_conversation_logs_db_path(self) -> str:
        """대화 로그 데이터베이스 경로"""
        return self.get('mlops.conversation_logs_db', 'data/ops/conversation_logs.db')
    
    def get_quality_metrics_db_path(self) -> str:
        """품질 메트릭스 데이터베이스 경로"""
        return self.get('mlops.quality_metrics_db', 'data/ops/quality_metrics.db')
    
    def get_monitoring_interval(self) -> int:
        """모니터링 간격 (초)"""
        return self.get('mlops.monitoring_interval', 300)
    
    def get_alert_thresholds(self) -> Dict[str, float]:
        """알림 임계값 설정"""
        return self.get('mlops.alert_thresholds', {
            'overall_score_low': 0.6,
            'overall_score_critical': 0.4,
            'accuracy_score_low': 0.5,
            'relevance_score_low': 0.5,
            'completeness_score_low': 0.5,
            'clarity_score_low': 0.5,
            'structure_score_low': 0.5
        })
    
    def get_embedding_config(self) -> Dict[str, Any]:
        """임베딩 설정"""
        return self.get('embedding', {
            'model_name': 'BAAI/bge-m3',
            'batch_size': 32,
            'max_length': 512
        })
    
    def get_llm_config(self) -> Dict[str, Any]:
        """LLM 설정"""
        return self.get('llm', {
            'provider': 'openai',
            'model': 'gpt-4.1-mini',
            'temperature': 0.1,
            'max_tokens': 2000,
            'timeout': 30,
            'api_key_env': 'OPENAI_API_KEY'
        })
    
    def get_generation_config(self) -> Dict[str, Any]:
        """Generation 모듈 설정"""
        llm_config = self.get_llm_config()
        mlops_config = self.get('mlops', {})
        prompts_config = self.get('prompts', {})
        
        return {
            'model': llm_config.get('model', 'gpt-4.1-mini'),
            'temperature': llm_config.get('temperature', 0.1),
            'max_tokens': llm_config.get('max_tokens', 2000),
            'api_key_env': llm_config.get('api_key_env', 'OPENAI_API_KEY'),
            'enable_quality_evaluation': mlops_config.get('enable_quality_evaluation', True),
            'enable_conversation_logging': mlops_config.get('enable_conversation_logging', True),
            'conversation_history_limit': mlops_config.get('conversation_history_limit', 6),
            'prompt_manager_config': prompts_config.get('manager', {}),
            'legacy_prompts': prompts_config.get('legacy', {})
        }
    
    def reload(self):
        """설정 파일 다시 로드"""
        self.config = self._load_config()
        print(f"✅ 설정 파일 다시 로드됨: {self.config_path}")

# 전역 설정 인스턴스
yaml_config = YAMLConfig()
