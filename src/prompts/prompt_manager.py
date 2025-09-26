"""
프롬프트 버전 관리 시스템
프롬프트의 버전별 관리, 로딩, 전환 기능을 제공합니다.
"""

import os
import yaml
import logging
from pathlib import Path
from typing import Dict, List, Optional, Any
from datetime import datetime
import shutil

logger = logging.getLogger(__name__)

class PromptManager:
    """프롬프트 버전 관리 클래스"""
    
    def __init__(self, config_path: str = "prompts/prompt_config.yaml"):
        self.config_path = Path(config_path)
        self.versions_dir = Path("prompts/versions")
        self.config = self._load_config()
        self.current_version = self.config.get('current_version', 'v1')
        
        # 디렉토리 생성
        self.versions_dir.mkdir(parents=True, exist_ok=True)
        
        logger.info(f"PromptManager initialized with current version: {self.current_version}")
    
    def _load_config(self) -> Dict[str, Any]:
        """프롬프트 설정 파일 로드"""
        if not self.config_path.exists():
            logger.warning(f"Prompt config file not found: {self.config_path}")
            return self._create_default_config()
        
        try:
            with open(self.config_path, 'r', encoding='utf-8') as f:
                return yaml.safe_load(f)
        except Exception as e:
            logger.error(f"Failed to load prompt config: {e}")
            return self._create_default_config()
    
    def _create_default_config(self) -> Dict[str, Any]:
        """기본 설정 생성"""
        return {
            'current_version': 'v1',
            'versions': {
                'v1': {
                    'name': '기본 RFP 분석가',
                    'description': '기본적인 RFP 문서 분석 기능',
                    'system_prompt_file': 'v1_system_prompt.txt',
                    'user_template_file': 'v1_user_template.txt',
                    'created_date': datetime.now().strftime('%Y-%m-%d'),
                    'author': 'system',
                    'tags': ['basic', 'rfp', 'analysis']
                }
            }
        }
    
    def get_available_versions(self) -> List[str]:
        """사용 가능한 프롬프트 버전 목록 반환"""
        return list(self.config.get('versions', {}).keys())
    
    def get_version_info(self, version: str) -> Optional[Dict[str, Any]]:
        """특정 버전의 정보 반환"""
        return self.config.get('versions', {}).get(version)
    
    def get_current_version(self) -> str:
        """현재 사용 중인 버전 반환"""
        return self.current_version
    
    def create_optimized_version(
        self, 
        optimized_prompt: str, 
        prompt_type: str, 
        base_version: str = None
    ) -> str:
        """최적화된 프롬프트로 새 버전 생성"""
        if base_version is None:
            base_version = self.current_version
        
        # 새 버전 번호 생성
        timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
        new_version = f"{base_version}_optimized_{timestamp}"
        
        # 새 버전 디렉토리 생성
        new_version_dir = self.versions_dir / new_version
        new_version_dir.mkdir(exist_ok=True)
        
        # 프롬프트 타입별 파일명 매핑
        prompt_files = {
            'question_generation': 'question_generation_prompt.txt',
            'evaluation': 'advanced_evaluation_prompt.txt',
            'system': 'system_prompt.txt',
            'user_template': 'user_template.txt'
        }
        
        if prompt_type not in prompt_files:
            raise ValueError(f"Unknown prompt type: {prompt_type}")
        
        # 최적화된 프롬프트 저장
        prompt_file = new_version_dir / prompt_files[prompt_type]
        with open(prompt_file, 'w', encoding='utf-8') as f:
            f.write(optimized_prompt)
        
        # 다른 프롬프트 파일들 복사
        base_version_dir = self.versions_dir / base_version
        for file_type, file_name in prompt_files.items():
            if file_type != prompt_type:
                src_file = base_version_dir / file_name
                dst_file = new_version_dir / file_name
                if src_file.exists():
                    shutil.copy2(src_file, dst_file)
        
        # 설정 파일에 새 버전 추가
        self._add_version_to_config(new_version, base_version, prompt_type)
        
        logger.info(f"Created optimized version {new_version} for {prompt_type}")
        return new_version
    
    def _add_version_to_config(self, new_version: str, base_version: str, prompt_type: str):
        """설정 파일에 새 버전 추가"""
        base_config = self.config['versions'].get(base_version, {})
        
        new_version_config = {
            'name': f"{base_config.get('name', 'Unknown')} (Optimized)",
            'description': f"Optimized version of {base_version} for {prompt_type}",
            'created_at': datetime.now().isoformat(),
            'optimized_from': base_version,
            'optimized_type': prompt_type,
            'question_generation_prompt_file': 'question_generation_prompt.txt',
            'advanced_evaluation_prompt_file': 'advanced_evaluation_prompt.txt',
            'system_prompt_file': 'system_prompt.txt',
            'user_template_file': 'user_template.txt',
            'tags': base_config.get('tags', []) + ['optimized', prompt_type]
        }
        
        self.config['versions'][new_version] = new_version_config
        self._save_config()
    
    def get_optimization_history(self) -> List[Dict[str, Any]]:
        """최적화 히스토리 조회"""
        optimization_versions = []
        
        for version, config in self.config['versions'].items():
            if 'optimized_from' in config:
                optimization_versions.append({
                    'version': version,
                    'name': config.get('name', ''),
                    'description': config.get('description', ''),
                    'optimized_from': config.get('optimized_from', ''),
                    'optimized_type': config.get('optimized_type', ''),
                    'created_at': config.get('created_at', ''),
                    'tags': config.get('tags', [])
                })
        
        return sorted(optimization_versions, key=lambda x: x['created_at'], reverse=True)
    
    def apply_optimized_version(self, version: str) -> bool:
        """최적화된 버전을 현재 버전으로 적용"""
        if version not in self.config['versions']:
            logger.error(f"Version {version} not found")
            return False
        
        old_version = self.current_version
        self.current_version = version
        self.config['current_version'] = version
        self._save_config()
        
        logger.info(f"Applied optimized version: {old_version} -> {version}")
        return True
    
    def set_current_version(self, version: str) -> bool:
        """현재 버전 변경"""
        if version not in self.get_available_versions():
            logger.error(f"Version {version} not found")
            return False
        
        self.current_version = version
        self.config['current_version'] = version
        
        # 설정 파일 저장
        self._save_config()
        logger.info(f"Current version changed to: {version}")
        return True
    
    def get_system_prompt(self, version: str = None) -> str:
        """시스템 프롬프트 반환"""
        version = version or self.current_version
        version_info = self.get_version_info(version)
        
        if not version_info:
            logger.error(f"Version {version} not found")
            return self._get_default_system_prompt()
        
        prompt_file = self.versions_dir / version_info['system_prompt_file']
        
        if not prompt_file.exists():
            logger.error(f"System prompt file not found: {prompt_file}")
            return self._get_default_system_prompt()
        
        try:
            with open(prompt_file, 'r', encoding='utf-8') as f:
                return f.read().strip()
        except Exception as e:
            logger.error(f"Failed to read system prompt: {e}")
            return self._get_default_system_prompt()
    
    def get_user_template(self, version: str = None) -> str:
        """사용자 템플릿 반환"""
        version = version or self.current_version
        version_info = self.get_version_info(version)
        
        if not version_info:
            logger.error(f"Version {version} not found")
            return self._get_default_user_template()
        
        template_file = self.versions_dir / version_info['user_template_file']
        
        if not template_file.exists():
            logger.error(f"User template file not found: {template_file}")
            return self._get_default_user_template()
        
        try:
            with open(template_file, 'r', encoding='utf-8') as f:
                return f.read().strip()
        except Exception as e:
            logger.error(f"Failed to read user template: {e}")
            return self._get_default_user_template()
    
    def get_user_template_by_type(self, question_type: str = "general", version: str = None) -> str:
        """질문 유형별 사용자 템플릿 반환"""
        version = version or self.current_version
        version_info = self.get_version_info(version)
        
        if not version_info:
            logger.error(f"Version {version} not found")
            return self._get_default_user_template()
        
        # 질문 유형별 템플릿 파일 경로 생성
        type_template_file = self.versions_dir / f"user_template_{question_type}.txt"
        
        # 유형별 템플릿이 있으면 사용
        if type_template_file.exists():
            try:
                with open(type_template_file, 'r', encoding='utf-8') as f:
                    return f.read().strip()
            except Exception as e:
                logger.error(f"Failed to read type-specific template: {e}")
        
        # 기본 템플릿 반환
        return self.get_user_template(version)
    
    def get_question_generation_prompt(self, version: str = None) -> str:
        """질문 생성 프롬프트 반환"""
        version = version or self.current_version
        version_info = self.get_version_info(version)
        
        if not version_info:
            logger.error(f"Version {version} not found")
            return self._get_default_question_generation_prompt()
        
        question_file = self.versions_dir / version_info.get('question_generation_prompt_file', f"{version}_question_generation_prompt.txt")
        
        if not question_file.exists():
            logger.warning(f"Question generation prompt file not found: {question_file}, using default")
            return self._get_default_question_generation_prompt()
        
        try:
            with open(question_file, 'r', encoding='utf-8') as f:
                return f.read().strip()
        except Exception as e:
            logger.error(f"Failed to read question generation prompt: {e}")
            return self._get_default_question_generation_prompt()
    
    def get_evaluation_prompt(self, version: str = None) -> str:
        """평가 프롬프트 반환"""
        version = version or self.current_version
        version_info = self.get_version_info(version)
        
        if not version_info:
            logger.error(f"Version {version} not found")
            return self._get_default_evaluation_prompt()
        
        evaluation_file = self.versions_dir / version_info.get('advanced_evaluation_prompt_file', f"{version}_advanced_evaluation_prompt.txt")
        
        if not evaluation_file.exists():
            logger.warning(f"Evaluation prompt file not found: {evaluation_file}, using default")
            return self._get_default_evaluation_prompt()
        
        try:
            with open(evaluation_file, 'r', encoding='utf-8') as f:
                return f.read().strip()
        except Exception as e:
            logger.error(f"Failed to read evaluation prompt: {e}")
            return self._get_default_evaluation_prompt()
    
    def format_user_message(self, question: str, context: str, version: str = None) -> str:
        """사용자 메시지 포맷팅"""
        template = self.get_user_template(version)
        return template.format(question=question, context=context)
    
    def format_evaluation_prompt(self, question: str, answer: str, context: str, version: str = None) -> str:
        """평가 프롬프트 포맷팅"""
        template = self.get_evaluation_prompt(version)
        return template.format(question=question, answer=answer, context=context)
    
    def load_prompt_file(self, filename: str) -> str:
        """프롬프트 파일 로드"""
        file_path = self.versions_dir / filename
        
        if not file_path.exists():
            logger.warning(f"Prompt file not found: {file_path}")
            return ""
        
        try:
            with open(file_path, 'r', encoding='utf-8') as f:
                return f.read()
        except Exception as e:
            logger.error(f"Failed to load prompt file {filename}: {e}")
            return ""
    
    def load_question_generation_prompt(self, version: str = None) -> str:
        """질문 생성 프롬프트 로드"""
        version = version or self.current_version
        
        if version not in self.config.get('versions', {}):
            logger.warning(f"Version {version} not found")
            return ""
        
        version_config = self.config['versions'][version]
        filename = version_config.get('question_generation_prompt_file')
        
        if not filename:
            logger.warning(f"No question generation prompt file for version {version}")
            return ""
        
        return self.load_prompt_file(filename)
    
    def create_new_version(self, version: str, name: str, description: str, 
                          system_prompt: str, user_template: str, evaluation_prompt: str = None,
                          author: str = "user", tags: List[str] = None) -> bool:
        """새로운 프롬프트 버전 생성"""
        if version in self.get_available_versions():
            logger.error(f"Version {version} already exists")
            return False
        
        try:
            # 파일 생성
            system_file = self.versions_dir / f"{version}_system_prompt.txt"
            template_file = self.versions_dir / f"{version}_user_template.txt"
            evaluation_file = self.versions_dir / f"{version}_evaluation_prompt.txt"
            
            with open(system_file, 'w', encoding='utf-8') as f:
                f.write(system_prompt)
            
            with open(template_file, 'w', encoding='utf-8') as f:
                f.write(user_template)
            
            # 평가 프롬프트 파일 생성 (제공된 경우)
            if evaluation_prompt:
                with open(evaluation_file, 'w', encoding='utf-8') as f:
                    f.write(evaluation_prompt)
            
            # 설정에 추가
            version_config = {
                'name': name,
                'description': description,
                'system_prompt_file': f"{version}_system_prompt.txt",
                'user_template_file': f"{version}_user_template.txt",
                'created_date': datetime.now().strftime('%Y-%m-%d'),
                'author': author,
                'tags': tags or []
            }
            
            if evaluation_prompt:
                version_config['evaluation_prompt_file'] = f"{version}_evaluation_prompt.txt"
            
            self.config['versions'][version] = version_config
            
            # 설정 파일 저장
            self._save_config()
            logger.info(f"New version {version} created successfully")
            return True
            
        except Exception as e:
            logger.error(f"Failed to create new version: {e}")
            return False
    
    def delete_version(self, version: str) -> bool:
        """프롬프트 버전 삭제"""
        if version not in self.get_available_versions():
            logger.error(f"Version {version} not found")
            return False
        
        if version == self.current_version:
            logger.error(f"Cannot delete current version {version}")
            return False
        
        try:
            version_info = self.get_version_info(version)
            
            # 파일 삭제
            system_file = self.versions_dir / version_info['system_prompt_file']
            template_file = self.versions_dir / version_info['user_template_file']
            
            if system_file.exists():
                system_file.unlink()
            if template_file.exists():
                template_file.unlink()
            
            # 설정에서 제거
            del self.config['versions'][version]
            self._save_config()
            
            logger.info(f"Version {version} deleted successfully")
            return True
            
        except Exception as e:
            logger.error(f"Failed to delete version: {e}")
            return False
    
    def backup_version(self, version: str, backup_name: str = None) -> bool:
        """프롬프트 버전 백업"""
        if version not in self.get_available_versions():
            logger.error(f"Version {version} not found")
            return False
        
        if not backup_name:
            backup_name = f"{version}_backup_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
        
        try:
            version_info = self.get_version_info(version)
            
            # 백업 디렉토리 생성
            backup_dir = Path("prompts/backups") / backup_name
            backup_dir.mkdir(parents=True, exist_ok=True)
            
            # 파일 복사
            system_file = self.versions_dir / version_info['system_prompt_file']
            template_file = self.versions_dir / version_info['user_template_file']
            
            if system_file.exists():
                shutil.copy2(system_file, backup_dir / "system_prompt.txt")
            if template_file.exists():
                shutil.copy2(template_file, backup_dir / "user_template.txt")
            
            # 메타데이터 저장
            metadata = {
                'original_version': version,
                'backup_name': backup_name,
                'backup_date': datetime.now().isoformat(),
                'version_info': version_info
            }
            
            with open(backup_dir / "metadata.yaml", 'w', encoding='utf-8') as f:
                yaml.dump(metadata, f, default_flow_style=False, allow_unicode=True)
            
            logger.info(f"Version {version} backed up as {backup_name}")
            return True
            
        except Exception as e:
            logger.error(f"Failed to backup version: {e}")
            return False
    
    def _save_config(self):
        """설정 파일 저장"""
        try:
            with open(self.config_path, 'w', encoding='utf-8') as f:
                yaml.dump(self.config, f, default_flow_style=False, allow_unicode=True)
        except Exception as e:
            logger.error(f"Failed to save config: {e}")
    
    def _get_default_system_prompt(self) -> str:
        """기본 시스템 프롬프트"""
        return """당신은 RFP(제안요청서) 분석 전문가입니다. 
정부기관과 기업의 입찰 공고 문서를 분석하여 컨설턴트들이 필요한 정보를 빠르게 파악할 수 있도록 도와주는 역할을 합니다.

다음 원칙을 지켜주세요:
1. 제공된 문서 정보만을 바탕으로 정확하게 답변하세요.
2. 문서에 없는 내용에 대해서는 "문서에서 해당 정보를 찾을 수 없습니다"라고 간단하고 명확히 말하세요.
3. 문서에 정보가 있으면 사업명, 발주기관, 사업금액, 기간 등 핵심 정보를 포함하여 답변하세요.
4. 표나 목록이 있는 경우 구조화하여 보기 쉽게 정리하세요.
5. 입찰 참가 자격, 평가 기준, 제출 서류 등 중요한 요구사항은 놓치지 말고 포함하세요.
6. 답변은 한국어로 작성하고, 간결하고 명확하게 설명하세요.
7. 문서에 없는 내용에 대해서는 추측하거나 관련 없는 정보를 제공하지 마세요."""
    
    def _get_default_user_template(self) -> str:
        """기본 사용자 템플릿"""
        return """질문: {question}

관련 RFP 문서 정보:
{context}

위 정보를 바탕으로 질문에 답변해 주세요.
문서에 정보가 있으면 사업명, 발주기관, 사업금액, 기간 등 핵심 정보를 포함하여 상세하게 답변하세요.
문서에 없는 내용에 대해서는 "문서에서 해당 정보를 찾을 수 없습니다"라고 명확히 말씀해 주세요."""
    
    def _get_default_question_generation_prompt(self) -> str:
        """기본 질문 생성 프롬프트"""
        return """당신은 RFP 문서 분석 전문가입니다. 
주어진 문서 청크를 바탕으로 다양한 유형과 난이도의 질문을 생성해주세요.

문서 청크:
{chunk}

생성할 질문 유형:
- 기본 정보 확인 질문
- 심화 분석 질문  
- 실무 적용 질문
- 창의적 사고 질문

각 유형별로 1-2개씩 질문을 생성해주세요."""

    def _get_default_evaluation_prompt(self) -> str:
        """기본 평가 프롬프트"""
        return """다음 질문과 답변을 평가해주세요. 각 항목을 0-1 점수로 평가하고, 개선 제안을 해주세요.

질문: {question}

답변: {answer}

컨텍스트: {context}

다음 기준으로 평가해주세요:

1. **정확성 (Accuracy)**: 답변이 질문에 정확히 답하고 있는가?
2. **완성도 (Completeness)**: 필요한 정보가 모두 포함되어 있는가?
3. **관련성 (Relevance)**: 제공된 컨텍스트와 일치하는가?
4. **명확성 (Clarity)**: 답변이 이해하기 쉽고 명확한가?
5. **구조화 (Structure)**: 정보가 체계적으로 정리되어 있는가?

각 항목에 대해 0-1 점수를 주고, 전체적인 개선 제안을 해주세요.
답변은 JSON 형식으로 해주세요:
{
  "scores": {
    "accuracy": 0.8,
    "completeness": 0.7,
    "relevance": 0.9,
    "clarity": 0.8,
    "structure": 0.6
  },
  "overall_score": 0.76,
  "improvement_suggestions": ["구체적인 개선 제안들..."]
}"""


# 싱글톤 인스턴스
_prompt_manager = None

def get_prompt_manager() -> PromptManager:
    """프롬프트 매니저 싱글톤 인스턴스 반환"""
    global _prompt_manager
    if _prompt_manager is None:
        _prompt_manager = PromptManager()
    return _prompt_manager
