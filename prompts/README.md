# 📝 프롬프트 버전 관리 시스템

이 디렉토리는 RFP RAG 시스템의 프롬프트 버전을 관리하는 시스템입니다.

## 📁 디렉토리 구조

```
prompts/
├── README.md                    # 이 파일
├── prompt_config.yaml          # 프롬프트 버전 설정 파일
├── versions/                   # 프롬프트 버전별 파일들
│   ├── v1_system_prompt.txt    # v1 시스템 프롬프트
│   ├── v1_user_template.txt    # v1 사용자 템플릿
│   ├── v2_system_prompt.txt    # v2 시스템 프롬프트
│   └── v2_user_template.txt    # v2 사용자 템플릿
└── backups/                    # 백업 파일들 (자동 생성)
    └── [backup_name]/
        ├── system_prompt.txt
        ├── user_template.txt
        └── metadata.yaml
```

## 🚀 사용법

### 1. 프롬프트 버전 확인
```python
from src.prompts.prompt_manager import get_prompt_manager

prompt_manager = get_prompt_manager()
print(f"현재 버전: {prompt_manager.get_current_version()}")
print(f"사용 가능한 버전: {prompt_manager.get_available_versions()}")
```

### 2. 프롬프트 버전 변경
```python
# v2 버전으로 변경
prompt_manager.set_current_version("v2")

# 시스템 프롬프트 가져오기
system_prompt = prompt_manager.get_system_prompt()

# 사용자 메시지 포맷팅
user_message = prompt_manager.format_user_message(question, context)
```

### 3. 새 프롬프트 버전 생성
```python
# 새 버전 생성
prompt_manager.create_new_version(
    version="v3",
    name="도메인 전문가",
    description="특정 도메인에 특화된 프롬프트",
    system_prompt="새로운 시스템 프롬프트...",
    user_template="새로운 사용자 템플릿...",
    author="user",
    tags=["domain", "expert"]
)
```

### 4. 프롬프트 백업
```python
# 현재 버전 백업
prompt_manager.backup_version("v2", "v2_backup_20240924")
```

## 📋 현재 버전들

### v1 - 기본 RFP 분석가
- **설명**: 기본적인 RFP 문서 분석 기능을 제공하는 프롬프트
- **특징**: 간단하고 명확한 답변 제공
- **태그**: basic, rfp, analysis

### v2 - 전략적 RFP 컨설턴트
- **설명**: 컨설턴트 관점에서 전략적 인사이트를 제공하는 고도화된 프롬프트
- **특징**: 구조화된 정보 제공, 전략적 관점 강화
- **태그**: strategic, consultant, insights

## ⚙️ 설정 파일

`prompt_config.yaml`에서 다음을 설정할 수 있습니다:

- **current_version**: 현재 사용 중인 버전
- **versions**: 각 버전의 메타데이터
- **evaluation**: A/B 테스트 및 평가 설정
- **management**: 백업, 롤백 등 관리 설정

## 🔄 Streamlit UI에서 사용

Streamlit 앱의 사이드바에서 "📝 프롬프트 관리" 섹션을 통해:

1. **현재 버전 확인**: 현재 사용 중인 프롬프트 버전 표시
2. **버전 선택**: 드롭다운에서 다른 버전 선택
3. **버전 변경**: 선택한 버전으로 즉시 전환
4. **버전 정보**: 각 버전의 상세 정보 확인
5. **프롬프트 미리보기**: 현재 프롬프트 내용 확인

## 🧪 A/B 테스트

프롬프트 버전 간 성능 비교를 위한 A/B 테스트 기능:

- 자동 메트릭 수집
- 사용자 만족도 추적
- 정확도 및 완성도 평가
- 비교 기간 설정

## 📊 모니터링

각 프롬프트 버전의 성능을 모니터링할 수 있습니다:

- 응답 품질 점수
- 사용자 만족도
- 정확도
- 완성도
- 응답 시간

## 🛠️ 고급 기능

### 자동 백업
- 설정된 간격으로 자동 백업
- 버전 변경 전 자동 백업
- 백업 파일 메타데이터 관리

### 롤백 기능
- 이전 버전으로 안전한 롤백
- 백업에서 복원
- 변경 이력 추적

### 버전 관리
- 최대 버전 수 제한
- 오래된 버전 자동 정리
- 버전별 태그 관리

## 🔧 문제 해결

### 프롬프트 매니저 초기화 실패
```python
# 수동으로 프롬프트 매니저 초기화
from src.prompts.prompt_manager import PromptManager
prompt_manager = PromptManager("prompts/prompt_config.yaml")
```

### 버전 변경 실패
- 설정 파일 권한 확인
- 파일 경로 확인
- YAML 문법 오류 확인

### 프롬프트 파일 없음
- `versions/` 디렉토리 확인
- 파일명 규칙 확인 (`{version}_system_prompt.txt`)
- 파일 인코딩 확인 (UTF-8)

## 📈 향후 계획

- [ ] 프롬프트 성능 자동 평가
- [ ] 사용자 피드백 기반 자동 개선
- [ ] 도메인별 특화 프롬프트
- [ ] 다국어 프롬프트 지원
- [ ] 프롬프트 템플릿 엔진
- [ ] 실시간 프롬프트 편집
