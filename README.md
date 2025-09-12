# RFP RAG 시스템 - 입찰메이트

## 📋 프로젝트 개요

복잡한 기업 및 정부 제안요청서(RFP) 내용을 효과적으로 추출하고 요약하여 필요한 정보를 제공하는 RAG(Retrieval-Augmented Generation) 시스템을 구축합니다.

### 🎯 프로젝트 목표
- 하루 수백 건의 RFP 문서를 자동으로 분석하여 주요 정보를 추출
- 컨설턴트들이 수십 페이지의 문서를 일일이 읽지 않고도 핵심 요구사항을 파악할 수 있도록 지원
- 고객사에게 적합한 입찰 기회를 빠르게 찾아 추천하는 시스템 구축

### 🔧 기술 스택 (미정)

| 시나리오 | LLM 실행 | 임베딩 | Vector DB |
|---------|---------|--------|-----------|
| **A: GCP 실행 기반** | HuggingFace 모델 (LLaMA, Gemma 등) | HuggingFace 임베딩 | FAISS, Chroma |
| **B: 클라우드 API 기반** | OpenAI, Claude, Gemini | OpenAI Embedding, Cohere | FAISS, Chroma, Supabase |

## 📁 프로젝트 구조

```
ai03_prj2_rfp_rag(projectmission2)/
├── data/
│   ├── raw/           # 원본 RFP 문서 (PDF, HWP)
│   └── processed/     # 전처리된 데이터
├── src/
│   ├── data_processing/   # 문서 처리 모듈
│   ├── embedding/         # 임베딩 생성 모듈
│   ├── retrieval/         # 검색 모듈
│   ├── generation/        # 텍스트 생성 모듈
│   └── utils/            # 유틸리티 함수들
├── models/
│   ├── scenario_a/       # GCP 기반 모델 파일들
│   └── scenario_b/       # 클라우드 API 기반 설정
├── config/               # 설정 파일들
├── notebooks/            # 실험 및 분석용 Jupyter 노트북
├── tests/
│   ├── unit/            # 단위 테스트
│   └── integration/     # 통합 테스트
├── docs/                # 프로젝트 문서
├── scripts/             # 실행 스크립트
└── requirements.txt     # Python 의존성
```

## 🚀 설치 및 실행

### 환경 설정

1. **Python 환경 생성**
```bash
python -m venv venv
source venv/bin/activate  # Linux/Mac
# 또는
venv\Scripts\activate     # Windows
```

2. **의존성 설치**
```bash
pip install -r requirements.txt
```

### 데이터 준비

1. **RFP 문서 및 메타데이터 다운로드**
   - 구글 드라이브 > 프로젝트 > 중급 프로젝트 > 원본데이터에서 파일들을 다운로드
   - `data/raw/` 폴더에 문서 파일들을 배치
   - `data_list.csv` 파일을 `data/` 폴더에 배치

## 🔬 주요 기능

### 📄 문서 처리
- PDF 및 HWP 파일 파싱
- 텍스트 추출 및 전처리
- 메타데이터 처리 및 활용

### 🧩 문서 청킹
- 청크 크기 및 중첩 설정 최적화
- 의미 단위 기반 청킹
- 다양한 청킹 전략 실험

### 🔍 임베딩 및 검색
- 다중 임베딩 모델 비교 실험
- Vector DB 구축 및 최적화
- 메타데이터 기반 필터링
- 다양한 검색 기법 (MMR, Hybrid Search 등)

### 🤖 텍스트 생성
- 다중 LLM 모델 비교
- 프롬프트 엔지니어링 최적화
- 대화 맥락 유지
- 응답 포맷 및 톤 조정

## 📊 성능 평가

### 평가 기준
- **정확성**: 요청된 내용을 정확하게 추출하는지
- **종합성**: 여러 문서에 대한 요청을 잘 종합하는지
- **맥락 이해**: 후속 질문의 맥락을 잘 이해하는지
- **정합성**: 문서에 없는 내용에 대해서는 모른다고 답변하는지

### 평가 지표
- Precision@K
- Recall@K
- F1-Score
- 응답 시간
- 토큰 사용량

## 🎯 사용 예시

```python
from src.rfp_rag import RFPRAGSystem

# 시스템 초기화
rag_system = RFPRAGSystem(config_path='config/scenario_b.yaml')

# 질문 처리
question = "국민연금공단이 발주한 이러닝시스템 관련 사업 요구사항을 정리해 줘."
response = rag_system.ask(question)
print(response)

# 후속 질문
follow_up = "콘텐츠 개발 관리 요구 사항에 대해서 더 자세히 알려 줘."
response = rag_system.ask(follow_up)
print(response)
```

## 👥 팀 역할 분배

| 역할명 | 담당 | 담당 업무 |
|-------|-------|----------|
| **Project Manager** | 이영호 | 프로젝트 관리, 성능 평가, 협업 조율 |
| **데이터 처리 담당** | 김도영, 배진석 | 문서 파싱, 메타데이터 처리, 청킹 전략 |
| **Retrieval 담당** | 김지영 | 임베딩 생성, Vector DB 구축, 검색 최적화 |
| **Generation 담당** | 이영호 | LLM 모델 선정, 프롬프트 엔지니어링, 응답 최적화, Streamlit |

## 🔄 개발 워크플로우

1. **데이터 준비** → 문서 로딩 및 전처리
2. **문서 청킹** → 최적의 청크 크기 및 전략 결정
3. **임베딩 생성** → 모델 선택 및 벡터화
4. **Retrieval 구현** → 검색 알고리즘 개발 및 최적화
5. **Generation 구현** → LLM 통합 및 프롬프트 튜닝
6. **평가 및 개선** → 성능 측정 및 반복 개선

## 📈 실험 계획

### 비교 실험 항목
- **시나리오 A vs B**: 온프레미스 vs 클라우드 성능 비교
- **임베딩 모델**: 다양한 모델의 검색 성능 비교
- **청킹 전략**: 청크 크기, 중첩, 의미 기반 청킹 비교
- **검색 기법**: 유사도 검색, MMR, Hybrid Search 비교
- **LLM 모델**: 다양한 생성 모델의 응답 품질 비교

### 하이퍼파라미터 튜닝 (미정)
- 청크 크기: 256, 512, 1024 토큰
- 중첩 크기: 50, 100, 200 토큰
- Top-K: 3, 5, 10, 20
- Temperature: 0.1, 0.3, 0.7
- Max tokens: 512, 1024, 2048

## 🤝 협업 가이드

### 브랜치 전략
- `main`: 배포 가능한 안정 버전
- `develop`: 개발 중인 통합 브랜치
- `feature/*`: 기능 개발 브랜치
- `experiment/*`: 실험용 브랜치

### 커밋 컨벤션
```
feat: 새로운 기능 추가
fix: 버그 수정
docs: 문서 수정
style: 코드 포맷팅
refactor: 코드 리팩토링
test: 테스트 코드 추가
chore: 기타 작업
```

## 📋 체크리스트

### 필수 구현 사항


### 심화



---

**AI03기 Part3 1팀** | AI03 스프린트 중급 프로젝트
