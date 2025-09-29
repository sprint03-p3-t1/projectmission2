# RFP RAG 시스템 - 모듈 구조

## 📁 모듈 구조

```
src/
├── data_models.py      # 공통 데이터 클래스 정의
├── data_loader.py      # JSON 데이터 로딩
├── retriever.py        # 문서 검색
├── generator.py        # 답변 생성
├── rfp_rag_main.py     # 메인 시스템 통합
├── streamlit_app.py    # 웹 인터페이스
└── utils/              # 유틸리티 함수들
```

## 🔧 모듈별 역할

### 1. `data_models.py`
- **공통 데이터 클래스 정의**
- `RFPDocument`: RFP 문서 데이터
- `DocumentChunk`: 문서 청크 데이터
- `RetrievalResult`: 검색 결과
- `RAGResponse`: RAG 응답
- `RAGSystemInterface`: 인터페이스 정의

### 2. `data_loader.py`
- **JSON 파일 로딩 및 문서 관리**
- JSON 파일을 RFPDocument 객체로 변환
- 메타데이터 기반 문서 검색
- 문서 통계 정보 제공

### 3. `retriever.py`
- **문서 청킹, 임베딩, 벡터 검색**
- `RFPChunker`: 문서를 청크로 분할
- `RFPEmbedder`: 임베딩 생성
- `RFPVectorStore`: 벡터 저장소
- `RFPRetriever`: 통합 검색기

### 4. `generator.py`
- **답변 생성 및 대화 관리**
- `RFPGenerator`: OpenAI 기반 답변 생성
- 대화 히스토리 관리
- 다양한 생성 옵션 (요약, 비교 등)

### 5. `rfp_rag_main.py`
- **전체 시스템 통합**
- 모든 모듈을 조합하여 완전한 RAG 시스템 구성
- 사용자 친화적 인터페이스 제공

### 6. `streamlit_app.py`
- **웹 사용자 인터페이스**
- 대시보드, 검색, 질의응답 기능

## 🚀 사용 방법

### 1. 전체 시스템 사용
```python
from rfp_rag_main import RFPRAGSystem

# 시스템 초기화
rag_system = RFPRAGSystem(
    json_dir="data/preprocess/json",
    openai_api_key="your-api-key"
)
rag_system.initialize()

# 질문하기
answer = rag_system.ask("한국사학진흥재단 사업 요구사항을 알려주세요.")
```

### 2. 리트리버만 독립 사용
```python
from data_loader import RFPDataLoader
from retriever import RFPRetriever

# 데이터 로드
data_loader = RFPDataLoader("data/preprocess/json")
data_loader.initialize()
documents = data_loader.get_documents()

# 리트리버 초기화
retriever = RFPRetriever()
retriever.initialize(documents)

# 검색
results = retriever.retrieve("시스템 구축 사업", k=5)
```

### 3. 제네레이터만 독립 사용
```python
from generator import RFPGenerator
from data_models import DocumentChunk, RetrievalResult

# 제네레이터 초기화
generator = RFPGenerator("your-api-key")
generator.initialize()

# 가짜 검색 결과로 테스트
test_chunk = DocumentChunk(...)
test_results = [RetrievalResult(chunk=test_chunk, score=0.95, rank=1)]

# 답변 생성
response = generator.generate_response("질문", test_results)
```

## 🤝 팀 협업 가이드

### 리트리버 작업 사항
1. **`retriever.py` 모듈 개선**
   - 청킹 전략 최적화
   - 임베딩 모델 실험
   - 검색 알고리즘 개선
   - 필터링 로직 강화

2. **독립 테스트**
   ```bash
   python src/retriever.py
   ```

### 제네레이션 담당자 작업 사항
1. **`generator.py` 모듈 개선**
   - 프롬프트 엔지니어링
   - 응답 품질 향상
   - 대화 맥락 관리
   - 다양한 생성 모드 추가

2. **독립 테스트**
   ```bash
   python src/generator.py
   ```

### 통합 테스트
```bash
python src/rfp_rag_main.py
```

## 📋 개발 체크리스트

### 리트리버 모듈
- [ ] 청킹 크기 최적화
- [ ] 임베딩 모델 성능 비교
- [ ] MMR(Maximum Marginal Relevance) 구현
- [ ] 하이브리드 검색 (키워드 + 의미 검색)
- [ ] 메타데이터 필터링 강화

### 제네레이터 모듈
- [ ] 시스템 프롬프트 최적화
- [ ] 응답 포맷팅 개선
- [ ] 대화 맥락 유지 로직
- [ ] 다양한 응답 모드 (요약, 비교, 분석)
- [ ] 토큰 사용량 최적화

### 통합 시스템
- [ ] 성능 벤치마킹
- [ ] 오류 처리 강화
- [ ] 로깅 시스템 개선
- [ ] 캐싱 메커니즘
- [ ] API 문서화

## 🔧 설정 파일

### `.env` 파일
```
OPENAI_API_KEY=your-key
EMBEDDING_MODEL=BAAI/bge-m3
MODEL_NAME=gpt-3.5-turbo
TEMPERATURE=0.1
MAX_TOKENS=2000
```

### `config/rag_config.yaml`
모듈별 상세 설정 관리

## 📊 성능 모니터링

각 모듈은 독립적으로 성능을 측정할 수 있습니다:
- **리트리버**: 검색 정확도, 응답 시간
- **제네레이터**: 토큰 사용량, 생성 시간, 품질 점수
- **전체 시스템**: End-to-End 성능
