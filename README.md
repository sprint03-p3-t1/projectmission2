# RFP RAG 시스템 - 입찰메이트

## 📋 프로젝트 개요

복잡한 기업 및 정부 제안요청서(RFP) 내용을 효과적으로 추출하고 요약하여 필요한 정보를 제공하는 통합 RAG(Retrieval-Augmented Generation) 시스템을 구축합니다.

### 🎯 프로젝트 목표
- 하루 수백 건의 RFP 문서를 자동으로 분석하여 주요 정보를 추출
- 컨설턴트들이 수십 페이지의 문서를 일일이 읽지 않고도 핵심 요구사항을 파악할 수 있도록 지원
- 고객사에게 적합한 입찰 기회를 빠르게 찾아 추천하는 시스템 구축
- **FAISS와 ChromaDB 두 가지 검색 시스템을 통합하여 비교 및 선택 가능**

### 🔧 기술 스택

| 구성 요소 | 기술 | 모델/라이브러리 |
|---------|------|----------------|
| **LLM** | OpenAI API | GPT-4.1-mini |
| **임베딩 (FAISS)** | HuggingFace | BAAI/bge-m3 |
| **임베딩 (ChromaDB)** | HuggingFace | nlpai-lab/KURE-v1 |
| **Vector DB** | FAISS + ChromaDB | 하이브리드 검색 지원 |
| **웹 인터페이스** | Streamlit | 통합 대시보드 |
| **검색 기법** | 하이브리드 | BM25 + Vector + Reranking |

## 📁 프로젝트 구조

```
projectmission2/
├── data/
│   ├── raw/                    # 원본 RFP 문서 (PDF, HWP)
│   ├── processed/              # 전처리된 데이터
│   │   ├── json/              # JSON 변환된 문서
│   │   └── csv/               # CSV 메타데이터
│   └── cache/                 # 통합 캐시 디렉토리
│       ├── faiss/             # FAISS 시스템 캐시
│       └── chromadb/          # ChromaDB 시스템 캐시
├── src/
│   ├── config/                # 통합 설정 모듈
│   │   └── unified_config.py  # 통합 설정 관리
│   ├── data_processing/       # 문서 처리 모듈
│   ├── embedding/             # 임베딩 생성 모듈
│   │   ├── embedder.py        # RFPEmbedder - SentenceTransformer 기반 임베딩
│   │   ├── vector_store.py    # RFPVectorStore - FAISS 기반 벡터 검색
│   │   └── cache_manager.py   # EmbeddingCacheManager - 임베딩 캐시 관리
│   ├── retrieval/             # 검색 모듈
│   │   ├── retriever.py       # RFPRetriever - FAISS 기반 검색기
│   │   ├── hybrid_retriever.py # Retriever - ChromaDB 하이브리드 검색
│   │   ├── chunker.py         # RFPChunker - 문서 청킹 처리
│   │   ├── tokenizer_wrapper.py # TokenizerWrapper - 한국어 토크나이저
│   │   └── README.md          # 검색 시스템 상세 문서
│   ├── generation/            # 텍스트 생성 모듈
│   │   ├── generator.py       # RFPGenerator - OpenAI GPT 기반 답변 생성
│   │   └── readme_generation.md # 생성 모듈 상세 문서
│   ├── systems/               # 시스템 선택기 모듈
│   │   └── system_selector.py # FAISS/ChromaDB 선택기
│   ├── rfp_rag_main.py        # 메인 RAG 시스템
│   └── unified_streamlit_app.py # 통합 Streamlit 앱
├── run_unified_system.py      # 통합 시스템 실행 스크립트
├── UNIFIED_SYSTEM_README.md   # 통합 시스템 문서
└── requirements.txt           # Python 의존성
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

3. **환경 변수 설정**
```bash
# .env 파일 생성
echo "OPENAI_API_KEY=your_openai_api_key_here" > .env
echo "MODEL_NAME=gpt-4.1-mini" >> .env
```

### 데이터 준비

1. **RFP 문서 및 메타데이터 다운로드**
   - 구글 드라이브 > 프로젝트 > 중급 프로젝트 > 원본데이터에서 파일들을 다운로드
   - `data/raw/` 폴더에 문서 파일들을 배치
   - `data_list.csv` 파일을 `data/` 폴더에 배치

### 통합 시스템 실행

1. **통합 Streamlit 앱 실행**
```bash
python run_unified_system.py
```

2. **웹 브라우저에서 접속**
   - URL: `http://localhost:8501`
   - FAISS 또는 ChromaDB 시스템 선택 가능
   - 두 시스템 비교 모드 지원

## 🔬 주요 기능

### 🎯 통합 RAG 시스템
- **FAISS 시스템**: GPU 가속 벡터 검색 (21,219개 청크)
- **ChromaDB 시스템**: 하이브리드 검색 (BM25 + Vector + Reranking, 7,569개 문서)
- **시스템 선택**: 웹 인터페이스에서 실시간 시스템 전환
- **비교 모드**: 두 시스템의 검색 결과 동시 비교

### 📄 문서 처리
- PDF 및 HWP 파일 파싱
- JSON 변환 및 메타데이터 추출
- 자동 청킹 및 전처리

### 🔍 고급 검색 기능
- **하이브리드 검색**: BM25 + 벡터 유사도 + 리랭킹
- **메타데이터 필터링**: 발주기관, 사업금액, 기간 등
- **GPU 가속**: 임베딩 생성 및 검색 최적화
- **캐시 시스템**: 빠른 재시작 및 성능 향상

### 🧩 모듈별 상세 기능

#### 📁 `src/embedding/` - 임베딩 모듈
- **`embedder.py`**: `RFPEmbedder` 클래스
  - SentenceTransformer 기반 임베딩 생성
  - GPU/CPU 자동 감지 및 최적화
  - BAAI/bge-m3, nlpai-lab/KURE-v1 모델 지원
- **`vector_store.py`**: `RFPVectorStore` 클래스
  - FAISS 기반 벡터 검색 인덱스
  - Inner Product (cosine similarity) 검색
  - 배치 검색 및 메타데이터 관리
- **`cache_manager.py`**: `EmbeddingCacheManager` 클래스
  - 임베딩 결과 디스크 캐싱
  - MD5 해시 기반 캐시 무효화
  - 청크, 임베딩, 메타데이터 통합 관리

#### 📁 `src/retrieval/` - 검색 모듈
- **`retriever.py`**: `RFPRetriever` 클래스 (FAISS 시스템)
  - 문서 청킹, 임베딩, 벡터 검색 통합
  - 캐시 기반 빠른 재시작
  - 21,219개 청크 관리
- **`hybrid_retriever.py`**: `Retriever` 클래스 (ChromaDB 시스템)
  - BM25 + 벡터 + 리랭킹 하이브리드 검색
  - 자연어 필터 자동 추출
  - 한국어 토크나이저 통합 (Kiwi, KoNLPy)
  - 7,569개 문서 관리
- **`chunker.py`**: `RFPChunker` 클래스
  - tiktoken 기반 토큰 단위 청킹
  - 의미 단위 보존 청킹
  - 중첩 설정으로 맥락 유지
- **`tokenizer_wrapper.py`**: `TokenizerWrapper` 클래스
  - Kiwi, KoNLPy 토크나이저 통합
  - 바이그램 자동 생성
  - 불용어 제거 및 정규화

#### 📁 `src/generation/` - 텍스트 생성 모듈
- **`generator.py`**: `RFPGenerator` 클래스
  - OpenAI GPT-4.1-mini 기반 답변 생성
  - 대화 히스토리 관리 (최근 6턴 유지)
  - Pydantic 에러 처리 및 폴백 메커니즘
  - 토큰 사용량 및 생성 시간 메타데이터 추적
  - RetrievalResult → Dictionary 변환 처리
  - 상세한 답변 생성 (사업명, 발주기관, 금액, 기간 포함)
- **`readme_generation.md`**: 생성 모듈 상세 사용법
  - 빠른 시작 가이드
  - 질문 예시 및 활용 시나리오
  - 고급 사용법 및 프로그래밍 방식 사용
  - 성능 최적화 팁 및 문제 해결

### 🤖 AI 생성
- **GPT-4.1-mini**: 고품질 답변 생성
- **맥락 유지**: 대화 히스토리 기반 연속 질의응답
- **상세 답변**: 사업명, 발주기관, 금액, 기간 등 핵심 정보 포함
- **에러 처리**: Pydantic 에러 시 폴백 메커니즘
- **메타데이터**: 토큰 사용량, 생성 시간, 모델 정보 추적

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

### 통합 시스템 사용

```python
from src.systems.system_selector import SystemSelector

# 시스템 선택기 초기화
selector = SystemSelector()

# FAISS 시스템으로 질문
faiss_response = selector.ask("국민연금공단이 발주한 이러닝시스템 관련 사업 요구사항을 정리해 줘.", "faiss")
print("FAISS 결과:", faiss_response)

# ChromaDB 시스템으로 질문
chroma_response = selector.ask("국민연금공단이 발주한 이러닝시스템 관련 사업 요구사항을 정리해 줘.", "chromadb")
print("ChromaDB 결과:", chroma_response)
```

### 개별 시스템 사용

```python
from src.rfp_rag_main import RFPRAGSystem

# FAISS 시스템 초기화
rag_system = RFPRAGSystem(
    json_dir="data/processed/json",
    openai_api_key="your_api_key",
    embedding_model="BAAI/bge-m3"
)
rag_system.initialize()

# 질문 처리
question = "국민연금공단이 발주한 이러닝시스템 관련 사업 요구사항을 정리해 줘."
response = rag_system.ask(question)
print(response.answer)
```

### 모듈별 직접 사용

```python
# 임베딩 모듈 사용
from src.embedding import RFPEmbedder, RFPVectorStore, EmbeddingCacheManager

# 임베딩 생성
embedder = RFPEmbedder("BAAI/bge-m3")
embedder.initialize()
embeddings = embedder.embed_texts(["문서 내용 1", "문서 내용 2"])

# 벡터 저장소 사용
vector_store = RFPVectorStore(embedder.dimension)
vector_store.add_embeddings(embeddings, chunks)

# 검색 모듈 사용
from src.retrieval import RFPRetriever, RFPChunker

# 문서 청킹
chunker = RFPChunker(chunk_size=1000, overlap=200)
chunks = chunker.chunk_document(document)

# 검색기 사용
retriever = RFPRetriever(embedding_model="BAAI/bge-m3")
retriever.initialize(documents)
results = retriever.search("검색 쿼리", top_k=5)

# 생성 모듈 사용
from src.generation import RFPGenerator

# 답변 생성기 초기화
generator = RFPGenerator(
    api_key="your_openai_api_key",
    model="gpt-4.1-mini",
    temperature=0.3,
    max_tokens=2048
)
generator.initialize()

# 답변 생성
response = generator.generate_response(
    question="질문 내용",
    retrieved_results=search_results,
    use_conversation_history=True
)
print(response.answer)
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

## 📈 시스템 성능

### 현재 구현된 기능
- ✅ **FAISS 시스템**: GPU 가속 벡터 검색 완료
- ✅ **ChromaDB 시스템**: 하이브리드 검색 완료
- ✅ **통합 인터페이스**: Streamlit 웹 앱 완료
- ✅ **캐시 시스템**: 빠른 재시작 지원
- ✅ **GPU 가속**: 임베딩 생성 최적화

### 성능 지표
- **FAISS**: 21,219개 청크, GPU 가속 검색
- **ChromaDB**: 7,569개 문서, BM25 + Vector + Reranking
- **응답 시간**: 평균 2-3초 (GPU 사용 시)
- **정확도**: 상세한 답변 생성 (사업명, 발주기관, 금액, 기간 포함)

### 최적화된 하이퍼파라미터
- 청크 크기: 1000 토큰
- 중첩 크기: 200 토큰
- Top-K: 3-5개 문서
- Temperature: 0.3
- Max tokens: 2048

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

## 📋 프로젝트 완성도

### ✅ 완료된 기능
- [x] **문서 처리**: PDF/HWP 파싱 및 JSON 변환
- [x] **FAISS 시스템**: GPU 가속 벡터 검색
- [x] **ChromaDB 시스템**: 하이브리드 검색 (BM25 + Vector + Reranking)
- [x] **통합 인터페이스**: Streamlit 웹 앱
- [x] **시스템 선택**: FAISS/ChromaDB 실시간 전환
- [x] **비교 모드**: 두 시스템 결과 동시 비교
- [x] **캐시 시스템**: 빠른 재시작 지원
- [x] **GPU 가속**: 임베딩 생성 최적화
- [x] **대화 맥락**: 연속 질의응답 지원
- [x] **상세 답변**: 핵심 정보 포함 답변 생성

### 🚀 추가 개선 가능 사항
- [ ] **멀티쿼리 검색**: 질문을 여러 개로 분해하여 검색
- [ ] **실시간 모니터링**: 시스템 성능 대시보드
- [ ] **A/B 테스트**: 두 시스템 성능 비교 분석
- [ ] **API 서버**: REST API 제공
- [ ] **배치 처리**: 대량 문서 일괄 처리

### 📚 상세 문서
- **검색 시스템**: `src/retrieval/README.md` - 하이브리드 검색 시스템 상세 가이드
- **생성 시스템**: `src/generation/readme_generation.md` - 답변 생성 모듈 상세 사용법
- **통합 시스템**: `UNIFIED_SYSTEM_README.md` - 통합 RAG 시스템 사용법
- **모듈별 문서**: 각 모듈 내부에 상세한 docstring 및 주석 포함

---

**AI03기 Part3 1팀** | AI03 스프린트 중급 프로젝트
