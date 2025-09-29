import os
import re
import json
import pickle
import logging
from typing import List, Dict, Set, Tuple

from tqdm import tqdm
from tqdm.asyncio import tqdm_asyncio
from langchain_chroma import Chroma
from langchain.schema import Document
from rank_bm25 import BM25Okapi 
from more_itertools import chunked

import aiofiles

# 로컬 임포트
from ..utils.exceptions import RetrieverError, ChunkLoadingError
from ..utils.filtering import extract_filters, check_filter_match, normalize_keywords
from ..utils.scaling import minmax_scale
from .tokenizer_wrapper import TokenizerWrapper
from .rerank import RerankModel

# 로컬 임포트
# 로깅 설정
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')



class Retriever:
    def __init__(self,
                 meta_df=None,
                 embedder=None,
                 reranker: RerankModel =None,
                 tokenizer=None,
                 persist_directory=None,
                 rerank_max_length=512,
                 bm25_weight=0.5,
                 rerank_weight=0.5,
                 bm25_path="bm25_index.pkl",
                 debug_mode=False
                 ):
        self.meta_df = meta_df
        self.embedder = embedder
        self.reranker = reranker
        self.tokenizer = tokenizer or TokenizerWrapper("kiwi")
        self.persist_directory = persist_directory
        self.rerank_max_length = rerank_max_length

        self.bm25_weight = bm25_weight
        self.rerank_weight = rerank_weight

        self.db = None
        self.bm25 = None
        self.bm25_ready = False
        self.bm25_path = bm25_path
        self.documents = []
        self.debug_mode = debug_mode

        self.last_scores = {}

    def set_weights(self, bm25_weight: float, rerank_weight: float):
        self.bm25_weight = bm25_weight
        self.rerank_weight = rerank_weight
        logging.info(f"🔧 가중치 설정됨 | BM25: {bm25_weight} | Rerank: {rerank_weight}")

    def get_doc_key(self, doc: Document) -> str:
        chunk_id = doc.metadata.get("chunk_id")
        if chunk_id:
            return chunk_id
        return str(hash(doc.page_content.strip()))

    def save_bm25_index(self):
        os.makedirs(os.path.dirname(self.bm25_path), exist_ok=True)
        with open(self.bm25_path, "wb") as f:
            pickle.dump(self.bm25, f)
        logging.info(f"✅ BM25 인덱스 저장 완료: {self.bm25_path}")

    def load_bm25_index(self):
        path = self.bm25_path
        if os.path.exists(path):
            try:
                with open(path, "rb") as f:
                    self.bm25 = pickle.load(f)
                self.bm25_ready = True
                logging.info(f"✅ BM25 인덱스 로드 완료: {path}")
            except Exception as e:
                self.bm25_ready = False
                logging.warning(f"❌ BM25 인덱스 로드 실패: {e}")
        else:
            self.bm25_ready = False
            logging.warning(f"❌ BM25 인덱스 파일 없음: {path}")

    def deduplicate_documents(self, documents: List[Document]) -> List[Document]:
        seen = set()
        unique_docs = []
        for doc in documents:
            chunk_id = doc.metadata.get("chunk_id")
            if chunk_id and chunk_id not in seen:
                seen.add(chunk_id)
                unique_docs.append(doc)
        removed = len(documents) - len(unique_docs)
        logging.info(f"🧹 중복 제거: {removed}개 제거됨")
        return unique_docs

    
    async def load_or_cache_json_docs(self, folder_path: str, cache_path: str) -> List[Document]:
        # 캐시 디렉토리 자동 생성
        os.makedirs(os.path.dirname(cache_path), exist_ok=True)
    
        if os.path.exists(cache_path):
            with open(cache_path, "rb") as f:
                logging.info("📦 캐시된 JSON 문서 로드 중...")
                return pickle.load(f)
        else:
            logging.info("📂 JSON 폴더에서 문서 로딩 중...")
            docs = await self.async_load_chunks_from_folder(folder_path)
            with open(cache_path, "wb") as f:
                pickle.dump(docs, f)
            logging.info("✅ JSON 캐시 저장 완료")
            return docs

    async def async_load_chunks_from_folder(self, folder_path: str) -> List[Document]:
        if not os.path.isdir(folder_path):
            raise ChunkLoadingError(f"❌ 폴더 경로가 존재하지 않음: {folder_path}")

        file_list = sorted([f for f in os.listdir(folder_path) if f.endswith(".json")])
        existing_sources = self.get_existing_chunk_ids()
        logging.info(f"📁 기존 DB에 저장된 source 수: {len(existing_sources)}")

        tasks = []
        for filename in file_list:
            if filename in existing_sources:
                logging.info(f"⏩ 이미 처리된 파일 건너뜀: {filename}")
                continue
            file_path = os.path.join(folder_path, filename)
            tasks.append(self._load_single_file(file_path, filename))

        # ✅ 고급 tqdm 적용: 실제 완료 기준으로 진행률 표시
        all_chunks = []
        for coro in tqdm_asyncio.as_completed(tasks, desc="📂 파일 처리 중", total=len(tasks)):
            result = await coro
            all_chunks.append(result)

        documents = [doc for sublist in all_chunks for doc in sublist]
        logging.info(f"✅ 새로 로드된 문서 수: {len(documents)}")

        documents = self.deduplicate_documents(documents)
        logging.info(f"🧹 중복 제거 후 문서 수: {len(documents)}")
        return documents

    
    async def _load_single_file(self, file_path: str, filename: str) -> list[Document]:
        try:
            async with aiofiles.open(file_path, "r", encoding="utf-8") as f:
                content = await f.read()
                data = json.loads(content)
    
            metadata_base = data.get("csv_metadata", {})
            docs = []
    
            for idx, page in enumerate(data.get("pdf_data", [])):
                page_num = page.get("page", idx)
                text = page.get("text", "").strip()
    
                if text:
                    metadata = metadata_base.copy()
                    metadata["chunk_id"] = f"{filename}::page::{page_num}::type::text"
                    metadata["page"] = page_num
    
                    docs.append(Document(page_content=text, metadata=metadata))
    
            return docs
    
        except Exception as e:
            logging.warning(f"⚠️ 파일 로딩 실패: {filename} | 오류: {e}")
            return []

    def get_all_documents_from_db(self) -> List[Document]:
        if self.db is None:
            raise ValueError("Vector DB가 초기화되지 않았습니다.")

        all_docs = []
        collection = self.db._collection
        count = collection.count()
        offset = 0
        limit = 1000

        while offset < count:
            results = collection.get(
                limit=limit,
                offset=offset,
                include=["metadatas", "documents"]
            )
            for doc, meta in zip(results["documents"], results["metadatas"]):
                if not meta.get("chunk_id"):
                    logging.warning("⚠️ chunk_id 누락된 문서 발견")
                all_docs.append(Document(page_content=doc, metadata=meta))
            offset += limit

        return all_docs


    def load_or_build_vector_db(self, documents: List[Document], force_rebuild: bool = False):
        """Vector DB와 BM25 인덱스를 로드하거나 새로 구축합니다."""
        
        # 1단계: Vector DB 초기화 (로드 또는 생성)
        self._initialize_vector_db(documents, force_rebuild)
        
        # 2단계: 새 문서가 있다면 DB에 추가
        new_docs = self._update_db_with_new_docs(documents)
        
        # 3단계: BM25 인덱스 동기화 (새 문서 추가 여부에 따라 처리)
        self.documents = self.get_all_documents_from_db()
        self._synchronize_bm25_index(was_db_updated=(len(new_docs) > 0))
        
        # 4단계: 리랭커 임베딩 캐싱
        self._cache_reranker_embeddings()
        logging.info("🚀 모든 DB 및 인덱스 준비 완료.")
    

    def _initialize_vector_db(self, documents: List[Document], force_rebuild: bool):
        """Vector DB를 생성하거나 로드합니다."""
        if force_rebuild or not self._db_exists():
            logging.info("🆕 벡터 DB 생성 중...")
            wrapped_docs = list(tqdm(documents, desc="🔄 문서 임베딩 중"))
            self.db = Chroma.from_documents(wrapped_docs, self.embedder, persist_directory=self.persist_directory)
            logging.info("✅ 새 DB 구축 완료.")
        else:
            logging.info("✅ 기존 벡터 DB 로드 중...")
            self.load_vector_db()
    
    def _update_db_with_new_docs(self, documents: List[Document]) -> List[Document]:
        """새로운 문서를 필터링하여 DB에 추가합니다."""
        new_docs = self._filter_new_documents(documents)
        if not new_docs:
            logging.info("⏩ 새 문서 없음, DB 추가 생략")
            return []
    
        logging.info(f"➕ 새 문서 {len(new_docs)}개 추가 중...")
        batch_size = 100
        total_batches = (len(new_docs) + batch_size - 1) // batch_size
        
        for batch in tqdm(chunked(new_docs, batch_size), desc="📦 배치 추가 중", total=total_batches):
            self.db.add_documents(batch)
            
        return new_docs
    
    def _synchronize_bm25_index(self, was_db_updated: bool):
        """DB 상태에 따라 BM25 인덱스를 생성하거나 로드합니다."""
        # 새 문서가 추가됐다면 BM25 인덱스는 무조건 새로 만들어야 함
        if was_db_updated:
            logging.info("🔧 새 문서 추가됨, BM25 인덱스 재생성...")
            self.build_bm25_index()
            self.save_bm25_index()
            logging.info("✅ BM25 인덱싱 완료.")
            return
    
        # 새 문서가 없다면, 기존 인덱스를 로드해보고 없으면 생성
        if not hasattr(self, "bm25_ready") or not self.bm25_ready:
            self.load_bm25_index()
            if not self.bm25_ready:
                logging.info("📚 기존 BM25 인덱스 없음, 새로 구축 시작...")
                self.build_bm25_index()
                self.save_bm25_index()
                logging.info("✅ BM25 인덱싱 완료.")
    
    def _cache_reranker_embeddings(self):
        """리랭커 모델을 위한 임베딩을 캐싱합니다."""
        if not self.documents:
            logging.warning("⚠️ 캐싱할 문서가 없습니다.")
            return
            
        logging.info("💡 리랭커 임베딩 캐싱 중...")
        texts = [doc.page_content[:self.rerank_max_length] for doc in self.documents]
        self.reranker.cache_embeddings(texts, max_length=self.rerank_max_length)
        logging.info("✅ 리랭커 캐싱 완료.")

    def load_vector_db(self):
        if not self.persist_directory or not os.path.exists(self.persist_directory):
            raise ValueError("저장된 DB가 없거나 persist_directory가 잘못 설정되었습니다.")
        self.db = Chroma(persist_directory=self.persist_directory, embedding_function=self.embedder)

    def _db_exists(self) -> bool:
        if not self.persist_directory:
            return False
        required_files = ["chroma.sqlite3"]
        return all(os.path.exists(os.path.join(self.persist_directory, f)) for f in required_files)

    def get_existing_chunk_ids(self) -> Set[str]:
        if self.db is None:
            try:
                self.load_vector_db()
            except Exception as e:
                logging.warning(f"⚠️ DB 로드 실패: {e}")
                self.db = None
                return set()

        try:
            result = self.db.get(include=["metadatas"])
            chunk_ids = set()
            for i, meta in enumerate(result.get("metadatas", [])):
                cid = meta.get("chunk_id")
                if not cid:
                    logging.warning(f"⚠️ chunk_id 누락된 문서 발견 (index={i})")
                    continue
                chunk_ids.add(cid)
            return chunk_ids
        except Exception as e:
            logging.warning(f"⚠️ chunk_id 목록 추출 실패: {e}")
            return set()


    def _filter_new_documents(self, documents: List[Document]) -> List[Document]:
        existing_chunk_ids  = self.get_existing_chunk_ids()
        new_docs = []
        for doc in documents:
            chunk_id  = doc.metadata.get("chunk_id")
            if chunk_id  and chunk_id  not in existing_chunk_ids :
                new_docs.append(doc)
        return new_docs

    # ✅ BM25 관련 함수 추가
    def build_bm25_index(self):
        if self.bm25 is not None:
            logging.info("⏩ BM25 인덱스 이미 존재함, 재생성 생략")
            return
        if not self.documents:
            raise ValueError("BM25 인덱스를 생성할 문서가 없습니다.")

        tokenized_corpus = [self.tokenizer.tokenize_korean(doc.page_content) 
                            for doc in tqdm(self.documents, desc="🧠 문서 토큰화 중")]
        self.bm25 = BM25Okapi(tokenized_corpus)
        self.bm25_ready = True
        logging.info("✅ BM25 인덱스 생성 완료")

    def _debug_print_bm25_scores(self, top_docs: List[Tuple[float, Document]]):
        """BM25 점수 디버그 출력 전용 함수"""
        print("\n📈 BM25 상위 문서 및 점수:")
        for i, (score, doc) in enumerate(top_docs):
            print(f"BM25 문서 {i+1} | 점수: {score:.4f} | 출처: {doc.metadata.get('파일명')}")
            print(doc.page_content[:300])
            print("-" * 40)
        
    def bm25_search(self, query: str, k: int = 5, filter: Dict = None, debug: bool = False) -> List[Tuple[float, Document]]:
        if not self.bm25_ready:
            raise ValueError("BM25 인덱스가 준비되지 않았습니다.")

        tokenized_query = self.tokenizer.tokenize_korean(query)
        doc_scores = self.bm25.get_scores(tokenized_query)
           
        # ✅ 필터링 적용
        filtered_docs = []
        for score, doc in zip(doc_scores, self.documents):
            if filter:
                match = all(doc.metadata.get(k) == v for k, v in filter.items())
                if not match:
                    continue
            filtered_docs.append((score, doc))

        top_docs = sorted(filtered_docs, key=lambda x: x[0], reverse=True)[:k]

        if self.debug_mode:
            self._debug_print_bm25_scores(top_docs)
            
        return top_docs

    def rerank_documents(self, query: str, documents: List[Document]) -> Dict[str, float]:
        """쿼리와 문서들의 유사도를 계산하여 재순위화 점수 반환"""
        if not self.reranker:
            logging.warning("⚠️ 재순위화 모델이 로드되지 않았습니다. 점수가 0으로 반환됩니다.")
            return {self.get_doc_key(doc): 0.0 for doc in documents}

        if not documents:
            logging.info("ℹ️ 재순위화할 문서가 없어 빈 결과를 반환합니다.")
            return {}
            
        texts_to_rerank = [doc.page_content[:self.rerank_max_length] for doc in documents]
        base_scores = self.reranker.rerank(query, texts_to_rerank)
    
        query_tokens = self.tokenizer.tokenize_korean(query)
        bonus_weight = 1.0
    
        final_scores = {}
        for doc, base_score in zip(documents, base_scores.values()):
            bonus = 0
            if query in doc.page_content:
                bonus += 2.0
            bonus += sum(1 for token in query_tokens if token in doc.page_content) * bonus_weight
    
            key = self.get_doc_key(doc)
            final_scores[key] = base_score + bonus
    
        logging.info("🧠 재순위화 점수 계산 완료 (보너스 포함)")
        return final_scores
    
    def _calculate_combined_scores(self, documents: List[Document], query: str, 
                                 bm25_scores: Dict[str, float], rerank_scores: Dict[str, float]) -> List[Tuple[float, Document]]:
        """문서들에 대한 BM25 + 재순위화 점수를 계산하여 반환"""
        final_scored = []
        self.last_scores= {}

        # 1. 점수 정규화
        scaled_bm25 = minmax_scale(bm25_scores)
        scaled_rerank = minmax_scale(rerank_scores)

        for doc in documents:
            key = self.get_doc_key(doc)
            bm25 = scaled_bm25.get(key, 0.0)
            rerank = scaled_rerank.get(key, 0.0)
            combined = self.bm25_weight * bm25 + self.rerank_weight * rerank
            
            self.last_scores[key] = {
                "bm25": bm25,
                "rerank": rerank,
                "combined": combined
            }
            final_scored.append((combined, doc))
    
        return final_scored

    def _debug_print_scores(self, documents: List[Document], search_type: str):
        """디버그 모드일 때 점수 정보 출력"""
        if not self.debug_mode:
            return
            
        logging.info(f"{search_type} 점수 정보")
        for doc in documents:
            key = self.get_doc_key(doc)
            scores = self.last_scores.get(key, {})
            bm25 = scores.get("bm25", 0.0)
            rerank = scores.get("rerank", 0.0)
            combined = scores.get("combined", 0.0)

            source = doc.metadata.get("파일명", "❓")
            chunk_index = doc.metadata.get("chunk_id", "❓")
            print(f"🔍 {source} | Chunk {chunk_index} | BM25: {bm25:.2f} | Rerank: {rerank:.2f} | Combined: {combined:.2f}")

        
    def _merge_search_results(self, vector_results: List[Document], bm25_results: List[Tuple[float, Document]]) -> Tuple[List[Document], Dict[str, float]]:
        """벡터와 BM25 검색 결과를 병합하고 점수 딕셔너리 반환"""
        merged = {}
        bm25_scores = {}
        
        # BM25 결과 우선 추가
        for score, doc in bm25_results:
            key = self.get_doc_key(doc)
            merged[key] = doc
            bm25_scores[key] = score
        
        # 벡터 결과에서 새로운 문서만 추가
        for doc in vector_results:
            key = self.get_doc_key(doc)
            if key not in merged:
                merged[key] = doc
                bm25_scores[key] = 0.0
        
        return list(merged.values()), bm25_scores
    
    def hybrid_search(self, query: str, top_k: int = 3, candidate_size: int = 10,
                     candidate_filenames: List[str] = None) -> List[Document]:
        """하이브리드 검색: BM25 + 벡터 + rerank (후보군이 있으면 그 안에서만 실행)"""
        if self.db is None:
            raise ValueError("Vector DB가 초기화되지 않았습니다.")
        if not self.bm25_ready:
            raise ValueError("BM25 인덱스가 준비되지 않았습니다.")
    
        # 1. 기본 검색
        vector_results = self.db.similarity_search(query, k=candidate_size, filter=None)
        bm25_results = self.bm25_search(query, k=candidate_size, filter=None)

        logging.info(f"📁 벡터 검색 결과 파일명: {[doc.metadata.get('파일명') for doc in vector_results]}")

         # 2. 후보군 필터링
        if candidate_filenames:
            logging.info(f"📁 후보군 제한 적용: {len(candidate_filenames)}개")
            vector_results = [doc for doc in vector_results if doc.metadata.get("파일명") in candidate_filenames]
            bm25_results = [(score, doc) for score, doc in bm25_results if doc.metadata.get("파일명") in candidate_filenames]
    
            # ✅ fallback: 후보군이 검색 결과에 아예 없으면 meta_df 기반 dummy 문서라도 추가
            if not vector_results and not bm25_results:
                logging.warning("⚠️ 후보군이 검색 결과에 없음 → meta_df 후보군 강제 추가")
                candidate_chunks = []
                for fname in candidate_filenames:
                    try:
                        # meta_df에서 row 찾아 dummy Document 생성
                        row = self.meta_df[self.meta_df["파일명"] == fname].iloc[0].to_dict()
                        
                        # page_content는 '사업 요약'만 쓰고, metadata에서는 제거
                        page_content = row.get("사업 요약", "")
                        
                        # metadata에서 '사업 요약' 키 제거 (중복 방지)
                        metadata = {k: v for k, v in row.items() if k != "사업 요약"}
                        
                        doc = Document(page_content=page_content, metadata=metadata)
                        candidate_chunks.append(doc)
                    except Exception as e:
                        logging.error(f"❌ 후보군 강제 추가 실패: {fname}, {e}")
    
                if candidate_chunks:
                    vector_results = candidate_chunks  # 강제 투입
        
        # 3. 결과 병합
        merged_docs, bm25_scores = self._merge_search_results(vector_results, bm25_results)

        if not merged_docs:
            logging.warning("⚠️ 검색 결과 없음 → 빈 결과 반환")
            return []
    
        # 4. rerank
        rerank_scores = self.rerank_documents(query, merged_docs)
    
        # 5. 점수 계산 + 정렬
        scored_docs = self._calculate_combined_scores(merged_docs, query, bm25_scores, rerank_scores)
        final_results = sorted(scored_docs, key=lambda x: x[0], reverse=True)
        
        if self.debug_mode:
            self._debug_print_scores([doc for _, doc in final_results], "하이브리드 검색")
    
        logging.info(f"📊 하이브리드 검색 완료: {len(final_results)} → {top_k}개 반환")
        return [doc for _, doc in final_results[:top_k]]


    def detect_query_type(self, query: str, filters: Dict[str, Dict]) -> str:
        normalized_query = query.replace(" ", "").lower()
        
        explicit_summary_keywords = normalize_keywords([
            "사업요약", "공고요약", "사업개요", "공고개요"
        ])
        
        metadata_keywords = normalize_keywords([
            "사업금액",  "입찰일", "입찰시작일", "참여시작일",
            "입찰마감일", "참여마감일", "공고번호", "공개일자", "입찰공고일"
        ])

        if any(k in normalized_query for k in explicit_summary_keywords):
            return "metadata"
        if any(filters for field in metadata_keywords):
            return "metadata"
        return "semantic"

    
    def smart_search(self, query: str, top_k: int = 5, candidate_size: int = 10) -> List[Document]:
        """스마트 검색: 필터 추출 + 쿼리 유형 판단 + 하이브리드 검색"""
        if self.db is None or not self.bm25_ready:
            raise ValueError("❌ Vector DB 또는 BM25 인덱스가 준비되지 않았습니다.")
    
        # 1. 필터 추출
        filters = extract_filters(query, self.meta_df, self.tokenizer)
        logging.info(f"🧠 추출된 필터: {filters}")
    
        # 2. 쿼리 유형 판단
        query_type = self.detect_query_type(query, filters)
    
        # 3. 발주기관 제거 (쿼리에서 직접 빼줌)
        if filters.get("발주 기관"):
            agency_name = filters["발주 기관"]["value"]
            query = re.sub(rf"\b{re.escape(agency_name)}\b", "", query).strip()
            logging.info(f"🧹 쿼리에서 발주기관 키워드 제거됨: '{agency_name}'")
    
        # 4. 메타데이터 기반 후보군 뽑기
        matched_records = []
        candidate_filenames = None
        if query_type == "metadata" and self.meta_df is not None:
            matched_df = self.meta_df[
                self.meta_df.apply(lambda row: check_filter_match(row, filters), axis=1)
            ]
            logging.info(f"📊 메타데이터 필터링 완료: {len(matched_df)}개")
    
            if not matched_df.empty:
                matched_records = matched_df.head(10).to_dict(orient="records")
                candidate_filenames = matched_df["파일명"].dropna().unique().tolist()
                logging.info(f"📁 의미 검색 대상 제한됨 (파일명 기준): {len(candidate_filenames)}개")
            else:
                logging.warning("⚠️ 메타데이터 필터링 결과 없음 → 전체 문서 대상으로 검색")
    
        # 5. 하이브리드 검색 실행
        logging.info("🔍 의미 기반 하이브리드 검색 실행")
        semantic_docs = self.hybrid_search(
            query=query,
            top_k=top_k,
            candidate_size=candidate_size,
            candidate_filenames=candidate_filenames
        )
        
        return matched_records, semantic_docs
        
        