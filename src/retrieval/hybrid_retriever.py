import os
import json
import logging
from typing import List, Dict, Set, Tuple

import torch
from tqdm import tqdm
from tqdm.asyncio import tqdm_asyncio
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_chroma import Chroma
from langchain.schema import Document
from sentence_transformers import CrossEncoder
from rank_bm25 import BM25Okapi 
from more_itertools import chunked

import aiofiles
import asyncio
import pickle

# 로컬 임포트
from ..utils.exceptions import RetrieverError, ChunkLoadingError
from ..utils.filtering import extract_filters, check_filter_match, normalize_keywords
from .tokenizer_wrapper import TokenizerWrapper

# 로깅 설정
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

class Retriever:
    def __init__(self,
                 meta_df=None,
                 embedder=None,
                 reranker=None,
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

    def tokenize_korean(self, text: str) -> List[str]:
        tokens = self.tokenizer.tokenize(text)
        stopwords = {"에서", "는", "은", "이", "가", "하", "어야", "에", "을", "를", "도", "로", "과", "와", "의", "?", "다"}
        tokens = [t for t in tokens if t not in stopwords]
    
        # ✅ bi-gram 생성
        bigrams = [tokens[i] + tokens[i+1] for i in range(len(tokens) - 1)]
    
        # 최종 토큰 = 원래 토큰 + bigram
        return tokens + bigrams

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

    
    async def load_or_cache_json_docs(self, folder_path: str, cache_path: str = "cached_json_docs.pkl") -> List[Document]:
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
        if force_rebuild or not self._db_exists():
            logging.info("🆕 벡터 DB 생성 중...")

            # ✅ 문서 임베딩 진행 상황 표시
            wrapped_docs = list(tqdm(documents, desc="🔄 문서 임베딩 중"))
            self.db = Chroma.from_documents(wrapped_docs, self.embedder, persist_directory=self.persist_directory)
            
            logging.info("✅ 새 DB 구축 완료.")
        else:
            logging.info("✅ 기존 벡터 DB 로드 중...")
            self.load_vector_db()

        # ✅ 중복 문서 필터링 후 추가
        new_docs = self._filter_new_documents(documents)
        if new_docs:
            logging.info(f"➕ 새 문서 {len(new_docs)}개 추가 중...")
            
            # ✅ 새 문서 추가 진행 상황 표시
            wrapped_new_docs = list(tqdm(new_docs, desc="📥 새 문서 추가 중"))
            for batch in tqdm(chunked(wrapped_new_docs, 1000), desc="📥 배치 문서 추가 중"):
                self.db.add_documents(batch)

            # ✅ 전체 문서 로딩 + BM25 인덱싱 (새 문서 있을 때만)
            self.documents = self.get_all_documents_from_db()
            for _ in tqdm(range(1), desc="🔧 BM25 인덱스 구축 중"):  # 단일 작업이지만 시각적 피드백용
                self.build_bm25_index()
            self.save_bm25_index()
            logging.info("✅ BM25 인덱싱 완료.")
        else:
            logging.info("⏩ 새 문서 없음, DB 추가 생략")
            if not hasattr(self, "bm25_ready") or not self.bm25_ready:
                self.documents = self.get_all_documents_from_db()
                self.load_bm25_index()
                
                if not self.bm25_ready:
                    logging.info("📚 BM25 인덱싱 시작...")
                    for _ in tqdm(range(1), desc="🔧 BM25 인덱스 구축 중"):
                        self.build_bm25_index()
                    self.save_bm25_index()
                    logging.info("✅ BM25 인덱싱 완료.")

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

        tokenized_corpus = [self.tokenize_korean(doc.page_content) for doc in self.documents]
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

        tokenized_query = self.tokenize_korean(query)
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
        if not self.reranker:
            return {self.get_doc_key(doc): 0.0 for doc in documents}

        # 1. 입력 쌍 생성
        max_length = getattr(self, "rerank_max_length", 512)
        pairs = [[query, doc.page_content[:max_length]] for doc in documents]

        # 2. 점수 예측
        scores = self.reranker.predict(pairs).flatten()

        # 3. 고유 키 기준으로 점수 매핑 후 반환
        return {
            self.get_doc_key(doc): float(score)
            for score, doc in zip(scores, documents)
        }

    def _calculate_combined_scores(self, documents: List[Document], query: str, 
                                 bm25_scores: Dict[str, float], rerank_scores: Dict[str, float]) -> List[Tuple[float, Document]]:
        """문서들에 대한 BM25 + 재순위화 점수를 계산하여 반환"""
        final_scored = []
        self.last_scores= {}
        
        for doc in documents:
            key = self.get_doc_key(doc)
            bm25 = bm25_scores.get(key, 0.0)
            rerank = rerank_scores.get(key, 0.0)
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
    
    def hybrid_search(self, query: str, top_k: int = 3, candidate_size: int = 10, filter_dict: Dict = None) -> List[Document]:
        """하이브리드 검색: 벡터 + BM25 + 재순위화 + 필터링"""
        if self.db is None:
            raise ValueError("Vector DB가 초기화되지 않았습니다.")
        if not self.bm25_ready:
            raise ValueError("BM25 인덱스가 준비되지 않았습니다.")
        
        if filter_dict:
            logging.info(f"🔍 하이브리드 필터 적용: {filter_dict}")

        # 1. 벡터 + BM25 하이브리드 검색. 단순 필터(=) 적용. 검색범위 좁히고 성능 향상
        simple_filter = {
            key: val["value"]
            for key, val in filter_dict.items()
            if val.get("operator") == "=" and key != "사업 요약"
        }
        if not simple_filter:
            simple_filter = None

        vector_results = self.db.similarity_search(query, k=candidate_size, filter=simple_filter)
        bm25_results = self.bm25_search(query, k=candidate_size, filter=simple_filter)

        merged_docs, bm25_scores = self._merge_search_results(vector_results, bm25_results)

        # 2. 고급 조건 필터링 적용 (>, < 등 연산자 필터링)
        filtered_docs = [doc for doc in merged_docs if check_filter_match(doc.metadata, filter_dict)]
        logging.info(f"✅ 고급 필터링 후 문서 수: {len(filtered_docs )}")

        if not filtered_docs :
            logging.warning("⚠️ 필터링 결과가 없습니다. 원본 결과에서 상위 {top_k}개 반환합니다.")
            final_docs_to_score = merged_docs
        else:
            # 필터링 결과가 있으면 그 문서들에만 점수 계산
            final_docs_to_score = filtered_docs 

         # 3. 재순위화 점수 계산
        rerank_scores = self.rerank_documents(query, final_docs_to_score)
   
        # 4. 최종 점수 계산 및 정렬
        scored_docs = self._calculate_combined_scores(final_docs_to_score,  query, bm25_scores, rerank_scores)
        final_results = sorted(scored_docs, key=lambda x: x[0], reverse=True)
        
        # 디버그 출력
        if self.debug_mode:
            self._debug_print_scores(final_docs_to_score , "하이브리드 검색")
        
        logging.info(f"📊 하이브리드 검색 완료: {len(final_results)} → {top_k}개 반환")
        return [doc for _, doc in final_results[:top_k]]

    def detect_query_type(self, query: str) -> str:
        normalized_query = query.replace(" ", "").lower()
        
        explicit_summary_keywords = normalize_keywords([
            "사업요약", "공고요약", "사업개요", "공고개요"
        ])
        
        metadata_keywords = normalize_keywords([
            "사업금액", "예산", "금액", "입찰일", "입찰시작일", "참여시작일",
            "입찰마감일", "참여마감일", "공고번호", "발주기관", "공개일자", "파일형식", "입찰공고일"
        ])

        if any(k in normalized_query for k in explicit_summary_keywords):
            return "metadata"
        if any(k in normalized_query for k in metadata_keywords):
            return "metadata"
        return "semantic"

    
    def smart_search(self, query: str, top_k: int = 5, candidate_size: int = 10) -> List[Document]:
        """스마트 검색: 필터 추출 + 하이브리드 검색 + 고급 필터링"""
        if self.db is None or not self.bm25_ready:
            raise ValueError("❌ Vector DB 또는 BM25 인덱스가 준비되지 않았습니다.")

        # 1. 쿼리에서 필터 조건 추출
        filters  = extract_filters(query, self.meta_df, self.tokenizer)
        logging.info(f"🧠 추출된 필터: {filters}")

        # 2️. 쿼리 유형 판단
        query_type = self.detect_query_type(query)

        # 🚨 발주기관 키워드가 추출되었으면 쿼리에서 삭제
        if filters.get("발주 기관"):
            agency_name = filters["발주 기관"]["value"]
            query = query.replace(agency_name, "").strip()
            logging.info(f"🧹 쿼리에서 발주기관 키워드 제거됨: '{agency_name}'")


        # 3. 메타데이터 기반 검색
        if query_type == "metadata" and self.meta_df is not None:
            logging.info("📊 메타데이터 기반 검색 실행")
            matched_docs = self.meta_df[
                self.meta_df.apply(lambda row: check_filter_match(row, filters), axis=1)
            ]
            return matched_docs.to_dict(orient="records")  # ✅ 전체 행을 dict로 반환

        
        # 4. 하이브리드 검색 수행 (모든 필터 정보 전달)
        logging.info("🔍 의미 기반 하이브리드 검색 실행")
        return self.hybrid_search(
            query=query, 
            top_k=top_k, 
            candidate_size=candidate_size,
            filter_dict=filters 
        )
        