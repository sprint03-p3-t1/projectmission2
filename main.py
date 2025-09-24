import os
import sys
import asyncio
import logging
import pandas as pd

from config import Config
from src.retrieval.hybrid_retriever import Retriever
from src.retrieval.tokenizer_wrapper import TokenizerWrapper
from langchain_huggingface import HuggingFaceEmbeddings
from sentence_transformers import CrossEncoder

# 로깅 설정
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    datefmt="%H:%M:%S"
)

async def main():
    # 설정 로드
    cfg = Config()
    
    # 기본 설정
    LOAD_MODE = "json"
    TOKENIZER = "kiwi"
    
    # 메타데이터 로드
    meta_df = pd.read_csv(cfg.meta_csv_path)
    
    # 디바이스 설정
    import torch
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    print(f"✅ 디바이스: {device}")
    
    # 모델 초기화 (외부에서 로드하여 캐시화)
    embedder = HuggingFaceEmbeddings(
        model_name=cfg.embedder_model,
        model_kwargs={"device": device}
    )
    
    reranker = CrossEncoder(
        cfg.reranker_model,
        device=device,
        max_length=cfg.rerank_max_length
    )
    
    tokenizer = TokenizerWrapper(TOKENIZER)
    
    # Retriever 초기화 (모든 기존 기능 유지)
    retriever = Retriever(
        meta_df=meta_df,
        embedder=embedder,
        reranker=reranker,
        tokenizer=tokenizer,
        persist_directory=cfg.chroma_db_path,
        rerank_max_length=cfg.rerank_max_length,
        bm25_path=cfg.bm25_path,
        debug_mode=True
    )
    
    try:
        # 문서 로딩 (기존 로직 100% 보존)
        if LOAD_MODE == "json":
            docs = await retriever.load_or_cache_json_docs(
                cfg.processed_dir, 
                cache_path=cfg.cached_json_path
            )
        
        # 가중치 설정 및 벡터 DB 구축 (기존과 동일)
        retriever.set_weights(bm25_weight=0.5, rerank_weight=0.5)
        retriever.load_or_build_vector_db(docs)
        
        # 검색 실행 (기존 로직 그대로)
        query_text = "국립인천해양박물관 최종검수 기간은"
        logging.info(f"\n[스마트 검색 실행] 질문: {query_text}")
        
        tokenized_query = retriever.tokenize_korean(query_text)
        print(f"\n🔍 BM25 키워드 토큰: {tokenized_query}")
        
        results = retriever.smart_search(
            query=query_text,
            top_k=3,
            candidate_size=10,
        )
        
        # 결과 출력 (기존 로직 100% 유지)
        print("\n📈 최종 결과:")
        
        if results and isinstance(results[0], dict):
            # 메타데이터 기반 검색 결과
            for i, record in enumerate(results):
                print(f"\n📄 문서 {i+1}")
                for key, value in record.items():
                    print(f"🔹 {key}: {value}")
                print("=" * 50)
        else:
            # 의미 기반 검색 결과
            for i, doc in enumerate(results):
                key = retriever.get_doc_key(doc)
                scores = retriever.last_scores.get(key, {})
                bm25 = scores.get("bm25", 0.0)
                rerank = scores.get("rerank", 0.0)
                combined = scores.get("combined", 0.0)
        
                print(f"문서 {i+1} | 출처: {doc.metadata.get('chunk_id')}")
                print(f"🔹 BM25 점수: {bm25:.2f} | 🔹 Rerank 점수: {rerank:.2f} | 🔹 Combined: {combined:.2f}")
                print(doc.page_content[:500])
                print("=" * 50)

    except Exception as e:
        logging.error(f"오류: {e}")

if __name__ == "__main__":
    asyncio.run(main())
