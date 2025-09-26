import re
import os
import hashlib
import pickle
from typing import List, Dict

import torch
from transformers import AutoModel, AutoTokenizer


class RerankModel:
    def __init__(self, model_name: str, cache_dir: str, device: str = "cuda"):
        """모델과 토크나이저를 로드하여 초기화"""
        self.model = AutoModel.from_pretrained(model_name).to(device)
        self.tokenizer = AutoTokenizer.from_pretrained(model_name)
        self.device = device
        self.cache_dir = cache_dir
        os.makedirs(self.cache_dir, exist_ok=True)

    def _get_cache_path(self, text: str) -> str:
        """텍스트를 기반으로 캐시 파일 경로 생성"""
        key = hashlib.md5(text.encode()).hexdigest()
        return os.path.join(self.cache_dir, f"{key}.pkl")

    def _load_embedding_from_cache(self, text: str) -> torch.Tensor:
        """캐시에서 임베딩 로드"""
        path = self._get_cache_path(text)
        if os.path.exists(path):
            with open(path, "rb") as f:
                return pickle.load(f)
        return None

    def _save_embedding_to_cache(self, text: str, embedding: torch.Tensor):
        """임베딩을 캐시에 저장"""
        path = self._get_cache_path(text)
        with open(path, "wb") as f:
            pickle.dump(embedding.cpu(), f)

    def _embed(self, text: str) -> torch.Tensor:
        """텍스트를 의미 벡터로 변환"""
        inputs = self.tokenizer(text, padding=True, truncation=True, return_tensors="pt").to(self.device)
        with torch.no_grad():
            hidden = self.model(**inputs).last_hidden_state
            mask = inputs["attention_mask"].unsqueeze(-1)
            pooled = (hidden * mask).sum(dim=1) / mask.sum(dim=1)
        return pooled[0]

    def _get_embedding(self, text: str, is_query: bool = False) -> torch.Tensor:
        """임베딩 반환 (문서는 캐싱, 쿼리는 계산)"""
        if is_query:
            return self._embed(text)

        cached = self._load_embedding_from_cache(text)
        if cached is not None:
            return cached.to(self.device)

        emb = self._embed(text)
        self._save_embedding_to_cache(text, emb)
        return emb

    def rerank(self, query: str, documents: List[str]) -> Dict[str, float]:
        """쿼리와 문서들의 유사도를 계산해 점수 반환"""
        query_embedding = self._get_embedding(query, is_query=True).unsqueeze(0)
        doc_embeddings = [self._get_embedding(doc).unsqueeze(0) for doc in documents]
        doc_tensor = torch.cat(doc_embeddings, dim=0)

        cos_scores = torch.nn.functional.cosine_similarity(query_embedding, doc_tensor)
        return {doc: score.item() for doc, score in zip(documents, cos_scores)}

    def cache_embeddings(self, texts: List[str], max_length: int = 512):
        """전체 문서에 대해 의미 임베딩 캐싱"""
        skipped, cached = 0, 0
        for text in texts:
            text = text[:max_length]
            path = self._get_cache_path(text)
            if os.path.exists(path):
                skipped += 1
                continue
            emb = self._embed(text)
            self._save_embedding_to_cache(text, emb)
            cached += 1
        print(f"✅ 의미 임베딩 캐싱 완료 | 새로 캐싱: {cached}개 | 스킵: {skipped}개")


