"""
텍스트 처리 모듈
문서 텍스트의 전처리 및 청킹을 담당합니다.
"""

import re
import logging
from typing import List, Dict, Any
from dataclasses import dataclass

logger = logging.getLogger(__name__)


@dataclass
class ChunkConfig:
    """청킹 설정 클래스"""
    chunk_size: int = 512
    overlap: int = 50
    separator: str = "\n\n"


class TextProcessor:
    """텍스트 처리 클래스"""

    def __init__(self, config: ChunkConfig = None):
        """
        TextProcessor 초기화

        Args:
            config: 청킹 설정 (기본값 사용 시 None)
        """
        self.config = config or ChunkConfig()

    def preprocess_text(self, text: str) -> str:
        """
        텍스트 전처리 수행

        Args:
            text: 원본 텍스트

        Returns:
            전처리된 텍스트
        """
        if not text:
            return ""

        # 기본적인 텍스트 정제
        processed_text = self._clean_text(text)
        processed_text = self._normalize_whitespace(processed_text)
        processed_text = self._remove_noise(processed_text)

        return processed_text

    def _clean_text(self, text: str) -> str:
        """텍스트에서 불필요한 문자 제거"""
        # 연속된 공백 제거
        text = re.sub(r'\s+', ' ', text)
        # 특수 문자 정제 (필요한 특수문자는 유지)
        text = re.sub(r'[^\w\s가-힣.!?(),-]', '', text)
        return text.strip()

    def _normalize_whitespace(self, text: str) -> str:
        """공백 정규화"""
        # 줄바꿈을 단일 공백으로 변환
        text = re.sub(r'\n+', ' ', text)
        # 탭을 공백으로 변환
        text = text.replace('\t', ' ')
        return text

    def _remove_noise(self, text: str) -> str:
        """노이즈 제거"""
        # 매우 짧은 단어들 제거 (1-2글자 한글 단어는 유지)
        text = re.sub(r'\b[a-zA-Z]{1,2}\b', '', text)
        return text.strip()

    def chunk_text(self, text: str, config: ChunkConfig = None) -> List[Dict[str, Any]]:
        """
        텍스트를 청크로 분할

        Args:
            text: 분할할 텍스트
            config: 청킹 설정 (기본값 사용 시 None)

        Returns:
            청크 리스트 (각 청크는 텍스트, 시작 위치, 끝 위치 포함)
        """
        if not text:
            return []

        config = config or self.config

        # 전처리 수행
        processed_text = self.preprocess_text(text)

        # 기본적인 청킹 (문장 단위 또는 고정 길이)
        chunks = self._create_chunks(processed_text, config)

        return chunks

    def _create_chunks(self, text: str, config: ChunkConfig) -> List[Dict[str, Any]]:
        """
        텍스트를 청크로 분할하는 내부 메서드

        Args:
            text: 분할할 텍스트
            config: 청킹 설정

        Returns:
            청크 리스트
        """
        chunks = []
        words = text.split()
        current_chunk = []
        current_length = 0

        for word in words:
            word_length = len(word) + 1  # 공백 포함

            # 현재 청크가 최대 크기를 초과할 경우
            if current_length + word_length > config.chunk_size and current_chunk:
                # 청크 생성
                chunk_text = ' '.join(current_chunk)
                chunks.append({
                    'text': chunk_text,
                    'start_pos': len(' '.join(words[:len(current_chunk) - len(current_chunk)])),
                    'end_pos': len(' '.join(words[:len(current_chunk)])),
                    'word_count': len(current_chunk)
                })

                # 중첩을 위한 일부 단어 유지
                overlap_words = self._get_overlap_words(current_chunk, config.overlap)
                current_chunk = overlap_words
                current_length = sum(len(w) + 1 for w in overlap_words)

            current_chunk.append(word)
            current_length += word_length

        # 마지막 청크 추가
        if current_chunk:
            chunk_text = ' '.join(current_chunk)
            chunks.append({
                'text': chunk_text,
                'start_pos': len(text) - len(chunk_text),
                'end_pos': len(text),
                'word_count': len(current_chunk)
            })

        return chunks

    def _get_overlap_words(self, words: List[str], overlap_size: int) -> List[str]:
        """
        중첩을 위한 단어들 추출

        Args:
            words: 원본 단어 리스트
            overlap_size: 중첩할 단어 수

        Returns:
            중첩 단어 리스트
        """
        if len(words) <= overlap_size:
            return words[-overlap_size:] if overlap_size > 0 else []

        return words[-overlap_size:]

    def chunk_by_sections(self, text: str, section_patterns: List[str] = None) -> List[Dict[str, Any]]:
        """
        섹션 기반으로 텍스트를 청킹 (심화 기능)

        Args:
            text: 분할할 텍스트
            section_patterns: 섹션 패턴 리스트

        Returns:
            섹션별 청크 리스트
        """
        if not section_patterns:
            # 기본적인 RFP 섹션 패턴들
            section_patterns = [
                r'제\s*\d+\s*조',  # 제1조, 제2조 등
                r'제\s*\d+\s*항',  # 제1항, 제2항 등
                r'\d+\.',  # 1., 2. 등
                r'[가-힣]+\s*요구사항',  # 기능 요구사항, 성능 요구사항 등
                r'[가-힣]+\s*사항',  # 세부 사항, 기술 사항 등
            ]

        chunks = []
        current_pos = 0

        # 섹션 패턴으로 분할
        for pattern in section_patterns:
            matches = list(re.finditer(pattern, text))

            for match in matches:
                start_pos = match.start()
                end_pos = match.end()

                # 이전 청크와 현재 매치 사이의 텍스트
                if start_pos > current_pos:
                    chunk_text = text[current_pos:start_pos].strip()
                    if chunk_text:
                        chunks.append({
                            'text': chunk_text,
                            'start_pos': current_pos,
                            'end_pos': start_pos,
                            'section_type': 'content',
                            'word_count': len(chunk_text.split())
                        })

                # 섹션 헤더
                section_text = text[start_pos:end_pos].strip()
                chunks.append({
                    'text': section_text,
                    'start_pos': start_pos,
                    'end_pos': end_pos,
                    'section_type': 'header',
                    'word_count': len(section_text.split())
                })

                current_pos = end_pos

        # 마지막 부분 추가
        if current_pos < len(text):
            chunk_text = text[current_pos:].strip()
            if chunk_text:
                chunks.append({
                    'text': chunk_text,
                    'start_pos': current_pos,
                    'end_pos': len(text),
                    'section_type': 'content',
                    'word_count': len(chunk_text.split())
                })

        return chunks


# 사용 예시
if __name__ == "__main__":
    processor = TextProcessor()

    sample_text = """
    제1조 (목적)
    이 사업은 이러닝 시스템을 구축하는 것을 목적으로 한다.

    제2조 (정의)
    이 약관에서 사용하는 용어의 정의는 다음과 같다.

    제3조 (기능 요구사항)
    시스템은 다음과 같은 기능을 제공해야 한다.
    1. 사용자 관리 기능
    2. 콘텐츠 관리 기능
    3. 학습 관리 기능
    """

    # 기본 청킹
    chunks = processor.chunk_text(sample_text)
    print(f"기본 청킹 결과: {len(chunks)}개 청크")

    # 섹션 기반 청킹
    section_chunks = processor.chunk_by_sections(sample_text)
    print(f"섹션 청킹 결과: {len(section_chunks)}개 청크")
