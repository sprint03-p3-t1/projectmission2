"""
개선된 HWP 텍스트 추출기
올바른 HWP 텍스트 추출을 위한 전용 모듈
"""

import os
import re
import zlib
import struct
import logging
from typing import List, Dict, Optional, Tuple
from pathlib import Path
import unicodedata

try:
    import olefile
    HWP_AVAILABLE = True
except ImportError:
    HWP_AVAILABLE = False

logger = logging.getLogger(__name__)


class HWPExtractor:
    """HWP 파일 텍스트 추출기"""
    
    def __init__(self):
        if not HWP_AVAILABLE:
            raise ImportError("olefile이 설치되지 않았습니다. 'pip install olefile' 실행해주세요.")
    
    def extract_text(self, file_path: Path) -> Tuple[str, Dict]:
        """
        HWP 파일에서 텍스트 추출
        
        Args:
            file_path: HWP 파일 경로
            
        Returns:
            (추출된 텍스트, 메타데이터)
        """
        try:
            if self._is_hwpx_zip(file_path):
                return self._extract_hwpx_text(file_path)
            elif self._is_hwp_ole(file_path):
                return self._extract_hwp_ole_text(file_path)
            else:
                raise ValueError(f"지원하지 않는 HWP 파일 형식: {file_path}")
        except Exception as e:
            logger.error(f"HWP 텍스트 추출 실패: {file_path.name} - {str(e)}")
            return "", {"error": str(e), "file": str(file_path)}
    
    def _is_hwp_ole(self, path: Path) -> bool:
        """OLE 기반 HWP 파일 확인"""
        try:
            with path.open("rb") as f:
                head = f.read(8)
            return head.startswith(b"\xD0\xCF\x11\xE0")
        except Exception:
            return False
    
    def _is_hwpx_zip(self, path: Path) -> bool:
        """HWPX (ZIP) 파일 확인"""
        try:
            import zipfile
            return zipfile.is_zipfile(path)
        except Exception:
            return False
    
    def _extract_hwp_ole_text(self, path: Path) -> Tuple[str, Dict]:
        """OLE 기반 HWP 파일에서 텍스트 추출"""
        with olefile.OleFileIO(path) as ole:
            header = self._read_ole_stream(ole, "FileHeader") or b""
            is_compressed = self._is_compressed_from_header(header)
            
            texts: List[str] = []
            
            # 1) PrvText 프리뷰 텍스트
            prv = self._decompress_if_needed(
                self._read_ole_stream(ole, "PrvText"), is_compressed
            )
            if prv:
                try:
                    text = prv.decode("utf-16le", errors="ignore")
                    if self._is_meaningful_text(text):
                        texts.append(text)
                except Exception:
                    text = self._utf16_sweep_recover(prv)
                    if self._is_meaningful_text(text):
                        texts.append(text)
            
            # 2) BodyText/Section* 본문 텍스트
            for entry in ole.listdir(streams=True, storages=True):
                if (len(entry) == 2 and 
                    entry[0] == 'BodyText' and 
                    entry[1].startswith('Section')):
                    
                    raw = self._read_ole_stream(ole, "/".join(entry))
                    data = self._decompress_if_needed(raw, is_compressed)
                    text = self._utf16_sweep_recover(data)
                    
                    if self._is_meaningful_text(text):
                        texts.append(text)
            
            # 3) DocInfo 문서 정보
            docinfo = self._decompress_if_needed(
                self._read_ole_stream(ole, "DocInfo"), is_compressed
            )
            if docinfo:
                text = self._utf16_sweep_recover(docinfo)
                if self._is_meaningful_text(text):
                    texts.append(text)
        
        full_text = "\n\n".join(texts)
        meta = {
            "format": "hwp-ole", 
            "compressed": bool(is_compressed), 
            "file": str(path),
            "sections_found": len(texts)
        }
        
        return self._clean_extracted_text(full_text), meta
    
    def _extract_hwpx_text(self, path: Path) -> Tuple[str, Dict]:
        """HWPX (ZIP) 파일에서 텍스트 추출"""
        import zipfile
        
        with zipfile.ZipFile(path, 'r') as zf:
            names = [n for n in zf.namelist() if n.lower().endswith(".xml")]
            pref = [n for n in names if "contents/" in n.lower() and "section" in n.lower()]
            targets = pref or names
            
            texts: List[str] = []
            for n in sorted(targets):
                try:
                    raw = zf.read(n)
                    # XML 태그 제거
                    raw = re.sub(rb"<[^>]+>", b" ", raw)
                    
                    try:
                        text = raw.decode("utf-8", errors="ignore")
                    except Exception:
                        text = raw.decode("cp949", errors="ignore")
                    
                    if self._is_meaningful_text(text):
                        texts.append(text)
                except KeyError:
                    continue
        
        full_text = "\n\n".join(texts)
        meta = {
            "format": "hwpx-zipxml", 
            "file": str(path),
            "xml_files_processed": len(texts)
        }
        
        return self._clean_extracted_text(full_text), meta
    
    def _read_ole_stream(self, ole, stream_name: str) -> Optional[bytes]:
        """OLE 스트림 읽기"""
        if ole.exists(stream_name):
            with ole.openstream(stream_name) as s:
                return s.read()
        return None
    
    def _is_compressed_from_header(self, hdr: bytes) -> bool:
        """FileHeader에서 압축 여부 확인"""
        if len(hdr) > 37:
            return bool(hdr[36] & 0x01)
        return False
    
    def _decompress_if_needed(self, data: Optional[bytes], is_compressed: bool) -> bytes:
        """필요 시 압축 해제"""
        if not data:
            return b""
        if not is_compressed:
            return data
        
        try:
            return zlib.decompress(data, -15)  # raw deflate
        except zlib.error:
            try:
                return zlib.decompress(data)   # zlib header
            except zlib.error:
                return data  # 압축 해제 실패 시 원본 반환
    
    def _utf16_sweep_recover(self, data: bytes) -> str:
        """UTF-16LE 디코딩 시도"""
        if not data:
            return ""
        
        # 직접 디코드 시도
        try:
            return data.decode("utf-16le", errors="ignore")
        except Exception:
            pass
        
        # 오프셋 0, 1에서 스윕 시도
        out = []
        view = memoryview(data)
        for offset in (0, 1):
            chunk = view[offset:]
            try:
                text = chunk.tobytes().decode("utf-16le", errors="ignore")
                if text and self._is_meaningful_text(text):
                    out.append(text)
            except Exception:
                continue
        
        return "\n".join(out)
    
    def _is_meaningful_text(self, text: str) -> bool:
        """의미 있는 텍스트인지 확인"""
        if not text or len(text.strip()) < 10:
            return False
        
        # 한글 문자 비율 확인
        korean_chars = len(re.findall(r'[가-힣]', text))
        total_chars = len(text.strip())
        
        if total_chars == 0:
            return False
        
        korean_ratio = korean_chars / total_chars
        
        # 한글이 5% 이상 포함되거나, 영문이 많이 포함된 경우
        english_chars = len(re.findall(r'[a-zA-Z]', text))
        english_ratio = english_chars / total_chars
        
        return korean_ratio >= 0.05 or english_ratio >= 0.3
    
    def _clean_extracted_text(self, text: str) -> str:
        """추출된 텍스트 정제"""
        if not text:
            return ""
        
        # 유니코드 정규화
        text = unicodedata.normalize("NFKC", text)
        
        # 제어 문자 제거
        text = re.sub(r'[\x00-\x1f\x7f-\x9f]', '', text)
        
        # 연속된 공백 정리
        text = re.sub(r'\s+', ' ', text)
        text = re.sub(r'\n\s*\n', '\n\n', text)  # 빈 줄 정리
        
        # 의미 없는 반복 문자 제거
        text = re.sub(r'(.)\1{10,}', r'\1\1\1', text)  # 10개 이상 반복 → 3개로
        
        return text.strip()


# 사용 예시
if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)
    
    extractor = HWPExtractor()
    
    # 테스트 파일 경로
    test_file = Path("../data/processed/hwp").glob("*.hwp")
    
    for hwp_file in list(test_file)[:3]:  # 처음 3개만 테스트
        print(f"\n=== {hwp_file.name} ===")
        text, meta = extractor.extract_text(hwp_file)
        
        print(f"메타데이터: {meta}")
        print(f"텍스트 길이: {len(text)} 글자")
        
        if text:
            # 한글 포함 여부 확인
            korean_count = len(re.findall(r'[가-힣]', text))
            print(f"한글 글자 수: {korean_count}")
            print(f"미리보기: {text[:200]}...")
        else:
            print("텍스트 추출 실패")

