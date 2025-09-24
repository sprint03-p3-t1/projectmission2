"""
문서 로더 모듈
PDF 및 HWP 파일을 로드하고 텍스트를 추출합니다.
"""

import os
import logging
from typing import List, Dict, Optional
from pathlib import Path
import pandas as pd

# PDF 처리
try:
    import PyPDF2
    PDF_AVAILABLE = True
except ImportError:
    PDF_AVAILABLE = False
    print("PyPDF2가 설치되지 않았습니다. PDF 파일 처리를 위해 설치해주세요.")

# HWP 처리 (olefile 필요)
try:
    import olefile
    HWP_AVAILABLE = True
except ImportError:
    HWP_AVAILABLE = False
    print("olefile이 설치되지 않았습니다. HWP 파일 처리를 위해 설치해주세요.")

logger = logging.getLogger(__name__)


class DocumentLoader:
    """RFP 문서 로더 클래스"""

    def __init__(self, data_dir: str = "data/raw"):
        """
        DocumentLoader 초기화

        Args:
            data_dir: 원본 데이터가 저장된 디렉토리 경로
        """
        self.data_dir = Path(data_dir)
        self.supported_extensions = ['.pdf', '.hwp']

        if not self.data_dir.exists():
            logger.warning(f"데이터 디렉토리가 존재하지 않습니다: {self.data_dir}")
            self.data_dir.mkdir(parents=True, exist_ok=True)

    def load_documents(self) -> List[Dict]:
        """
        지정된 디렉토리에서 모든 RFP 문서를 로드합니다.

        Returns:
            문서 정보 리스트 (파일명, 내용, 메타데이터 등)
        """
        documents = []

        for file_path in self.data_dir.rglob('*'):
            if file_path.suffix.lower() in self.supported_extensions:
                try:
                    doc_info = self._load_single_document(file_path)
                    if doc_info:
                        documents.append(doc_info)
                        logger.info(f"문서 로드 성공: {file_path.name}")
                except Exception as e:
                    logger.error(f"문서 로드 실패: {file_path.name} - {str(e)}")

        logger.info(f"총 {len(documents)}개 문서 로드 완료")
        return documents

    def _load_single_document(self, file_path: Path) -> Optional[Dict]:
        """
        단일 문서를 로드합니다.

        Args:
            file_path: 문서 파일 경로

        Returns:
            문서 정보 딕셔너리 또는 None (실패 시)
        """
        file_extension = file_path.suffix.lower()

        if file_extension == '.pdf':
            return self._load_pdf(file_path)
        elif file_extension == '.hwp':
            return self._load_hwp(file_path)
        else:
            logger.warning(f"지원하지 않는 파일 형식: {file_extension}")
            return None

    def _load_pdf(self, file_path: Path) -> Optional[Dict]:
        """
        PDF 파일을 로드합니다.

        Args:
            file_path: PDF 파일 경로

        Returns:
            문서 정보 딕셔너리
        """
        if not PDF_AVAILABLE:
            logger.error("PDF 처리를 위한 PyPDF2가 설치되지 않았습니다.")
            return None

        try:
            with open(file_path, 'rb') as file:
                pdf_reader = PyPDF2.PdfReader(file)
                text = ""

                for page in pdf_reader.pages:
                    text += page.extract_text() + "\n"

                return {
                    'file_name': file_path.name,
                    'file_path': str(file_path),
                    'content': text.strip(),
                    'file_type': 'pdf',
                    'page_count': len(pdf_reader.pages),
                    'metadata': {}
                }
        except Exception as e:
            logger.error(f"PDF 파일 로드 실패: {file_path.name} - {str(e)}")
            return None

    def _load_hwp(self, file_path: Path) -> Optional[Dict]:
        """
        HWP 파일을 로드합니다.

        Args:
            file_path: HWP 파일 경로

        Returns:
            문서 정보 딕셔너리
        """
        if not HWP_AVAILABLE:
            logger.error("HWP 처리를 위한 olefile이 설치되지 않았습니다.")
            return None

        try:
            # 개선된 HWP 추출기 사용
            from .hwp_extractor import HWPExtractor
            
            extractor = HWPExtractor()
            text, extract_meta = extractor.extract_text(file_path)

            return {
                'file_name': file_path.name,
                'file_path': str(file_path),
                'content': text.strip(),
                'file_type': 'hwp',
                'page_count': None,
                'metadata': extract_meta
            }
        except Exception as e:
            logger.error(f"HWP 파일 로드 실패: {file_path.name} - {str(e)}")
            return None

    def _extract_text_from_hwp(self, ole_file) -> str:
        """
        레거시 HWP 텍스트 추출 메서드 (하위 호환성을 위해 유지)
        새로운 구현에서는 hwp_extractor.HWPExtractor 사용 권장

        Args:
            ole_file: OLE 파일 객체

        Returns:
            추출된 텍스트
        """
        logger.warning("레거시 HWP 추출 메서드 사용 중. hwp_extractor.HWPExtractor 사용을 권장합니다.")
        
        text = ""
        try:
            # 기본적인 텍스트 추출 시도
            if ole_file.exists('BodyText'):
                body_text = ole_file.openstream('BodyText').read()
                text = body_text.decode('utf-16', errors='ignore')
        except Exception as e:
            logger.warning(f"HWP 텍스트 추출 실패: {str(e)}")

        return text

    def load_metadata(self, metadata_file: str = "data/data_list.csv") -> pd.DataFrame:
        """
        메타데이터 CSV 파일을 로드합니다.

        Args:
            metadata_file: 메타데이터 파일 경로

        Returns:
            메타데이터 DataFrame
        """
        try:
            metadata_path = Path(metadata_file)
            if metadata_path.exists():
                df = pd.read_csv(metadata_path)
                logger.info(f"메타데이터 로드 성공: {len(df)}개 레코드")
                return df
            else:
                logger.warning(f"메타데이터 파일이 존재하지 않습니다: {metadata_file}")
                return pd.DataFrame()
        except Exception as e:
            logger.error(f"메타데이터 로드 실패: {str(e)}")
            return pd.DataFrame()


# 사용 예시
if __name__ == "__main__":
    loader = DocumentLoader()
    documents = loader.load_documents()
    metadata = loader.load_metadata()

    print(f"로드된 문서 수: {len(documents)}")
    print(f"메타데이터 레코드 수: {len(metadata)}")
