# projectmission2/tests/unit/single_pdf_preprocessing.py

import os
import sys
from pathlib import Path

# 프로젝트 루트
PROJECT_ROOT = Path(__file__).resolve().parents[2]  
sys.path.insert(0, str(PROJECT_ROOT))

# ✅ config import
from config.data_preprocess_config import (
    RAW_DIR, PDF_PROCESSED_DIR,
    ALL_PDF_JSON_DIR, MAX_CHUNK_SIZE
)

SRC_DIR = PROJECT_ROOT / "src"
sys.path.insert(0, str(SRC_DIR))

# PDF 파이프라인 import
from src.data_processing.pdf.pdf_pipeline import process_single_pdf
from src.data_processing.pdf.pdf_extractor import extract_text_and_tables
from src.data_processing.pdf.pdf_chunker import create_structured_chunks
from src.data_processing.pdf.pdf_analyzer import analyze_chunks

# 처리할 파일명
pdf_file = "대전대학교_대전대학교 2024학년도 다층적 융합 학습경험 플랫폼(MILE) 전.pdf"

# process_single_pdf 호출 시 config에서 가져온 경로를 인자로 전달

def main():
    
    process_single_pdf(
        file_name=pdf_file,
        raw_dir=str(RAW_DIR),
        all_pdf_json_dir=str(ALL_PDF_JSON_DIR),
        max_chunk_size=MAX_CHUNK_SIZE
    )

if __name__ == "__main__":
    main()