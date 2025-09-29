# projectmission2/config/data_preprocess_config.py

import os
from pathlib import Path

# 프로젝트 기본 경로
PROJECT_ROOT = Path(__file__).resolve().parents[1]

# 원본 HWP 파일이 있는 디렉터리
RAW_DIR = PROJECT_ROOT / "data/raw/files"

# ========================================================================================================
# 전처리된 HWP 데이터가 저장될 디렉터리
HWP_PROCESSED_DIR = PROJECT_ROOT / "data/processed/datapreprocessingbjs(hwp5proc)"

# hwp5proc 실행 파일 경로 (사용자 환경에 맞게 수정 필요)
HWP5PROC_EXECUTABLE = "/home/spai0323/myenv/bin/hwp5proc"

# HWP JSON 저장 설정
ALL_HWP_JSON_DIR = PROJECT_ROOT / "data/processed/datapreprocessingbjs(hwp5proc)/processed_json"

# =============================================================================================================

# 전처리된 PDF 데이터가 저장될 디렉터리
PDF_PROCESSED_DIR = PROJECT_ROOT / "data/processed/datapreprocessingbjs(pdfplumber)"

# PDF TEXT 저장 설정
ALL_PDF_TEXT_DIR = PROJECT_ROOT / "data/processed/datapreprocessingbjs(pdfplumber)/text_all_files"

# PDF JSON 저장 설정
ALL_PDF_JSON_DIR = PROJECT_ROOT / "data/processed/datapreprocessingbjs(pdfplumber)/pdf_processed_json"

# ===============================================================================================================

# 한 청크당 최대 토큰 수 (문단 분할 기준)
MAX_CHUNK_SIZE = 1000

# ------------------------
# 기타 설정
# ------------------------
