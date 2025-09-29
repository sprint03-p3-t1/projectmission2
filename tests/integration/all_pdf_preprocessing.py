# projectmission2/tests/integration/all_pdf_preprocessing.py

import os
import sys
from pathlib import Path
from tqdm import tqdm

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
from src.data_processing.pdf.json_merger import merge_pdf_jsons

def main():
    
    # RAW_DIR 내 모든 PDF 파일
    pdf_files = [f for f in os.listdir(RAW_DIR) if f.lower().endswith(".pdf")]    
    print(f"총 {len(pdf_files)}개의 PDF 파일 발견 ✅")

    os.makedirs(ALL_PDF_JSON_DIR, exist_ok=True)

    for pdf_file in tqdm(pdf_files, desc="전체 PDF 처리"):
        print(f"\n🚀 [시작] {pdf_file}")
        pdf_path = os.path.join(RAW_DIR, pdf_file)
        file_base_name = os.path.splitext(pdf_file)[0]

        process_single_pdf(
            file_name=pdf_file,
            raw_dir=str(RAW_DIR),
            all_pdf_json_dir=str(ALL_PDF_JSON_DIR),
            max_chunk_size=MAX_CHUNK_SIZE
        )
        print(f"✅ [완료] {pdf_file} \n")

    merged_output_file = os.path.join(ALL_PDF_JSON_DIR, "merged_all_pdfs.json")
    
    # 이전 병합 JSON 삭제
    if os.path.exists(merged_output_file):
        os.remove(merged_output_file)
        
    merge_pdf_jsons(ALL_PDF_JSON_DIR, merged_output_file)

if __name__ == "__main__":
    main()