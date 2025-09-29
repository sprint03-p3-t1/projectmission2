# projectmission2/tests/unit/single_hwp_preprocessing.py

import os
import sys
from pathlib import Path

# 프로젝트 루트
PROJECT_ROOT = Path(__file__).resolve().parents[2]  
sys.path.insert(0, str(PROJECT_ROOT))

# ✅ config/single_data_preprocess_config.py import
from config.data_preprocess_config import (
    RAW_DIR, HWP_PROCESSED_DIR, HWP5PROC_EXECUTABLE,
    ALL_HWP_JSON_DIR, MAX_CHUNK_SIZE
)

SRC_DIR = PROJECT_ROOT / "src"
sys.path.insert(0, str(SRC_DIR))

# ✅ data_processing/hwp 모듈 import
from data_processing.hwp.single.hwp_converter import convert_hwp_single_file
from data_processing.hwp.single.xhtml_parser import parse_hwp5proc_xhtml
from data_processing.hwp.single.chunking import main_chunking
from data_processing.hwp.single.save_json import save_processed_data
from data_processing.hwp.single.load_json import load_chunks_from_json
from data_processing.hwp.single.prepare_chunks import prepare_chunks_for_embedding

def main():
    # 1️⃣ HWP 파일 경로
    hwp_file = os.path.join(RAW_DIR, "(재)예술경영지원센터_통합 정보시스템 구축 사전 컨설팅.hwp")
    xhtml_output_dir = os.path.join(HWP_PROCESSED_DIR, 'processed_xhtml')
    os.makedirs(xhtml_output_dir, exist_ok=True)

    print("=== 1. HWP → XHTML 변환 ===")
    convert_hwp_single_file(hwp_file, xhtml_output_dir, hwp5proc_executable=HWP5PROC_EXECUTABLE)

    # 변환된 XHTML 파일 경로
    xhtml_file = os.path.join(
        xhtml_output_dir,
        os.path.basename(hwp_file).replace(".hwp", ".xhtml")
    )

    # 2️⃣ XHTML 파싱
    print("\n=== 2. XHTML → 구조화 데이터 파싱 ===")
    parsed_data = parse_hwp5proc_xhtml(xhtml_file)

    # 3️⃣ 청킹
    print("\n=== 3. 구조화 데이터 → 청크 생성 ===")
    chunks = main_chunking(parsed_data)

    # 🔑 JSON 파일명 (동적으로 생성)
    hwp_base = os.path.splitext(os.path.basename(hwp_file))[0]
    single_json_filename = f"{hwp_base}_processed.json"

    # 🔑 ALL_HWP_JSON_DIR 디렉터리 생성 (없으면 자동 생성)
    os.makedirs(ALL_HWP_JSON_DIR, exist_ok=True)

    # 4️⃣ JSON 저장 (all_json/{파일명}_processed.json)
    print("\n=== 4. 처리 결과 JSON 저장 ===")
    all_processed_data = {os.path.basename(hwp_file): chunks}
    save_processed_data(all_processed_data, ALL_HWP_JSON_DIR, single_json_filename)

    # 5️⃣ JSON 로드 & 청크 준비
    print("\n=== 5. JSON 로드 및 임베딩 준비 ===")
    loaded_data = load_chunks_from_json(os.path.join(ALL_HWP_JSON_DIR, single_json_filename))
    if loaded_data is None:
        print("데이터 로드 실패, 종료합니다.")
        return

    prepared_chunks = prepare_chunks_for_embedding(loaded_data)   
    
    print(f"✅ 전체 파이프라인 완료 - 총 {len(prepared_chunks)}개 청크 준비 완료")

if __name__ == "__main__":
    main()