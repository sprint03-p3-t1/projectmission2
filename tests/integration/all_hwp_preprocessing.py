# projectmission2/tests/integration/all_hwp_preprocessing.py

import os
import sys
from pathlib import Path
from tqdm import tqdm

# 프로젝트 루트
PROJECT_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(PROJECT_ROOT))

# config import
from config.data_preprocess_config import RAW_DIR, HWP_PROCESSED_DIR, HWP5PROC_EXECUTABLE, ALL_HWP_JSON_DIR, MAX_CHUNK_SIZE

sys.path.insert(0, str(PROJECT_ROOT / "src"))

# 모듈 import
from data_processing.hwp.single.hwp_converter import convert_hwp_single_file
from data_processing.hwp.single.xhtml_parser import parse_hwp5proc_xhtml
from data_processing.hwp.single.chunking import main_chunking
from data_processing.hwp.single.save_json import save_processed_data
from data_processing.hwp.single.load_json import load_chunks_from_json
from data_processing.hwp.single.prepare_chunks import prepare_chunks_for_embedding

def main():
    # all_hwp용 디렉토리 생성
    xhtml_output_base = os.path.join(HWP_PROCESSED_DIR, 'processed_xhtml')
    os.makedirs(xhtml_output_base, exist_ok=True)
    os.makedirs(ALL_HWP_JSON_DIR, exist_ok=True)

    # RAW_DIR 내 모든 HWP 파일
    hwp_files = [f for f in os.listdir(RAW_DIR) if f.lower().endswith(".hwp")]
    
    # 순서 제어
    hwp_files = sorted(hwp_files)            # 알파벳 순 정렬
    # hwp_files = sorted(hwp_files, reverse=True)  # 역순
    # import random; random.shuffle(hwp_files)     # 무작위 순서
    
    print(f"총 {len(hwp_files)}개의 HWP 파일 발견 ✅")

    for hwp_file in tqdm(hwp_files, desc="전체 HWP 처리"):
        hwp_path = os.path.join(RAW_DIR, hwp_file)
        xhtml_output_dir = xhtml_output_base

        # 1️⃣ HWP → XHTML
        convert_hwp_single_file(hwp_path, xhtml_output_dir, hwp5proc_executable=HWP5PROC_EXECUTABLE)
            
        xhtml_file = os.path.join(
            xhtml_output_dir,
            os.path.basename(hwp_file).replace(".hwp", ".xhtml")
        )

        if not os.path.exists(xhtml_file):
            print(f"⚠️ {hwp_file} 변환된 XHTML 없음 → 스킵")
            continue
           
        # 2️⃣ XHTML → 구조화 데이터 파싱
        parsed_data = parse_hwp5proc_xhtml(xhtml_file, verbose=False)

        # 3️⃣ 구조화 데이터 → 청크 생성
        chunks = main_chunking(parsed_data, max_chunk_size=MAX_CHUNK_SIZE, verbose=False)

        # 4️⃣ JSON 저장
        hwp_base = os.path.splitext(hwp_file)[0]
        single_json_filename = f"{hwp_base}_processed.json"
        all_processed_data = {hwp_file: chunks}
        save_processed_data(all_processed_data, ALL_HWP_JSON_DIR, single_json_filename)

    print(f"\n✅ 전체 파이프라인 완료 - 총 {len(hwp_files)}개 파일 처리 완료")

    # 모든 HWP 처리 결과를 하나로 합치기 
    all_processed_path = os.path.join(HWP_PROCESSED_DIR, "all_processed_hwp.json")
    all_hwp_data = {}
    for json_file in os.listdir(ALL_HWP_JSON_DIR):
        if json_file.lower().endswith("_processed.json"):
            file_path = os.path.join(ALL_HWP_JSON_DIR, json_file)
            data = load_chunks_from_json(file_path)
            if data:
                all_hwp_data.update(data)
    
    save_processed_data(all_hwp_data, HWP_PROCESSED_DIR, "all_processed_hwp.json")
    print(f"✅ 모든 HWP JSON을 하나로 합쳐서 저장 완료: {all_processed_path}")
    
    # --- 이제 통합 JSON 기준으로 청크 준비 ---
    prepared_chunks = prepare_chunks_for_embedding(all_hwp_data)
    print(f"✅ 모든 HWP 통합 JSON 기준으로 청크 준비 완료 - 총 {len(prepared_chunks)}개 청크")

if __name__ == "__main__":
    main()