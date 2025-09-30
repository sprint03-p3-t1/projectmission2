# projectmission2/src/data_processing/merge_all_json.py

import json
import os
from pathlib import Path
import sys

# ===============================
# 프로젝트 루트 설정
# ===============================
PROJECT_ROOT = Path(__file__).resolve().parents[2]  # projectmission2 루트
sys.path.insert(0, str(PROJECT_ROOT))

# ===============================
# config import
# ===============================
from config.data_preprocess_config import (
    RAW_DIR, PROCESSED_DIR, HWP_PROCESSED_DIR, PDF_PROCESSED_DIR,
    ALL_PDF_JSON_DIR, MAX_CHUNK_SIZE
)

# ===============================
# 파일 경로
# ===============================
HWP_JSON_PATH = HWP_PROCESSED_DIR / "all_processed_hwp.json"
PDF_JSON_PATH = PDF_PROCESSED_DIR / "merged_all_pdfs.json"
OUTPUT_JSON_PATH = PROCESSED_DIR / "all_documents.json"

# ===============================
# 통합 함수
# ===============================
def merge_hwp_pdf_json(hwp_json_path: str | Path, pdf_json_path: str | Path, output_path: str | Path):
    merged_chunks = []

    for path in [hwp_json_path, pdf_json_path]:
        if not os.path.exists(path):
            print(f"⚠️ 파일이 존재하지 않음: {path}")
            continue

        with open(path, 'r', encoding='utf-8') as f:
            data = json.load(f)

            if isinstance(data, dict):
                # HWP JSON 처리
                for file_name, file_chunks in data.items():
                    for c in file_chunks:
                        source_file = c.get('metadata', {}).get('source_file', file_name)
                        c['chunk_id'] = f"{source_file}_{c.get('chunk_id', 'unknown')}"
                        merged_chunks.append(c)

            elif isinstance(data, list):
                # PDF JSON 처리
                for c in data:
                    source_file = c.get('metadata', {}).get('source_file', 'unknown')
                    c['chunk_id'] = f"{source_file}_{c.get('chunk_id', 'unknown')}"
                    merged_chunks.append(c)
            else:
                print(f"⚠️ 알 수 없는 JSON 구조: {path}")

    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    with open(output_path, 'w', encoding='utf-8') as f:
        json.dump(merged_chunks, f, ensure_ascii=False, indent=4)

    print(f"✅ HWP+PDF 통합 JSON 저장 완료: {output_path}")
    print(f"총 청크 수: {len(merged_chunks)}")
    return merged_chunks

# ===============================
# CLI 실행 가능
# ===============================
if __name__ == "__main__":
    merge_hwp_pdf_json(HWP_JSON_PATH, PDF_JSON_PATH, OUTPUT_JSON_PATH)