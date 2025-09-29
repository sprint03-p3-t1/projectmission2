# projectmission2/src/data_processing/pdf/json_merger.py

import os
import json
from typing import List, Dict

def merge_pdf_jsons(json_dir: str, output_file: str) -> List[Dict]:
    """
    디렉터리 내 모든 PDF JSON 파일을 합치고, 각 청크 ID를 고유화합니다.
    
    Args:
        json_dir (str): 개별 PDF JSON 파일들이 있는 디렉터리
        output_file (str): 합쳐진 JSON 파일 경로

    Returns:
        List[Dict]: 합쳐진 청크 리스트
    """
    merged_chunks = []
    json_files = [f for f in os.listdir(json_dir) if f.lower().endswith('.json')]
    
    for jf in json_files:
        path = os.path.join(json_dir, jf)
        with open(path, 'r', encoding='utf-8') as f:
            chunks = json.load(f)
            for c in chunks:
                # chunk_id 고유화: 파일명 + 기존 chunk_id
                base_name = os.path.splitext(jf)[0]
                c['chunk_id'] = f"{base_name}_{c['chunk_id']}"
                merged_chunks.append(c)
    
    # 합친 JSON 저장
    os.makedirs(os.path.dirname(output_file), exist_ok=True)
    with open(output_file, 'w', encoding='utf-8') as f:
        json.dump(merged_chunks, f, ensure_ascii=False, indent=4)
    
    print(f"✅ 총 {len(merged_chunks)}개의 청크를 합쳐서 저장 완료: {output_file}")
    return merged_chunks