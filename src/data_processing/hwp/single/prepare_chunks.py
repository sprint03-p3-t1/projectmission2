# projectmission2/src/data_processing/hwp/single/prepare_chunks.py

import os
from tqdm import tqdm

def prepare_chunks_for_embedding(single_processed_data: dict):
    chunks = []
    for file_name, file_chunks in single_processed_data.items():
        for chunk in file_chunks:
            chunk['metadata']['source_file'] = file_name
            chunks.append(chunk)

    # 청크 ID 고유화
    for idx, chunk in enumerate(tqdm(chunks, desc="청크 ID 고유화")):
        file_name = chunk['metadata'].get("source_file", "unknown_file")
        original_id = chunk.get('chunk_id', f"chunk_{idx:04d}")
        chunk['chunk_id'] = f"{os.path.basename(file_name)}_{original_id}"

    print(f"총 {len(chunks)}개의 청크 준비 완료 ✅")
    return chunks