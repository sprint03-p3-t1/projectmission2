# projectmission2/src/data_processing/pdf/pdf_pipeline.py

import os
import json
from .pdf_extractor import extract_text_and_tables
from .pdf_chunker import create_structured_chunks
from .pdf_analyzer import analyze_chunks

def save_chunks_to_json(chunks, output_path):
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    with open(output_path,'w',encoding='utf-8') as f:
        json.dump(chunks,f,ensure_ascii=False,indent=4)

def process_single_pdf(
    file_name: str,
    raw_dir: str,
    all_pdf_json_dir: str,
    max_chunk_size: int
):
    """
    단일 PDF를 처리하고 JSON으로 저장합니다.
    config를 내부에서 import하지 않고, 외부에서 경로를 전달받음
    """
    input_path = os.path.join(raw_dir, file_name)
    if not os.path.exists(input_path):
        print(f"파일 없음: {input_path}")
        return

    # PDF 파싱
    parsed = extract_text_and_tables(input_path)
    
    # 청크 생성
    chunks = create_structured_chunks(parsed, max_chunk_size)
    
    # 청크 ID와 메타데이터 처리
    for c in chunks:
        c['metadata']['source_file'] = file_name
        c['chunk_id'] = f"{file_name}_{c['chunk_id']}"
    
    # 청크 분석
    analyze_chunks(chunks)
    
    # JSON 저장
    base_name = os.path.splitext(file_name)[0]  # 확장자 제거
    output_path = os.path.join(all_pdf_json_dir, f"{base_name}.json")
    save_chunks_to_json(chunks, output_path)
    print(f"✅ 저장 완료: {output_path}")
