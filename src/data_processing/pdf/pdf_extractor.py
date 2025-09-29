# projectmission2/src/data_processing/pdf/pdf_extractor.py

import pdfplumber
import os
import re
from typing import List, Dict, Any
from tqdm import tqdm

def convert_table_to_natural_language(table_rows: List[List[str]]) -> str:
    if not table_rows or len(table_rows) < 2:
        return ""
    headers = [h if h is not None else "" for h in table_rows[0]]
    content_rows = table_rows[1:]
    natural_text = []

    if any('월' in h for h in headers) or any('일정' in h for h in headers):
        natural_text.append("다음은 추진일정표입니다:")
        for row in content_rows:
            row = [cell if cell is not None else "" for cell in row]
            if len(row) > 1 and row[0].strip():
                task = row[0].strip()
                schedule_info = []
                for i, cell in enumerate(row[1:], 1):
                    if cell.strip() and i < len(headers):
                        header = headers[i]
                        schedule_info.append(f"{header}에 {task}")
                if schedule_info:
                    natural_text.append(". ".join(schedule_info))
    else:
        for row in content_rows:
            row = [cell if cell is not None else "" for cell in row]
            if len(row) >= len(headers):
                row_text = [f"{header}: {cell.strip()}" for header, cell in zip(headers, row) if cell and cell.strip()]
                if row_text:
                    natural_text.append(", ".join(row_text))

    return ". ".join(natural_text)

def extract_text_and_tables(file_path: str) -> Dict[str, Any]:
    parsed_data = {'sections': [], 'tables': [], 'metadata': {}}

    try:
        with pdfplumber.open(file_path) as pdf:
            parsed_data['metadata']['total_pages'] = len(pdf.pages)

            for page_idx, page in enumerate(tqdm(pdf.pages, desc=f"📄 적용 중... {os.path.basename(file_path)}", leave=False)):
                # 표 추출
                for i, table in enumerate(page.extract_tables()):
                    table_data = {
                        'id': f"table_{page_idx}_{i}",
                        'type': 'table',
                        'rows': table,
                        'raw_content': convert_table_to_natural_language(table),
                        'metadata': {'page': page_idx + 1}
                    }
                    if table_data['raw_content']:
                        parsed_data['tables'].append(table_data)

                # 텍스트 추출
                page_text = page.extract_text(x_tolerance=2, y_tolerance=2, layout=True)
                if not page_text:
                    continue
                paragraphs = [para for para in page_text.split('\n\n') if para.strip()]
                for i, para in enumerate(paragraphs):
                    parsed_data['sections'].append({
                        'id': f"para_{page_idx}_{i}",
                        'type': 'paragraph',
                        'content': para.strip(),
                        'metadata': {'page': page_idx + 1}
                    })

    except Exception as e:
        print(f"PDF 파싱 중 오류 발생: {e}")

    parsed_data['metadata']['total_paragraphs'] = len(parsed_data['sections'])
    parsed_data['metadata']['total_tables'] = len(parsed_data['tables'])
    return parsed_data