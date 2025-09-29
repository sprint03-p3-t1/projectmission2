# projectmission2/src/data_processing/hwp/single/chunking.py

import re
from typing import List, Dict, Any

def create_structured_chunks(parsed_data: Dict[str, Any], max_chunk_size: int = 1000) -> List[Dict[str, Any]]:
    chunks = []

    # 1. 표 데이터 청킹
    for table in parsed_data.get('tables', []):
        table_chunk = create_table_chunk(table)
        if table_chunk:
            chunks.append(table_chunk)

    # 2. 문단 데이터 청킹
    paragraph_chunks = create_paragraph_chunks(parsed_data.get('sections', []), max_chunk_size)
    chunks.extend(paragraph_chunks)

    # 3. 청크 ID 부여
    for i, chunk in enumerate(chunks):
        chunk['chunk_id'] = f"chunk_{i:04d}"

    return chunks

# --- 표 청킹 ---
def create_table_chunk(table_data: Dict[str, Any]) -> Dict[str, Any] | None:
    if not table_data.get('raw_content'):
        return None

    table_type = identify_table_type(table_data)

    return {
        'type': 'table',
        'subtype': table_type,
        'content': table_data['raw_content'],
        'original_rows': len(table_data.get('rows', [])),
        'metadata': {
            'table_id': table_data.get('id'),
            'source': 'table_extraction',
            'priority': get_table_priority(table_type)
        }
    }

def identify_table_type(table_data: Dict[str, Any]) -> str:
    content = table_data.get('raw_content', '').lower()
    rows = table_data.get('rows', [])

    if rows:
        header = ' '.join(rows[0]).lower()
        if any(k in header for k in ['월', '일정', '추진']):
            return 'schedule'
        if any(k in header for k in ['예산', '비용', '금액', '원']):
            return 'budget'
        if any(k in header for k in ['담당', '역할', '조직']):
            return 'organization'

    if any(k in content for k in ['일정', '월', '추진']):
        return 'schedule'
    if any(k in content for k in ['예산', '비용', '천원', '만원', '십만원']):
        return 'budget'
    return 'general'

def get_table_priority(table_type: str) -> int:
    return {'schedule':10, 'budget':8, 'organization':6, 'general':4}.get(table_type, 4)

# --- 문단 청킹 ---
def create_paragraph_chunks(sections: List[Dict[str, Any]], max_chunk_size: int) -> List[Dict[str, Any]]:
    chunks = []
    current_chunk = []
    current_size = 0

    for section in sections:
        content = section.get('content', '')
        if not content.strip():
            continue

        estimated_tokens = len(content) * 0.7

        if current_size + estimated_tokens <= max_chunk_size:
            current_chunk.append(content)
            current_size += estimated_tokens
        else:
            if current_chunk:
                chunks.append(create_paragraph_chunk(current_chunk))
            if estimated_tokens > max_chunk_size:
                chunks.extend(split_large_paragraph(content, max_chunk_size))
                current_chunk = []
                current_size = 0
            else:
                current_chunk = [content]
                current_size = estimated_tokens

    if current_chunk:
        chunks.append(create_paragraph_chunk(current_chunk))

    return chunks

def create_paragraph_chunk(content_list: List[str]) -> Dict[str, Any]:
    combined_content = ' '.join(content_list)
    paragraph_type = identify_paragraph_type(combined_content)
    return {
        'type': 'paragraph',
        'subtype': paragraph_type,
        'content': combined_content,
        'paragraph_count': len(content_list),
        'metadata': {
            'source': 'paragraph_extraction',
            'priority': get_paragraph_priority(paragraph_type)
        }
    }

def identify_paragraph_type(content: str) -> str:
    content_lower = content.lower()
    if any(k in content_lower for k in ['제', '장', '절', '항']):
        return 'header'
    if any(k in content_lower for k in ['목적', '개요', '배경']):
        return 'overview'
    if any(k in content_lower for k in ['결론', '요약', '종합']):
        return 'conclusion'
    if any(k in content_lower for k in ['방법', '절차', '과정']):
        return 'methodology'
    return 'general'

def get_paragraph_priority(paragraph_type: str) -> int:
    return {'overview':9, 'conclusion':8, 'methodology':7, 'header':6, 'general':5}.get(paragraph_type, 5)

def split_large_paragraph(content: str, max_chunk_size: int) -> List[Dict[str, Any]]:
    chunks = []
    sentences = re.split(r'[.!?]\s+', content)
    current_chunk = []
    current_size = 0

    for sentence in sentences:
        estimated_tokens = len(sentence) * 0.7
        if current_size + estimated_tokens <= max_chunk_size:
            current_chunk.append(sentence)
            current_size += estimated_tokens
        else:
            if current_chunk:
                chunk_content = '. '.join(current_chunk) + '.'
                chunks.append({
                    'type':'paragraph', 'subtype':'split_general',
                    'content': chunk_content, 'paragraph_count':1,
                    'metadata': {'source':'paragraph_split', 'priority':5}
                })
            current_chunk = [sentence]
            current_size = estimated_tokens

    if current_chunk:
        chunk_content = '. '.join(current_chunk) + '.'
        chunks.append({
            'type':'paragraph', 'subtype':'split_general',
            'content': chunk_content, 'paragraph_count':1,
            'metadata': {'source':'paragraph_split', 'priority':5}
        })

    return chunks

# --- 청크 분석 ---
def analyze_chunks(chunks: List[Dict[str, Any]]) -> None:
    type_counts = {}
    subtype_counts = {}
    priority_counts = {}

    for chunk in chunks:
        t = chunk['type']
        s = chunk['subtype']
        p = chunk['metadata']['priority']
        type_counts[t] = type_counts.get(t,0)+1
        subtype_counts[s] = subtype_counts.get(s,0)+1
        priority_counts[p] = priority_counts.get(p,0)+1

    print("\n=== 청크 분석 ===")
    print("타입별 분포:", type_counts)
    print("세부 타입별 분포:", subtype_counts)
    print("우선순위별 분포:", priority_counts)

def main_chunking(parsed_result: Dict[str, Any], max_chunk_size:int=1000, verbose=True) -> List[Dict[str,Any]]:
    print("청킹 시작")
    chunks = create_structured_chunks(parsed_result, max_chunk_size=max_chunk_size)
    if verbose:
        analyze_chunks(chunks)
    print("청킹 완료")
    return chunks