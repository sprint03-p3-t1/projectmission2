# projectmission2/src/data_processing/pdf/pdf_chunker.py

import re
from typing import List, Dict, Any

def identify_table_type(table_data: Dict[str, Any]) -> str:
    content = table_data.get('raw_content', '').lower()
    rows = table_data.get('rows', [])
    if rows and rows[0]:
        header = ' '.join(cell for cell in rows[0] if cell).lower()
        if any(k in header for k in ['월','일정','추진']):
            return 'schedule'
        if any(k in header for k in ['예산','비용','금액','원']):
            return 'budget'
        if any(k in header for k in ['담당','역할','조직']):
            return 'organization'
    if any(k in content for k in ['일정','월','추진']):
        return 'schedule'
    elif any(k in content for k in ['예산','비용','천원']):
        return 'budget'
    return 'general'

def get_table_priority(table_type: str) -> int:
    return {'schedule':10,'budget':8,'organization':6,'general':4}.get(table_type,4)

def create_table_chunk(table_data: Dict[str, Any]) -> Dict[str, Any]:
    if not table_data.get('raw_content'):
        return None
    table_type = identify_table_type(table_data)
    return {
        'type':'table',
        'subtype':table_type,
        'content': table_data['raw_content'],
        'original_rows': len(table_data.get('rows',[])),
        'metadata': {
            'table_id': table_data.get('id'),
            'source':'pdf_table_extraction',
            'priority': get_table_priority(table_type)
        }
    }

def identify_paragraph_type(content: str) -> str:
    c = content.lower()
    if any(k in c for k in ['제','장','절','항']): return 'header'
    if any(k in c for k in ['목적','개요','배경']): return 'overview'
    if any(k in c for k in ['결론','요약','종합']): return 'conclusion'
    if any(k in c for k in ['방법','절차','과정']): return 'methodology'
    return 'general'

def get_paragraph_priority(ptype: str) -> int:
    return {'overview':9,'conclusion':8,'methodology':7,'header':6,'general':5}.get(ptype,5)

def split_large_paragraph(content: str, max_chunk_size: int) -> List[Dict[str, Any]]:
    chunks=[]
    sentences = re.split(r'[.!?]\s+', content)
    current_chunk,current_size=[],0
    for s in sentences:
        est=len(s)*0.7
        if current_size+est<=max_chunk_size:
            current_chunk.append(s)
            current_size+=est
        else:
            if current_chunk:
                chunks.append({'type':'paragraph','subtype':'split_general','content':'. '.join(current_chunk)+'.','paragraph_count':1,'metadata':{'source':'paragraph_split','priority':5}})
            current_chunk=[s]
            current_size=est
    if current_chunk:
        chunks.append({'type':'paragraph','subtype':'split_general','content':'. '.join(current_chunk)+'.','paragraph_count':1,'metadata':{'source':'paragraph_split','priority':5}})
    return chunks

def create_paragraph_chunk(content_list: List[str]) -> Dict[str, Any]:
    combined = ' '.join(content_list)
    ptype = identify_paragraph_type(combined)
    return {
        'type':'paragraph',
        'subtype':ptype,
        'content': combined,
        'paragraph_count': len(content_list),
        'metadata': {'source':'pdf_paragraph_extraction','priority':get_paragraph_priority(ptype)}
    }

def create_paragraph_chunks(sections: List[Dict[str, Any]], max_chunk_size: int) -> List[Dict[str, Any]]:
    chunks=[]
    current_chunk,current_size=[],0
    for sec in sections:
        content=sec.get('content','')
        if not content.strip(): continue
        est=len(content)*0.7
        if current_size+est<=max_chunk_size:
            current_chunk.append(content)
            current_size+=est
        else:
            if current_chunk:
                chunks.append(create_paragraph_chunk(current_chunk))
            if est>max_chunk_size:
                chunks.extend(split_large_paragraph(content,max_chunk_size))
                current_chunk,current_size=[],0
            else:
                current_chunk=[content]
                current_size=est
    if current_chunk:
        chunks.append(create_paragraph_chunk(current_chunk))
    return chunks

def create_structured_chunks(parsed_data: Dict[str, Any], max_chunk_size: int) -> List[Dict[str, Any]]:
    chunks=[]
    for t in parsed_data.get('tables',[]):
        c=create_table_chunk(t)
        if c: chunks.append(c)
    chunks.extend(create_paragraph_chunks(parsed_data.get('sections',[]), max_chunk_size))
    for i,chunk in enumerate(chunks):
        chunk['chunk_id']=f"chunk_{i:04d}"
    return chunks