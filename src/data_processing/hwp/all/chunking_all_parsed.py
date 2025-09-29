# projectmission2/src/data_processing/hwp/all/chunking_all_parsed.py

from typing import Dict, Any, List
import re

def create_structured_chunks(parsed_data: Dict[str, Any], max_chunk_size: int = 1000) -> List[Dict[str, Any]]:
    """
    파싱된 데이터를 구조화된 청크로 변환
    
    Args:
        parsed_data: parse_hwp5proc_xhtml 함수의 결과
        max_chunk_size: 최대 청크 크기 (토큰 수 기준)
    
    Returns:
        List[Dict]: 구조화된 청크들의 리스트
    """
    chunks = []
    
    print("청킹 작업 시작...")
    
    # 1. 표 데이터 청킹
    for table in parsed_data['tables']:
        table_chunk = create_table_chunk(table)
        if table_chunk:
            chunks.append(table_chunk)
    
    # 2. 문단 데이터 청킹
    paragraph_chunks = create_paragraph_chunks(parsed_data['sections'], max_chunk_size)
    chunks.extend(paragraph_chunks)
    
    # 3. 청크에 고유 ID 부여
    for i, chunk in enumerate(chunks):
        chunk['chunk_id'] = f"chunk_{i:04d}"
    
    print(f"청킹 완료 - 총 {len(chunks)}개 청크 생성")
    return chunks

def create_table_chunk(table_data: Dict[str, Any]) -> Dict[str, Any]:
    """
    표 데이터를 청크로 변환
    """
    if not table_data.get('raw_content') or not table_data['raw_content'].strip():
        return None
    
    # 표 유형 판단
    table_type = identify_table_type(table_data)
    
    chunk = {
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
    
    return chunk

def identify_table_type(table_data: Dict[str, Any]) -> str:
    """
    표의 유형을 식별 (일정표, 예산표, 일반표 등)
    """
    content = table_data.get('raw_content', '').lower()
    rows = table_data.get('rows', [])
    
    # 헤더 행 확인
    if rows:
        header = ' '.join(rows[0]).lower()
        
        # 일정표 판단
        if any(keyword in header for keyword in ['월', '일정', '추진']):
            return 'schedule'
        
        # 예산표 판단
        if any(keyword in header for keyword in ['예산', '비용', '금액', '원']):
            return 'budget'
        
        # 조직표 판단
        if any(keyword in header for keyword in ['담당', '역할', '조직']):
            return 'organization'
    
    # 내용 기반 판단
    if any(keyword in content for keyword in ['일정', '월', '추진']):
        return 'schedule'
    elif any(keyword in content for keyword in ['예산', '비용', '천원', '만원', '십만원']):
        return 'budget'
    else:
        return 'general'

def get_table_priority(table_type: str) -> int:
    """
    표 유형별 우선순위 설정
    """
    priority_map = {
        'schedule': 10,  # 일정표는 높은 우선순위
        'budget': 8,
        'organization': 6,
        'general': 4
    }
    return priority_map.get(table_type, 4)

def create_paragraph_chunks(sections: List[Dict[str, Any]], max_chunk_size: int) -> List[Dict[str, Any]]:
    """
    문단 데이터를 적절한 크기의 청크로 분할
    """
    chunks = []
    current_chunk = []
    current_size = 0
    
    for section in sections:
        content = section.get('content', '')
        if not content.strip():
            continue
        
        # 대략적인 토큰 수 계산 (한국어: 글자수 * 0.7)
        estimated_tokens = len(content) * 0.7
        
        # 현재 청크에 추가할 수 있는지 확인
        if current_size + estimated_tokens <= max_chunk_size:
            current_chunk.append(content)
            current_size += estimated_tokens
        else:
            # 현재 청크 저장하고 새 청크 시작
            if current_chunk:
                chunks.append(create_paragraph_chunk(current_chunk))
            
            # 새 청크 시작
            if estimated_tokens > max_chunk_size:
                # 너무 큰 문단은 분할
                split_chunks = split_large_paragraph(content, max_chunk_size)
                chunks.extend(split_chunks)
                current_chunk = []
                current_size = 0
            else:
                current_chunk = [content]
                current_size = estimated_tokens
    
    # 마지막 청크 처리
    if current_chunk:
        chunks.append(create_paragraph_chunk(current_chunk))
    
    return chunks

def create_paragraph_chunk(content_list: List[str]) -> Dict[str, Any]:
    """
    문단 리스트를 하나의 청크로 변환
    """
    combined_content = ' '.join(content_list)
    
    # 문단 유형 판단
    paragraph_type = identify_paragraph_type(combined_content)
    
    chunk = {
        'type': 'paragraph',
        'subtype': paragraph_type,
        'content': combined_content,
        'paragraph_count': len(content_list),
        'metadata': {
            'source': 'paragraph_extraction',
            'priority': get_paragraph_priority(paragraph_type)
        }
    }
    
    return chunk

def identify_paragraph_type(content: str) -> str:
    """
    문단의 유형을 식별
    """
    content_lower = content.lower()
    
    # 제목/헤더 판단
    if any(keyword in content_lower for keyword in ['제', '장', '절', '항']):
        return 'header'
    
    # 목적/개요 판단
    if any(keyword in content_lower for keyword in ['목적', '개요', '배경']):
        return 'overview'
    
    # 결론/요약 판단
    if any(keyword in content_lower for keyword in ['결론', '요약', '종합']):
        return 'conclusion'
    
    # 방법론 판단
    if any(keyword in content_lower for keyword in ['방법', '절차', '과정']):
        return 'methodology'
    
    return 'general'

def get_paragraph_priority(paragraph_type: str) -> int:
    """
    문단 유형별 우선순위 설정
    """
    priority_map = {
        'overview': 9,
        'conclusion': 8,
        'methodology': 7,
        'header': 6,
        'general': 5
    }
    return priority_map.get(paragraph_type, 5)

def split_large_paragraph(content: str, max_chunk_size: int) -> List[Dict[str, Any]]:
    """
    큰 문단을 여러 청크로 분할
    """
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
                    'type': 'paragraph',
                    'subtype': 'split_general',
                    'content': chunk_content,
                    'paragraph_count': 1,
                    'metadata': {
                        'source': 'paragraph_split',
                        'priority': 5
                    }
                })
            
            current_chunk = [sentence]
            current_size = estimated_tokens
    
    # 마지막 청크
    if current_chunk:
        chunk_content = '. '.join(current_chunk) + '.'
        chunks.append({
            'type': 'paragraph',
            'subtype': 'split_general',
            'content': chunk_content,
            'paragraph_count': 1,
            'metadata': {
                'source': 'paragraph_split',
                'priority': 5
            }
        })
    
    return chunks

def analyze_chunks(chunks: List[Dict[str, Any]]) -> None:
    """
    생성된 청크들을 분석하고 통계를 출력
    """
    print("\n=== 청크 분석 결과 ===")
    
    # 타입별 통계
    type_counts = {}
    subtype_counts = {}
    
    for chunk in chunks:
        chunk_type = chunk['type']
        chunk_subtype = chunk['subtype']
        
        type_counts[chunk_type] = type_counts.get(chunk_type, 0) + 1
        subtype_counts[chunk_subtype] = subtype_counts.get(chunk_subtype, 0) + 1
    
    print("타입별 분포:")
    for chunk_type, count in type_counts.items():
        print(f"  {chunk_type}: {count}")
    
    print("\n세부 타입별 분포:")
    for subtype, count in subtype_counts.items():
        print(f"  {subtype}: {count}")
    
    # 우선순위별 분포
    priority_counts = {}
    for chunk in chunks:
        priority = chunk['metadata']['priority']
        priority_counts[priority] = priority_counts.get(priority, 0) + 1
    
    print("\n우선순위별 분포:")
    for priority in sorted(priority_counts.keys(), reverse=True):
        print(f"  우선순위 {priority}: {priority_counts[priority]}개")

# 실행 함수
def main_chunking(parsed_result):
    """
    파싱 결과를 받아서 청킹 실행
    """
    chunks = create_structured_chunks(parsed_result, max_chunk_size=1000)
    
    # 분석 결과 출력
    analyze_chunks(chunks)
    
    # 샘플 청크 출력
    print("\n=== 샘플 청크 ===")
    
    # 일정표 청크 찾기
    schedule_chunks = [c for c in chunks if c.get('subtype') == 'schedule']
    if schedule_chunks:
        print("일정표 청크:")
        sample_schedule = schedule_chunks[0]
        print(f"  타입: {sample_schedule['type']}-{sample_schedule['subtype']}")
        print(f"  내용: {sample_schedule['content'][:200]}...")
        print(f"  우선순위: {sample_schedule['metadata']['priority']}")
    
    # 일반 문단 청크
    paragraph_chunks = [c for c in chunks if c['type'] == 'paragraph']
    if paragraph_chunks:
        print("\n문단 청크:")
        sample_paragraph = paragraph_chunks[0]
        print(f"  타입: {sample_paragraph['type']}-{sample_paragraph['subtype']}")
        print(f"  내용: {sample_paragraph['content'][:200]}...")
        print(f"  문단 수: {sample_paragraph['paragraph_count']}")
    
    return chunks