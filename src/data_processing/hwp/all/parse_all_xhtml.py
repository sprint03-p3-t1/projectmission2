# projectmission2/src/data_processing/hwp/all/parse_all_hwp.py

import os
from bs4 import BeautifulSoup
from typing import Dict, Any, List
import re

def parse_hwp5proc_xhtml(file_path):
    """
    hwp5proc으로 변환된 XHTML 파일을 파싱하여 구조화된 데이터로 변환
    
    Args:
        file_path (str): XHTML 파일 경로
    
    Returns:
        dict: 파싱된 섹션과 표 데이터
    """
    print(f"XHTML 파일 파싱 시작: {file_path}")
    
    try:
        with open(file_path, 'r', encoding='utf-8') as f:
            content = f.read()
    except UnicodeDecodeError:
        # UTF-8로 읽기 실패 시 다른 인코딩 시도
        with open(file_path, 'r', encoding='cp949') as f:
            content = f.read()
    
    soup = BeautifulSoup(content, 'xml')  # XML 파서 사용
    
    # 결과 저장용 딕셔너리
    parsed_data = {
        'sections': [],
        'tables': [],
        'metadata': {
            'total_paragraphs': 0,
            'total_tables': 0,
            'file_size': len(content)
        }
    }
    
    # 1. 표(TableControl) 추출
    tables = soup.find_all('TableControl')
    print(f"발견된 표 개수: {len(tables)}")
    
    for i, table in enumerate(tables):
        table_data = extract_table_structure(table, table_id=i)
        if table_data:
            parsed_data['tables'].append(table_data)
    
    # 2. 일반 텍스트 문단 추출
    paragraphs = soup.find_all('Paragraph')
    print(f"발견된 문단 개수: {len(paragraphs)}")
    
    for i, para in enumerate(paragraphs):
        text_content = extract_paragraph_text(para, para_id=i)
        if text_content and text_content.get('content', '').strip():
            parsed_data['sections'].append(text_content)
    
    # 메타데이터 업데이트
    parsed_data['metadata']['total_paragraphs'] = len(parsed_data['sections'])
    parsed_data['metadata']['total_tables'] = len(parsed_data['tables'])
    
    print(f"파싱 완료 - 문단: {len(parsed_data['sections'])}, 표: {len(parsed_data['tables'])}")
    return parsed_data

def extract_table_structure(table_element, table_id):
    """
    TableControl 요소에서 표 데이터 추출
    """
    try:
        table_data = {
            'id': table_id,
            'type': 'table',
            'rows': [],
            'raw_content': '',
            'metadata': {}
        }
        
        # TableBody 찾기
        table_body = table_element.find('TableBody')
        if not table_body:
            return None
            
        # TableRow들 찾기
        rows = table_body.find_all('TableRow')
        
        for row in rows:
            row_data = []
            cells = row.find_all('TableCell')
            
            for cell in cells:
                # 셀 내의 Text 요소들 추출
                cell_text = ""
                text_elements = cell.find_all('Text')
                for text_elem in text_elements:
                    if text_elem.string:
                        cell_text += text_elem.string.strip() + " "
                
                row_data.append(cell_text.strip())
            
            if any(row_data):  # 빈 행이 아닌 경우만 추가
                table_data['rows'].append(row_data)
        
        # 표 내용을 자연어로 변환
        table_data['raw_content'] = convert_table_to_natural_language(table_data['rows'])
        
        return table_data
        
    except Exception as e:
        print(f"표 추출 중 오류 발생: {e}")
        return None

def extract_paragraph_text(paragraph_element, para_id):
    """
    Paragraph 요소에서 텍스트 추출
    """
    try:
        text_content = ""
        
        # LineSeg 요소들 찾기
        line_segs = paragraph_element.find_all('LineSeg')
        
        for line_seg in line_segs:
            # Text 요소들 추출
            text_elements = line_seg.find_all('Text')
            for text_elem in text_elements:
                if text_elem.string:
                    text_content += text_elem.string.strip() + " "
        
        return {
            'id': para_id,
            'type': 'paragraph',
            'content': text_content.strip(),
            'metadata': {}
        }
        
    except Exception as e:
        print(f"문단 추출 중 오류 발생: {e}")
        return None

def convert_table_to_natural_language(table_rows):
    """
    표 데이터를 자연어 형태로 변환
    """
    if not table_rows or len(table_rows) < 2:
        return ""
    
    # 첫 번째 행을 헤더로 가정
    headers = table_rows[0]
    content_rows = table_rows[1:]
    
    natural_text = []
    
    # 추진일정표 특별 처리
    if any('월' in str(header) for header in headers) and '구분' in str(headers):
        natural_text.append("프로젝트 추진일정은 다음과 같습니다:")
        
        for row in content_rows:
            if len(row) > 1 and row[0].strip():  # 첫 번째 컬럼이 비어있지 않은 경우
                task = row[0].strip()
                # 각 월별 일정 정보 추출
                schedule_info = []
                for i, cell in enumerate(row[1:], 1):
                    if cell.strip() and i < len(headers):
                        month = headers[i]
                        schedule_info.append(f"{month}에 {task}")
                
                if schedule_info:
                    natural_text.extend(schedule_info)
    else:
        # 일반 표 처리
        for row in content_rows:
            if len(row) >= len(headers):
                row_text = []
                for i, (header, cell) in enumerate(zip(headers, row)):
                    if cell.strip():
                        row_text.append(f"{header}: {cell.strip()}")
                
                if row_text:
                    natural_text.append(", ".join(row_text))
    
    return ". ".join(natural_text)

# # 실제 파일로 테스트 실행
# def main():
#     # 실제 경로
#     xhtml_file_path = '../data/processed/datapreprocessingbjs(hwp5proc)/all_xhtml/(재)예술경영지원센터_통합 정보시스템 구축 사전 컨설팅.xhtml'
    
    
#     # 파일 존재 확인
#     if not os.path.exists(xhtml_file_path):
#         print(f"파일을 찾을 수 없습니다: {xhtml_file_path}")
#         return
    
#     # 파싱 실행
#     parsed_result = parse_hwp5proc_xhtml(xhtml_file_path)
    
#     # 결과 출력
#     print("\n=== 파싱 결과 요약 ===")
#     print(f"추출된 문단 수: {len(parsed_result['sections'])}")
#     print(f"추출된 표 수: {len(parsed_result['tables'])}")
#     print(f"파일 크기: {parsed_result['metadata']['file_size']:,} bytes")
    
#     # 첫 번째 표 샘플 출력 (추진일정표)
#     if parsed_result['tables']:
#         print("\n=== 첫 번째 표 샘플 ===")
#         first_table = parsed_result['tables'][0]
#         print(f"표 ID: {first_table['id']}")
#         print(f"행 수: {len(first_table['rows'])}")
#         if first_table['raw_content']:
#             print("자연어 변환 결과:")
#             print(first_table['raw_content'][:500] + "..." if len(first_table['raw_content']) > 500 else first_table['raw_content'])
    
#     # 첫 번째 문단 샘플 출력
#     if parsed_result['sections']:
#         print("\n=== 첫 번째 문단 샘플 ===")
#         first_section = parsed_result['sections'][0]
#         print(f"문단 ID: {first_section['id']}")
#         content = first_section['content']
#         print("내용:", content[:200] + "..." if len(content) > 200 else content)
    
#     return parsed_result

# # 실행
# if __name__ == "__main__":
#     result = main()