# projectmission2/src/data_processing/hwp/all/process_parse_and_chunk_for_folder.py

import os
import glob
import io
import contextlib
from parse_all_hwp import parse_hwp5proc_xhtml
from chunking_all_parsed import main_chunking

def process_all_xhtml_files(directory_path):
    """
    지정된 디렉토리의 모든 XHTML 파일을 순회하며 파싱 및 청킹을 실행합니다.

    Args:
        directory_path (str): XHTML 파일들이 있는 디렉토리 경로
    """
    print(f"디렉토리 내 모든 XHTML 파일 처리 시작: {directory_path}")
    
    # 디렉토리 내 모든 .xhtml 파일 경로를 가져옵니다.
    file_paths = glob.glob(os.path.join(directory_path, '*.xhtml'))
    
    if not file_paths:
        print(f"경로에 .xhtml 파일이 없습니다: {directory_path}")
        return

    total_files = len(file_paths)
    print(f"총 {total_files}개의 파일이 발견되었습니다.")
    
    all_chunks = {}

    for i, file_path in enumerate(file_paths):
        print("\n" + "="*50)
        print(f"[{i+1}/{total_files}] 파일 처리 중: {os.path.basename(file_path)}")
        print("="*50)
        
        try:
            # 출력 임시 차단
            with io.StringIO() as buf, contextlib.redirect_stdout(buf):
                
                # 1. 파일 파싱
                parsed_result = parse_hwp5proc_xhtml(file_path)
                
                # 2. 파싱 결과를 바탕으로 청킹
                chunks = main_chunking(parsed_result)
                
            # 각 파일의 청크를 딕셔너리에 저장
            all_chunks[os.path.basename(file_path)] = chunks
                
        except Exception as e:
            print(f"파일 처리 중 오류 발생: {file_path}, 오류: {e}")
            continue
            
    print("\n\n모든 파일 처리 완료.")
    
    # 모든 파일의 청크 결과를 반환하거나, 필요에 따라 추가 작업을 수행할 수 있습니다.
    return all_chunks

# 메인 실행 코드
if __name__ == "__main__":
    # 파일들이 위치한 디렉토리 경로를 설정합니다.
    target_directory = '../data/processed/datapreprocessingbjs(hwp5proc)/all_xhtml'
    
    # 함수 실행
    all_processed_data = process_all_xhtml_files(target_directory)
    
    # 필요하다면 all_processed_data를 사용하여 다음 작업을 진행할 수 있습니다.
    # 예: print(f"처리된 파일 개수: {len(all_processed_data)}")