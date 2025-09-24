import json
import os

def load_chunks_from_json(file_path: str):
    """JSON 파일에서 청크 데이터를 불러옵니다."""
    try:
        with open(file_path, 'r', encoding='utf-8') as f:
            data = json.load(f)
        print(f"✅ {file_path}에서 데이터를 성공적으로 불러왔습니다.")
        return data
    except FileNotFoundError:
        print(f"❌ 파일을 찾을 수 없습니다: {file_path}")
        return None
    except json.JSONDecodeError:
        print(f"❌ JSON 디코딩 실패. 파일 형식을 확인해주세요.")
        return None