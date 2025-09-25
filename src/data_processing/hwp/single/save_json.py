import json
import os

def save_processed_data(all_processed_data: dict, output_dir: str, output_file: str):
    os.makedirs(output_dir, exist_ok=True)
    output_path = os.path.join(output_dir, output_file)

    try:
        with open(output_path, 'w', encoding='utf-8') as f:
            json.dump(all_processed_data, f, ensure_ascii=False, indent=4)
        print(f"✅ '{output_path}'에 처리된 데이터가 성공적으로 저장되었습니다.")
    except Exception as e:
        print(f"❌ 파일 저장 중 오류: {e}")