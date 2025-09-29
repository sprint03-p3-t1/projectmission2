# projectmission2/src/data_processing/hwp/all/convert_all_hwp_to_xhtml.py

from config.all_data_preprocessing_config import HWP5PROC_EXECUTABLE
import os
import subprocess

def convert_all_hwps_in_folder(input_directory, output_directory, hwp5proc_executable):
    """
    지정된 디렉터리 내의 모든 HWP 파일을 외부 hwp5proc 명령어로 XHTML로 변환합니다.

    Args:
        input_directory (str): HWP 파일들이 있는 디렉터리 경로.
        output_directory (str): 변환된 XHTML 파일을 저장할 디렉터리.
    """
    # 1. 출력 디렉터리 생성 (없을 경우)
    if not os.path.exists(output_directory):
        os.makedirs(output_directory)
        print(f"출력 디렉터리가 없어 생성했습니다: {output_directory}")

    # 2. hwp5proc 실행 파일 경로 설정 (사용자 환경에 맞게 수정 필요)
    # hwp5proc_executable = HWP5PROC_EXECUTABLE # <-- 이 경로를 실제 사용하시는 경로로 수정해주세요.

    print(f"HWP 파일 일괄 변환 시작 (입력 폴더: {input_directory})")

    converted_count = 0
    error_files = []

    # 3. 입력 디렉터리 내 모든 파일 순회
    for filename in os.listdir(input_directory):
        if filename.lower().endswith(".hwp"): # .hwp 파일만 처리 (대소문자 구분 없음)
            input_path = os.path.join(input_directory, filename)
            base_name, _ = os.path.splitext(filename)
            output_path = os.path.join(output_directory, f'{base_name}.xhtml')

            # 4. hwp5proc 명령어 구성
            command = [
                hwp5proc_executable,
                "xml",  # hwp5proc의 xml 하위 명령어 사용
                input_path
            ]

            print(f"\n--- 변환 중: {filename} ---")
            try:
                # 5. subprocess.run으로 명령어 실행
                result = subprocess.run(
                    command,
                    capture_output=True,
                    text=True,
                    check=True
                )

                # 6. 캡처된 출력을 파일에 저장
                with open(output_path, 'w', encoding='utf-8') as f:
                    f.write(result.stdout)

                print(f"✅ 성공: {filename} -> {output_path}")
                converted_count += 1

            except subprocess.CalledProcessError as e:
                print(f"❌ 오류 발생 ({filename}):")
                print(f"   명령어: {' '.join(e.cmd)}")
                print(f"   반환 코드: {e.returncode}")
                # 오류가 발생한 파일은 따로 기록
                error_files.append({
                    "filename": filename,
                    "error": f"Return code: {e.returncode}, Stderr: {e.stderr.strip()}"
                })
            except FileNotFoundError:
                print(f"❌ 오류: hwp5proc 실행 파일을 찾을 수 없습니다. '{hwp5proc_executable}' 경로를 확인해주세요.")
                # hwp5proc 실행 파일이 없는 경우, 전체 작업 중단
                return
            except Exception as e:
                print(f"❌ 예기치 않은 오류 발생 ({filename}): {e}")
                error_files.append({
                    "filename": filename,
                    "error": str(e)
                })

    print("\n--- 변환 작업 완료 ---")
    print(f"총 변환된 파일 수: {converted_count}")

    if error_files:
        print(f"오류가 발생한 파일 수: {len(error_files)}")
        print("오류가 발생한 파일 목록:")
        for error_info in error_files:
            print(f"- {error_info['filename']}: {error_info['error']}")
    else:
        print("모든 HWP 파일 변환에 성공했습니다.")

if __name__ == "__main__":
    # 이 스크립트가 직접 실행될 때만 아래 코드를 실행합니다.
    # 사용 예시:
    # 1. 'input_hwp' 폴더에 변환할 HWP 파일을 넣어주세요.
    # 2. 'output_xhtml' 폴더에 변환된 XHTML 파일이 저장됩니다.
    
    # 예시 입력 및 출력 디렉터리 경로를 설정합니다.
    # 이 부분은 필요에 따라 수정하여 사용하세요.
    # --- 실행 부분 ---
    # HWP 파일이 있는 입력 폴더 경로
    input_folder = '../data/raw/files'
    # 변환된 XHTML 파일을 저장할 출력 디렉터리
    output_directory_batch = '../data/processed/datapreprocessingbjs(hwp5proc)/all_xhtml'
    
    # 함수 호출
    convert_all_hwps_in_folder(input_folder, output_directory_batch, HWP5PROC_EXECUTABLE)