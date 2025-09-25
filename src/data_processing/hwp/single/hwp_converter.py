from config.data_preprocess_config import HWP5PROC_EXECUTABLE
import os
import subprocess

def convert_hwp_single_file(input_path, output_directory, hwp5proc_executable):
    """
    외부 hwp5proc 명령어를 사용하여 단일 HWP 파일을 XHTML로 변환합니다.

    Args:
        input_path (str): 변환할 HWP 파일의 전체 경로.
        output_directory (str): 변환된 파일을 저장할 디렉터리.
    """
    # 1. 출력 디렉터리 생성
    if not os.path.exists(output_directory):
        os.makedirs(output_directory)
        print(f"출력 디렉터리가 없어 생성했습니다: {output_directory}")

    # 2. 출력 파일 경로 설정
    file_name = os.path.basename(input_path)
    base_name, _ = os.path.splitext(file_name)
    output_path = os.path.join(output_directory, f'{base_name}.xhtml')

    # 3. hwp5proc 명령어 및 인자 리스트 구성
    # 실제 hwp5proc 실행 파일 경로를 지정합니다. (사용자 환경에 맞게 수정 필요)
    hwp5proc_executable = HWP5PROC_EXECUTABLE # <-- 이 경로를 실제 사용하시는 경로로 수정해주세요.
    command = [
        hwp5proc_executable,
        "xml",  # hwp5proc의 xml 하위 명령어 사용
        input_path
    ]

    print(f"HWP 파일 변환 시작: {input_path}")
    print(f"변환 결과 저장 경로: {output_path}")

    try:
        # 4. subprocess.run을 사용하여 명령어 실행
        result = subprocess.run(
            command,
            capture_output=True,
            text=True,
            check=True
        )

        # 5. 캡처된 출력을 파일에 쓰기
        with open(output_path, 'w', encoding='utf-8') as f:
            f.write(result.stdout)

        print("✅ 변환이 성공적으로 완료되었습니다.")
        print(f"   변환된 파일: {output_path}")

    except subprocess.CalledProcessError as e:
        print("❌ hwp5proc 실행 중 오류가 발생했습니다.")
        print(f"   명령어: {' '.join(e.cmd)}")
        print(f"   반환 코드: {e.returncode}")
        print(f"   표준 출력:\n{e.stdout}")
        print(f"   표준 오류:\n{e.stderr}")
    except FileNotFoundError:
        print(f"❌ 오류: hwp5proc 실행 파일을 찾을 수 없습니다. '{hwp5proc_executable}' 경로를 확인해주세요.")
    except Exception as e:
        print(f"❌ 예기치 않은 오류가 발생했습니다: {e}")