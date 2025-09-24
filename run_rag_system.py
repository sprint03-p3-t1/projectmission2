#!/usr/bin/env python3
"""
RFP RAG 시스템 실행 스크립트
"""

import os
import sys
import subprocess
from pathlib import Path

def check_requirements():
    """필요 조건 확인"""
    from dotenv import load_dotenv
    
    print("🔍 시스템 요구사항 확인 중...")
    
    # .env 파일 로드
    load_dotenv()
    
    # Python 버전 확인
    if sys.version_info < (3, 8):
        print("❌ Python 3.8 이상이 필요합니다.")
        return False
    
    # OpenAI API 키 확인
    if not os.getenv("OPENAI_API_KEY"):
        print("❌ OPENAI_API_KEY 환경변수가 설정되지 않았습니다.")
        print("   다음 중 하나의 방법으로 설정하세요:")
        print("   1. .env 파일에 OPENAI_API_KEY=your-key 추가")
        print("   2. export OPENAI_API_KEY='your-api-key-here' 실행")
        return False
    
    # JSON 데이터 디렉토리 확인
    json_dir = Path("data/preprocess/json")
    if not json_dir.exists():
        print(f"❌ JSON 데이터 디렉토리를 찾을 수 없습니다: {json_dir}")
        print("   preprocess.py를 먼저 실행하여 데이터를 전처리하세요.")
        return False
    
    json_files = list(json_dir.glob("*.json"))
    if not json_files:
        print(f"❌ JSON 파일을 찾을 수 없습니다: {json_dir}")
        return False
    
    print(f"✅ {len(json_files)}개의 JSON 파일 발견")
    print("✅ 모든 요구사항이 충족되었습니다.")
    return True

def install_requirements():
    """필요한 패키지 설치"""
    print("📦 필요한 패키지 설치 중...")
    
    try:
        subprocess.run([
            sys.executable, "-m", "pip", "install", "-r", "requirements.txt"
        ], check=True)
        print("✅ 패키지 설치 완료")
        return True
    except subprocess.CalledProcessError as e:
        print(f"❌ 패키지 설치 실패: {e}")
        return False

def run_streamlit():
    """Streamlit 앱 실행"""
    print("🚀 RFP RAG 시스템을 시작합니다...")
    print("   브라우저에서 http://localhost:8501 로 접속하세요.")
    
    try:
        subprocess.run([
            sys.executable, "-m", "streamlit", "run", "src/streamlit_app.py",
            "--server.port", "8501",
            "--server.address", "0.0.0.0"
        ])
    except KeyboardInterrupt:
        print("\n👋 시스템을 종료합니다.")
    except Exception as e:
        print(f"❌ 실행 중 오류 발생: {e}")

def run_console_demo():
    """콘솔 데모 실행"""
    print("💻 콘솔 모드로 RFP RAG 시스템을 실행합니다...")
    
    try:
        from src import RFPRAGSystem
        
        # 시스템 초기화
        api_key = os.getenv("OPENAI_API_KEY")
        json_dir = "data/preprocess/json"
        
        rag_system = RFPRAGSystem(json_dir, api_key)
        rag_system.initialize()
        
        # 문서 요약 출력
        summary = rag_system.get_document_summary()
        print("\n=== 📊 문서 요약 ===")
        print(f"총 문서 수: {summary.get('총_문서_수', 0)}")
        print(f"총 청크 수: {summary.get('총_청크_수', 0)}")
        
        if "사업금액_통계" in summary:
            stats = summary["사업금액_통계"]
            print(f"평균 사업금액: {stats.get('평균', 0):,}원")
            print(f"총 사업금액: {stats.get('총합', 0):,}원")
        
        # 대화형 질의응답
        print("\n=== 💬 질의응답 시작 ===")
        print("질문을 입력하세요 (종료: 'quit' 또는 'exit')")
        
        while True:
            question = input("\n👤 질문: ").strip()
            
            if question.lower() in ['quit', 'exit', '종료']:
                break
            
            if not question:
                continue
            
            print("🤖 답변을 생성하고 있습니다...")
            response = rag_system.ask(question)
            print(f"🤖 답변: {response}")
    
    except Exception as e:
        print(f"❌ 콘솔 모드 실행 중 오류 발생: {e}")

def main():
    """메인 함수"""
    print("📋 입찰메이트 - RFP RAG 시스템")
    print("=" * 50)
    
    # 요구사항 확인
    if not check_requirements():
        return
    
    # 실행 모드 선택 (자동으로 웹 인터페이스 모드 선택)
    print("\n🎯 실행 모드를 선택하세요:")
    print("1. 웹 인터페이스 (Streamlit)")
    print("2. 콘솔 모드")
    print("3. 패키지 설치만")
    
    # 자동으로 웹 인터페이스 모드 선택
    choice = "1"
    print(f"\n선택 (1-3): {choice}")
    
    if choice == "1":
        if install_requirements():
            run_streamlit()
    elif choice == "2":
        if install_requirements():
            run_console_demo()
    elif choice == "3":
        install_requirements()

if __name__ == "__main__":
    main()
