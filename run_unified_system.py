#!/usr/bin/env python3
"""
통합 RAG 시스템 실행 스크립트
두 시스템(FAISS, ChromaDB)을 모두 지원하는 통합 시스템 실행
"""

import os
import sys
import logging
from pathlib import Path

# 프로젝트 루트를 Python 경로에 추가
project_root = Path(__file__).parent
sys.path.insert(0, str(project_root))

# 환경 변수 설정
os.environ.setdefault('PYTHONPATH', str(project_root))

# 로깅 설정
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s [%(levelname)s] %(name)s: %(message)s',
    datefmt='%H:%M:%S'
)

logger = logging.getLogger(__name__)

def main():
    """메인 실행 함수"""
    try:
        logger.info("🚀 통합 RAG 시스템 시작")
        
        # Streamlit 앱 실행
        import subprocess
        import sys
        
        app_path = project_root / "src" / "unified_streamlit_app.py"
        
        cmd = [
            sys.executable, "-m", "streamlit", "run", 
            str(app_path),
            "--server.port", "8501",
            "--server.address", "0.0.0.0",
            "--server.headless", "true"
        ]
        
        # 로그 파일을 새로 덮어쓰기 위해 기존 로그 파일 삭제
        log_file = project_root / "streamlit.log"
        if log_file.exists():
            log_file.unlink()
            logger.info("🗑️ 기존 로그 파일 삭제됨")
        
        logger.info(f"📱 Streamlit 앱 실행: {app_path}")
        logger.info("🌐 접속 URL: http://localhost:8501")
        
        subprocess.run(cmd)
        
    except KeyboardInterrupt:
        logger.info("👋 사용자에 의해 종료됨")
    except Exception as e:
        logger.error(f"❌ 실행 중 오류 발생: {e}")
        sys.exit(1)

if __name__ == "__main__":
    main()
