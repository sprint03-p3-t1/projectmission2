#!/bin/bash

echo "🚀 FastAPI RAG 시스템 시작 중..."

# 1. 기존 프로세스 종료
echo "1. 기존 프로세스 종료 중..."
pkill -f "uvicorn api.main:app"
sleep 2

# 2. 가상 환경 활성화
echo "2. 가상 환경 활성화..."
cd /home/spai0316/projectmission2 || exit
source ~/myenv/bin/activate

# 3. FastAPI 의존성 설치
echo "3. FastAPI 의존성 설치..."
pip install -r api/requirements.txt

# 4. CUDA 환경 변수 설정 (CPU 모드 강제)
export CUDA_VISIBLE_DEVICES=""
export PYTORCH_CUDA_ALLOC_CONF=""

# 5. FastAPI 서버 시작
echo "4. FastAPI 서버 시작 중..."
nohup uvicorn api.main:app --host 0.0.0.0 --port 8501 --workers 4 > fastapi.log 2>&1 &

echo "✅ FastAPI 서버가 백그라운드에서 시작되었습니다!"
echo "   API 문서: http://35.225.142.54:8501/docs"
echo "   웹 인터페이스: http://35.225.142.54:8501"
echo "   로그 확인: tail -f fastapi.log"

# 5초 대기 후 상태 확인
sleep 5
echo ""
echo "🔍 서버 상태 확인:"
ps aux | grep "uvicorn api.main:app" | grep -v grep
