#!/bin/bash

# Streamlit 재시작 스크립트

echo "🔄 Streamlit 재시작 중..."

# 기존 Streamlit 프로세스 종료
echo "1. 기존 프로세스 종료 중..."
pkill -f streamlit
sleep 2

# 가상환경 활성화 및 Streamlit 시작
echo "2. Streamlit 시작 중..."
cd /home/spai0316/projectmission2
source ~/myenv/bin/activate

# CPU 모드로 실행
export CUDA_VISIBLE_DEVICES=""
export PYTORCH_CUDA_ALLOC_CONF=""

# Streamlit 백그라운드 실행
nohup streamlit run src/streamlit_app.py --server.port 8501 --server.address 0.0.0.0 > streamlit.log 2>&1 &

echo "✅ Streamlit이 백그라운드에서 시작되었습니다."
echo "📱 접속 URL: http://35.225.142.54:8501"
echo "📋 로그 확인: tail -f streamlit.log"
echo "🛑 종료: pkill -f streamlit"
