#!/bin/bash

# Streamlit 종료 스크립트

echo "🛑 Streamlit 종료 중..."

# Streamlit 프로세스 종료
pkill -f streamlit

echo "✅ Streamlit이 종료되었습니다."
echo "📋 실행 중인 프로세스 확인: ps aux | grep streamlit"
