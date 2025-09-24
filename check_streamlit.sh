#!/bin/bash

# Streamlit 상태 확인 스크립트

echo "🔍 Streamlit 상태 확인 중..."

# 프로세스 확인
echo "1. 실행 중인 Streamlit 프로세스:"
ps aux | grep streamlit | grep -v grep

echo ""
echo "2. 포트 8501 사용 상태:"
netstat -tlnp | grep 8501 || echo "포트 8501이 사용되지 않음"

echo ""
echo "3. HTTP 응답 확인:"
curl -I http://localhost:8501 2>/dev/null | head -1 || echo "HTTP 응답 없음"

echo ""
echo "4. 최근 로그 (마지막 10줄):"
if [ -f streamlit.log ]; then
    tail -10 streamlit.log
else
    echo "로그 파일이 없습니다."
fi
