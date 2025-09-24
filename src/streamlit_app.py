"""
RFP RAG 시스템 Streamlit 웹 인터페이스 - 입찰메이트
"""

import streamlit as st
import os
import json
import pandas as pd
from datetime import datetime
import plotly.express as px
import plotly.graph_objects as go
from typing import Dict, Any, List

# RFP RAG 시스템 import (새로운 모듈 구조)
import sys
import os
# 프로젝트 루트를 Python path에 추가
project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, project_root)
from src import RFPRAGSystem

# 페이지 설정
st.set_page_config(
    page_title="입찰메이트 - RFP RAG 시스템",
    page_icon="📋",
    layout="wide",
    initial_sidebar_state="expanded"
)

# CSS 스타일링 - 어두운 테마로 가독성 개선
st.markdown("""
<style>
    /* 전체 배경색을 어둡게 설정 */
    .stApp {
        background-color: #1e1e1e;
        color: #ffffff;
    }
    
    /* 메인 컨테이너 배경 */
    .main .block-container {
        background-color: #2d2d2d;
        padding: 2rem;
        border-radius: 10px;
        margin: 1rem;
    }
    
    /* 사이드바 배경 */
    .css-1d391kg {
        background-color: #2d2d2d;
    }
    
    /* 텍스트 색상 개선 */
    .stMarkdown {
        color: #ffffff !important;
    }
    
    .stText {
        color: #ffffff !important;
    }
    
    .main-header {
        font-size: 2.5rem;
        font-weight: bold;
        color: #4fc3f7;
        text-align: center;
        margin-bottom: 2rem;
        text-shadow: 2px 2px 4px rgba(0,0,0,0.5);
    }
    .sub-header {
        font-size: 1.5rem;
        color: #e0e0e0;
        margin-top: 1rem;
        margin-bottom: 1rem;
        font-weight: 600;
    }
    .metric-card {
        background-color: #3d3d3d;
        padding: 1rem;
        border-radius: 0.5rem;
        border-left: 4px solid #4fc3f7;
        margin: 0.5rem 0;
        color: #ffffff;
        box-shadow: 0 2px 4px rgba(0,0,0,0.3);
    }
    .chat-message {
        padding: 1rem;
        border-radius: 0.5rem;
        margin: 0.5rem 0;
        color: #ffffff;
    }
    .user-message {
        background-color: #1e3a5f;
        border-left: 4px solid #2196f3;
        color: #ffffff;
    }
    .assistant-message {
        background-color: #1b5e20;
        border-left: 4px solid #4caf50;
        color: #ffffff;
    }
    .warning-box {
        background-color: #3d2c00;
        border: 1px solid #ffb74d;
        color: #ffcc02;
        padding: 1rem;
        border-radius: 0.5rem;
        margin: 1rem 0;
        font-weight: 500;
    }
    
    /* 입력 필드 스타일 개선 */
    .stTextInput > div > div > input {
        background-color: #3d3d3d;
        color: #ffffff;
        border: 1px solid #555555;
    }
    
    .stTextArea > div > div > textarea {
        background-color: #3d3d3d;
        color: #ffffff;
        border: 1px solid #555555;
    }
    
    /* 버튼 스타일 개선 */
    .stButton > button {
        background-color: #4fc3f7;
        color: #000000;
        border: none;
        border-radius: 5px;
        font-weight: 600;
    }
    
    .stButton > button:hover {
        background-color: #29b6f6;
        color: #000000;
    }
    
    /* 체크박스와 라디오 버튼 스타일 */
    .stCheckbox > label {
        color: #ffffff !important;
    }
    
    .stRadio > label {
        color: #ffffff !important;
    }
    
    /* 셀렉트박스 스타일 */
    .stSelectbox > div > div {
        background-color: #3d3d3d;
        color: #ffffff;
    }
    
    /* 탭 스타일 */
    .stTabs [data-baseweb="tab-list"] {
        background-color: #2d2d2d;
    }
    
    .stTabs [data-baseweb="tab"] {
        background-color: #3d3d3d;
        color: #ffffff;
    }
    
    /* 메트릭 스타일 */
    [data-testid="metric-container"] {
        background-color: #3d3d3d;
        border: 1px solid #555555;
        padding: 1rem;
        border-radius: 0.5rem;
        color: #ffffff;
    }
    
    [data-testid="metric-container"] > div {
        color: #ffffff !important;
    }
</style>
""", unsafe_allow_html=True)

@st.cache_resource
def initialize_rag_system():
    """RAG 시스템 초기화 (캐시됨)"""
    from dotenv import load_dotenv
    import os
    
    # .env 파일 로드 (현재 디렉토리에서)
    env_path = os.path.join(os.getcwd(), '.env')
    load_dotenv(env_path)
    
    # 프로젝트 루트에서도 시도
    if not os.getenv("OPENAI_API_KEY"):
        project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
        env_path = os.path.join(project_root, '.env')
        load_dotenv(env_path)
    
    # API 키를 여러 방법으로 시도
    api_key = None
    
    # 1. Streamlit secrets에서 시도
    try:
        api_key = st.secrets["OPENAI_API_KEY"]
        st.success("✅ Streamlit secrets에서 API 키 로드")
    except:
        pass
    
    # 2. 환경변수에서 시도
    if not api_key:
        api_key = os.getenv("OPENAI_API_KEY")
        if api_key:
            st.success("✅ 환경변수에서 API 키 로드")
    
    if not api_key:
        st.error("OPENAI_API_KEY를 찾을 수 없습니다.")
        st.error("다음 중 하나의 방법으로 설정하세요:")
        st.error("1. .streamlit/secrets.toml 파일에 OPENAI_API_KEY 추가")
        st.error("2. .env 파일에 OPENAI_API_KEY=your-key 추가")
        st.error("3. export OPENAI_API_KEY='your-key' 실행")
        st.error(f"현재 작업 디렉토리: {os.getcwd()}")
        st.stop()
    
    st.success(f"✅ OpenAI API 키 로드 성공: {api_key[:10]}...")
    
    # 현재 작업 디렉토리를 기준으로 상대 경로 사용
    json_dir = os.path.join(os.getcwd(), "data", "preprocess", "json")
    if not os.path.exists(json_dir):
        st.error(f"JSON 디렉토리를 찾을 수 없습니다: {json_dir}")
        st.error(f"현재 작업 디렉토리: {os.getcwd()}")
        st.stop()
    
    with st.spinner("RFP RAG 시스템을 초기화하고 있습니다... (최초 1회만 실행됩니다)"):
        rag_system = RFPRAGSystem(json_dir, api_key)
        rag_system.initialize()
        
        # 디버깅: 초기화 상태 확인
        st.write(f"🔍 디버깅: 초기화 상태 = {rag_system.is_initialized}")
        st.write(f"🔍 디버깅: 문서 수 = {len(rag_system.data_loader.documents)}")
        
        # 테스트 검색
        test_results = rag_system.search_documents(키워드='구미아시아육상경기')
        st.write(f"🔍 디버깅: 테스트 검색 결과 = {len(test_results)}개")
    
    return rag_system

def format_currency(amount):
    """금액을 한국 원화 형식으로 포맷"""
    # 문자열인 경우 그대로 반환
    if isinstance(amount, str):
        return amount
    
    # 숫자가 아닌 경우 처리
    if not isinstance(amount, (int, float)) or amount <= 0:
        return "금액 미정"
    
    if amount >= 1000000000:
        return f"{amount/1000000000:.1f}억원"
    elif amount >= 10000000:
        return f"{amount/10000000:.0f}천만원"
    elif amount >= 10000:
        return f"{amount/10000:.0f}만원"
    else:
        return f"{int(amount):,}원"

def create_document_overview(rag_system):
    """문서 개요 대시보드"""
    summary = rag_system.get_document_summary()
    
    # 메트릭 카드들
    col1, col2, col3, col4 = st.columns(4)
    
    with col1:
        st.metric(
            label="총 문서 수",
            value=summary.get("총_문서_수", 0)
        )
    
    with col2:
        st.metric(
            label="총 청크 수",
            value=summary.get("총_청크_수", 0)
        )
    
    with col3:
        if "사업금액_통계" in summary:
            avg_amount = summary["사업금액_통계"].get("평균", 0)
            st.metric(
                label="평균 사업금액",
                value=format_currency(avg_amount)
            )
    
    with col4:
        if "사업금액_통계" in summary:
            total_amount = summary["사업금액_통계"].get("총합", 0)
            st.metric(
                label="총 사업금액",
                value=format_currency(total_amount)
            )
    
    # 발주기관별 분포 차트
    if "발주기관별_문서_수" in summary:
        st.subheader("📊 발주기관별 공고 분포")
        
        agency_data = summary["발주기관별_문서_수"]
        df_agency = pd.DataFrame(list(agency_data.items()), columns=["발주기관", "문서수"])
        df_agency = df_agency.sort_values("문서수", ascending=False)
        
        fig = px.bar(
            df_agency, 
            x="발주기관", 
            y="문서수",
            title="발주기관별 RFP 문서 수",
            color="문서수",
            color_continuous_scale="viridis"
        )
        fig.update_layout(xaxis_tickangle=-45)
        st.plotly_chart(fig, use_container_width=True)
    
    # 최근 공고 테이블
    if "최근_공고" in summary:
        st.subheader("📋 최근 공고")
        
        recent_docs = summary["최근_공고"]
        if recent_docs:
            df_recent = pd.DataFrame(recent_docs)
            df_recent["사업금액"] = df_recent["사업금액"].apply(format_currency)
            
            st.dataframe(
                df_recent,
                use_container_width=True,
                hide_index=True
            )

def create_search_interface(rag_system):
    """문서 검색 인터페이스"""
    st.subheader("🔍 문서 검색")
    
    # 검색 필터
    col1, col2 = st.columns(2)
    
    with col1:
        keyword = st.text_input("키워드 검색", placeholder="예: 정보시스템, 고도화")
        
        # 최소 사업금액 입력 (억원 단위로 표시)
        min_amount_input = st.number_input(
            "최소 사업금액 (억원)", 
            min_value=0.0, 
            value=0.0, 
            step=1.0,
            format="%.1f",
            help="0으로 설정하면 제한 없음"
        )
        min_amount = int(min_amount_input * 100000000)  # 억원을 원으로 변환
    
    with col2:
        # 발주기관 선택
        summary = rag_system.get_document_summary()
        agencies = list(summary.get("발주기관별_문서_수", {}).keys())
        selected_agency = st.selectbox("발주기관", ["전체"] + agencies)
        
        # 최대 사업금액 입력 (억원 단위로 표시, 기본값 1000억원)
        max_amount_input = st.number_input(
            "최대 사업금액 (억원)", 
            min_value=0.0, 
            value=1000.0, 
            step=100.0,
            format="%.1f",
            help="비워두려면 0으로 설정"
        )
        max_amount = int(max_amount_input * 100000000) if max_amount_input > 0 else None
    
    # 검색 실행
    if st.button("검색", type="primary"):
        filters = {}
        
        if keyword:
            filters["키워드"] = keyword
        if selected_agency != "전체":
            filters["발주기관"] = selected_agency
        if min_amount > 0:
            filters["최소금액"] = min_amount
        if max_amount is not None and max_amount > 0:
            filters["최대금액"] = max_amount
        
        # 디버깅: 필터 정보 출력
        st.write(f"🔍 디버깅: 검색 필터 = {filters}")
        st.write(f"🔍 디버깅: RAG 시스템 초기화 상태 = {rag_system.is_initialized}")
        
        results = rag_system.search_documents(**filters)
        
        # 디버깅: 검색 결과 출력
        st.write(f"🔍 디버깅: 검색 결과 수 = {len(results)}")
        
        if results:
            st.success(f"{len(results)}개의 문서를 찾았습니다.")
            
            # 결과를 DataFrame으로 변환
            df_results = pd.DataFrame(results)
            df_results["사업금액"] = df_results["사업금액"].apply(format_currency)
            
            st.dataframe(
                df_results,
                use_container_width=True,
                hide_index=True
            )
        else:
            st.warning("검색 조건에 맞는 문서를 찾을 수 없습니다.")

def create_chat_interface(rag_system):
    """채팅 인터페이스"""
    st.subheader("💬 RFP 질의응답")
    
    # 채팅 히스토리 초기화
    if "messages" not in st.session_state:
        st.session_state.messages = []
    
    # 예시 질문들
    st.markdown("**💡 예시 질문:**")
    example_questions = [
        "한국사학진흥재단이 발주한 사업의 요구사항을 알려주세요.",
        "대학재정정보시스템 고도화 사업의 사업금액과 기간은 얼마인가요?",
        "고려대학교 차세대 포털시스템 구축 사업의 주요 내용을 요약해주세요.",
        "입찰 참가 자격 요건이 있는 사업들을 알려주세요.",
        "사업금액이 10억원 이상인 프로젝트들의 특징을 분석해주세요."
    ]
    
    cols = st.columns(2)
    for i, question in enumerate(example_questions):
        col = cols[i % 2]
        with col:
            if st.button(f"Q{i+1}: {question[:30]}...", key=f"example_{i}"):
                st.session_state.messages.append({"role": "user", "content": question})
                with st.spinner("답변을 생성하고 있습니다..."):
                    response_text = rag_system.ask(question)
                st.session_state.messages.append({"role": "assistant", "content": response_text})
                st.rerun()
    
    # 채팅 히스토리 표시
    for message in st.session_state.messages:
        if message["role"] == "user":
            st.markdown(
                f'<div class="chat-message user-message"><strong>👤 사용자:</strong><br>{message["content"]}</div>',
                unsafe_allow_html=True
            )
        else:
            st.markdown(
                f'<div class="chat-message assistant-message"><strong>🤖 입찰메이트:</strong><br>{message["content"]}</div>',
                unsafe_allow_html=True
            )
    
    # 사용자 입력
    st.markdown("**💬 질문하기:**")
    user_input = st.text_input(
        "RFP에 대해 질문해보세요...",
        placeholder="예: 구미아시아육상경기 사업의 주요 요구사항은 무엇인가요?",
        key="user_question_input"
    )
    
    col1, col2 = st.columns([1, 4])
    with col1:
        ask_button = st.button("질문하기", type="primary", key="ask_button")
    
    if ask_button and user_input:
        # 사용자 메시지 추가
        st.session_state.messages.append({"role": "user", "content": user_input})
        
        # 응답 생성
        with st.spinner("답변을 생성하고 있습니다..."):
            response_text = rag_system.ask(user_input)
        
        # 응답 추가
        st.session_state.messages.append({"role": "assistant", "content": response_text})
        
        # 페이지 새로고침
        st.rerun()
    
    # 대화 초기화 버튼
    if st.session_state.messages:
        if st.button("🗑️ 대화 초기화", key="clear_chat"):
            st.session_state.messages = []
            st.rerun()

def main():
    """메인 애플리케이션"""
    
    # 헤더
    st.markdown('<h1 class="main-header">📋 입찰메이트 - RFP RAG 시스템</h1>', unsafe_allow_html=True)
    st.markdown("복잡한 RFP 문서를 빠르게 분석하고 핵심 정보를 추출하는 AI 어시스턴트")
    
    # 사이드바
    with st.sidebar:
        st.header("🛠️ 시스템 설정")
        
        # API 키 확인
        api_key = os.getenv("OPENAI_API_KEY")
        if api_key:
            st.success("✅ OpenAI API 키 설정됨")
        else:
            st.error("❌ OpenAI API 키가 설정되지 않음")
            st.markdown("""
            <div class="warning-box">
                <strong>설정 방법:</strong><br>
                터미널에서 다음 명령어를 실행하세요:<br>
                <code>export OPENAI_API_KEY="your-api-key"</code>
            </div>
            """, unsafe_allow_html=True)
            st.stop()
        
        st.markdown("---")
        
        # 탭 선택
        selected_tab = st.radio(
            "메뉴 선택",
            ["📊 대시보드", "🔍 문서 검색", "💬 질의응답"],
            index=0,
            help="💬 질의응답 탭에서 RFP에 대해 질문할 수 있습니다"
        )
        
        st.markdown("---")
        
        # 시스템 정보
        st.subheader("📋 시스템 정보")
        st.info("""
        **RFP RAG 시스템**
        - 한국어 RFP 문서 분석
        - 의미 기반 검색
        - GPT 기반 질의응답
        - 메타데이터 필터링
        """)
    
    # RAG 시스템 초기화
    try:
        rag_system = initialize_rag_system()
    except Exception as e:
        st.error(f"시스템 초기화 중 오류가 발생했습니다: {e}")
        st.stop()
    
    # 탭별 컨텐츠
    if selected_tab == "📊 대시보드":
        create_document_overview(rag_system)
    
    elif selected_tab == "🔍 문서 검색":
        create_search_interface(rag_system)
    
    elif selected_tab == "💬 질의응답":
        create_chat_interface(rag_system)
    
    # 푸터
    st.markdown("---")
    st.markdown(
        "<div style='text-align: center; color: #666; font-size: 0.9rem;'>"
        "📋 입찰메이트 - AI03기 Part3 1팀 | RFP RAG 시스템"
        "</div>",
        unsafe_allow_html=True
    )

if __name__ == "__main__":
    main()
