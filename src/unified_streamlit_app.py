"""
통합 RAG 시스템 Streamlit 앱
두 시스템(FAISS, ChromaDB)을 선택하고 비교할 수 있는 통합 인터페이스
"""

import streamlit as st
import pandas as pd
import time
import logging
from typing import Dict, Any, List
import asyncio
import matplotlib.pyplot as plt
import numpy as np

import sys
from pathlib import Path

# 프로젝트 루트를 Python 경로에 추가
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

# 직접 import하여 순환 import 문제 방지
import sys
sys.path.append(str(Path(__file__).parent.parent))

from src.config.unified_config import UnifiedConfig
from src.systems.system_selector import SystemSelector
from src.generation.generator import RFPGenerator
from src.ops import get_quality_visualizer, get_quality_metrics, get_quality_monitor, get_conversation_tracker

# 로깅 설정
import os
log_file = os.path.join(os.getcwd(), 'streamlit.log')
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s [%(levelname)s] %(name)s: %(message)s',
    handlers=[
        logging.FileHandler(log_file, mode='w'),  # 새로 덮어쓰기
        logging.StreamHandler()
    ],
    force=True  # 기존 핸들러 덮어쓰기
)
logger = logging.getLogger(__name__)
logger.info("🚀 Streamlit 앱 시작 - 로깅 초기화 완료")

# 페이지 설정
st.set_page_config(
    page_title="RFP RAG System - 통합 검색",
    page_icon="🤖",
    layout="wide",
    initial_sidebar_state="expanded"
)

# CSS 스타일
st.markdown("""
<style>
    .main-header {
        text-align: center;
        padding: 1rem 0;
        background: linear-gradient(90deg, #667eea 0%, #764ba2 100%);
        color: white;
        border-radius: 10px;
        margin-bottom: 2rem;
    }
    .system-card {
        border: 1px solid #ddd;
        border-radius: 10px;
        padding: 1rem;
        margin: 0.5rem 0;
        background-color: #f8f9fa;
    }
    .system-active {
        border-color: #28a745;
        background-color: #d4edda;
    }
    .comparison-container {
        display: flex;
        gap: 1rem;
        margin: 1rem 0;
    }
    .result-card {
        flex: 1;
        border: 1px solid #ddd;
        border-radius: 10px;
        padding: 1rem;
        background-color: #f8f9fa;
    }
    .metric-box {
        background-color: #e9ecef;
        padding: 0.5rem;
        border-radius: 5px;
        margin: 0.25rem 0;
        text-align: center;
    }
</style>
""", unsafe_allow_html=True)

@st.cache_resource
def initialize_config():
    """설정 초기화 (캐시됨)"""
    return UnifiedConfig()

@st.cache_resource
def initialize_system_selector(config):
    """시스템 선택기 초기화 (캐시됨)"""
    logger.info("🔧 시스템 선택기 초기화 시작")
    system_selector = SystemSelector(config)
    
    # 자동으로 모든 시스템 초기화
    logger.info("🚀 모든 시스템 자동 초기화 시작")
    try:
        # FAISS 시스템 초기화
        logger.info("📊 FAISS 시스템 초기화 중...")
        system_selector.initialize_system("faiss")
        logger.info("✅ FAISS 시스템 초기화 완료")
        
        # ChromaDB 시스템 초기화
        logger.info("🔍 ChromaDB 시스템 초기화 중...")
        system_selector.initialize_system("chromadb")
        logger.info("✅ ChromaDB 시스템 초기화 완료")
        
        logger.info("🎉 모든 시스템 자동 초기화 완료!")
        
    except Exception as e:
        logger.error(f"❌ 시스템 초기화 중 오류 발생: {e}")
        # 초기화 실패해도 시스템 선택기는 반환 (수동 초기화 가능)
    
    return system_selector

def display_header():
    """헤더 표시"""
    st.markdown("""
    <div class="main-header">
        <h1>🤖 RFP RAG System - 통합 검색</h1>
        <p>FAISS vs ChromaDB 하이브리드 검색 시스템 비교</p>
        <p>🚀 자동 초기화 지원 - 서버 시작 시 모든 시스템이 자동으로 준비됩니다</p>
    </div>
    """, unsafe_allow_html=True)

def display_system_selector(config, system_selector):
    """시스템 선택 UI"""
    st.sidebar.header("🔧 시스템 설정")
    
    # 시스템 선택
    available_systems = config.get_available_systems()
    system_names = [config.get_system_info(system)["name"] for system in available_systems]
    system_mapping = {name: system for name, system in zip(system_names, available_systems)}
    
    selected_system_name = st.sidebar.selectbox(
        "검색 시스템 선택",
        options=system_names,
        index=0,
        help="사용할 검색 시스템을 선택하세요"
    )
    
    selected_system = system_mapping[selected_system_name]
    
    # 시스템 정보 표시
    system_info = config.get_system_info(selected_system)
    
    st.sidebar.markdown("### 📊 선택된 시스템 정보")
    st.sidebar.markdown(f"**모델**: {system_info['embedder_model']}")
    st.sidebar.markdown(f"**벡터 DB**: {system_info['vector_db_type']}")
    st.sidebar.markdown(f"**재순위화**: {'✅' if system_info['has_reranker'] else '❌'}")
    
    # 시스템 상태
    system_status = system_selector.get_system_status()
    current_status = system_status[selected_system]
    
    if current_status["initialized"]:
        st.sidebar.success(f"✅ {system_info['name']} 자동 초기화됨")
        st.sidebar.info("🚀 서버 시작 시 자동으로 초기화되었습니다")
    else:
        st.sidebar.warning(f"⚠️ {system_info['name']} 초기화 필요")
        st.sidebar.error("❌ 자동 초기화 실패 - 수동 초기화가 필요합니다")
    
    return selected_system

def display_comparison_mode():
    """비교 모드 설정"""
    st.sidebar.markdown("---")
    comparison_mode = st.sidebar.checkbox(
        "🔄 비교 모드",
        help="두 시스템의 결과를 동시에 비교합니다"
    )
    
    if comparison_mode:
        st.sidebar.info("💡 비교 모드에서는 두 시스템의 결과를 동시에 표시합니다")
    
    return comparison_mode

def initialize_system_if_needed(system_selector, system_name):
    """시스템이 초기화되지 않았다면 초기화"""
    system_status = system_selector.get_system_status()
    
    if not system_status[system_name]["initialized"]:
        with st.spinner(f"🔄 {system_name} 시스템 초기화 중..."):
            try:
                system_selector.initialize_system(system_name)
                st.success(f"✅ {system_name} 시스템 초기화 완료!")
                time.sleep(1)
                st.rerun()
            except Exception as e:
                st.error(f"❌ 시스템 초기화 실패: {e}")
                return None
    
    return system_selector.get_system(system_name)

def display_search_interface(system_selector, selected_system, comparison_mode):
    """검색 인터페이스"""
    st.header("🔍 검색")
    
    # 검색 입력
    col1, col2 = st.columns([3, 1])
    
    with col1:
        query = st.text_input(
            "검색어를 입력하세요",
            placeholder="예: 국립인천해양박물관 최종검수 기간은?",
            help="자연어로 질문을 입력하세요"
        )
    
    with col2:
        search_button = st.button("🔍 검색", type="primary", use_container_width=True)
    
    if search_button and query:
        # 시스템 초기화 확인
        system = initialize_system_if_needed(system_selector, selected_system)
        if system is None:
            return
        
        if comparison_mode:
            # 비교 모드: 두 시스템 모두 실행
            display_comparison_results(system_selector, query)
        else:
            # 단일 모드: 선택된 시스템만 실행
            display_single_result(system, query, selected_system)

def display_single_result(system, query, system_name):
    """단일 시스템 결과 표시"""
    with st.spinner("🔍 검색 중..."):
        start_time = time.time()
        
        try:
            if system_name == "faiss":
                # 기존 FAISS 시스템
                response = system.ask(query)
                end_time = time.time()
                
                # 디버깅 로그 추가
                logger.info(f"🔍 Streamlit 검색 결과: {response[:200]}...")
                logger.info(f"🔍 응답 타입: {type(response)}")
                logger.info(f"🔍 응답 길이: {len(response) if response else 0}")
                
                st.success(f"✅ 검색 완료 ({end_time - start_time:.2f}초)")
                
                # 결과 표시
                st.markdown("### 📄 검색 결과")
                st.markdown(response)
                
            elif system_name == "chromadb":
                # 팀원 ChromaDB 시스템
                results = system.smart_search(query, top_k=3, candidate_size=10)
                end_time = time.time()
                
                st.success(f"✅ 검색 완료 ({end_time - start_time:.2f}초)")
                
                # 결과 표시
                st.markdown("### 📄 검색 결과")
                
                if results and isinstance(results[0], dict):
                    # 메타데이터 기반 결과
                    df = pd.DataFrame(results)
                    st.dataframe(df, use_container_width=True)
                else:
                    # 문서 기반 결과
                    for i, doc in enumerate(results):
                        with st.expander(f"📄 문서 {i+1}"):
                            st.markdown(f"**출처**: {doc.metadata.get('chunk_id', 'Unknown')}")
                            st.markdown(f"**내용**: {doc.page_content[:500]}...")
                            
                            # 점수 정보 표시
                            if hasattr(system, 'last_scores'):
                                key = system.get_doc_key(doc)
                                scores = system.last_scores.get(key, {})
                                if scores:
                                    col1, col2, col3 = st.columns(3)
                                    with col1:
                                        st.metric("BM25 점수", f"{scores.get('bm25', 0):.3f}")
                                    with col2:
                                        st.metric("재순위화 점수", f"{scores.get('rerank', 0):.3f}")
                                    with col3:
                                        st.metric("통합 점수", f"{scores.get('combined', 0):.3f}")
                
        except Exception as e:
            st.error(f"❌ 검색 중 오류 발생: {e}")
            logger.error(f"Search error: {e}")

def display_comparison_results(system_selector, query):
    """비교 모드 결과 표시"""
    st.markdown("### 🔄 시스템 비교 결과")
    
    # 두 시스템 모두 초기화
    systems_to_compare = ["faiss", "chromadb"]
    results = {}
    times = {}
    
    for system_name in systems_to_compare:
        system = initialize_system_if_needed(system_selector, system_name)
        if system is None:
            continue
            
        with st.spinner(f"🔄 {system_name} 시스템 검색 중..."):
            start_time = time.time()
            
            try:
                if system_name == "faiss":
                    response = system.ask(query)
                    results[system_name] = response
                elif system_name == "chromadb":
                    search_results = system.smart_search(query, top_k=3, candidate_size=10)
                    results[system_name] = search_results
                
                end_time = time.time()
                times[system_name] = end_time - start_time
                
            except Exception as e:
                st.error(f"❌ {system_name} 시스템 오류: {e}")
                results[system_name] = None
                times[system_name] = 0
    
    # 결과 비교 표시
    if results:
        col1, col2 = st.columns(2)
        
        with col1:
            st.markdown("#### 🔵 FAISS 시스템")
            if results.get("faiss"):
                st.markdown(f"⏱️ 검색 시간: {times.get('faiss', 0):.2f}초")
                st.markdown(results["faiss"])
            else:
                st.error("검색 결과 없음")
        
        with col2:
            st.markdown("#### 🟢 ChromaDB 하이브리드 시스템")
            if results.get("chromadb"):
                st.markdown(f"⏱️ 검색 시간: {times.get('chromadb', 0):.2f}초")
                
                if isinstance(results["chromadb"], list) and results["chromadb"]:
                    if isinstance(results["chromadb"][0], dict):
                        # 메타데이터 결과
                        df = pd.DataFrame(results["chromadb"])
                        st.dataframe(df, use_container_width=True)
                    else:
                        # 문서 결과
                        for i, doc in enumerate(results["chromadb"]):
                            with st.expander(f"📄 문서 {i+1}"):
                                st.markdown(f"**출처**: {doc.metadata.get('chunk_id', 'Unknown')}")
                                st.markdown(f"**내용**: {doc.page_content[:300]}...")
            else:
                st.error("검색 결과 없음")

def display_system_management(system_selector):
    """시스템 관리 UI"""
    st.sidebar.markdown("---")
    st.sidebar.header("⚙️ 시스템 관리")
    
    # 시스템 상태 표시
    system_status = system_selector.get_system_status()
    
    for system_name, status in system_status.items():
        system_info = system_selector.config.get_system_info(system_name)
        
        with st.sidebar.expander(f"{system_info['name']} 상태"):
            if status["initialized"]:
                st.success("✅ 자동 초기화됨")
                st.info("🚀 서버 시작 시 자동으로 초기화되었습니다")
            else:
                st.warning("⚠️ 초기화 필요")
                st.error("❌ 자동 초기화 실패")
            
            st.markdown(f"**모델**: {system_info['embedder_model']}")
            st.markdown(f"**벡터 DB**: {system_info['vector_db_type']}")
            st.markdown(f"**초기화 방식**: {'자동' if status['initialized'] else '수동 필요'}")
            
            if st.button(f"🗑️ {system_name} 캐시 정리", key=f"clear_{system_name}"):
                with st.spinner("캐시 정리 중..."):
                    system_selector.clear_cache(system_name)
                st.success("캐시 정리 완료!")
                st.rerun()

def display_quality_evaluation_result(quality_eval: Dict[str, Any]):
    """품질 평가 결과 표시"""
    if not quality_eval or "error" in quality_eval:
        return
    
    st.markdown("### 📊 품질 평가 결과")
    
    # 종합 점수 표시
    overall_score = quality_eval.get("overall_score", 0)
    score_color = "green" if overall_score >= 0.8 else "orange" if overall_score >= 0.6 else "red"
    
    col1, col2, col3 = st.columns([1, 1, 2])
    
    with col1:
        st.metric(
            "종합 점수",
            f"{overall_score:.3f}",
            delta=f"{overall_score - 0.7:.3f}" if overall_score > 0.7 else None
        )
    
    with col2:
        # 점수 등급 표시
        if overall_score >= 0.8:
            st.success("🟢 우수")
        elif overall_score >= 0.6:
            st.warning("🟡 보통")
        else:
            st.error("🔴 개선 필요")
    
    with col3:
        # 세부 점수 표시
        scores = quality_eval.get("scores", {})
        if scores:
            st.markdown("**세부 점수:**")
            for metric, score in scores.items():
                st.markdown(f"• {metric}: {score:.3f}")
    
    # 개선 제안 표시
    suggestions = quality_eval.get("suggestions", [])
    if suggestions:
        st.markdown("#### 💡 개선 제안")
        for i, suggestion in enumerate(suggestions, 1):
            st.info(f"**{i}.** {suggestion}")

def display_quality_dashboard():
    """품질 평가 대시보드 표시"""
    st.markdown("## 📊 품질 평가 대시보드")
    
    # 품질 평가 도구 초기화
    quality_visualizer = get_quality_visualizer()
    quality_metrics = get_quality_metrics()
    quality_monitor = get_quality_monitor()
    
    # 사이드바 - 대시보드 설정
    with st.sidebar:
        st.markdown("### 📈 대시보드 설정")
        
        # 기간 선택
        days = st.selectbox(
            "분석 기간",
            options=[1, 3, 7, 14, 30],
            index=2,  # 기본값: 7일
            help="품질 데이터 분석 기간을 선택하세요"
        )
        
        # 모니터링 상태
        st.markdown("### 🔍 모니터링 상태")
        monitoring_status = quality_monitor.get_monitoring_status()
        
        if monitoring_status.get("is_monitoring", False):
            st.success("✅ 실시간 모니터링 활성화")
            if st.button("모니터링 중지"):
                quality_monitor.stop_monitoring()
                st.rerun()
        else:
            st.warning("⚠️ 모니터링 비활성화")
            if st.button("모니터링 시작"):
                quality_monitor.start_monitoring()
                st.rerun()
        
        # 품질 통계 요약
        st.markdown("### 📋 품질 요약")
        stats = quality_metrics.get_quality_statistics(days)
        
        col1, col2 = st.columns(2)
        with col1:
            st.metric(
                "평균 품질",
                f"{stats['avg_overall_score']:.3f}",
                delta=f"{stats['avg_overall_score'] - 0.7:.3f}" if stats['avg_overall_score'] > 0.7 else None
            )
        with col2:
            st.metric(
                "총 평가 수",
                f"{stats['total_evaluations']}"
            )
    
    # 메인 대시보드 컨텐츠
    tab1, tab2, tab3, tab4 = st.tabs(["📊 개요", "📈 트렌드", "🎯 분석", "💡 개선"])
    
    with tab1:
        st.markdown("### 품질 평가 개요")
        
        # 품질 점수 게이지
        col1, col2 = st.columns([1, 1])
        
        with col1:
            current_score = stats['avg_overall_score']
            gauge_chart = quality_visualizer.create_quality_score_gauge(current_score)
            st.plotly_chart(gauge_chart, use_container_width=True)
        
        with col2:
            # 품질 분포 파이 차트
            distribution_chart = quality_visualizer.create_quality_distribution_chart(days)
            st.plotly_chart(distribution_chart, use_container_width=True)
        
        # 품질 지표 레이더 차트
        st.markdown("### 품질 지표 상세")
        overview_chart = quality_visualizer.create_quality_overview_chart(days)
        st.plotly_chart(overview_chart, use_container_width=True)
    
    with tab2:
        st.markdown("### 품질 트렌드 분석")
        
        # 트렌드 차트
        trend_chart = quality_visualizer.create_quality_trend_chart(days)
        st.plotly_chart(trend_chart, use_container_width=True)
        
        # 품질 지표별 비교
        st.markdown("### 품질 지표별 비교")
        comparison_chart = quality_visualizer.create_quality_metrics_comparison(days)
        st.plotly_chart(comparison_chart, use_container_width=True)
    
    with tab3:
        st.markdown("### 상세 분석")
        
        # 품질 인사이트
        insights = quality_monitor.get_quality_insights()
        
        col1, col2 = st.columns([1, 1])
        
        with col1:
            st.markdown("#### 📊 품질 현황")
            if insights.get("overall_quality"):
                overall = insights["overall_quality"]
                st.metric("7일 평균", f"{overall['7day_avg']:.3f}")
                st.metric("1일 평균", f"{overall['1day_avg']:.3f}")
                st.metric("트렌드", overall['trend'])
        
        with col2:
            st.markdown("#### 📈 품질 분포")
            if insights.get("quality_distribution"):
                dist = insights["quality_distribution"]
                st.metric("고품질 비율", f"{dist['high_quality_ratio']:.1%}")
                st.metric("저품질 비율", f"{dist['low_quality_ratio']:.1%}")
        
        # 인사이트 표시
        st.markdown("#### 💡 주요 인사이트")
        if insights.get("insights"):
            for insight in insights["insights"]:
                st.info(f"• {insight}")
        else:
            st.info("분석할 데이터가 충분하지 않습니다.")
    
    with tab4:
        st.markdown("### 개선 제안")
        
        # 개선 제안 차트
        suggestions_chart = quality_visualizer.create_improvement_suggestions_chart(days)
        st.plotly_chart(suggestions_chart, use_container_width=True)
        
        # 개선 제안 상세
        suggestions = quality_metrics.get_improvement_suggestions(days)
        if suggestions:
            st.markdown("#### 🎯 우선순위별 개선 제안")
            for i, suggestion in enumerate(suggestions, 1):
                st.markdown(f"**{i}위**: {suggestion}")
        else:
            st.info("개선 제안 데이터가 없습니다.")
        
        # 데이터 내보내기
        st.markdown("#### 📤 데이터 내보내기")
        if st.button("📊 차트를 HTML로 내보내기"):
            try:
                output_path = f"quality_dashboard_{days}days.html"
                quality_visualizer.export_charts_to_html(output_path, days)
                st.success(f"차트가 {output_path}에 저장되었습니다!")
                
                # 파일 다운로드 링크
                with open(output_path, 'rb') as f:
                    st.download_button(
                        label="📥 HTML 파일 다운로드",
                        data=f.read(),
                        file_name=output_path,
                        mime="text/html"
                    )
                     except Exception as e:
                         st.error(f"내보내기 실패: {e}")

def display_conversation_analytics_dashboard():
    """대화 로그 분석 대시보드 표시"""
    st.markdown("## 📊 대화 로그 분석 대시보드")
    
    # 대화 추적기 초기화
    conversation_tracker = get_conversation_tracker()
    
    # 사이드바 - 분석 설정
    with st.sidebar:
        st.markdown("### 📈 분석 설정")
        
        # 기간 선택
        days = st.selectbox(
            "분석 기간",
            options=[1, 3, 7, 14, 30],
            index=2,  # 기본값: 7일
            help="대화 로그 분석 기간을 선택하세요"
        )
        
        # 필터 옵션
        st.markdown("### 🔍 필터 옵션")
        system_filter = st.selectbox(
            "시스템 타입",
            options=["전체", "faiss", "chromadb"],
            help="특정 시스템의 대화만 분석"
        )
        
        min_quality = st.slider(
            "최소 품질 점수",
            min_value=0.0,
            max_value=1.0,
            value=0.0,
            step=0.1,
            help="이 점수 이상의 대화만 표시"
        )
    
    # 메인 대시보드 컨텐츠
    tab1, tab2, tab3, tab4 = st.tabs(["📊 개요", "🔍 검색", "📈 분석", "📋 상세"])
    
    with tab1:
        st.markdown("### 대화 로그 개요")
        
        # 기본 통계 조회
        analytics = conversation_tracker.get_conversation_analytics(days)
        basic_stats = analytics.get('basic_stats', {})
        
        # 통계 메트릭 표시
        col1, col2, col3, col4 = st.columns(4)
        
        with col1:
            st.metric(
                "총 대화 수",
                f"{basic_stats.get('total_conversations', 0)}",
                help="선택된 기간 동안의 총 질문-답변 쌍 수"
            )
        
        with col2:
            st.metric(
                "총 세션 수",
                f"{basic_stats.get('total_sessions', 0)}",
                help="선택된 기간 동안의 총 대화 세션 수"
            )
        
        with col3:
            avg_quality = basic_stats.get('avg_quality_score', 0)
            st.metric(
                "평균 품질 점수",
                f"{avg_quality:.3f}" if avg_quality else "N/A",
                delta=f"{avg_quality - 0.7:.3f}" if avg_quality and avg_quality > 0.7 else None,
                help="선택된 기간 동안의 평균 답변 품질 점수"
            )
        
        with col4:
            total_tokens = basic_stats.get('total_tokens_used', 0)
            st.metric(
                "총 토큰 사용량",
                f"{total_tokens:,}" if total_tokens else "0",
                help="선택된 기간 동안의 총 토큰 사용량"
            )
        
        # 시스템별 통계
        st.markdown("### 시스템별 성능 비교")
        system_stats = analytics.get('system_stats', {})
        
        if system_stats:
            system_df = pd.DataFrame([
                {
                    '시스템': system,
                    '대화 수': stats['count'],
                    '평균 품질': f"{stats['avg_quality']:.3f}" if stats['avg_quality'] else "N/A",
                    '평균 생성 시간(ms)': f"{stats['avg_generation_time']:.0f}" if stats['avg_generation_time'] else "N/A"
                }
                for system, stats in system_stats.items()
            ])
            st.dataframe(system_df, use_container_width=True)
        else:
            st.info("분석할 데이터가 없습니다.")
        
        # 시간대별 통계
        st.markdown("### 시간대별 대화 패턴")
        hourly_stats = analytics.get('hourly_stats', {})
        
        if hourly_stats:
            hourly_df = pd.DataFrame([
                {
                    '시간': f"{hour}:00",
                    '대화 수': stats['count'],
                    '평균 품질': f"{stats['avg_quality']:.3f}" if stats['avg_quality'] else "N/A"
                }
                for hour, stats in hourly_stats.items()
            ])
            st.dataframe(hourly_df, use_container_width=True)
        else:
            st.info("시간대별 데이터가 없습니다.")
    
    with tab2:
        st.markdown("### 대화 로그 검색")
        
        # 검색 폼
        with st.form("conversation_search_form"):
            col1, col2 = st.columns(2)
            
            with col1:
                search_query = st.text_input(
                    "검색어",
                    placeholder="질문이나 답변에서 검색할 키워드를 입력하세요",
                    help="질문 또는 답변 내용에서 검색합니다"
                )
                
                search_system = st.selectbox(
                    "시스템 타입",
                    options=["전체", "faiss", "chromadb"]
                )
            
            with col2:
                search_min_quality = st.slider(
                    "최소 품질 점수",
                    min_value=0.0,
                    max_value=1.0,
                    value=0.0,
                    step=0.1
                )
                
                search_limit = st.number_input(
                    "결과 수 제한",
                    min_value=10,
                    max_value=1000,
                    value=100,
                    step=10
                )
            
            search_submitted = st.form_submit_button("🔍 검색")
        
        # 검색 실행
        if search_submitted:
            with st.spinner("검색 중..."):
                try:
                    # 검색 파라미터 설정
                    search_params = {
                        'query': search_query if search_query else None,
                        'system_type': search_system if search_system != "전체" else None,
                        'min_quality_score': search_min_quality if search_min_quality > 0 else None,
                        'limit': search_limit
                    }
                    
                    # 검색 실행
                    conversations = conversation_tracker.search_conversations(**search_params)
                    
                    if conversations:
                        st.success(f"검색 결과: {len(conversations)}개의 대화를 찾았습니다.")
                        
                        # 검색 결과 표시
                        for i, conv in enumerate(conversations):
                            with st.expander(f"대화 {i+1}: {conv['question'][:50]}..."):
                                col1, col2 = st.columns([2, 1])
                                
                                with col1:
                                    st.markdown(f"**질문:** {conv['question']}")
                                    st.markdown(f"**답변:** {conv['answer']}")
                                
                                with col2:
                                    st.markdown(f"**시스템:** {conv['system_type']}")
                                    st.markdown(f"**모델:** {conv['model_name']}")
                                    st.markdown(f"**품질 점수:** {conv['overall_quality_score']:.3f}" if conv['overall_quality_score'] else "N/A")
                                    st.markdown(f"**시간:** {conv['question_timestamp']}")
                                    
                                    if st.button(f"상세 보기", key=f"detail_{conv['log_id']}"):
                                        st.session_state.selected_conversation_id = conv['log_id']
                                        st.rerun()
                    else:
                        st.warning("검색 조건에 맞는 대화를 찾을 수 없습니다.")
                        
                except Exception as e:
                    st.error(f"검색 중 오류가 발생했습니다: {e}")
    
    with tab3:
        st.markdown("### 대화 패턴 분석")
        
        # 최근 대화 로그 조회
        recent_conversations = conversation_tracker.search_conversations(limit=100)
        
        if recent_conversations:
            # 품질 점수 분포
            quality_scores = [conv['overall_quality_score'] for conv in recent_conversations if conv['overall_quality_score']]
            
            if quality_scores:
                st.markdown("#### 품질 점수 분포")
                fig, ax = plt.subplots(figsize=(10, 6))
                ax.hist(quality_scores, bins=20, alpha=0.7, color='skyblue', edgecolor='black')
                ax.set_xlabel('품질 점수')
                ax.set_ylabel('빈도')
                ax.set_title('대화 품질 점수 분포')
                ax.axvline(np.mean(quality_scores), color='red', linestyle='--', label=f'평균: {np.mean(quality_scores):.3f}')
                ax.legend()
                st.pyplot(fig)
            
            # 시스템별 성능 비교
            system_performance = {}
            for conv in recent_conversations:
                system = conv['system_type']
                if system not in system_performance:
                    system_performance[system] = {'count': 0, 'total_quality': 0, 'total_time': 0}
                
                system_performance[system]['count'] += 1
                if conv['overall_quality_score']:
                    system_performance[system]['total_quality'] += conv['overall_quality_score']
                if conv['generation_time_ms']:
                    system_performance[system]['total_time'] += conv['generation_time_ms']
            
            if system_performance:
                st.markdown("#### 시스템별 성능 비교")
                performance_data = []
                for system, stats in system_performance.items():
                    avg_quality = stats['total_quality'] / stats['count'] if stats['count'] > 0 else 0
                    avg_time = stats['total_time'] / stats['count'] if stats['count'] > 0 else 0
                    performance_data.append({
                        '시스템': system,
                        '대화 수': stats['count'],
                        '평균 품질': avg_quality,
                        '평균 생성 시간(ms)': avg_time
                    })
                
                performance_df = pd.DataFrame(performance_data)
                st.dataframe(performance_df, use_container_width=True)
        else:
            st.info("분석할 대화 데이터가 없습니다.")
    
    with tab4:
        st.markdown("### 대화 상세 정보")
        
        # 선택된 대화 상세 정보 표시
        if hasattr(st.session_state, 'selected_conversation_id'):
            conversation_id = st.session_state.selected_conversation_id
            
            try:
                # 대화 상세 정보 조회
                conversation_details = conversation_tracker.get_conversation_details(conversation_id)
                
                if conversation_details:
                    st.markdown(f"#### 대화 ID: {conversation_id}")
                    
                    # 기본 정보
                    col1, col2 = st.columns(2)
                    
                    with col1:
                        st.markdown("**기본 정보**")
                        st.markdown(f"- 질문 시간: {conversation_details['question_timestamp']}")
                        st.markdown(f"- 시스템 타입: {conversation_details['system_type']}")
                        st.markdown(f"- 모델: {conversation_details['model_name']}")
                        st.markdown(f"- 검색 방법: {conversation_details['search_method']}")
                    
                    with col2:
                        st.markdown("**성능 정보**")
                        st.markdown(f"- 검색 시간: {conversation_details['search_time_ms']}ms")
                        st.markdown(f"- 생성 시간: {conversation_details['generation_time_ms']}ms")
                        st.markdown(f"- 토큰 사용량: {conversation_details['total_tokens']}")
                        st.markdown(f"- 품질 점수: {conversation_details['overall_quality_score']:.3f}" if conversation_details['overall_quality_score'] else "N/A")
                    
                    # 질문과 답변
                    st.markdown("**질문**")
                    st.text_area("", conversation_details['question'], height=100, disabled=True)
                    
                    st.markdown("**답변**")
                    st.text_area("", conversation_details['answer'], height=200, disabled=True)
                    
                    # 검색된 청크 정보
                    if conversation_details['retrieved_chunks']:
                        st.markdown("**검색된 문서 청크**")
                        for i, chunk in enumerate(conversation_details['retrieved_chunks'][:5]):  # 상위 5개만 표시
                            with st.expander(f"청크 {i+1}: {chunk.get('content', '')[:50]}..."):
                                st.json(chunk)
                    
                    # 검색 단계별 상세 정보
                    if conversation_details.get('search_steps'):
                        st.markdown("**검색 단계별 상세 정보**")
                        for step in conversation_details['search_steps']:
                            with st.expander(f"단계: {step['step_type']} (순서: {step['step_order']})"):
                                st.markdown(f"**실행 시간:** {step['execution_time_ms']}ms")
                                st.markdown("**입력 데이터:**")
                                st.json(step['input_data'])
                                st.markdown("**출력 데이터:**")
                                st.json(step['output_data'])
                                if step['metadata']:
                                    st.markdown("**메타데이터:**")
                                    st.json(step['metadata'])
                else:
                    st.error("대화 상세 정보를 찾을 수 없습니다.")
                    
            except Exception as e:
                st.error(f"대화 상세 정보 조회 중 오류가 발생했습니다: {e}")
        else:
            st.info("상세 정보를 보려면 검색 탭에서 '상세 보기' 버튼을 클릭하세요.")

def main():
    """메인 함수"""
    try:
        # 설정 및 시스템 선택기 초기화
        config = initialize_config()
        system_selector = initialize_system_selector(config)
        
        # UI 표시
        display_header()
        
        # 시스템 선택
        selected_system = display_system_selector(config, system_selector)
        
                # 메인 탭 생성
                tab1, tab2, tab3, tab4 = st.tabs(["🔍 질의응답", "📊 품질 대시보드", "📈 대화 로그 분석", "⚙️ 시스템 관리"])
        
        with tab1:
            # 비교 모드 설정
            comparison_mode = display_comparison_mode()
            
            # 검색 인터페이스
            display_search_interface(system_selector, selected_system, comparison_mode)
        
        with tab2:
            # 품질 평가 대시보드
            display_quality_dashboard()
        
        with tab3:
            # 대화 로그 분석 대시보드
            display_conversation_analytics_dashboard()
        
        with tab4:
            # 시스템 관리
            display_system_management(system_selector)
        
    except Exception as e:
        st.error(f"❌ 애플리케이션 오류: {e}")
        logger.error(f"Application error: {e}")

if __name__ == "__main__":
    main()
