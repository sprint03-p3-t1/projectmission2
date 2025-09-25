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
import matplotlib.font_manager as fm
import numpy as np
import json

# 한글 폰트 설정
plt.rcParams['font.family'] = 'DejaVu Sans'
plt.rcParams['axes.unicode_minus'] = False

# 한글 폰트 설정 (YAML 설정 파일 사용)
try:
    import yaml
    import os
    
    # YAML 설정 파일 로드
    config_path = os.path.join(os.getcwd(), 'config', 'rag_config.yaml')
    if os.path.exists(config_path):
        with open(config_path, 'r', encoding='utf-8') as f:
            config = yaml.safe_load(f)
        
        font_config = config.get('ui', {}).get('korean_font', {})
        font_path = font_config.get('path', 'NanumGothic.ttf')
        fallback_fonts = font_config.get('fallback_fonts', ['NanumGothic', 'NanumBarunGothic'])
        
        # 절대 경로로 변환
        absolute_font_path = os.path.join(os.getcwd(), font_path)
        
        if os.path.exists(absolute_font_path):
            # 폰트 파일이 있으면 직접 등록
            fm.fontManager.addfont(absolute_font_path)
            plt.rcParams['font.family'] = 'NanumGothic'
            print(f"✅ 한글 폰트 설정 완료: {absolute_font_path}")
        else:
            # 시스템 폰트 찾기
            available_fonts = [f.name for f in fm.fontManager.ttflist]
            
            for font in fallback_fonts:
                if font in available_fonts:
                    plt.rcParams['font.family'] = font
                    print(f"✅ 시스템 한글 폰트 설정: {font}")
                    break
            else:
                print("⚠️ 한글 폰트를 찾을 수 없습니다. 기본 폰트를 사용합니다.")
                plt.rcParams['font.family'] = 'DejaVu Sans'
    else:
        print("⚠️ 설정 파일을 찾을 수 없습니다. 기본 폰트를 사용합니다.")
        plt.rcParams['font.family'] = 'DejaVu Sans'
        
except Exception as e:
    print(f"⚠️ 폰트 설정 중 오류: {e}")
    plt.rcParams['font.family'] = 'DejaVu Sans'

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
from src.ops import get_quality_visualizer, get_quality_metrics, get_quality_monitor, get_conversation_tracker, AutoEvaluator, PromptOptimizer, OptimizationResult, OptimizationConfig

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
    
    # 프롬프트 버전 관리
    display_prompt_management()

def display_prompt_management():
    """프롬프트 버전 관리 UI"""
    st.sidebar.markdown("---")
    st.sidebar.header("📝 프롬프트 관리")
    
    try:
        from src.prompts.prompt_manager import get_prompt_manager
        prompt_manager = get_prompt_manager()
        
        # 현재 프롬프트 버전 표시
        current_version = prompt_manager.get_current_version()
        st.sidebar.info(f"**현재 버전**: {current_version}")
        
        # 사용 가능한 버전 목록
        available_versions = prompt_manager.get_available_versions()
        
        if available_versions:
            # 버전 선택 드롭다운
            selected_version = st.sidebar.selectbox(
                "프롬프트 버전 선택",
                available_versions,
                index=available_versions.index(current_version) if current_version in available_versions else 0,
                help="다른 프롬프트 버전으로 전환할 수 있습니다"
            )
            
            # 버전 변경 버튼
            if selected_version != current_version:
                if st.sidebar.button("🔄 버전 변경", key="change_prompt_version"):
                    if prompt_manager.set_current_version(selected_version):
                        st.sidebar.success(f"✅ {selected_version} 버전으로 변경됨")
                        st.rerun()
                    else:
                        st.sidebar.error("❌ 버전 변경 실패")
            
            # 버전 정보 표시
            version_info = prompt_manager.get_version_info(selected_version)
            if version_info:
                with st.sidebar.expander(f"📋 {selected_version} 정보"):
                    st.markdown(f"**이름**: {version_info.get('name', 'N/A')}")
                    st.markdown(f"**설명**: {version_info.get('description', 'N/A')}")
                    st.markdown(f"**생성일**: {version_info.get('created_date', 'N/A')}")
                    st.markdown(f"**작성자**: {version_info.get('author', 'N/A')}")
                    
                    tags = version_info.get('tags', [])
                    if tags:
                        st.markdown(f"**태그**: {', '.join(tags)}")
        
        # 프롬프트 미리보기
        if st.sidebar.button("👁️ 프롬프트 미리보기", key="preview_prompt"):
            st.sidebar.markdown("---")
            st.sidebar.subheader("📄 시스템 프롬프트")
            system_prompt = prompt_manager.get_system_prompt()
            st.sidebar.text_area("", system_prompt, height=200, disabled=True, key="system_prompt_preview")
            
            st.sidebar.subheader("📝 사용자 템플릿")
            user_template = prompt_manager.get_user_template()
            st.sidebar.text_area("", user_template, height=150, disabled=True, key="user_template_preview")
            
            st.sidebar.subheader("📊 평가 프롬프트")
            evaluation_prompt = prompt_manager.get_evaluation_prompt()
            st.sidebar.text_area("", evaluation_prompt, height=200, disabled=True, key="evaluation_prompt_preview")
    
    except ImportError as e:
        st.sidebar.warning("⚠️ 프롬프트 매니저를 사용할 수 없습니다")
        st.sidebar.error(f"오류: {e}")
    except Exception as e:
        st.sidebar.error(f"❌ 프롬프트 관리 오류: {e}")

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
        
        # 프롬프트 버전 정보
        st.markdown("### 📝 프롬프트 버전")
        try:
            from src.prompts.prompt_manager import get_prompt_manager
            prompt_manager = get_prompt_manager()
            
            current_version = prompt_manager.get_current_version()
            version_info = prompt_manager.get_version_info(current_version)
            
            if version_info:
                st.info(f"**현재 버전**: {current_version}")
                st.markdown(f"**이름**: {version_info.get('name', 'N/A')}")
                st.markdown(f"**설명**: {version_info.get('description', 'N/A')}")
                
                # 버전 변경 옵션
                available_versions = prompt_manager.get_available_versions()
                if len(available_versions) > 1:
                    selected_version = st.selectbox(
                        "프롬프트 버전 변경",
                        available_versions,
                        index=available_versions.index(current_version) if current_version in available_versions else 0,
                        help="다른 프롬프트 버전으로 전환"
                    )
                    
                    if selected_version != current_version:
                        if st.button("🔄 버전 변경", key="change_prompt_version_analytics"):
                            if prompt_manager.set_current_version(selected_version):
                                st.success(f"✅ {selected_version} 버전으로 변경됨")
                                st.rerun()
                            else:
                                st.error("❌ 버전 변경 실패")
        except Exception as e:
            st.warning("⚠️ 프롬프트 매니저를 사용할 수 없습니다")
            st.error(f"오류: {e}")
    
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
        
        # 프롬프트 버전별 통계 (새로 추가)
        st.markdown("### 📝 프롬프트 버전별 성능")
        try:
            from src.prompts.prompt_manager import get_prompt_manager
            prompt_manager = get_prompt_manager()
            available_versions = prompt_manager.get_available_versions()
            
            if available_versions:
                version_stats = []
                for version in available_versions:
                    version_info = prompt_manager.get_version_info(version)
                    if version_info:
                        # 실제로는 데이터베이스에서 프롬프트 버전별 통계를 조회해야 하지만,
                        # 현재는 버전 정보만 표시
                        version_stats.append({
                            '버전': version,
                            '이름': version_info.get('name', 'N/A'),
                            '생성일': version_info.get('created_date', 'N/A'),
                            '태그': ', '.join(version_info.get('tags', []))
                        })
                
                if version_stats:
                    version_df = pd.DataFrame(version_stats)
                    st.dataframe(version_df, use_container_width=True)
                else:
                    st.info("프롬프트 버전 정보가 없습니다.")
            else:
                st.info("사용 가능한 프롬프트 버전이 없습니다.")
        except Exception as e:
            st.warning(f"프롬프트 버전 통계를 불러올 수 없습니다: {e}")
    
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
                ax.set_xlabel('품질 점수', fontsize=12)
                ax.set_ylabel('빈도', fontsize=12)
                ax.set_title('대화 품질 점수 분포', fontsize=14, fontweight='bold')
                ax.axvline(np.mean(quality_scores), color='red', linestyle='--', label=f'평균: {np.mean(quality_scores):.3f}')
                ax.legend(fontsize=10)
                ax.grid(True, alpha=0.3)
                
                # 한글 폰트 설정 적용
                for label in ax.get_xticklabels() + ax.get_yticklabels():
                    label.set_fontsize(10)
                
                plt.tight_layout()
                st.pyplot(fig)
                plt.close(fig)  # 메모리 절약
            
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
        
        # 대화 선택 인터페이스 추가
        try:
            # 최근 대화 목록 가져오기
            recent_conversations = conversation_tracker.get_recent_conversations(limit=50)
            
            if recent_conversations:
                st.markdown("#### 대화 선택")
                
                # 대화 선택 드롭다운
                conversation_options = {}
                for conv in recent_conversations:
                    display_text = f"[{conv['question_timestamp']}] {conv['question'][:50]}... (시스템: {conv['system_type']})"
                    conversation_options[display_text] = conv['log_id']
                
                selected_display = st.selectbox(
                    "대화를 선택하세요:",
                    options=list(conversation_options.keys()),
                    index=0,
                    help="상세 정보를 보고 싶은 대화를 선택하세요"
                )
                
                if selected_display:
                    selected_conversation_id = conversation_options[selected_display]
                    st.session_state.selected_conversation_id = selected_conversation_id
                    
                    # 상세보기 버튼
                    if st.button("📋 상세 정보 보기", key="view_details_button"):
                        st.rerun()
            else:
                st.info("표시할 대화가 없습니다.")
                
        except Exception as e:
            st.error(f"대화 목록 조회 중 오류가 발생했습니다: {e}")
        
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
                    st.text_area("질문 내용", conversation_details['question'], height=100, disabled=True, label_visibility="collapsed")
                    
                    st.markdown("**답변**")
                    st.text_area("답변 내용", conversation_details['answer'], height=200, disabled=True, label_visibility="collapsed")
                    
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

def display_auto_evaluation_dashboard():
    """자동 평가 대시보드"""
    st.markdown("## 🤖 자동 평가 시스템")
    st.markdown("LLM이 자동으로 질문을 생성하고, 답변을 생성한 후, 10개 기준으로 평가합니다.")
    
    # 자동 평가 시스템 초기화
    if 'auto_evaluator' not in st.session_state:
        try:
            with st.spinner("자동 평가 시스템을 초기화하는 중..."):
                st.session_state.auto_evaluator = AutoEvaluator()
                st.session_state.auto_evaluator.initialize()
            st.success("✅ 자동 평가 시스템이 초기화되었습니다.")
        except Exception as e:
            st.error(f"❌ 자동 평가 시스템 초기화 실패: {e}")
            logger.error(f"Auto evaluator initialization error: {e}")
            return
    
    # 자동 평가 실행 섹션
    st.markdown("### 📝 자동 평가 실행")
    
    col1, col2 = st.columns([2, 1])
    
    with col1:
        # 문서 청크 입력
        st.markdown("**문서 청크 입력**")
        document_chunks = st.text_area(
            "RFP 문서 청크를 입력하세요 (각 청크는 빈 줄로 구분)",
            height=200,
            placeholder="청크 1: 사업 개요 및 추진 배경...\n\n청크 2: 요구사항 및 평가 기준...\n\n청크 3: 일정 및 계약 조건..."
        )
    
    with col2:
        st.markdown("**설정**")
        num_questions = st.slider("청크당 질문 수", 1, 5, 3)
        run_evaluation = st.button("🚀 자동 평가 실행", type="primary")
    
    # 자동 평가 실행
    if run_evaluation and document_chunks.strip():
        try:
            # 청크 분리
            chunks = [chunk.strip() for chunk in document_chunks.split('\n\n') if chunk.strip()]
            
            if not chunks:
                st.error("유효한 문서 청크를 입력해주세요.")
                return
            
            with st.spinner("자동 평가를 실행 중입니다..."):
                # 전체 자동 평가 파이프라인 실행
                result = st.session_state.auto_evaluator.run_full_auto_evaluation(
                    chunks, num_questions
                )
                
                st.success(f"✅ 자동 평가 완료! {result['questions_generated']}개 질문 생성, {result['evaluations_completed']}개 평가 완료")
                
                # 결과 저장
                st.session_state.auto_evaluation_result = result
                
        except Exception as e:
            st.error(f"❌ 자동 평가 실행 중 오류: {e}")
            logger.error(f"Auto evaluation error: {e}")
    
    # 평가 결과 표시
    if 'auto_evaluation_result' in st.session_state:
        result = st.session_state.auto_evaluation_result
        
        st.markdown("### 📊 평가 결과")
        
        # 통계 요약
        stats = result['statistics']
        if stats:
            col1, col2, col3, col4 = st.columns(4)
            
            with col1:
                st.metric("총 평가 수", stats['total_evaluations'])
            with col2:
                st.metric("평균 종합 점수", f"{stats['average_scores']['overall']:.3f}")
            with col3:
                st.metric("평균 정확성", f"{stats['average_scores']['accuracy']:.3f}")
            with col4:
                st.metric("평균 완성도", f"{stats['average_scores']['completeness']:.3f}")
        
        # 상세 결과 표시
        st.markdown("### 📋 상세 평가 결과")
        
        for i, eval_result in enumerate(result['evaluation_results']):
            with st.expander(f"평가 {i+1}: {eval_result.question[:50]}..."):
                col1, col2 = st.columns([1, 1])
                
                with col1:
                    st.markdown("**질문**")
                    st.write(eval_result.question)
                    
                    st.markdown("**답변**")
                    st.write(eval_result.answer[:500] + "..." if len(eval_result.answer) > 500 else eval_result.answer)
                
                with col2:
                    st.markdown("**평가 점수**")
                    scores = eval_result.scores
                    
                    # 점수 표시
                    score_cols = st.columns(2)
                    with score_cols[0]:
                        st.metric("정확성", f"{scores.get('accuracy', 0):.3f}")
                        st.metric("완성도", f"{scores.get('completeness', 0):.3f}")
                        st.metric("관련성", f"{scores.get('relevance', 0):.3f}")
                        st.metric("명확성", f"{scores.get('clarity', 0):.3f}")
                        st.metric("구조화", f"{scores.get('structure', 0):.3f}")
                    
                    with score_cols[1]:
                        st.metric("실용성", f"{scores.get('practicality', 0):.3f}")
                        st.metric("전문성", f"{scores.get('expertise', 0):.3f}")
                        st.metric("창의성", f"{scores.get('creativity', 0):.3f}")
                        st.metric("실행가능성", f"{scores.get('feasibility', 0):.3f}")
                        st.metric("리스크분석", f"{scores.get('risk_analysis', 0):.3f}")
                    
                    st.metric("종합 점수", f"{eval_result.overall_score:.3f}", delta=None)
                
                # 강점, 약점, 개선제안
                col1, col2, col3 = st.columns(3)
                
                with col1:
                    st.markdown("**강점**")
                    for strength in eval_result.strengths:
                        st.write(f"• {strength}")
                
                with col2:
                    st.markdown("**약점**")
                    for weakness in eval_result.weaknesses:
                        st.write(f"• {weakness}")
                
                with col3:
                    st.markdown("**개선제안**")
                    for suggestion in eval_result.improvement_suggestions:
                        st.write(f"• {suggestion}")
                
                # 전반 평가
                st.markdown("**전반 평가**")
                st.write(eval_result.evaluation_notes)

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
        tab1, tab2, tab3, tab4, tab5, tab6 = st.tabs(["🔍 질의응답", "📊 품질 대시보드", "📈 대화 로그 분석", "🤖 자동 평가", "🔧 프롬프트 최적화", "⚙️ 시스템 관리"])
        
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
            # 자동 평가
            display_auto_evaluation_dashboard()
            
        with tab5:
            # 프롬프트 최적화
            display_prompt_optimization_dashboard()
            
        with tab6:
            # 시스템 관리
            display_system_management(system_selector)
        
    except Exception as e:
        st.error(f"❌ 애플리케이션 오류: {e}")
        logger.error(f"Application error: {e}")

def display_prompt_optimization_dashboard():
    """프롬프트 최적화 대시보드 표시"""
    st.markdown("## 🔧 프롬프트 최적화 시스템")
    st.markdown("LLM 기반 자동 프롬프트 개선 파이프라인 - 실제 대화 로그 데이터를 활용한 스마트 최적화")
    
    # 프롬프트 최적화 실행
    with st.container():
        st.markdown("### ⚙️ 프롬프트 최적화 실행")
        
        # 데이터 소스 선택
        st.markdown("**데이터 소스 선택:**")
        data_source = st.radio(
            "최적화에 사용할 데이터를 선택하세요:",
            ["자동 데이터 사용 (대화 로그)", "수동 입력 (기존 방식)"],
            help="자동 데이터 사용 시 실제 대화 로그에서 질문-답변 쌍을 자동으로 추출합니다"
        )
        
        if data_source == "자동 데이터 사용 (대화 로그)":
            # 자동 데이터 설정
            col1, col2 = st.columns([2, 1])
            
            with col1:
                st.markdown("**자동 데이터 설정:**")
                
                # 대화 로그 필터 옵션
                col1_1, col1_2 = st.columns(2)
                
                with col1_1:
                    # 최근 질문 수 선택
                    num_questions = st.selectbox(
                        "최근 질문 수",
                        options=[10, 20, 50, 100],
                        index=1,  # 기본값: 20
                        help="최적화에 사용할 최근 질문 수"
                    )
                    
                    # 날짜 범위 선택
                    date_range = st.selectbox(
                        "날짜 범위",
                        options=["최근 1주일", "최근 1개월", "최근 3개월", "전체"],
                        index=1,  # 기본값: 최근 1개월
                        help="분석할 대화 로그의 날짜 범위"
                    )
                
                with col1_2:
                    # 시스템 타입 필터
                    system_filter = st.selectbox(
                        "시스템 타입",
                        options=["전체", "faiss", "chromadb"],
                        help="특정 시스템의 대화만 사용"
                    )
                    
                    # 품질 점수 필터
                    min_quality = st.slider(
                        "최소 품질 점수",
                        min_value=0.0,
                        max_value=1.0,
                        value=0.0,
                        step=0.1,
                        help="이 점수 이상의 대화만 사용"
                    )
                
                # 데이터 미리보기
                if st.button("📊 데이터 미리보기", key="preview_data"):
                    try:
                        conversation_tracker = get_conversation_tracker()
                        
                        # 필터 파라미터 설정
                        filter_params = {
                            'limit': num_questions,
                            'system_type': system_filter if system_filter != "전체" else None,
                            'min_quality_score': min_quality if min_quality > 0 else None
                        }
                        
                        # 날짜 범위 적용
                        if date_range == "최근 1주일":
                            filter_params['days'] = 7
                        elif date_range == "최근 1개월":
                            filter_params['days'] = 30
                        elif date_range == "최근 3개월":
                            filter_params['days'] = 90
                        
                        # 대화 로그 조회
                        conversations = conversation_tracker.search_conversations(**filter_params)
                        
                        if conversations:
                            st.success(f"✅ {len(conversations)}개의 대화를 찾았습니다.")
                            
                            # 미리보기 표시
                            preview_df = pd.DataFrame([
                                {
                                    '질문': conv['question'][:50] + "..." if len(conv['question']) > 50 else conv['question'],
                                    '시스템': conv['system_type'],
                                    '품질점수': f"{conv['overall_quality_score']:.3f}" if conv['overall_quality_score'] else "N/A",
                                    '시간': conv['question_timestamp']
                                }
                                for conv in conversations[:10]  # 상위 10개만 미리보기
                            ])
                            st.dataframe(preview_df, use_container_width=True)
                            
                            # 통계 정보
                            if conversations:
                                quality_scores = [conv['overall_quality_score'] for conv in conversations if conv['overall_quality_score']]
                                if quality_scores:
                                    avg_quality = sum(quality_scores) / len(quality_scores)
                                    st.info(f"📊 평균 품질 점수: {avg_quality:.3f}")
                        else:
                            st.warning("⚠️ 조건에 맞는 대화를 찾을 수 없습니다.")
                            
                    except Exception as e:
                        st.error(f"❌ 데이터 미리보기 중 오류: {e}")
            
            with col2:
                st.markdown("**최적화 설정:**")
                target_satisfaction = st.slider("목표 만족도", 0.7, 0.95, 0.9, 0.05)
                max_iterations = st.slider("최대 반복 수", 3, 10, 5)
                min_improvement = st.slider("최소 개선도", 0.01, 0.1, 0.05, 0.01)
                
                # 프롬프트 타입 선택
                st.markdown("**최적화할 프롬프트 타입:**")
                prompt_type = st.selectbox(
                    "프롬프트 타입 선택",
                    ["question_generation", "evaluation", "system"],
                    format_func=lambda x: {
                        "question_generation": "질문 생성 프롬프트",
                        "evaluation": "평가 프롬프트", 
                        "system": "시스템 프롬프트"
                    }[x],
                    key="auto_prompt_type"
                )
                
                # 프롬프트 버전 선택
                st.markdown("**최적화할 프롬프트 버전:**")
                try:
                    from src.prompts.prompt_manager import get_prompt_manager
                    prompt_manager = get_prompt_manager()
                    available_versions = prompt_manager.get_available_versions()
                    current_version = prompt_manager.get_current_version()
                    
                    selected_version = st.selectbox(
                        "프롬프트 버전 선택",
                        available_versions,
                        index=available_versions.index(current_version) if current_version in available_versions else 0,
                        help="최적화할 프롬프트 버전을 선택하세요",
                        key="auto_prompt_version"
                    )
                    
                    # 버전 정보 표시
                    version_info = prompt_manager.get_version_info(selected_version)
                    if version_info:
                        st.info(f"**선택된 버전**: {selected_version} - {version_info.get('name', 'N/A')}")
                        st.caption(f"설명: {version_info.get('description', 'N/A')}")
                        
                except Exception as e:
                    st.error(f"프롬프트 버전 정보를 불러올 수 없습니다: {e}")
                    selected_version = "v3"  # 기본값
        
        else:
            # 기존 수동 입력 방식
            col1, col2 = st.columns([2, 1])
            
            with col1:
                st.markdown("**테스트 문서 청크 입력:**")
                document_chunks_text = st.text_area(
                    "최적화 테스트용 RFP 문서 청크를 입력하세요",
                    height=200,
                    placeholder="사업 개요 및 추진 배경\n\n본 사업은 디지털 전환을 통한 업무 효율성 향상을 목적으로 추진되는 사업입니다...",
                    key="manual_chunks"
                )
                
                st.markdown("**최적화할 프롬프트 타입:**")
                prompt_type = st.selectbox(
                    "프롬프트 타입 선택",
                    ["question_generation", "evaluation", "system"],
                    format_func=lambda x: {
                        "question_generation": "질문 생성 프롬프트",
                        "evaluation": "평가 프롬프트", 
                        "system": "시스템 프롬프트"
                    }[x],
                    key="manual_prompt_type"
                )
                
                # 프롬프트 버전 선택
                st.markdown("**최적화할 프롬프트 버전:**")
                try:
                    from src.prompts.prompt_manager import get_prompt_manager
                    prompt_manager = get_prompt_manager()
                    available_versions = prompt_manager.get_available_versions()
                    current_version = prompt_manager.get_current_version()
                    
                    selected_version = st.selectbox(
                        "프롬프트 버전 선택",
                        available_versions,
                        index=available_versions.index(current_version) if current_version in available_versions else 0,
                        help="최적화할 프롬프트 버전을 선택하세요",
                        key="manual_prompt_version"
                    )
                    
                    # 버전 정보 표시
                    version_info = prompt_manager.get_version_info(selected_version)
                    if version_info:
                        st.info(f"**선택된 버전**: {selected_version} - {version_info.get('name', 'N/A')}")
                        st.caption(f"설명: {version_info.get('description', 'N/A')}")
                        
                except Exception as e:
                    st.error(f"프롬프트 버전 정보를 불러올 수 없습니다: {e}")
                    selected_version = "v3"  # 기본값
            
            with col2:
                st.markdown("**최적화 설정:**")
                target_satisfaction = st.slider("목표 만족도", 0.7, 0.95, 0.9, 0.05, key="manual_target")
                max_iterations = st.slider("최대 반복 수", 3, 10, 5, key="manual_max_iter")
                min_improvement = st.slider("최소 개선도", 0.01, 0.1, 0.05, 0.01, key="manual_min_improve")
        
        # 최적화 실행 버튼
        if st.button("🚀 프롬프트 최적화 시작", type="primary"):
            if data_source == "자동 데이터 사용 (대화 로그)":
                # 자동 데이터로 최적화 실행
                try:
                    conversation_tracker = get_conversation_tracker()
                    
                    # 필터 파라미터 설정
                    filter_params = {
                        'limit': num_questions,
                        'system_type': system_filter if system_filter != "전체" else None,
                        'min_quality_score': min_quality if min_quality > 0 else None
                    }
                    
                    # 날짜 범위 적용
                    if date_range == "최근 1주일":
                        filter_params['days'] = 7
                    elif date_range == "최근 1개월":
                        filter_params['days'] = 30
                    elif date_range == "최근 3개월":
                        filter_params['days'] = 90
                    
                    # 대화 로그 조회
                    conversations = conversation_tracker.search_conversations(**filter_params)
                    
                    if not conversations:
                        st.error("❌ 조건에 맞는 대화를 찾을 수 없습니다.")
                        return
                    
                    # 대화 로그에서 질문-답변 쌍 추출
                    document_chunks = []
                    for conv in conversations:
                        # 질문과 답변을 하나의 청크로 결합
                        chunk = f"질문: {conv['question']}\n\n답변: {conv['answer']}"
                        document_chunks.append(chunk)
                    
                    st.info(f"📊 {len(document_chunks)}개의 대화를 최적화 데이터로 사용합니다.")
                    
                    # 최적화 설정
                    config = OptimizationConfig(
                        target_satisfaction=target_satisfaction,
                        max_iterations=max_iterations,
                        min_improvement=min_improvement
                    )
                    
                    # 프롬프트 최적화 실행
                    optimizer = PromptOptimizer()
                    
                    # AutoEvaluator 초기화
                    auto_evaluator = AutoEvaluator()
                    auto_evaluator.initialize()
                    
                    optimizer.initialize(
                        client=get_openai_client(),
                        prompt_manager=get_prompt_manager(),
                        auto_evaluator=auto_evaluator
                    )
                    
                    with st.spinner("실제 대화 로그 데이터로 프롬프트 최적화 실행 중..."):
                        result = optimizer.optimize_prompt(
                            prompt_type=prompt_type,
                            document_chunks=document_chunks,
                            config=config,
                            base_version=selected_version
                        )
                    
                    if result:
                        st.success(f"✅ 프롬프트 최적화 완료! 만족도: {result.satisfaction_score:.3f}")
                        
                        # 결과 표시
                        display_optimization_results(result)
                    else:
                        st.error("❌ 프롬프트 최적화 실패")
                        
                except Exception as e:
                    st.error(f"❌ 프롬프트 최적화 중 오류 발생: {str(e)}")
                    logger.error(f"Prompt optimization error: {e}")
            
            else:
                # 기존 수동 입력 방식
                if document_chunks_text.strip():
                    # 문서 청크 분할
                    document_chunks = [chunk.strip() for chunk in document_chunks_text.split('\n\n') if chunk.strip()]
                    
                    if document_chunks:
                        try:
                            # 최적화 설정
                            config = OptimizationConfig(
                                target_satisfaction=target_satisfaction,
                                max_iterations=max_iterations,
                                min_improvement=min_improvement
                            )
                            
                            # 프롬프트 최적화 실행
                            optimizer = PromptOptimizer()
                            
                            # AutoEvaluator 초기화
                            auto_evaluator = AutoEvaluator()
                            auto_evaluator.initialize()
                            
                            optimizer.initialize(
                                client=get_openai_client(),
                                prompt_manager=get_prompt_manager(),
                                auto_evaluator=auto_evaluator
                            )
                            
                            with st.spinner("프롬프트 최적화 실행 중..."):
                                result = optimizer.optimize_prompt(
                                    prompt_type=prompt_type,
                                    document_chunks=document_chunks,
                                    config=config,
                                    base_version=selected_version
                                )
                            
                            if result:
                                st.success(f"✅ 프롬프트 최적화 완료! 만족도: {result.satisfaction_score:.3f}")
                                
                                # 결과 표시
                                display_optimization_results(result)
                            else:
                                st.error("❌ 프롬프트 최적화 실패")
                                
                        except Exception as e:
                            st.error(f"❌ 프롬프트 최적화 중 오류 발생: {str(e)}")
                            logger.error(f"Prompt optimization error: {e}")
                    else:
                        st.warning("⚠️ 유효한 문서 청크를 입력해주세요")
                else:
                    st.warning("⚠️ 테스트용 문서 청크를 입력해주세요")
    
    # 최적화 히스토리
    with st.container():
        st.markdown("### 📊 최적화 히스토리")
        
        try:
            optimizer = PromptOptimizer()
            history = optimizer.get_optimization_results()
            
            if history:
                # 최적화 결과 테이블
                df_history = pd.DataFrame(history)
                df_history['created_at'] = pd.to_datetime(df_history['created_at'])
                df_history = df_history.sort_values('created_at', ascending=False)
                
                st.dataframe(
                    df_history[['version', 'satisfaction_score', 'iteration_count', 'status', 'created_at']],
                    use_container_width=True
                )
                
                # 상세 결과 보기
                if st.button("📈 상세 결과 보기"):
                    display_optimization_history_details(history)
            else:
                st.info("아직 최적화 결과가 없습니다.")
                
        except Exception as e:
            st.error(f"히스토리 조회 중 오류: {str(e)}")
            logger.error(f"Optimization history error: {e}")

def display_optimization_results(result: OptimizationResult):
    """최적화 결과 표시"""
    st.markdown("### 📈 최적화 결과")
    
    col1, col2, col3 = st.columns(3)
    
    with col1:
        st.metric("만족도 점수", f"{result.satisfaction_score:.3f}")
    
    with col2:
        st.metric("반복 횟수", result.iteration_count)
    
    with col3:
        status_color = "🟢" if result.status == "success" else "🟡" if result.status == "in_progress" else "🔴"
        st.metric("상태", f"{status_color} {result.status}")
    
    # 프롬프트 비교
    st.markdown("### 📝 프롬프트 비교")
    
    col1, col2 = st.columns(2)
    
    with col1:
        st.markdown("**원본 프롬프트:**")
        st.text_area("", result.original_prompt, height=300, disabled=True, label_visibility="collapsed", key="original_prompt")
    
    with col2:
        st.markdown("**최적화된 프롬프트:**")
        st.text_area("", result.optimized_prompt, height=300, disabled=True, label_visibility="collapsed", key="optimized_prompt")
    
    # 개선 사항
    if result.improvement_reasons:
        st.markdown("### 💡 개선 사항")
        for i, reason in enumerate(result.improvement_reasons, 1):
            st.write(f"{i}. {reason}")
    
    # 실패 사례
    if result.failed_cases:
        st.markdown("### ❌ 실패 사례")
        for i, case in enumerate(result.failed_cases[:3], 1):  # 최대 3개만 표시
            with st.expander(f"실패 사례 {i}"):
                st.write(f"**질문:** {case.get('question', 'N/A')}")
                st.write(f"**답변:** {case.get('answer', 'N/A')[:200]}...")
                st.write(f"**점수:** {case.get('overall_score', 0):.3f}")

def display_optimization_history_details(history: List[Dict[str, Any]]):
    """최적화 히스토리 상세 표시"""
    st.markdown("### 📊 최적화 히스토리 상세")
    
    for i, record in enumerate(history[:5]):  # 최근 5개만 표시
        with st.expander(f"최적화 {i+1}: {record['version']} (만족도: {record['satisfaction_score']:.3f})"):
            col1, col2 = st.columns(2)
            
            with col1:
                st.markdown("**기본 정보:**")
                st.write(f"- 버전: {record['version']}")
                st.write(f"- 만족도: {record['satisfaction_score']:.3f}")
                st.write(f"- 반복 횟수: {record['iteration_count']}")
                st.write(f"- 상태: {record['status']}")
                st.write(f"- 생성일: {record['created_at']}")
            
            with col2:
                st.markdown("**개선 사항:**")
                try:
                    improvement_reasons = json.loads(record['improvement_reasons'])
                    for reason in improvement_reasons:
                        st.write(f"- {reason}")
                except:
                    st.write("개선 사항 정보 없음")
            
            # 프롬프트 미리보기
            st.markdown("**최적화된 프롬프트 미리보기:**")
            optimized_prompt = record['optimized_prompt'][:500] + "..." if len(record['optimized_prompt']) > 500 else record['optimized_prompt']
            st.text(optimized_prompt)

def get_openai_client():
    """OpenAI 클라이언트 반환"""
    try:
        from openai import OpenAI
        return OpenAI()
    except Exception as e:
        logger.error(f"Failed to create OpenAI client: {e}")
        return None

def get_prompt_manager():
    """프롬프트 매니저 반환"""
    try:
        from src.prompts.prompt_manager import PromptManager
        return PromptManager()
    except Exception as e:
        logger.error(f"Failed to create PromptManager: {e}")
        return None

if __name__ == "__main__":
    main()
