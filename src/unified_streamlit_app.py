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
    return SystemSelector(config)

def display_header():
    """헤더 표시"""
    st.markdown("""
    <div class="main-header">
        <h1>🤖 RFP RAG System - 통합 검색</h1>
        <p>FAISS vs ChromaDB 하이브리드 검색 시스템 비교</p>
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
        st.sidebar.success(f"✅ {system_info['name']} 초기화됨")
    else:
        st.sidebar.warning(f"⚠️ {system_info['name']} 초기화 필요")
    
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
                st.success("✅ 초기화됨")
            else:
                st.warning("⚠️ 초기화 필요")
            
            st.markdown(f"**모델**: {system_info['embedder_model']}")
            st.markdown(f"**벡터 DB**: {system_info['vector_db_type']}")
            
            if st.button(f"🗑️ {system_name} 캐시 정리", key=f"clear_{system_name}"):
                with st.spinner("캐시 정리 중..."):
                    system_selector.clear_cache(system_name)
                st.success("캐시 정리 완료!")
                st.rerun()

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
        
        # 비교 모드 설정
        comparison_mode = display_comparison_mode()
        
        # 검색 인터페이스
        display_search_interface(system_selector, selected_system, comparison_mode)
        
        # 시스템 관리
        display_system_management(system_selector)
        
    except Exception as e:
        st.error(f"❌ 애플리케이션 오류: {e}")
        logger.error(f"Application error: {e}")

if __name__ == "__main__":
    main()
