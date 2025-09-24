"""
FastAPI 기반 RFP RAG 시스템 API
"""
from fastapi import FastAPI, HTTPException, BackgroundTasks
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import HTMLResponse
from pydantic import BaseModel
from typing import List, Optional, Dict, Any
import asyncio
import time
import sys
import os

# 프로젝트 루트를 Python 경로에 추가
sys.path.append('/home/spai0316/projectmission2/src')

from rfp_rag_main import RFPRAGSystem

# FastAPI 앱 생성
app = FastAPI(
    title="RFP RAG System API",
    description="RFP 문서 검색 및 질의응답 API",
    version="1.0.0"
)

# CORS 설정 (프론트엔드 연결용)
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # 프로덕션에서는 특정 도메인으로 제한
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# 전역 RAG 시스템 인스턴스
rag_system = None

# Pydantic 모델들
class SearchRequest(BaseModel):
    keyword: Optional[str] = None
    발주기관: Optional[str] = None
    최소금액: Optional[int] = None
    최대금액: Optional[int] = None
    limit: int = 10

class QuestionRequest(BaseModel):
    question: str
    use_conversation_history: bool = True

class SearchResponse(BaseModel):
    results: List[Dict[str, Any]]
    total_count: int
    search_time: float

class QuestionResponse(BaseModel):
    answer: str
    sources: List[Dict[str, Any]] = []
    generation_time: float = 0.0
    search_time: float = 0.0

# RAG 시스템 초기화
@app.on_event("startup")
async def startup_event():
    global rag_system
    print("🚀 RAG 시스템 초기화 중...")
    rag_system = RFPRAGSystem('data/preprocess/json')
    rag_system.initialize()
    print("✅ RAG 시스템 초기화 완료!")

# 헬스 체크
@app.get("/health")
async def health_check():
    return {
        "status": "healthy",
        "rag_initialized": rag_system is not None,
        "timestamp": time.time()
    }

# 문서 검색 API
@app.post("/api/search", response_model=SearchResponse)
async def search_documents(request: SearchRequest):
    if not rag_system:
        raise HTTPException(status_code=503, detail="RAG 시스템이 초기화되지 않았습니다")
    
    start_time = time.time()
    
    try:
        # 검색 필터 구성
        filters = {}
        if request.keyword:
            filters['키워드'] = request.keyword
        if request.발주기관:
            filters['발주기관'] = request.발주기관
        if request.최소금액:
            filters['최소금액'] = request.최소금액
        if request.최대금액:
            filters['최대금액'] = request.최대금액
        
        # 문서 검색
        results = rag_system.search_documents(**filters)
        
        # 결과 제한
        if request.limit:
            results = results[:request.limit]
        
        # 응답 형식 변환
        search_results = []
        for doc in results:
            search_results.append({
                "사업명": doc.사업명,
                "발주기관": doc.발주기관,
                "사업금액": doc.사업금액,
                "계약기간": doc.계약기간,
                "사업개요": doc.사업개요[:200] + "..." if len(doc.사업개요) > 200 else doc.사업개요
            })
        
        search_time = time.time() - start_time
        
        return SearchResponse(
            results=search_results,
            total_count=len(search_results),
            search_time=search_time
        )
        
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"검색 중 오류 발생: {str(e)}")

# 질의응답 API
@app.post("/api/ask", response_model=QuestionResponse)
async def ask_question(request: QuestionRequest):
    if not rag_system:
        raise HTTPException(status_code=503, detail="RAG 시스템이 초기화되지 않았습니다")
    
    start_time = time.time()
    
    try:
        # 질문 처리 (비동기로 실행) - ask_detailed 사용
        rag_response = await asyncio.get_event_loop().run_in_executor(
            None, 
            rag_system.ask_detailed, 
            request.question
        )
        
        total_time = time.time() - start_time
        
        # RAGResponse 객체를 dict로 변환하여 처리
        if hasattr(rag_response, 'answer'):
            answer = str(rag_response.answer)
            sources = []
            # retrieved_chunks가 있으면 소스 정보 추가
            if hasattr(rag_response, 'retrieved_chunks') and rag_response.retrieved_chunks:
                try:
                    # retrieved_chunks가 이미 dict 형태로 변환되어 있음
                    sources = [{"chunk_id": chunk.get('chunk_id', 'unknown'), 
                              "content": chunk.get('content', '')[:200] + "..."} 
                              for chunk in rag_response.retrieved_chunks[:3]]
                except:
                    sources = []
        else:
            # 문자열 응답인 경우
            answer = str(rag_response)
            sources = []
        
        return QuestionResponse(
            answer=answer,
            sources=sources,
            generation_time=total_time,
            search_time=0.0
        )
        
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"질의응답 중 오류 발생: {str(e)}")

# 문서 통계 API
@app.get("/api/stats")
async def get_document_stats():
    if not rag_system:
        raise HTTPException(status_code=503, detail="RAG 시스템이 초기화되지 않았습니다")
    
    try:
        stats = rag_system.get_document_summary()
        return stats
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"통계 조회 중 오류 발생: {str(e)}")

# 간단한 웹 인터페이스 (개발용)
@app.get("/", response_class=HTMLResponse)
async def read_root():
    return """
    <!DOCTYPE html>
    <html>
    <head>
        <title>RFP RAG System</title>
        <meta charset="utf-8">
        <style>
            body { font-family: Arial, sans-serif; margin: 40px; background: #1e1e1e; color: #fff; }
            .container { max-width: 800px; margin: 0 auto; }
            .header { text-align: center; margin-bottom: 40px; }
            .api-section { background: #2d2d2d; padding: 20px; margin: 20px 0; border-radius: 8px; }
            .endpoint { background: #3d3d3d; padding: 15px; margin: 10px 0; border-radius: 5px; }
            .method { color: #4fc3f7; font-weight: bold; }
            .url { color: #81c784; }
            code { background: #1e1e1e; padding: 2px 6px; border-radius: 3px; }
        </style>
    </head>
    <body>
        <div class="container">
            <div class="header">
                <h1>🚀 RFP RAG System API</h1>
                <p>RFP 문서 검색 및 질의응답 API 서버</p>
            </div>
            
            <div class="api-section">
                <h2>📚 API 엔드포인트</h2>
                
                <div class="endpoint">
                    <div class="method">GET</div>
                    <div class="url">/health</div>
                    <p>서버 상태 확인</p>
                </div>
                
                <div class="endpoint">
                    <div class="method">POST</div>
                    <div class="url">/api/search</div>
                    <p>RFP 문서 검색</p>
                    <code>{"keyword": "구미아시아육상경기", "limit": 10}</code>
                </div>
                
                <div class="endpoint">
                    <div class="method">POST</div>
                    <div class="url">/api/ask</div>
                    <p>질의응답</p>
                    <code>{"question": "구미아시아육상경기 사업의 주요 요구사항은?"}</code>
                </div>
                
                <div class="endpoint">
                    <div class="method">GET</div>
                    <div class="url">/api/stats</div>
                    <p>문서 통계 조회</p>
                </div>
            </div>
            
            <div class="api-section">
                <h2>📖 API 문서</h2>
                <p><a href="/docs" style="color: #4fc3f7;">Swagger UI</a> | <a href="/redoc" style="color: #4fc3f7;">ReDoc</a></p>
            </div>
        </div>
    </body>
    </html>
    """

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8501)
