"""
비동기 RFP Generator
OpenAI API 호출을 비동기로 처리하여 동시 사용자 지원
"""
import asyncio
import time
from typing import List, Dict, Any, Optional
import httpx
import json
import sys
import os

# 프로젝트 루트를 Python 경로에 추가
sys.path.append('/home/spai0316/projectmission2/src')

from generation.generator import RFPGenerator
from retrieval.retriever import RetrievalResult

class AsyncRFPGenerator(RFPGenerator):
    """비동기 RFP Generator"""
    
    def __init__(self, model: str = "gpt-4.1-mini", temperature: float = 0.7, max_tokens: int = 2000):
        super().__init__(model, temperature, max_tokens)
        self.client = None  # httpx.AsyncClient로 초기화
    
    async def initialize(self):
        """비동기 초기화"""
        import openai
        from dotenv import load_dotenv
        
        load_dotenv()
        
        # OpenAI API 키 설정
        api_key = os.getenv("OPENAI_API_KEY")
        if not api_key:
            raise ValueError("OPENAI_API_KEY 환경변수가 설정되지 않았습니다")
        
        # httpx.AsyncClient 생성
        self.client = httpx.AsyncClient(
            base_url="https://api.openai.com/v1",
            headers={
                "Authorization": f"Bearer {api_key}",
                "Content-Type": "application/json"
            },
            timeout=60.0
        )
        
        print(f"✅ AsyncRFPGenerator 초기화 완료: {self.model}")
    
    async def generate_response_async(
        self, 
        question: str, 
        retrieved_results: List[RetrievalResult],
        use_conversation_history: bool = True
    ) -> str:
        """비동기 응답 생성"""
        start_time = time.time()
        
        try:
            # 컨텍스트 구성
            context = self._build_context(retrieved_results)
            
            # 메시지 구성
            messages = []
            if use_conversation_history:
                # TODO: 대화 히스토리 구현
                pass
            
            # 시스템 프롬프트
            system_prompt = self._create_system_prompt()
            messages.append({"role": "system", "content": system_prompt})
            
            # 사용자 쿼리와 컨텍스트
            user_message = self._create_user_message(question, context)
            messages.append({"role": "user", "content": user_message})
            
            # 비동기 OpenAI API 호출
            response = await self.client.post(
                "/chat/completions",
                json={
                    "model": self.model,
                    "messages": messages,
                    "temperature": self.temperature,
                    "max_tokens": self.max_tokens
                }
            )
            
            if response.status_code != 200:
                raise Exception(f"OpenAI API 오류: {response.status_code} - {response.text}")
            
            result = response.json()
            answer = result["choices"][0]["message"]["content"]
            
            generation_time = time.time() - start_time
            print(f"✅ 비동기 응답 생성 완료: {generation_time:.2f}초")
            
            return answer
            
        except Exception as e:
            print(f"❌ 비동기 응답 생성 오류: {str(e)}")
            return f"죄송합니다. 응답 생성 중 오류가 발생했습니다: {str(e)}"
    
    async def close(self):
        """리소스 정리"""
        if self.client:
            await self.client.aclose()
