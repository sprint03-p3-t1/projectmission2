import os
from dotenv import load_dotenv
from typing import List, Union

from langchain.chat_models import ChatOpenAI
from langchain.prompts import ChatPromptTemplate
from langchain.chains import LLMChain

# .env 파일 로드
load_dotenv()
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")

# LangChain QA 체인 초기화 함수
def get_qa_chain():
    llm = ChatOpenAI(
        openai_api_key=OPENAI_API_KEY,
        model_name="gpt-4o-mini",# gpt5는 너무 느림. 심지어 5는 temperature 무조건 1
        temperature=1,
    )

    prompt = ChatPromptTemplate.from_template("""
    다음은 사용자의 질문과 관련된 검색 결과입니다:

    {context}

    위 내용을 바탕으로 사용자의 질문에 대해 간결하고 정확하게 답변해주세요:
    질문: {question}
    """)

    return LLMChain(llm=llm, prompt=prompt)

# 문서 병합 함수
def merge_docs_to_text(docs):
    """
    의미 기반 검색 결과(Document 리스트)를 받아서
    LLM context로 합치는 함수.
    (문서 전체 내용을 합침, 잘라내기 제한 없음 : 필요시 제한.)
    """
    if not docs:
        return ""

    merged = []
    for doc in docs:
        text = doc.page_content.strip()
        merged.append(f"[출처: {doc.metadata.get('파일명', '❓')}]\n{text}")

    return "\n\n".join(merged)
