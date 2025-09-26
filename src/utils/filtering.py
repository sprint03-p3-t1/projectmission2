import re
import logging
import pandas as pd
from datetime import datetime
from typing import List, Dict, Union, Any, Optional

FILTER_MAPPER = {
    "사업금액": {
        "field": "사업 금액", 
        "type": int,
        "pattern": r"(사업\s?금액)?\s*(\d+[억만천백조]+)\s*(이상|이하|초과|미만)?"
    },
    "입찰시작일": {
        "field": "입찰 참여 시작일",
        "type": "date",
        "pattern": r"(입찰\s?시작일|참여\s?시작일)[^\d]*(\d{4})[년\s]*(\d{1,2})?[월]?"
    },
    "입찰마감일": {
        "field": "입찰 참여 마감일",
        "type": "date",
        "pattern": r"(입찰\s?마감일|참여\s?마감일)[^\d]*(\d{4})[년\s]*(\d{1,2})?[월]?"
    },
    "입찰공고일": {
        "field": "공개 일자",
        "type": "date",
        "pattern": r"(입찰\s?공고일)[^\d]*(\d{4})[년\s]*(\d{1,2})?[월]?"
    },
    "발주기관": {
        "field": "발주 기관",  
        "type": str,
        "pattern": r"(한국농어촌공사|조달청|도로공사|[가-힣]{2,})"
    },
    "공고번호": {
        "field": "공고 번호", 
        "type": str,
        "pattern": r"(공고번호\s?\d{4}-?\d{3,})"
    },
}
def normalize_keywords(keywords: list[str]) -> set[str]:
    """키워드 리스트를 정규화하여 비교 가능하게 변환"""
    return {k.replace(" ", "").lower() for k in keywords}

def safe_parse_date(value: str) -> Optional[datetime]:
    if not isinstance(value, str):
        return None
    try:
        parts = [int(p) for p in re.findall(r"\d+", value)]
        if len(parts) >= 2:
            year, month = parts[0], parts[1]
            day = parts[2] if len(parts) > 2 else 1
            return datetime(year, month, day)
    except Exception as e:
        logging.warning(f"❌ 날짜 파싱 실패: {value} → {e}")
        return None

def parse_korean_number(text: str) -> int:
    unit_values = {
        "십": 10,
        "백": 100,
        "천": 1000,
        "만": 10_000,
        "억": 100_000_000,
        "조": 1_000_000_000_000
    }

    # 정규화
    text = text.replace(",", "").replace("억원", "억").replace("백만원", "백만") \
               .replace("천만원", "천만").replace("만원", "만").replace("원", "").strip()
    print("🌸 처리전 text:", text)

    # 단위별 블록 추출
    blocks = re.findall(r"(\d+)([십백천만억조]+)", text)

    total = 0
    current_block = 0
    last_big_unit = 1

    for num_str, unit_str in blocks:
        num = int(num_str)
        small_unit = 1
        big_unit = 1

        for char in unit_str:
            if char in ["십", "백", "천"]:
                small_unit *= unit_values[char]
            elif char in ["만", "억", "조"]:
                big_unit = unit_values[char]

        current_block += num * small_unit

        # 큰 단위가 붙었으면 전체 블록에 곱해서 total에 더함
        if big_unit > 1:
            total += current_block * big_unit
            current_block = 0

    total += current_block
    print("🌸 처리완료후:", total)
    return total


# 예시:
# parse_korean_number("5천만원")        # 50000000
# parse_korean_number("1억 2천만원")    # 120000000
# parse_korean_number("2천만")         # 20000000
# parse_korean_number("3백억")         # 30000000000
# parse_korean_number("456백만")       # 456000000


def convert_value(raw: str, value_type):
    """필터 추출용 값 변환"""
    if value_type in ("int", int):
        return parse_korean_number(raw)
    elif value_type in ("date", datetime):
        return safe_parse_date(raw)
    elif value_type in ("float", float):
        try:
            return float(str(raw).replace(",", "").strip())
        except ValueError:
            return None
    else:
        return raw.strip()


OPERATOR_FUNC = {
    ">=": lambda v, t: v >= t,
    "<=": lambda v, t: v <= t,
    ">":  lambda v, t: v > t,
    "<":  lambda v, t: v < t,
    "=":  lambda v, t: v == t,
    "~":  lambda v, t: abs(v - t) <= t * 0.1
}

def extract_operator(text: str, context: str = "") -> str:
    full_text = text + " " + context
    if "이후" in full_text or "부터" in full_text or "최소" in full_text or "이상" in full_text:
        return ">="
    elif "이전" in full_text or "까지" in full_text or "최대" in full_text or "이하" in full_text:
        return "<="
    elif "초과" in full_text:
        return ">"
    elif "미만" in full_text:
        return "<"
    elif "약" in full_text or "정도" in full_text:
        return "~"
    return "="


# 🚨 기관명/파일명 필터링 시 제거할 잡음 단어 목록
NOISE_WORDS = {
     # 날짜/시점 관련
    "년", "월", "일", "년도", "2024", "2025",

    # 입찰/공고 관련
    "입찰", "공고", "재공고", "긴급", "협상", "사전공개",

    # 금액 관련
    "원", "예산",

}

def extract_field_filter_by_tokens(
    query: str,
    field_values: List[str],
    tokenizer,
    field_name: str,
    use_exact_match: bool = True,
    threshold: float = 0.5
) -> Optional[Dict[str, Dict]]:
    
    query_tokens = set(tokenizer.tokenize_korean(query, use_bigrams=False)) - NOISE_WORDS
    print("❤️query_tokens", query_tokens)
    logging.debug(f"🧹 필터링용 토큰셋: {query_tokens}")

    # 1️⃣ 정확 매칭 우선
    if use_exact_match:
        for value in field_values:
            if value and value in query:
                return {field_name: {"value": value, "operator": "="}}

    # 유사도 기반 매칭
    best_match = None
    best_score = 0
    for value in field_values:
        match_count = sum(1 for token in query_tokens if token in value)
        score = match_count / len(query_tokens) if query_tokens else 0
        if score > best_score and score > threshold:
            best_match = value
            best_score = score
           
    if best_match:
        print("❤️best_match", best_match, best_score)
        return {field_name: {"value": best_match, "operator": "="}}

    return None


def extract_filters(query: str, meta_df: pd.DataFrame, tokenizer) -> Dict[str, Dict]:
    filters = {}
    
    # 3️⃣ 정규식 기반 필터 추출
    for keyword, filter_info in FILTER_MAPPER.items():
        field_name = filter_info.get("field")
    
        # ✅ 기관명/파일명은 정규식으로 추출하지 않음
        if field_name in filters or field_name in ["발주 기관", "파일명"]:
            continue
            
        match = re.search(filter_info['pattern'], query)
        if match:
            value_type = filter_info.get('type')
            if value_type == "date":
                year = match.group(2)
                month = match.group(3) if match.lastindex and match.lastindex >= 3 and match.group(3) else "1"
                raw_value = f"{year}년 {month}월"
            elif value_type == int:
                raw_value = match.group(2)
                condition = match.group(3) or ""
                operator = extract_operator(condition, query)
            else:
                raw_value = match.group(1)
                operator = "="
    
            value = convert_value(raw_value, value_type)
            operator = extract_operator(raw_value, query) if value_type in ["date", int] else "="
            if value is not None:
                filters[field_name] = {"value": value, "operator": operator}
                logging.info(f"📌 {field_name} 필터 적용됨: {value} ({operator})")

    # 발주 기관 필터링
    agency_filter_applied = False
    if "발주 기관" in meta_df.columns:
        agency_list = meta_df["발주 기관"].dropna().unique().tolist()
        agency_filter = extract_field_filter_by_tokens(
            query=query,
            field_values=agency_list,
            tokenizer=tokenizer,
            field_name="발주 기관",
            use_exact_match=True,
            threshold=0.5
        )
        
        print("❤️agency_filter : ", agency_filter)
        if agency_filter:
            filters.update(agency_filter)
            agency_filter_applied = True
            logging.info(f"🏢 발주 기관 필터 적용됨: {agency_filter['발주 기관']['value']}")
   
    # 파일명 보조 필터링 (기관 필터 없을 때만)
    if not agency_filter_applied and "파일명" in meta_df.columns:
        filename_list = meta_df["파일명"].dropna().unique().tolist()
        print("❤️파일필터작동 ")
        filename_filter = extract_field_filter_by_tokens(
            query=query,
            field_values=filename_list,
            tokenizer=tokenizer,
            field_name="파일명",
            use_exact_match=True,
            threshold=0.5  # ✅ 더 유연하게
        )
        print("❤️file_filter : ", filename_filter)
        if filename_filter:
            filters.update(filename_filter)
            logging.info(f"📁 파일명 필터 적용됨: {filename_filter['파일명']['value']}")

    return filters


def is_valid_value(value):
    """값이 유효한지 확인하는 헬퍼 함수"""
    if value is None or str(value).strip() in ["", "미정", "nan"]:
        return False
    return True

def check_filter_match(data: Union[Dict, Any], filters: Dict[str, Dict]) -> bool:
    for field, condition in filters.items():
        raw_value = data.get(field)

        if not is_valid_value(raw_value):
            return False

        target = condition["value"]
        operator = condition["operator"]

        # ✅ raw_value가 target과 같은 타입인지 확인
        try:
            if isinstance(target, int):
                value = int(str(raw_value).replace(",", "").strip())
            elif isinstance(target, float):
                value = float(str(raw_value).replace(",", "").strip())
            elif isinstance(target, datetime):
                value = safe_parse_date(str(raw_value))
            else:
                value = str(raw_value).strip()
        except Exception:
            return False

        compare_func = OPERATOR_FUNC.get(operator)
        if not compare_func:
            return False

        try:
            return compare_func(value, target)
        except TypeError:
            return False

    return True

# 예시: '사업금액 5천만원 이상인 공고 찾아줘'