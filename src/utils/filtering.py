import re
import logging
import pandas as pd
from datetime import datetime
from typing import List, Dict, Set, Tuple, Union, Any, Optional


FILTER_MAPPER = {
    "사업금액": {
        "field": "사업 금액", 
        "type": int,
        "pattern": r"(\d+[억만천백조])(\s*이상|\s*이하|\s*초과|\s*미만)"
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
        "일": 1,
        "십": 10,
        "백": 100,
        "천": 1_000,
        "만": 10_000,
        "억": 100_000_000,
        "조": 1_000_000_000_000
    }

    text = text.replace(",", "").strip()
    total = 0

    # 예: "2천만" → 숫자 + 단위들 분리
    blocks = re.findall(r"(\d+)([가-힣]+)", text)

    for num_str, unit_str in blocks:
        num = int(num_str)
        multiplier = 1
        for char in unit_str:
            if char in unit_values:
                multiplier *= unit_values[char]
        total += num * multiplier

    # 단위 없는 숫자 처리
    if not blocks:
        digits = re.findall(r"\d+", text)
        if digits:
            total += int(digits[0])

    return total

# 예시:
# parse_korean_number("5천만원")        # 50000000
# parse_korean_number("1억 2천만원")    # 120000000
# parse_korean_number("2천만")         # 20000000
# parse_korean_number("3백억")         # 30000000000
# parse_korean_number("456백만")       # 456000000


def convert_value(raw: str, value_type):
    if value_type == int:
        return parse_korean_number(raw)
    elif value_type == "date":
        return safe_parse_date(raw)  
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


# 🚨 기관명 필터링 시 제거할 잡음 단어 목록
NOISE_WORDS: Set[str] = {"입찰", "공고", "입찰공고", "입찰공고일", "참여"}

def extract_agency_filter_by_tokens(query: str, agency_list: list, tokenizer) -> str:
    query_tokens = set(tokenizer.tokenize(query))
    filtered_query_tokens = query_tokens - NOISE_WORDS
    
    best_match = None
    best_score = 0

    for agency in agency_list:
        # 2. 기관명 토큰화 및 잡음 제거
        agency_tokens = set(tokenizer.tokenize(agency))
        filtered_agency_tokens = agency_tokens - NOISE_WORDS

        # 🚨 필터링 후 기관명 토큰이 없을 경우 건너뜀 (ZeroDivisionError 방지)
        if not filtered_agency_tokens:
            continue

        # 3. 필터링된 토큰으로 유사도 계산
        overlap = filtered_query_tokens & filtered_agency_tokens
        score = len(overlap) / len(filtered_agency_tokens)

        # 4. 임계값(0.5)을 넘는 가장 높은 점수를 가진 기관을 선택
        if score > best_score and score > 0.5:
            best_match = agency
            best_score = score

    return best_match

def extract_filters(query: str, meta_df: pd.DataFrame, tokenizer) -> Dict[str, Dict]:
    filters = {}

    # 1️⃣ 기관명 우선 추출 (토큰 기반)
    agency_list = meta_df["발주 기관"].dropna().unique().tolist()  # ✅ JSON 필드명 기준
    agency_match = extract_agency_filter_by_tokens(query, agency_list, tokenizer)
    if agency_match:
        filters["발주 기관"] = {"value": agency_match, "operator": "="}  # ✅ JSON 필드명 기준
        logging.info(f"🏢 기관명 우선 필터 적용됨: {agency_match}")
        #query = query.replace(agency_match, "").strip() # ✅ 질문에서 기관명 제거

    # 2️⃣ 나머지 필터 정규식 기반 추출
    for keyword, filter_info in FILTER_MAPPER.items():
        field_name = filter_info.get("field")  # ✅ JSON 필드명 기준

        # 기관명은 이미 처리했으므로 건너뜀
        if field_name == "발주 기관":
            continue

        match = re.search(filter_info['pattern'], query)
        if match:
            value_type = filter_info.get('type')

            if value_type == "date":
                year = match.group(2)
                month = match.group(3) if match.lastindex and match.lastindex >= 3 and match.group(3) else "1"
                raw_value = f"{year}년 {month}월"
            else:
                raw_value = match.group(1)

            value = convert_value(raw_value, value_type)
            operator = extract_operator(raw_value, query) if value_type in ["date", int] else "="
            if value is not None:
                filters[field_name] = {"value": value, "operator": operator}

    return filters


def is_valid_value(value):
    """값이 유효한지 확인하는 헬퍼 함수"""
    if value is None or str(value).strip() in ["", "미정", "nan"]:
        return False
    return True

def convert_value_to_target_type(raw_value, target_type):
    """대상 타입에 맞게 값을 변환"""
    try:
        if target_type is datetime:
            return safe_parse_date(str(raw_value))
        elif target_type in (int, float):
            return float(str(raw_value).replace(",", "").strip())
        return raw_value
    except (ValueError, TypeError):
        return None
    
def check_filter_match(data: Union[Dict, Any], filters: Dict[str, Dict]) -> bool:
    """
    필터 조건을 만족하는지 확인하는 범용 함수
    - data: dict 또는 row 또는 doc.metadata
    - filters: {field: {"value": ..., "operator": ...}}
    """
    for field, condition in filters.items():
        raw_value = data.get(field)
        
        # 1. 값의 유효성 검사
        if not is_valid_value(raw_value):
            return False

        target = condition["value"]
        operator = condition["operator"]

        # 2. 값의 타입 변환
        converted_value = convert_value_to_target_type(raw_value, type(target))
        if converted_value is None:
            return False

        # 3. 연산자 적용 및 비교
        compare_func = OPERATOR_FUNC.get(operator)
        if not compare_func:
            return False

        try:
            return compare_func(converted_value, target)
        except TypeError:
            return False
        
    return True

# 예시: '사업금액 5천만원 이상인 공고 찾아줘'