"""
메타데이터 처리 모듈
RFP 문서의 메타데이터를 처리하고 필터링합니다.
"""

import logging
from typing import Dict, List, Any, Optional
from dataclasses import dataclass
import pandas as pd
from fuzzywuzzy import fuzz
from fuzzywuzzy.process import extractOne

logger = logging.getLogger(__name__)


@dataclass
class MetadataFilter:
    """메타데이터 필터 설정"""
    organization: Optional[str] = None
    business_name: Optional[str] = None
    category: Optional[str] = None
    budget_min: Optional[float] = None
    budget_max: Optional[float] = None
    date_from: Optional[str] = None
    date_to: Optional[str] = None


class MetadataProcessor:
    """메타데이터 처리 클래스"""

    def __init__(self, metadata_df: pd.DataFrame = None):
        """
        MetadataProcessor 초기화

        Args:
            metadata_df: 메타데이터 DataFrame
        """
        self.metadata_df = metadata_df.copy() if metadata_df is not None else pd.DataFrame()
        self._preprocess_metadata()

    def _preprocess_metadata(self):
        """메타데이터 전처리"""
        if self.metadata_df.empty:
            return

        # 결측치 처리
        self.metadata_df = self.metadata_df.fillna('')

        # 날짜 형식 통일
        date_columns = [col for col in self.metadata_df.columns if 'date' in col.lower() or '일자' in col]
        for col in date_columns:
            if col in self.metadata_df.columns:
                self.metadata_df[col] = pd.to_datetime(self.metadata_df[col], errors='coerce')

        # 예산 관련 컬럼 숫자형 변환
        budget_columns = [col for col in self.metadata_df.columns if 'budget' in col.lower() or '예산' in col or '금액' in col]
        for col in budget_columns:
            if col in self.metadata_df.columns:
                # 숫자만 추출하여 변환
                self.metadata_df[col] = self.metadata_df[col].astype(str).str.replace(r'[^\d]', '', regex=True)
                self.metadata_df[col] = pd.to_numeric(self.metadata_df[col], errors='coerce')

        logger.info(f"메타데이터 전처리 완료: {len(self.metadata_df)}개 레코드")

    def filter_documents(self, filter_criteria: MetadataFilter,
                        fuzzy_threshold: int = 80) -> pd.DataFrame:
        """
        메타데이터 필터링 수행

        Args:
            filter_criteria: 필터링 기준
            fuzzy_threshold: 퍼지 매칭 임계값 (0-100)

        Returns:
            필터링된 메타데이터 DataFrame
        """
        if self.metadata_df.empty:
            return pd.DataFrame()

        filtered_df = self.metadata_df.copy()

        # 기관명 필터링
        if filter_criteria.organization:
            filtered_df = self._filter_by_organization(
                filtered_df, filter_criteria.organization, fuzzy_threshold
            )

        # 사업명 필터링
        if filter_criteria.business_name:
            filtered_df = self._filter_by_business_name(
                filtered_df, filter_criteria.business_name, fuzzy_threshold
            )

        # 카테고리 필터링
        if filter_criteria.category:
            filtered_df = self._filter_by_category(
                filtered_df, filter_criteria.category, fuzzy_threshold
            )

        # 예산 범위 필터링
        if filter_criteria.budget_min is not None or filter_criteria.budget_max is not None:
            filtered_df = self._filter_by_budget_range(
                filtered_df, filter_criteria.budget_min, filter_criteria.budget_max
            )

        # 날짜 범위 필터링
        if filter_criteria.date_from or filter_criteria.date_to:
            filtered_df = self._filter_by_date_range(
                filtered_df, filter_criteria.date_from, filter_criteria.date_to
            )

        logger.info(f"필터링 결과: {len(filtered_df)}개 문서")
        return filtered_df

    def _filter_by_organization(self, df: pd.DataFrame, org_name: str,
                               threshold: int) -> pd.DataFrame:
        """기관명으로 필터링"""
        org_columns = [col for col in df.columns if 'org' in col.lower() or '기관' in col or '발주' in col]

        if not org_columns:
            logger.warning("기관명 관련 컬럼을 찾을 수 없습니다.")
            return df

        # 가장 적합한 컬럼 선택
        org_column = org_columns[0]

        # 퍼지 매칭으로 필터링
        mask = df[org_column].astype(str).apply(
            lambda x: fuzz.partial_ratio(org_name.lower(), x.lower()) >= threshold
        )

        return df[mask]

    def _filter_by_business_name(self, df: pd.DataFrame, business_name: str,
                                threshold: int) -> pd.DataFrame:
        """사업명으로 필터링"""
        name_columns = [col for col in df.columns if 'name' in col.lower() or '사업' in col or '프로젝트' in col]

        if not name_columns:
            logger.warning("사업명 관련 컬럼을 찾을 수 없습니다.")
            return df

        # 가장 적합한 컬럼 선택
        name_column = name_columns[0]

        # 퍼지 매칭으로 필터링
        mask = df[name_column].astype(str).apply(
            lambda x: fuzz.partial_ratio(business_name.lower(), x.lower()) >= threshold
        )

        return df[mask]

    def _filter_by_category(self, df: pd.DataFrame, category: str,
                           threshold: int) -> pd.DataFrame:
        """카테고리로 필터링"""
        cat_columns = [col for col in df.columns if 'category' in col.lower() or '분류' in col or '유형' in col]

        if not cat_columns:
            logger.warning("카테고리 관련 컬럼을 찾을 수 없습니다.")
            return df

        cat_column = cat_columns[0]

        # 퍼지 매칭으로 필터링
        mask = df[cat_column].astype(str).apply(
            lambda x: fuzz.partial_ratio(category.lower(), x.lower()) >= threshold
        )

        return df[mask]

    def _filter_by_budget_range(self, df: pd.DataFrame,
                               budget_min: Optional[float],
                               budget_max: Optional[float]) -> pd.DataFrame:
        """예산 범위로 필터링"""
        budget_columns = [col for col in df.columns if 'budget' in col.lower() or '예산' in col or '금액' in col]

        if not budget_columns:
            logger.warning("예산 관련 컬럼을 찾을 수 없습니다.")
            return df

        budget_column = budget_columns[0]

        # 유효한 예산 값만 필터링
        valid_budget = df[budget_column].notna() & (df[budget_column] > 0)

        if budget_min is not None:
            valid_budget &= (df[budget_column] >= budget_min)

        if budget_max is not None:
            valid_budget &= (df[budget_column] <= budget_max)

        return df[valid_budget]

    def _filter_by_date_range(self, df: pd.DataFrame,
                             date_from: Optional[str],
                             date_to: Optional[str]) -> pd.DataFrame:
        """날짜 범위로 필터링"""
        date_columns = [col for col in df.columns if 'date' in col.lower() or '일자' in col]

        if not date_columns:
            logger.warning("날짜 관련 컬럼을 찾을 수 없습니다.")
            return df

        date_column = date_columns[0]

        # 날짜 형식 변환
        df[date_column] = pd.to_datetime(df[date_column], errors='coerce')

        mask = df[date_column].notna()

        if date_from:
            from_date = pd.to_datetime(date_from)
            mask &= (df[date_column] >= from_date)

        if date_to:
            to_date = pd.to_datetime(date_to)
            mask &= (df[date_column] <= to_date)

        return df[mask]

    def get_document_info(self, doc_id: str) -> Optional[Dict[str, Any]]:
        """
        특정 문서의 메타데이터 조회

        Args:
            doc_id: 문서 ID

        Returns:
            문서 메타데이터 딕셔너리
        """
        if self.metadata_df.empty:
            return None

        # ID 컬럼 찾기
        id_columns = [col for col in self.metadata_df.columns if 'id' in col.lower() or '번호' in col]

        if not id_columns:
            logger.warning("문서 ID 관련 컬럼을 찾을 수 없습니다.")
            return None

        id_column = id_columns[0]

        # 해당 문서 찾기
        doc_row = self.metadata_df[self.metadata_df[id_column] == doc_id]

        if doc_row.empty:
            return None

        return doc_row.iloc[0].to_dict()

    def get_statistics(self) -> Dict[str, Any]:
        """
        메타데이터 통계 정보 반환

        Returns:
            통계 정보 딕셔너리
        """
        if self.metadata_df.empty:
            return {}

        stats = {
            'total_documents': len(self.metadata_df),
            'organizations': self._get_column_statistics('org', '기관', '발주'),
            'categories': self._get_column_statistics('category', '분류', '유형'),
            'budget_stats': self._get_budget_statistics(),
            'date_range': self._get_date_range()
        }

        return stats

    def _get_column_statistics(self, *keywords) -> List[str]:
        """특정 키워드가 포함된 컬럼의 고유값 통계"""
        for keyword in keywords:
            columns = [col for col in self.metadata_df.columns if keyword in col.lower()]
            if columns:
                column = columns[0]
                unique_values = self.metadata_df[column].dropna().unique().tolist()
                return unique_values[:10]  # 상위 10개만 반환

        return []

    def _get_budget_statistics(self) -> Dict[str, Any]:
        """예산 관련 통계"""
        budget_columns = [col for col in self.metadata_df.columns if 'budget' in col.lower() or '예산' in col or '금액' in col]

        if not budget_columns:
            return {}

        budget_column = budget_columns[0]
        valid_budgets = self.metadata_df[budget_column].dropna()

        if valid_budgets.empty:
            return {}

        return {
            'min': float(valid_budgets.min()),
            'max': float(valid_budgets.max()),
            'mean': float(valid_budgets.mean()),
            'median': float(valid_budgets.median())
        }

    def _get_date_range(self) -> Dict[str, str]:
        """날짜 범위 통계"""
        date_columns = [col for col in self.metadata_df.columns if 'date' in col.lower() or '일자' in col]

        if not date_columns:
            return {}

        date_column = date_columns[0]
        valid_dates = pd.to_datetime(self.metadata_df[date_column], errors='coerce').dropna()

        if valid_dates.empty:
            return {}

        return {
            'min_date': valid_dates.min().strftime('%Y-%m-%d'),
            'max_date': valid_dates.max().strftime('%Y-%m-%d')
        }


# 사용 예시
if __name__ == "__main__":
    # 샘플 메타데이터 생성
    sample_data = {
        'doc_id': ['DOC001', 'DOC002', 'DOC003'],
        'organization': ['국민연금공단', '한국원자력연구원', '고려대학교'],
        'business_name': ['이러닝시스템', '선량평가시스템', '차세대포털시스템'],
        'budget': [100000000, 50000000, 80000000],
        'category': ['교육', '연구', '행정']
    }

    df = pd.DataFrame(sample_data)
    processor = MetadataProcessor(df)

    # 필터링 예시
    filter_criteria = MetadataFilter(organization='국민연금')
    filtered = processor.filter_documents(filter_criteria)
    print(f"필터링 결과: {len(filtered)}개 문서")

    # 통계 정보
    stats = processor.get_statistics()
    print(f"총 문서 수: {stats.get('total_documents', 0)}")
