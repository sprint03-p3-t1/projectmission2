"""
RFP RAG 시스템 - 품질 평가 시각화 모듈
그래프와 차트를 통한 품질 데이터 시각화
"""

import plotly.graph_objects as go
import plotly.express as px
from plotly.subplots import make_subplots
import pandas as pd
import numpy as np
from typing import Dict, List, Any, Optional
import logging

from .quality_metrics import QualityMetrics

# 로깅 설정
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class QualityVisualizer:
    """품질 평가 데이터 시각화 클래스"""
    
    def __init__(self, quality_metrics: QualityMetrics = None):
        self.quality_metrics = quality_metrics or QualityMetrics()
    
    def create_quality_overview_chart(self, days: int = 7) -> go.Figure:
        """품질 개요 차트 생성"""
        try:
            stats = self.quality_metrics.get_quality_statistics(days)
            
            # 품질 지표 데이터
            metrics = ['관련성', '완성도', '정확성', '명확성', '구조화']
            scores = [
                stats['avg_relevance_score'],
                stats['avg_completeness_score'], 
                stats['avg_accuracy_score'],
                stats['avg_clarity_score'],
                stats['avg_structure_score']
            ]
            
            # 레이더 차트 생성
            fig = go.Figure()
            
            fig.add_trace(go.Scatterpolar(
                r=scores,
                theta=metrics,
                fill='toself',
                name='평균 점수',
                line_color='rgb(32, 201, 151)',
                fillcolor='rgba(32, 201, 151, 0.3)'
            ))
            
            fig.update_layout(
                polar=dict(
                    radialaxis=dict(
                        visible=True,
                        range=[0, 1],
                        tickfont=dict(size=12),
                        gridcolor='rgba(0,0,0,0.1)'
                    ),
                    angularaxis=dict(
                        tickfont=dict(size=14),
                        gridcolor='rgba(0,0,0,0.1)'
                    )
                ),
                showlegend=True,
                title={
                    'text': f'품질 평가 개요 (최근 {days}일)',
                    'x': 0.5,
                    'xanchor': 'center',
                    'font': {'size': 16}
                },
                height=500,
                margin=dict(t=60, b=40, l=40, r=40)
            )
            
            return fig
            
        except Exception as e:
            logger.error(f"Error creating quality overview chart: {e}")
            return self._create_error_chart("품질 개요 차트 생성 실패")
    
    def create_quality_trend_chart(self, days: int = 30) -> go.Figure:
        """품질 트렌드 차트 생성"""
        try:
            trends_df = self.quality_metrics.get_quality_trends(days)
            
            if trends_df.empty:
                return self._create_empty_chart("트렌드 데이터가 없습니다")
            
            # 날짜 형식 변환
            trends_df['date'] = pd.to_datetime(trends_df['date'])
            
            # 서브플롯 생성
            fig = make_subplots(
                rows=2, cols=1,
                subplot_titles=('전체 품질 점수 트렌드', '품질 분포 변화'),
                vertical_spacing=0.1,
                specs=[[{"secondary_y": False}], [{"secondary_y": False}]]
            )
            
            # 전체 품질 점수 트렌드
            fig.add_trace(
                go.Scatter(
                    x=trends_df['date'],
                    y=trends_df['avg_overall_score'],
                    mode='lines+markers',
                    name='전체 품질',
                    line=dict(color='rgb(32, 201, 151)', width=3),
                    marker=dict(size=8)
                ),
                row=1, col=1
            )
            
            # 품질 분포 변화 (스택 바 차트)
            fig.add_trace(
                go.Bar(
                    x=trends_df['date'],
                    y=trends_df['high_quality_count'],
                    name='고품질',
                    marker_color='rgb(46, 204, 113)',
                    opacity=0.8
                ),
                row=2, col=1
            )
            
            fig.add_trace(
                go.Bar(
                    x=trends_df['date'],
                    y=trends_df['medium_quality_count'],
                    name='중품질',
                    marker_color='rgb(241, 196, 15)',
                    opacity=0.8
                ),
                row=2, col=1
            )
            
            fig.add_trace(
                go.Bar(
                    x=trends_df['date'],
                    y=trends_df['low_quality_count'],
                    name='저품질',
                    marker_color='rgb(231, 76, 60)',
                    opacity=0.8
                ),
                row=2, col=1
            )
            
            # 레이아웃 업데이트
            fig.update_layout(
                title={
                    'text': f'품질 트렌드 분석 (최근 {days}일)',
                    'x': 0.5,
                    'xanchor': 'center',
                    'font': {'size': 16}
                },
                height=700,
                showlegend=True,
                margin=dict(t=80, b=40, l=40, r=40)
            )
            
            # X축 설정
            fig.update_xaxes(title_text="날짜", row=1, col=1)
            fig.update_xaxes(title_text="날짜", row=2, col=1)
            
            # Y축 설정
            fig.update_yaxes(title_text="품질 점수", range=[0, 1], row=1, col=1)
            fig.update_yaxes(title_text="답변 수", row=2, col=1)
            
            return fig
            
        except Exception as e:
            logger.error(f"Error creating quality trend chart: {e}")
            return self._create_error_chart("품질 트렌드 차트 생성 실패")
    
    def create_quality_distribution_chart(self, days: int = 7) -> go.Figure:
        """품질 분포 차트 생성"""
        try:
            stats = self.quality_metrics.get_quality_statistics(days)
            quality_dist = stats['quality_distribution']
            
            # 파이 차트 데이터
            labels = ['고품질 (≥0.8)', '중품질 (0.6-0.8)', '저품질 (<0.6)']
            values = [
                quality_dist['high_quality'],
                quality_dist['medium_quality'],
                quality_dist['low_quality']
            ]
            colors = ['rgb(46, 204, 113)', 'rgb(241, 196, 15)', 'rgb(231, 76, 60)']
            
            # 파이 차트 생성
            fig = go.Figure(data=[go.Pie(
                labels=labels,
                values=values,
                hole=0.4,
                marker_colors=colors,
                textinfo='label+percent+value',
                textfont_size=12
            )])
            
            fig.update_layout(
                title={
                    'text': f'품질 분포 (최근 {days}일)',
                    'x': 0.5,
                    'xanchor': 'center',
                    'font': {'size': 16}
                },
                height=500,
                margin=dict(t=60, b=40, l=40, r=40),
                annotations=[dict(text=f'총 {sum(values)}개', x=0.5, y=0.5, font_size=16, showarrow=False)]
            )
            
            return fig
            
        except Exception as e:
            logger.error(f"Error creating quality distribution chart: {e}")
            return self._create_error_chart("품질 분포 차트 생성 실패")
    
    def create_improvement_suggestions_chart(self, days: int = 7) -> go.Figure:
        """개선 제안 차트 생성"""
        try:
            suggestions = self.quality_metrics.get_improvement_suggestions(days)
            
            if not suggestions:
                return self._create_empty_chart("개선 제안 데이터가 없습니다")
            
            # 개선 제안 빈도 차트
            fig = go.Figure(data=[
                go.Bar(
                    y=suggestions,
                    x=list(range(1, len(suggestions) + 1)),
                    orientation='h',
                    marker_color='rgb(52, 152, 219)',
                    text=[f"{i+1}위" for i in range(len(suggestions))],
                    textposition='inside',
                    textfont=dict(color='white', size=12)
                )
            ])
            
            fig.update_layout(
                title={
                    'text': f'개선 제안 우선순위 (최근 {days}일)',
                    'x': 0.5,
                    'xanchor': 'center',
                    'font': {'size': 16}
                },
                xaxis_title="우선순위",
                yaxis_title="개선 제안",
                height=max(400, len(suggestions) * 50),
                margin=dict(t=60, b=40, l=200, r=40)
            )
            
            return fig
            
        except Exception as e:
            logger.error(f"Error creating improvement suggestions chart: {e}")
            return self._create_error_chart("개선 제안 차트 생성 실패")
    
    def create_quality_metrics_comparison(self, days: int = 7) -> go.Figure:
        """품질 지표 비교 차트 생성"""
        try:
            stats = self.quality_metrics.get_quality_statistics(days)
            
            # 지표별 점수
            metrics = ['관련성', '완성도', '정확성', '명확성', '구조화']
            scores = [
                stats['avg_relevance_score'],
                stats['avg_completeness_score'],
                stats['avg_accuracy_score'],
                stats['avg_clarity_score'],
                stats['avg_structure_score']
            ]
            
            # 바 차트 생성
            fig = go.Figure(data=[
                go.Bar(
                    x=metrics,
                    y=scores,
                    marker_color=['rgb(231, 76, 60)' if score < 0.6 else 
                                 'rgb(241, 196, 15)' if score < 0.8 else 
                                 'rgb(46, 204, 113)' for score in scores],
                    text=[f"{score:.3f}" for score in scores],
                    textposition='outside',
                    textfont=dict(size=12, color='black')
                )
            ])
            
            # 임계값 라인 추가
            fig.add_hline(y=0.8, line_dash="dash", line_color="green", 
                         annotation_text="우수 (0.8)", annotation_position="top right")
            fig.add_hline(y=0.6, line_dash="dash", line_color="orange", 
                         annotation_text="보통 (0.6)", annotation_position="top right")
            
            fig.update_layout(
                title={
                    'text': f'품질 지표별 점수 비교 (최근 {days}일)',
                    'x': 0.5,
                    'xanchor': 'center',
                    'font': {'size': 16}
                },
                xaxis_title="품질 지표",
                yaxis_title="평균 점수",
                yaxis=dict(range=[0, 1]),
                height=500,
                margin=dict(t=60, b=40, l=40, r=40)
            )
            
            return fig
            
        except Exception as e:
            logger.error(f"Error creating quality metrics comparison: {e}")
            return self._create_error_chart("품질 지표 비교 차트 생성 실패")
    
    def create_quality_score_gauge(self, current_score: float, target_score: float = 0.8) -> go.Figure:
        """품질 점수 게이지 차트 생성"""
        try:
            # 게이지 차트 생성
            fig = go.Figure(go.Indicator(
                mode = "gauge+number+delta",
                value = current_score,
                domain = {'x': [0, 1], 'y': [0, 1]},
                title = {'text': "전체 품질 점수"},
                delta = {'reference': target_score},
                gauge = {
                    'axis': {'range': [None, 1]},
                    'bar': {'color': "darkblue"},
                    'steps': [
                        {'range': [0, 0.6], 'color': "lightgray"},
                        {'range': [0.6, 0.8], 'color': "yellow"},
                        {'range': [0.8, 1], 'color': "green"}
                    ],
                    'threshold': {
                        'line': {'color': "red", 'width': 4},
                        'thickness': 0.75,
                        'value': target_score
                    }
                }
            ))
            
            fig.update_layout(
                title={
                    'text': '실시간 품질 점수',
                    'x': 0.5,
                    'xanchor': 'center',
                    'font': {'size': 16}
                },
                height=400,
                margin=dict(t=60, b=40, l=40, r=40)
            )
            
            return fig
            
        except Exception as e:
            logger.error(f"Error creating quality score gauge: {e}")
            return self._create_error_chart("품질 점수 게이지 생성 실패")
    
    def _create_error_chart(self, message: str) -> go.Figure:
        """에러 차트 생성"""
        fig = go.Figure()
        fig.add_annotation(
            text=message,
            xref="paper", yref="paper",
            x=0.5, y=0.5, xanchor='center', yanchor='middle',
            showarrow=False, font_size=16, font_color="red"
        )
        fig.update_layout(
            xaxis={'visible': False},
            yaxis={'visible': False},
            height=300
        )
        return fig
    
    def _create_empty_chart(self, message: str) -> go.Figure:
        """빈 데이터 차트 생성"""
        fig = go.Figure()
        fig.add_annotation(
            text=message,
            xref="paper", yref="paper",
            x=0.5, y=0.5, xanchor='center', yanchor='middle',
            showarrow=False, font_size=14, font_color="gray"
        )
        fig.update_layout(
            xaxis={'visible': False},
            yaxis={'visible': False},
            height=300
        )
        return fig
    
    def export_charts_to_html(self, output_path: str, days: int = 7):
        """차트들을 HTML 파일로 내보내기"""
        try:
            charts = {
                'quality_overview': self.create_quality_overview_chart(days),
                'quality_trend': self.create_quality_trend_chart(days),
                'quality_distribution': self.create_quality_distribution_chart(days),
                'improvement_suggestions': self.create_improvement_suggestions_chart(days),
                'quality_metrics_comparison': self.create_quality_metrics_comparison(days)
            }
            
            # HTML 템플릿 생성
            html_content = f"""
            <!DOCTYPE html>
            <html>
            <head>
                <title>RFP RAG 시스템 품질 평가 대시보드</title>
                <meta charset="utf-8">
                <style>
                    body {{ font-family: Arial, sans-serif; margin: 20px; }}
                    h1 {{ text-align: center; color: #2c3e50; }}
                    h2 {{ color: #34495e; border-bottom: 2px solid #3498db; padding-bottom: 10px; }}
                    .chart-container {{ margin: 30px 0; }}
                </style>
            </head>
            <body>
                <h1>📊 RFP RAG 시스템 품질 평가 대시보드</h1>
                <p style="text-align: center; color: #7f8c8d;">최근 {days}일 데이터 기준</p>
            """
            
            # 각 차트를 HTML에 추가
            for chart_name, chart in charts.items():
                html_content += f"""
                <div class="chart-container">
                    <h2>{self._get_chart_title(chart_name)}</h2>
                    {chart.to_html(include_plotlyjs=False, div_id=chart_name)}
                </div>
                """
            
            html_content += """
                <script src="https://cdn.plot.ly/plotly-latest.min.js"></script>
            </body>
            </html>
            """
            
            # HTML 파일 저장
            with open(output_path, 'w', encoding='utf-8') as f:
                f.write(html_content)
            
            logger.info(f"Charts exported to HTML: {output_path}")
            
        except Exception as e:
            logger.error(f"Error exporting charts to HTML: {e}")
    
    def _get_chart_title(self, chart_name: str) -> str:
        """차트 제목 반환"""
        titles = {
            'quality_overview': '🎯 품질 평가 개요',
            'quality_trend': '📈 품질 트렌드 분석',
            'quality_distribution': '🥧 품질 분포',
            'improvement_suggestions': '💡 개선 제안 우선순위',
            'quality_metrics_comparison': '📊 품질 지표별 비교'
        }
        return titles.get(chart_name, chart_name)

# 품질 시각화 도구 싱글톤
_quality_visualizer_instance = None

def get_quality_visualizer() -> QualityVisualizer:
    """품질 시각화 도구 인스턴스 반환 (싱글톤)"""
    global _quality_visualizer_instance
    if _quality_visualizer_instance is None:
        _quality_visualizer_instance = QualityVisualizer()
    return _quality_visualizer_instance
