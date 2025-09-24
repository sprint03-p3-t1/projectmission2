"""
RFP RAG 시스템 - 품질 모니터링 시스템
실시간 품질 추적 및 알림 기능
"""

import time
import logging
from typing import Dict, List, Any, Optional, Callable
from datetime import datetime, timedelta
from dataclasses import dataclass
from threading import Thread, Event
import json

from .quality_metrics import QualityMetrics

# 로깅 설정
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

@dataclass
class QualityAlert:
    """품질 알림 데이터 클래스"""
    alert_type: str
    message: str
    severity: str  # 'low', 'medium', 'high', 'critical'
    timestamp: str
    metrics: Dict[str, Any]
    recommendations: List[str]

class QualityMonitor:
    """품질 모니터링 시스템"""
    
    def __init__(self, 
                 quality_metrics: QualityMetrics = None,
                 alert_thresholds: Dict[str, float] = None,
                 monitoring_interval: int = 300):  # 5분마다 모니터링
        
        self.quality_metrics = quality_metrics or QualityMetrics()
        self.monitoring_interval = monitoring_interval
        self.is_monitoring = False
        self.monitor_thread = None
        self.stop_event = Event()
        
        # 알림 임계값 설정
        self.alert_thresholds = alert_thresholds or {
            "overall_score_low": 0.6,
            "overall_score_critical": 0.4,
            "accuracy_score_low": 0.5,
            "relevance_score_low": 0.5,
            "completeness_score_low": 0.5,
            "clarity_score_low": 0.5,
            "structure_score_low": 0.5,
            "trend_decline_days": 3,
            "consecutive_low_scores": 5
        }
        
        # 알림 콜백 함수들
        self.alert_callbacks: List[Callable[[QualityAlert], None]] = []
        
        # 모니터링 히스토리
        self.monitoring_history: List[Dict[str, Any]] = []
        
        logger.info("Quality monitor initialized")
    
    def add_alert_callback(self, callback: Callable[[QualityAlert], None]):
        """알림 콜백 함수 추가"""
        self.alert_callbacks.append(callback)
        logger.info(f"Alert callback added: {callback.__name__}")
    
    def start_monitoring(self):
        """모니터링 시작"""
        if self.is_monitoring:
            logger.warning("Monitoring is already running")
            return
        
        self.is_monitoring = True
        self.stop_event.clear()
        self.monitor_thread = Thread(target=self._monitoring_loop, daemon=True)
        self.monitor_thread.start()
        
        logger.info("Quality monitoring started")
    
    def stop_monitoring(self):
        """모니터링 중지"""
        if not self.is_monitoring:
            logger.warning("Monitoring is not running")
            return
        
        self.is_monitoring = False
        self.stop_event.set()
        
        if self.monitor_thread:
            self.monitor_thread.join(timeout=5)
        
        logger.info("Quality monitoring stopped")
    
    def _monitoring_loop(self):
        """모니터링 루프"""
        while not self.stop_event.is_set():
            try:
                self._check_quality_metrics()
                self._check_quality_trends()
                self._check_consecutive_low_scores()
                
                # 모니터링 결과 저장
                self._save_monitoring_result()
                
            except Exception as e:
                logger.error(f"Error in monitoring loop: {e}")
            
            # 다음 모니터링까지 대기
            self.stop_event.wait(self.monitoring_interval)
    
    def _check_quality_metrics(self):
        """품질 메트릭스 확인"""
        try:
            # 최근 1시간 통계 조회
            stats = self.quality_metrics.get_quality_statistics(days=1)
            
            if stats["total_evaluations"] == 0:
                return
            
            # 전체 점수 확인
            overall_score = stats["avg_overall_score"]
            if overall_score < self.alert_thresholds["overall_score_critical"]:
                self._trigger_alert(
                    alert_type="overall_score_critical",
                    message=f"전체 품질 점수가 매우 낮습니다: {overall_score:.3f}",
                    severity="critical",
                    metrics=stats,
                    recommendations=[
                        "시스템 점검 필요",
                        "프롬프트 개선 검토",
                        "모델 성능 확인"
                    ]
                )
            elif overall_score < self.alert_thresholds["overall_score_low"]:
                self._trigger_alert(
                    alert_type="overall_score_low",
                    message=f"전체 품질 점수가 낮습니다: {overall_score:.3f}",
                    severity="high",
                    metrics=stats,
                    recommendations=[
                        "품질 개선 방안 검토",
                        "사용자 피드백 수집"
                    ]
                )
            
            # 개별 지표 확인
            for metric in ["accuracy", "relevance", "completeness", "clarity", "structure"]:
                score_key = f"avg_{metric}_score"
                if score_key in stats:
                    score = stats[score_key]
                    threshold_key = f"{metric}_score_low"
                    if threshold_key in self.alert_thresholds and score < self.alert_thresholds[threshold_key]:
                        self._trigger_alert(
                            alert_type=f"{metric}_score_low",
                            message=f"{metric} 점수가 낮습니다: {score:.3f}",
                            severity="medium",
                            metrics=stats,
                            recommendations=[
                                f"{metric} 개선 방안 검토",
                                "관련 프롬프트 최적화"
                            ]
                        )
        
        except Exception as e:
            logger.error(f"Error checking quality metrics: {e}")
    
    def _check_quality_trends(self):
        """품질 트렌드 확인"""
        try:
            # 최근 7일 트렌드 조회
            trends_df = self.quality_metrics.get_quality_trends(days=7)
            
            if len(trends_df) < 3:
                return
            
            # 최근 3일 평균 점수 계산
            recent_scores = trends_df['avg_overall_score'].tail(3).values
            if len(recent_scores) >= 3:
                # 하향 트렌드 확인
                if all(recent_scores[i] > recent_scores[i+1] for i in range(len(recent_scores)-1)):
                    decline_days = len(recent_scores)
                    if decline_days >= self.alert_thresholds["trend_decline_days"]:
                        self._trigger_alert(
                            alert_type="quality_trend_decline",
                            message=f"품질이 {decline_days}일 연속 하락하고 있습니다",
                            severity="high",
                            metrics={"recent_scores": recent_scores.tolist()},
                            recommendations=[
                                "품질 하락 원인 분석",
                                "시스템 성능 점검",
                                "사용자 피드백 수집"
                            ]
                        )
        
        except Exception as e:
            logger.error(f"Error checking quality trends: {e}")
    
    def _check_consecutive_low_scores(self):
        """연속 낮은 점수 확인"""
        try:
            # 최근 평가 결과 조회
            recent_evaluations = self.quality_metrics.get_recent_evaluations(limit=20)
            
            if len(recent_evaluations) == 0:
                return
            
            # 연속 낮은 점수 확인
            low_score_threshold = self.alert_thresholds["overall_score_low"]
            consecutive_low_count = 0
            
            for _, row in recent_evaluations.iterrows():
                if row['overall_score'] < low_score_threshold:
                    consecutive_low_count += 1
                else:
                    break
            
            if consecutive_low_count >= self.alert_thresholds["consecutive_low_scores"]:
                self._trigger_alert(
                    alert_type="consecutive_low_scores",
                    message=f"연속 {consecutive_low_count}개의 낮은 품질 점수 발생",
                    severity="high",
                    metrics={"consecutive_low_count": consecutive_low_count},
                    recommendations=[
                        "즉시 시스템 점검",
                        "프롬프트 개선",
                        "모델 성능 확인"
                    ]
                )
        
        except Exception as e:
            logger.error(f"Error checking consecutive low scores: {e}")
    
    def _trigger_alert(self, 
                      alert_type: str, 
                      message: str, 
                      severity: str,
                      metrics: Dict[str, Any],
                      recommendations: List[str]):
        """알림 트리거"""
        
        alert = QualityAlert(
            alert_type=alert_type,
            message=message,
            severity=severity,
            timestamp=datetime.now().isoformat(),
            metrics=metrics,
            recommendations=recommendations
        )
        
        # 알림 콜백 함수들 실행
        for callback in self.alert_callbacks:
            try:
                callback(alert)
            except Exception as e:
                logger.error(f"Error in alert callback {callback.__name__}: {e}")
        
        logger.warning(f"Quality alert triggered: {alert_type} - {message}")
    
    def _save_monitoring_result(self):
        """모니터링 결과 저장"""
        try:
            stats = self.quality_metrics.get_quality_statistics(days=1)
            
            monitoring_result = {
                "timestamp": datetime.now().isoformat(),
                "total_evaluations": stats["total_evaluations"],
                "avg_overall_score": stats["avg_overall_score"],
                "quality_distribution": stats["quality_distribution"],
                "alert_count": len([h for h in self.monitoring_history if h.get("has_alert", False)])
            }
            
            self.monitoring_history.append(monitoring_result)
            
            # 히스토리 크기 제한 (최근 100개만 유지)
            if len(self.monitoring_history) > 100:
                self.monitoring_history = self.monitoring_history[-100:]
        
        except Exception as e:
            logger.error(f"Error saving monitoring result: {e}")
    
    def get_monitoring_status(self) -> Dict[str, Any]:
        """모니터링 상태 조회"""
        try:
            stats = self.quality_metrics.get_quality_statistics(days=1)
            
            return {
                "is_monitoring": self.is_monitoring,
                "monitoring_interval": self.monitoring_interval,
                "last_check": datetime.now().isoformat(),
                "current_metrics": stats,
                "alert_thresholds": self.alert_thresholds,
                "monitoring_history_count": len(self.monitoring_history),
                "alert_callbacks_count": len(self.alert_callbacks)
            }
        
        except Exception as e:
            logger.error(f"Error getting monitoring status: {e}")
            return {
                "is_monitoring": self.is_monitoring,
                "error": str(e)
            }
    
    def get_quality_insights(self) -> Dict[str, Any]:
        """품질 인사이트 생성"""
        try:
            # 최근 7일 통계
            stats_7d = self.quality_metrics.get_quality_statistics(days=7)
            stats_1d = self.quality_metrics.get_quality_statistics(days=1)
            
            # 트렌드 분석
            trends_df = self.quality_metrics.get_quality_trends(days=7)
            
            # 개선 제안
            improvement_suggestions = self.quality_metrics.get_improvement_suggestions(days=7)
            
            # 품질 분포 분석
            quality_dist = stats_7d["quality_distribution"]
            total = sum(quality_dist.values())
            if total > 0:
                high_quality_ratio = quality_dist["high_quality"] / total
                low_quality_ratio = quality_dist["low_quality"] / total
            else:
                high_quality_ratio = 0
                low_quality_ratio = 0
            
            return {
                "overall_quality": {
                    "7day_avg": stats_7d["avg_overall_score"],
                    "1day_avg": stats_1d["avg_overall_score"],
                    "trend": "improving" if stats_1d["avg_overall_score"] > stats_7d["avg_overall_score"] else "declining"
                },
                "quality_distribution": {
                    "high_quality_ratio": round(high_quality_ratio, 3),
                    "low_quality_ratio": round(low_quality_ratio, 3),
                    "total_evaluations": stats_7d["total_evaluations"]
                },
                "improvement_suggestions": improvement_suggestions,
                "trend_data": trends_df.to_dict('records') if not trends_df.empty else [],
                "insights": self._generate_insights(stats_7d, stats_1d, trends_df)
            }
        
        except Exception as e:
            logger.error(f"Error generating quality insights: {e}")
            return {"error": str(e)}
    
    def _generate_insights(self, stats_7d: Dict, stats_1d: Dict, trends_df) -> List[str]:
        """인사이트 생성"""
        insights = []
        
        # 전체 품질 트렌드
        if stats_1d["avg_overall_score"] > stats_7d["avg_overall_score"]:
            insights.append("최근 품질이 개선되고 있습니다")
        elif stats_1d["avg_overall_score"] < stats_7d["avg_overall_score"]:
            insights.append("최근 품질이 하락하고 있습니다")
        
        # 개별 지표 분석
        for metric in ["relevance", "completeness", "accuracy", "clarity", "structure"]:
            score_key = f"avg_{metric}_score"
            if score_key in stats_7d and stats_7d[score_key] < 0.7:
                insights.append(f"{metric} 점수가 개선이 필요합니다")
        
        # 품질 분포 분석
        quality_dist = stats_7d["quality_distribution"]
        total = sum(quality_dist.values())
        if total > 0:
            high_ratio = quality_dist["high_quality"] / total
            if high_ratio < 0.3:
                insights.append("고품질 답변 비율이 낮습니다")
            elif high_ratio > 0.7:
                insights.append("고품질 답변 비율이 높습니다")
        
        return insights

# 기본 알림 콜백 함수들
def console_alert_callback(alert: QualityAlert):
    """콘솔 알림 콜백"""
    severity_emoji = {
        "low": "ℹ️",
        "medium": "⚠️", 
        "high": "🚨",
        "critical": "🔥"
    }
    
    emoji = severity_emoji.get(alert.severity, "❓")
    print(f"{emoji} [QUALITY ALERT] {alert.alert_type}: {alert.message}")
    print(f"   Recommendations: {', '.join(alert.recommendations)}")

def file_alert_callback(alert: QualityAlert, log_file: str = "quality_alerts.log"):
    """파일 알림 콜백"""
    with open(log_file, "a", encoding="utf-8") as f:
        f.write(f"{alert.timestamp} [{alert.severity.upper()}] {alert.alert_type}: {alert.message}\n")
        f.write(f"  Recommendations: {', '.join(alert.recommendations)}\n")
        f.write(f"  Metrics: {json.dumps(alert.metrics, ensure_ascii=False)}\n\n")

# 품질 모니터 싱글톤
_quality_monitor_instance = None

def get_quality_monitor() -> QualityMonitor:
    """품질 모니터 인스턴스 반환 (싱글톤)"""
    global _quality_monitor_instance
    if _quality_monitor_instance is None:
        _quality_monitor_instance = QualityMonitor()
        # 기본 콜백 추가
        _quality_monitor_instance.add_alert_callback(console_alert_callback)
        _quality_monitor_instance.add_alert_callback(file_alert_callback)
    return _quality_monitor_instance
