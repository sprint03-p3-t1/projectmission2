# MLOps 데이터 저장소

이 폴더는 RFP RAG 시스템의 MLOps 관련 데이터 파일들을 저장합니다.

## 파일 구조

```
data/ops/
├── conversation_logs.db      # 대화 로그 데이터베이스
├── quality_metrics.db        # 품질 평가 메트릭스 데이터베이스
└── README.md                # 이 파일
```

## 데이터베이스 설명

### conversation_logs.db
- **용도**: 사용자와 시스템 간의 모든 대화 기록 저장
- **테이블**: conversations, search_steps
- **관리 모듈**: `src/ops/conversation_tracker.py`

### quality_metrics.db
- **용도**: 시스템 품질 평가 메트릭스 저장
- **테이블**: quality_evaluations, quality_trends
- **관리 모듈**: `src/ops/quality_metrics.py`

## 주의사항

- 이 폴더의 데이터베이스 파일들은 시스템 운영에 필수적입니다.
- 백업을 정기적으로 수행하세요.
- 파일 권한을 적절히 설정하여 보안을 유지하세요.

## 관련 코드

MLOps 관련 코드는 `src/ops/` 폴더에 위치합니다:
- `quality_metrics.py`: 품질 평가 메트릭스 관리
- `quality_monitor.py`: 실시간 품질 모니터링
- `quality_visualizer.py`: 품질 데이터 시각화
- `conversation_tracker.py`: 대화 로그 추적
