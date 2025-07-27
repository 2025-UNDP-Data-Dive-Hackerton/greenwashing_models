# 🚀 그린워싱 탐지 실행 가이드

## ⚡ 바로 실행하기

### 1단계: 의존성 설치
```bash
pip install -r requirements.txt
# DuckDB 포함 - 290만개 데이터 빠른 처리!
```

### 2단계: 예측 실행
```bash
python scripts/predict_greenwashing.py
```

### 3단계: 통계 생성 (NEW!)
```bash
python scripts/generate_statistics.py
```

## 📊 입력 파일 형식

### crs_processed.csv (전처리된 CRS 데이터)
```csv
ProjectTitle,USD_Commitment,ClimateMitigation,ClimateAdaptation,Environment,Biodiversity
"Solar Energy Project",1000000,2,1,1,0
"Green Initiative",50000,0,0,0,0
```

**컬럼 설명:**
- `ProjectTitle`: 프로젝트 제목 (문자열)
- `USD_Commitment`: 투자액 (숫자, USD)
- `ClimateMitigation`: 기후완화 마커 (0, 1, 2)
- `ClimateAdaptation`: 기후적응 마커 (0, 1, 2)  
- `Environment`: 환경 마커 (0, 1, 2)
- `Biodiversity`: 생물다양성 마커 (0, 1, 2)

## 📈 출력 결과 해석

### 위험도 기준
- **0.9 이상**: "위험" (🔴) - 즉시 검토 필요
- **0.5 이상**: "주의" (🟡) - 추가 조사 권장  
- **그 외**: "정상" (🟢) - 모니터링 지속

### log_commitment.csv 출력 형식
```csv
ProjectTitle,USD_Commitment,log_commitment,risk_score,risk_ratio,risk_level,text_inconsistency,investment_inconsistency,excessive_markers,total_climate_score
"Solar Energy Project",1000000,13.816,0,0.0,정상,false,false,false,4
"Green Initiative",50000,10.820,40,0.4,주의,true,false,false,0
```

**출력 컬럼 설명:**
- `log_commitment`: 투자액의 로그 변환값
- `risk_score`: 위험도 점수 (0-100)
- `risk_ratio`: 위험도 비율 (0.0-1.0)
- `risk_level`: 위험 레벨 (정상/주의/위험)
- `text_inconsistency`: 텍스트 불일치 여부
- `investment_inconsistency`: 투자 불일치 여부
- `excessive_markers`: 마커 과다 사용 여부

## 🔧 커스터마이징

### 위험도 임계값 수정
`scripts/predict_greenwashing.py` 파일에서:
```python
# 위험 레벨 결정 부분 수정
if risk_ratio >= 0.9:    # 0.9 → 원하는 값으로 변경
    risk_level = "위험"
elif risk_ratio >= 0.5:  # 0.5 → 원하는 값으로 변경
    risk_level = "주의"
```

### 환경 키워드 추가
`models/model_config.json` 파일에서 `environmental_keywords` 배열 수정

## 💡 사용 예시

### Python 코드에서 직접 사용
```python
from scripts.predict_greenwashing import load_models, predict_single_project

# 모델 로드
models, config = load_models()

# 프로젝트 데이터
project = {
    'ProjectTitle': 'Solar Energy Development',
    'USD_Commitment': 1000000,
    'ClimateMitigation': 2,
    'ClimateAdaptation': 1,
    'Environment': 1,
    'Biodiversity': 0
}

# 예측 실행
result = predict_single_project(models, config, project)
print(f"위험도: {result['risk_ratio']} ({result['risk_level']})")
```

## 🚨 문제 해결

### 모델 파일 오류
- 모든 .pkl 파일이 같은 폴더에 있는지 확인
- Python 버전 호환성 확인 (3.7+ 권장)

### CSV 파일 오류  
- 필수 컬럼명 정확히 입력
- 숫자 컬럼에 문자 입력 금지
- UTF-8 인코딩 사용

## 📊 생성되는 파일들

### 예측 결과
- `log_commitment.csv` - 프로젝트별 예측 결과

### 통계 파일들 (NEW!)
- `overall_statistics.json` - 전체 통계
- `country_statistics.csv` - 국가별 통계  
- `yearly_trends.csv` - 연도별 트렌드
- `pattern_analysis.csv` - 패턴 분석
- `analysis_report.md` - 종합 리포트

## 📞 지원

- 📊 통계 분석: `analysis_report.md`에서 주요 인사이트 확인
- 🧪 테스트: 전체 CRS 데이터로 실제 분석
- 📁 결과: 6개 파일로 완전한 대시보드 구축 가능 