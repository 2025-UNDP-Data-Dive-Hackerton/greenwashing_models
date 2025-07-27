# 🤖 그린워싱 탐지 모델 패키지

## 📦 개요
UNDP 그린워싱 탐지를 위한 완전한 ML 모델 패키지입니다. 개발자가 바로 사용할 수 있도록 모든 필요한 파일과 예시가 포함되어 있습니다.

## 📁 파일 구조
```
greenwashing_models/
├── 📖 README.md                    # 이 파일
├── 📋 requirements.txt             # Python 의존성
│
├── 📂 scripts/                     # 🚀 실행 스크립트들
│   ├── preprocess_crs_data.py     # CRS 데이터 전처리 스크립트
│   ├── predict_greenwashing.py    # 메인 예측 스크립트 (바로 실행)
│   └── generate_statistics.py     # 통계 생성 스크립트 (NEW!)
│
├── 📂 models/                      # 🤖 모델 파일들
│   ├── text_inconsistency_model.pkl    # 텍스트 불일치 탐지
│   ├── tfidf_vectorizer.pkl            # TF-IDF 벡터라이저
│   ├── investment_anomaly_model.pkl    # 투자 패턴 이상 탐지
│   ├── anomaly_scaler.pkl              # 이상 탐지용 스케일러
│   ├── marker_inflation_model.pkl      # 마커 부풀리기 탐지
│   └── model_config.json               # 모델 설정 및 키워드
│
└── 📂 docs/                        # 📖 문서들
    └── RUN_GUIDE.md               # 상세 실행 가이드
```

## ⚡ 빠른 시작 (1분)

### 1. 의존성 설치
```bash
pip install -r requirements.txt
# DuckDB 포함 - 빠른 대용량 데이터 처리!
```

### 2. CRS 데이터 전처리 (필수)
```bash
# 전체 CRS 데이터 전처리 (crs_data.csv 필요)
python scripts/preprocess_crs_data.py
```

### 3. 모델 실행
```bash
python scripts/predict_greenwashing.py
```

### 4. 통계 생성 (NEW!)
```bash
python scripts/generate_statistics.py
```

### 5. 결과 확인
- 📊 **예측 결과**: `log_commitment.csv`
- 📈 **국가별 통계**: `country_statistics.csv`
- 📅 **연도별 트렌드**: `yearly_trends.csv`
- 🔍 **패턴 분석**: `pattern_analysis.csv`
- 📄 **종합 리포트**: `analysis_report.md`

## 📊 사용법

### 입력 형식 (crs_processed.csv)
| 컬럼명 | 타입 | 설명 | 예시 |
|--------|------|------|------|
| ProjectTitle | string | 프로젝트 제목 | "Solar Energy Project" |
| USD_Commitment | number | 투자액 (USD) | 1000000 |
| ClimateMitigation | int | 기후완화 마커 (0,1,2) | 2 |
| ClimateAdaptation | int | 기후적응 마커 (0,1,2) | 1 |
| Environment | int | 환경 마커 (0,1,2) | 1 |
| Biodiversity | int | 생물다양성 마커 (0,1,2) | 0 |

### 출력 형식 (log_commitment.csv)
| 컬럼명 | 설명 |
|--------|------|
| risk_ratio | 위험도 비율 (0.0-1.0) |
| risk_level | 위험 레벨 (정상/주의/위험) |
| log_commitment | 투자액 로그 변환값 |
| text_inconsistency | 텍스트 불일치 여부 |
| investment_inconsistency | 투자 불일치 여부 |
| excessive_markers | 마커 과다 사용 여부 |

## 🎯 위험도 해석

| 위험도 | 레벨 | 색상 | 의미 |
|--------|------|------|------|
| 0.9+ | 위험 | 🔴 | 즉시 검토 필요 |
| 0.5+ | 주의 | 🟡 | 추가 조사 권장 |
| 0.5 미만 | 정상 | 🟢 | 모니터링 지속 |

## 💻 개발자 통합

### Python에서 직접 사용
```python
from scripts.predict_greenwashing import load_models, predict_single_project

models, config = load_models()

project = {
    'ProjectTitle': 'Green Energy Initiative',
    'USD_Commitment': 500000,
    'ClimateMitigation': 0,
    'ClimateAdaptation': 0,
    'Environment': 0,
    'Biodiversity': 0
}

result = predict_single_project(models, config, project)
print(f"위험도: {result['risk_ratio']} ({result['risk_level']})")
```

### 웹 API로 래핑
```python
from flask import Flask, request, jsonify
from scripts.predict_greenwashing import load_models, predict_single_project

app = Flask(__name__)
models, config = load_models()

@app.route('/predict', methods=['POST'])
def predict():
    data = request.json
    result = predict_single_project(models, config, data)
    return jsonify(result)
```

## 🔧 커스터마이징

### 위험도 임계값 변경
`scripts/predict_greenwashing.py`에서 임계값 수정:
```python
if risk_ratio >= 0.9:    # 위험 임계값
    risk_level = "위험"
elif risk_ratio >= 0.5:  # 주의 임계값
    risk_level = "주의"
```

### 환경 키워드 추가
`models/model_config.json`에서 키워드 추가:
```json
{
  "environmental_keywords": [
    "climate", "green", "renewable", 
    "새로운키워드"  // 여기에 추가
  ]
}
```

## 📈 성능 지표
- **텍스트 불일치 모델**: AUC 0.9955
- **투자 이상 탐지**: 5% 이상치 탐지율
- **마커 부풀리기 탐지**: AUC 0.9406
- **처리 속도**: 10만개 프로젝트/분

## 🚀 프로덕션 배포

### Docker 사용
```dockerfile
FROM python:3.9-slim
COPY . /app
WORKDIR /app
RUN pip install -r requirements.txt
CMD ["python", "scripts/predict_greenwashing.py"]
```

### 클라우드 배포
- AWS Lambda, Google Cloud Functions 지원
- 메모리 요구사항: 최소 1GB
- 처리 시간: 프로젝트당 ~0.1초

## 📊 생성되는 통계 파일들

### 🎯 **개발자가 받게 되는 완전한 분석 결과**
- **📈 overall_statistics.json** - 전체 통계 (JSON 형식)
- **🌍 country_statistics.csv** - 국가별 그린워싱 위험도
- **📅 yearly_trends.csv** - 연도별 트렌드 데이터
- **🔍 pattern_analysis.csv** - 그린워싱 패턴 상세 분석
- **📄 analysis_report.md** - 종합 분석 리포트 (한국어)

### 💻 **개발자 역할**
- ✅ **제공받은 통계를 시각화** (차트, 그래프, 지도)
- ✅ **웹 대시보드 UI 구현** (React, Vue, Angular 등)
- ✅ **사용자 인터페이스 개발** (검색, 필터링 등)

## 📞 지원 및 문의

- 🔧 **실행 가이드**: `docs/RUN_GUIDE.md` 참고
- 📊 **통계 분석**: `analysis_report.md`에서 주요 인사이트 확인
- 🧪 **테스트**: 전체 CRS 데이터로 실제 분석
- 📁 **결과 파일**: 6개 파일로 완전한 대시보드 구축 가능

---

**✨ 이 패키지는 바로 사용 가능하도록 설계되었습니다. 추가 설정 없이 `python scripts/predict_greenwashing.py` 명령어로 즉시 시작하세요!** 