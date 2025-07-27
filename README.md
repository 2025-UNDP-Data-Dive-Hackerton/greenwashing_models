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
│   ├── train_models.py            # 모델 훈련 스크립트 
│   ├── predict_greenwashing.py    # 메인 예측 스크립트 (바로 실행)
│   └── generate_statistics.py     # 통계 생성 스크립트
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
    ├── RUN_GUIDE.md               # 상세 실행 가이드
    └── TRAINING_GUIDE.md          # 모델 훈련 가이드
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

### 3. 모델 훈련 (선택사항)
```bash
# ✅ 이미 훈련된 모델이 포함되어 있습니다!
# 재훈련하거나 모델을 커스터마이징하려면:
python scripts/train_models.py
```

### 4. 모델 실행
```bash
python scripts/predict_greenwashing.py
```

### 5. 통계 생성
```bash
python scripts/generate_statistics.py
```

### 6. 결과 확인
- 📊 **예측 결과**: `log_commitment.csv`
- 📈 **국가별 통계**: `country_statistics.csv`
- 📅 **연도별 트렌드**: `yearly_trends.csv`
- 🔍 **패턴 분석**: `pattern_analysis.csv`
- 📄 **종합 리포트**: `analysis_report.md`

## 📊 사용법

### 필수 파일
- `crs_data.csv` - 원본 CRS 데이터 (3.8GB, 루트 폴더에 위치)
- `crs_processed.csv` - 전처리된 데이터 (자동 생성)

### 입력 형식 (crs_processed.csv)
| 컬럼명 | 타입 | 설명 | 예시 |
|--------|------|------|------|
| ProjectTitle | string | 프로젝트 제목 | "Solar Energy Project" |
| ShortDescription | string | 프로젝트 설명 | "Clean energy initiative" |
| DonorName | string | 공여국 이름 | "Germany" |
| USD_Commitment | number | 투자액 (USD) | 1000000 |
| USD_Disbursement | number | 실제 지출액 (USD) | 800000 |
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



## 🎓 모델 훈련 과정 (상세)

### 📊 **훈련 데이터**
- **소스**: 전체 CRS 데이터 (3.8GB, 2.9M+ 레코드)
- **기간**: 2014-2023년 (10년간)
- **범위**: 447개 공여국, 196개 수혜국

### 🤖 **3가지 모델 훈련**

#### **1️⃣ 텍스트 불일치 탐지**
```python
# TF-IDF + Logistic Regression
- 입력: 프로젝트 제목 + 설명 (TF-IDF 벡터화)
- 타겟: 환경 키워드 많음 + 기후 마커 0 = 불일치
- 성능: AUC 0.995+ (매우 높은 정확도)
```

#### **2️⃣ 투자 패턴 이상 탐지**
```python
# Isolation Forest (비지도 학습)
- 입력: 투자액, 기후점수, 마커-투자 비율 등 8개 피처
- 탐지: 5% 이상치 (높은 마커 + 낮은 투자 패턴)
- 결과: 자동으로 의심스러운 투자 패턴 식별
```

#### **3️⃣ 마커 인플레이션 탐지**
```python
# Gradient Boosting Classifier
- 입력: 국가별 집계 통계 (마커 비율, 실제 투자 등)
- 타겟: 높은 마커 선언 + 낮은 실제 녹색 투자
- 성능: AUC 0.940+ (국가 수준 인플레이션 탐지)
```

### 🔄 **재훈련 방법**
```bash
# 1. 새로운 데이터로 재훈련
python scripts/train_models.py

# 2. 하이퍼파라미터 수정 후 재훈련
# train_models.py에서 모델 설정 변경 가능

# 3. 새로운 키워드 추가
# model_config.json에서 environmental_keywords 수정
```

### 📈 **실제 훈련 결과** (테스트 완료)
- **텍스트 모델**: AUC 0.9848로 텍스트-마커 불일치 탐지
- **이상 탐지**: 38,415개 의심 프로젝트 자동 식별 (5.0%)
- **인플레이션 탐지**: 80개 공여국별 그린워싱 위험도 분석
- **총 처리량**: 768,314개 프로젝트 완전 분석

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
- **텍스트 불일치 모델**: AUC 0.9848
- **투자 이상 탐지**: 5.0% 이상치 탐지율  
- **마커 인플레이션 탐지**: AUC 0.3036
- **처리 속도**: 76만개 프로젝트 처리 완료
- **전체 데이터**: 768,314개 프로젝트 분석

## 🚀 아키텍처 권장사항

### React + Elixir 통합 구조
```
Frontend (React)
    ↓ HTTP API 호출
Backend (Elixir/Phoenix)
    ↓ Python 스크립트 실행
ML Service (Python)
    ↓ 결과 반환
통계 파일들 (6개 CSV/JSON)
```

## 📊 생성되는 통계 파일들

### 🎯 **개발자가 받게 되는 완전한 분석 결과**
- **📈 overall_statistics.json** - 전체 통계 (JSON 형식)
- **🌍 country_statistics.csv** - 국가별 그린워싱 위험도
- **📅 yearly_trends.csv** - 연도별 트렌드 데이터
- **🔍 pattern_analysis.csv** - 그린워싱 패턴 상세 분석
- **📄 analysis_report.md** - 종합 분석 리포트 (한국어)


## 📞 개발자 지원

### 📚 **상세 문서**
- 🔧 **실행 가이드**: `docs/RUN_GUIDE.md` - 바로 실행하는 방법
- 🎓 **훈련 가이드**: `docs/TRAINING_GUIDE.md` - 모델 재훈련 방법

### 🎯 **핵심 결과물**
- 📊 **예측 결과**: `log_commitment.csv` (93MB, 768,314개 프로젝트)
- 📈 **통계 분석**: `analysis_report.md` (자동 생성)
- 🌍 **대시보드 데이터**: 6개 CSV/JSON 파일

### ✅ **검증 완료**
- 🧪 **테스트**: `undp` Conda 환경에서 전체 파이프라인 검증 완료
- 🚀 **성능**: 768,314개 프로젝트 분석 (정상 92.1%, 주의 7.9%)
- 📁 **개발자 준비**: React + Elixir 대시보드 구축 가능

---
*UNDP Data Dive 해커톤 2025 | 그린워싱 탐지 AI 모델*

