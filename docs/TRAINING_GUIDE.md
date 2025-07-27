# 🎓 그린워싱 탐지 모델 훈련 가이드

## 📋 목차
1. [훈련 개요](#-훈련-개요)
2. [사전 준비](#-사전-준비)
3. [훈련 실행](#-훈련-실행)
4. [모델 커스터마이징](#-모델-커스터마이징)
5. [성능 튜닝](#-성능-튜닝)
6. [트러블슈팅](#-트러블슈팅)

## 🎯 훈련 개요

### **왜 모델을 재훈련하나요?**
- ✅ **새로운 데이터** 추가 시 (2024년 이후 데이터)
- ✅ **키워드 업데이트** (새로운 환경 용어)
- ✅ **성능 개선** (하이퍼파라미터 튜닝)
- ✅ **도메인 특화** (특정 국가/분야 집중)

### **훈련되는 3가지 모델**
1. **🔤 텍스트 불일치 탐지**: TF-IDF + Logistic Regression
2. **📈 투자 패턴 이상 탐지**: Isolation Forest
3. **🎯 마커 인플레이션 탐지**: Gradient Boosting

## 🚀 사전 준비

### **1. 데이터 확인**
```bash
# CRS 데이터가 있는지 확인 (상위 폴더)
ls ../crs_data.csv

# 전처리된 데이터 생성 (필수)
python scripts/preprocess_crs_data.py
# 결과: crs_processed.csv (121MB, 768,314개 프로젝트)
```

### **2. 시스템 요구사항**
- **메모리**: 최소 4GB RAM (8GB 권장)
- **디스크**: 10GB 여유 공간
- **시간**: 전체 훈련 약 10-15분

### **3. 환경 설정**
```bash
# 의존성 설치
pip install -r requirements.txt

# Python 버전 확인 (3.8+ 필요)
python --version
```

## ⚡ 훈련 실행

### **기본 훈련**
```bash
# 전체 모델 훈련 (기본 설정)
python scripts/train_models.py

# 예상 결과:
# - 텍스트 모델: AUC 0.9848
# - 이상 탐지: 5.0% 탐지율
# - 인플레이션 모델: AUC 0.3036
```


## 🔧 모델 커스터마이징

### **1. 환경 키워드 수정**
```python
# models/model_config.json 편집
{
  "environmental_keywords": [
    "climate", "green", "renewable",
    "새로운키워드1", "새로운키워드2"  # 추가
  ]
}
```

### **2. 하이퍼파라미터 튜닝**
```python
# scripts/train_models.py에서 수정 가능

# 텍스트 모델
model = LogisticRegression(
    random_state=42,
    max_iter=2000,        # 증가
    C=0.1,               # 정규화 강화
    class_weight='balanced'
)

# 이상 탐지 모델
model = IsolationForest(
    contamination=0.03,   # 3%로 감소
    n_estimators=200,     # 트리 수 증가
    random_state=42
)

# 인플레이션 모델
model = GradientBoostingClassifier(
    n_estimators=200,     # 트리 수 증가
    learning_rate=0.05,   # 학습률 감소
    max_depth=5,          # 깊이 증가
    random_state=42
)
```

### **3. 피처 엔지니어링**
```python
# scripts/train_models.py에서 피처 추가

def create_custom_features(self, df):
    """커스텀 피처 생성"""
    # 새로운 피처 아이디어들
    df['project_complexity'] = df['ProjectTitle'].str.len()
    df['multi_country'] = df['RecipientName'].str.contains(',').astype(int)
    df['recent_project'] = (df['Year'] >= 2020).astype(int)
    
    return df
```

## 📊 성능 튜닝

### **성능 지표 해석**

#### **텍스트 모델 (AUC 기준)**
- **0.95+ (우수)**: 텍스트-마커 불일치를 잘 탐지
- **0.90-0.95 (양호)**: 대부분의 불일치 탐지 가능
- **0.90 미만**: 키워드나 데이터 품질 점검 필요

#### **이상 탐지 (탐지율 기준)**
- **3-7% (적정)**: 합리적인 이상치 비율
- **10% 이상**: 너무 민감함, contamination 조정 필요
- **1% 미만**: 너무 보수적, 실제 이상치 놓칠 수 있음

#### **인플레이션 모델 (AUC 기준)**
- **0.90+ (우수)**: 국가별 인플레이션 잘 구분
- **0.80-0.90 (양호)**: 대부분의 인플레이션 탐지
- **0.80 미만**: 피처 엔지니어링 개선 필요

### **성능 개선 팁**

#### **1. 데이터 품질 향상**
```python
# 더 엄격한 데이터 정제
df = df[df['USD_Commitment'] > 1000]  # 최소 투자액 필터
df = df[df['ProjectTitle'].str.len() > 10]  # 제목 길이 필터
```

#### **2. 앙상블 가중치 조정**
```python
# predict_greenwashing.py에서 가중치 수정
final_score = (
    0.4 * text_score +      # 텍스트 가중치 증가
    0.3 * anomaly_score +   # 이상 탐지 가중치
    0.3 * inflation_score   # 인플레이션 가중치
)
```

#### **3. 교차 검증**
```python
# 모델 성능 검증
from sklearn.model_selection import cross_val_score

scores = cross_val_score(model, X_train, y_train, cv=5, scoring='roc_auc')
print(f"교차 검증 AUC: {scores.mean():.4f} ± {scores.std():.4f}")
```



### **성능 최적화**

#### **1. 멀티프로세싱 활용**
```python
# sklearn 모델에서 병렬 처리
model = LogisticRegression(n_jobs=-1)  # 모든 CPU 코어 사용
model = IsolationForest(n_jobs=-1)
```

#### **2. 조기 종료**
```python
# Gradient Boosting에서 조기 종료
model = GradientBoostingClassifier(
    n_estimators=1000,
    validation_fraction=0.1,
    n_iter_no_change=10,  # 10회 개선 없으면 중단
    tol=1e-4
)
```

## 🎉 훈련 완료 후 확인사항

### **1. 모델 파일 확인**
```bash
ls models/
# 다음 파일들이 있어야 함:
# - text_inconsistency_model.pkl
# - tfidf_vectorizer.pkl
# - investment_anomaly_model.pkl
# - anomaly_scaler.pkl
# - marker_inflation_model.pkl
# - model_config.json
```

### **2. 성능 테스트**
```bash
# 예측 실행으로 모델 작동 확인
python scripts/predict_greenwashing.py

# 결과 파일 확인
ls log_commitment.csv
```

### **3. 통계 생성**
```bash
# 통계 생성으로 전체 파이프라인 확인
python scripts/generate_statistics.py

# 통계 파일들 확인
ls *.csv *.json analysis_report.md
```

