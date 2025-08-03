# 📝 Model 1 텍스트 불일치 탐지 도구

## 📋 개요
**Model 1 (텍스트 불일치 탐지 모델)**을 활용한 그린워싱 분석 도구입니다.

**Model 1 정보:**
- **모델 타입**: TF-IDF + LogisticRegression
- **모델 파일**: `scripts/tuning/models/text_inconsistency/best_model_20250731_191849.pkl` (131MB)
- **성능**: F2-Score 0.7847, AUC 0.9848
- **핵심 기능**: 프로젝트 제목의 환경 키워드와 실제 환경 마커 간의 불일치 탐지

## 🛠️ 핵심 도구

### **`simple_model1_tools.py` - 통합 분석 도구**

**하나의 파일에 모든 분석 기능이 포함된 간편한 도구입니다.**

#### **5가지 분석 기능:**

1. **`analyze_title(title)`** - 단일 제목 분석
2. **`detect_greenwashing(project_data)`** - 그린워싱 탐지  
3. **`analyze_keyword_patterns(projects_df)`** - 키워드 패턴 분석
4. **`analyze_organization_patterns(projects_df)`** - 기관별 분석
5. **`batch_analyze(projects_df)`** - 대량 프로젝트 일괄 분석

#### **사용법:**
```python
from simple_model1_tools import SimpleModel1Tools

# 도구 초기화
tools = SimpleModel1Tools()

# 1. 단일 제목 분석
result = tools.analyze_title("Green Climate Initiative")
print(f"키워드: {result['detected_keywords']}")
print(f"위험도: {result['risk_level']}")

# 2. 그린워싱 탐지
project = {
    'ProjectTitle': 'Green Climate Initiative',
    'ClimateMitigation': 0,
    'ClimateAdaptation': 0,
    'Environment': 0,
    'Biodiversity': 0
}
result = tools.detect_greenwashing(project)
print(f"불일치: {result['is_inconsistent']}")
```

## 📊 데이터 입력 방법

### **1. 단일 제목 분석** (`analyze_title`)
```python
# 문자열 하나만 입력
result = tools.analyze_title("Green Sustainable Climate Development")
```

### **2. 그린워싱 탐지** (`detect_greenwashing`)  
```python
# 딕셔너리 형태로 프로젝트 정보 입력
project_data = {
    'ProjectTitle': 'Green Climate Initiative',
    'ClimateMitigation': 0,      # 기후완화 마커 (0,1,2)
    'ClimateAdaptation': 0,      # 기후적응 마커 (0,1,2)
    'Environment': 0,            # 환경 마커 (0,1,2)
    'Biodiversity': 0            # 생물다양성 마커 (0,1,2)
}
result = tools.detect_greenwashing(project_data)
```

### **3. 대량 분석** (`analyze_keyword_patterns`, `analyze_organization_patterns`, `batch_analyze`)
```python
import pandas as pd

# CSV 파일에서 읽기
df = pd.read_csv('projects.csv')

# 또는 직접 DataFrame 만들기
df = pd.DataFrame([
    {
        'ProjectTitle': 'Green Energy Project', 
        'ClimateMitigation': 2, 
        'ClimateAdaptation': 1, 
        'Environment': 0, 
        'Biodiversity': 0,
        'DonorName': 'Germany',           # 기관별 분석 시 필요
        'USD_Commitment': 2000000         # 선택사항
    },
    {
        'ProjectTitle': 'Solar Development', 
        'ClimateMitigation': 2, 
        'ClimateAdaptation': 0, 
        'Environment': 1, 
        'Biodiversity': 0,
        'DonorName': 'Korea',
        'USD_Commitment': 1500000
    }
])

# 분석 실행
keyword_result = tools.analyze_keyword_patterns(df)
org_result = tools.analyze_organization_patterns(df, org_column='DonorName')
batch_result = tools.batch_analyze(df)
```

### **필수 컬럼:**
- `ProjectTitle` (string): 프로젝트 제목
- `ClimateMitigation` (int): 기후완화 마커 (0,1,2)
- `ClimateAdaptation` (int): 기후적응 마커 (0,1,2)
- `Environment` (int): 환경 마커 (0,1,2)
- `Biodiversity` (int): 생물다양성 마커 (0,1,2)

### **선택 컬럼:**
- `DonorName` (string): 공여기관명 (기관별 분석 시 필수)
- `USD_Commitment` (float): 투자액 (USD)
- `Year` (int): 연도

## 🚀 빠른 시작

### 1. 환경 설정
```bash
# 필수 라이브러리 설치
pip install pandas numpy joblib scikit-learn
```

### 2. 모델 파일 확인
다음 파일들이 올바른 위치에 있는지 확인:
- `scripts/tuning/models/text_inconsistency/best_model_20250731_191849.pkl`
- `scripts/tuning/models/config/model_config.json`

### 3. 기본 사용법
```python
from simple_model1_tools import SimpleModel1Tools

# 도구 초기화
tools = SimpleModel1Tools()

# 단일 제목 분석
result = tools.analyze_title("Green Sustainable Climate Development")
print(f"위험도: {result['risk_level']}")
print(f"키워드: {result['detected_keywords']}")

# 그린워싱 탐지
project = {
    'ProjectTitle': 'Green Climate Initiative',
    'ClimateMitigation': 0,
    'ClimateAdaptation': 0,
    'Environment': 0,
    'Biodiversity': 0
}
result = tools.detect_greenwashing(project)
print(f"그린워싱 여부: {result['is_inconsistent']}")
print(f"위험도: {result['risk_level']}")
```





