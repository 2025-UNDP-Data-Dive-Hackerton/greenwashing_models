# 5. **`batch_analyze(projects_df)`** - 대량 프로젝트 일괄 분석 (ML 모델 필수)

import pandas as pd
import re
import joblib
import os

class IndependentGreenwashingDetector:
    """
    ML 기반 그린워싱 탐지기 (TF-IDF + LogisticRegression 필수)
    """
    
    def __init__(self, 
                 model_path='scripts/tuning/models/text_inconsistency/best_model_20250731_191849.pkl',
                 tfidf_path='scripts/tuning/models/individual/tfidf_vectorizer.pkl'):
        # 환경 키워드 리스트 (기본값)
        self.environmental_keywords = [
            "climate", "green", "renewable", "environment", "environmental",
            "sustainable", "sustainability", "clean", "carbon", "emission",
            "solar", "wind", "energy efficiency", "biodiversity", "forest",
            "reforestation", "conservation", "ecosystem", "pollution"
        ]
        
        # ML 모델 로드 (필수)
        self.ml_model = None
        self.tfidf_vectorizer = None
        
        # 1. ML 모델 로드
        self.ml_model = self._load_model_file(model_path, "ML 모델")
        
        # 2. TF-IDF 벡터라이저 로드 (모델 파일에서 먼저 시도)
        self.tfidf_vectorizer = self._load_tfidf_vectorizer(tfidf_path)
        
        print("✅ 모든 모델 컴포넌트 로드 완료!")
        print(f"모델 타입: {type(self.ml_model).__name__}")
        print(f"벡터라이저 타입: {type(self.tfidf_vectorizer).__name__}")
        print(f"환경 키워드 개수: {len(self.environmental_keywords)}개")
        
        # 특성 수 호환성 확인
        if hasattr(self.ml_model, 'n_features_'):
            print(f"모델 예상 특성 수: {self.ml_model.n_features_}")
        if hasattr(self.tfidf_vectorizer, 'vocabulary_'):
            print(f"벡터라이저 특성 수: {len(self.tfidf_vectorizer.vocabulary_)}")
            
        # 전체 특성 수 계산 및 호환성 체크
        if hasattr(self, 'preprocessors') and self.preprocessors:
            total_features = 0
            for key, processor in self.preprocessors.items():
                if hasattr(processor, 'vocabulary_'):
                    total_features += len(processor.vocabulary_)
                elif hasattr(processor, 'classes_'):
                    total_features += 1  # 레이블 인코더는 1개 특성
            
            print(f"계산된 전체 특성 수: {total_features}")
            
            if hasattr(self.ml_model, 'n_features_') and total_features != self.ml_model.n_features_:
                raise ValueError(f"ML 모델과 전체 특성 수가 일치하지 않습니다. "
                               f"모델: {self.ml_model.n_features_}, 계산된 특성: {total_features}")
        
        print("✅ ML 모델과 전체 특성 호환성 확인 완료!")
    
    def _load_model_file(self, file_path, description):
        """모델 파일 로드 헬퍼 함수"""
        possible_paths = [
            file_path,
            f"../{file_path}",
            os.path.join(os.path.dirname(__file__), f"../{file_path}"),
            os.path.join(os.getcwd(), file_path),
            os.path.abspath(file_path),
            os.path.abspath(f"../{file_path}")
        ]
        
        for path in possible_paths:
            if os.path.exists(path):
                try:
                    print(f"📚 {description} 로드 중: {path}")
                    print(f"파일 크기: {os.path.getsize(path) / (1024*1024):.1f} MB")
                    
                    # 파일 로드
                    data = joblib.load(path)
                    
                    # 모델 파일인 경우 구조 분석 및 환경 키워드 추출
                    if description == "ML 모델" and isinstance(data, dict):
                        available_keys = list(data.keys())
                        print(f"📋 모델 파일 구조: {available_keys}")
                        
                        # Preprocessors 구조 확인
                        preprocessors = data.get('preprocessors', {})
                        if isinstance(preprocessors, dict):
                            print(f"📋 Preprocessors 구조: {list(preprocessors.keys())}")
                        
                        # 환경 키워드 가져오기
                        if 'env_keywords' in data:
                            file_keywords = data['env_keywords']
                            if isinstance(file_keywords, list) and len(file_keywords) > 0:
                                self.environmental_keywords = file_keywords
                                print(f"📋 모델 파일에서 {len(file_keywords)}개 환경 키워드 로드")
                        
                        # 모델 추출
                        model = data.get('model')
                        if model is None:
                            raise ValueError(f"모델을 찾을 수 없습니다. 사용 가능한 키: {available_keys}")
                        
                        # 모델과 함께 전체 데이터도 저장 (나중에 preprocessors 접근용)
                        self._model_data = data
                        return model
                    else:
                        # TF-IDF 벡터라이저는 직접 반환
                        return data
                        
                except Exception as e:
                    print(f"⚠️  {path}에서 {description} 로드 실패: {e}")
                    continue
        
        # 모든 경로에서 실패한 경우
        print(f"❌ {description} 파일을 찾을 수 없습니다!")
        print(f"시도한 경로들:")
        for path in possible_paths:
            print(f"  - {path} (존재: {os.path.exists(path)})")
        raise FileNotFoundError(f"{description}이 필수입니다. 파일을 찾을 수 없습니다.")
    
    def _load_tfidf_vectorizer(self, tfidf_path):
        """TF-IDF 벡터라이저 로드 및 특성 조합"""
        
        # 1. 모델 파일의 preprocessors에서 모든 특성 추출
        if hasattr(self, '_model_data') and self._model_data:
            preprocessors = self._model_data.get('preprocessors', {})
            if isinstance(preprocessors, dict):
                print("📚 모델 파일의 preprocessors 구조 분석 중...")
                
                # 각 preprocessor의 특성 수 확인
                for key, processor in preprocessors.items():
                    if hasattr(processor, 'vocabulary_'):
                        print(f"  - {key}: {len(processor.vocabulary_)}개 특성")
                    elif hasattr(processor, 'classes_'):
                        print(f"  - {key}: {len(processor.classes_)}개 클래스")
                
                # tfidf_title을 메인으로 사용하되, 전체 특성 구조 저장
                self.preprocessors = preprocessors
                
                # 주 TF-IDF 벡터라이저 선택 (tfidf_title 우선)
                main_tfidf = (preprocessors.get('tfidf_title') or 
                             preprocessors.get('tfidf_vectorizer') or 
                             preprocessors.get('tfidf'))
                
                if main_tfidf is not None:
                    print(f"📚 메인 TF-IDF 벡터라이저: tfidf_title")
                    print(f"벡터라이저 타입: {type(main_tfidf).__name__}")
                    if hasattr(main_tfidf, 'vocabulary_'):
                        print(f"메인 벡터라이저 특성 수: {len(main_tfidf.vocabulary_)}")
                    return main_tfidf
        
        # 2. 별도 파일에서 로드 시도
        print("📚 모델 파일에서 TF-IDF를 찾을 수 없어 별도 파일에서 시도")
        return self._load_model_file(tfidf_path, "TF-IDF 벡터라이저")
    
    def _create_full_features(self, title_clean):
        """모델이 기대하는 전체 특성 벡터 생성 (3002개)"""
        import numpy as np
        from scipy.sparse import hstack
        
        # 1. tfidf_title (1000개 특성)
        title_features = self.preprocessors['tfidf_title'].transform([title_clean])
        
        # 2. tfidf_desc (2000개 특성) - 빈 설명으로 처리
        desc_features = self.preprocessors['tfidf_desc'].transform([''])
        
        # 3. le_sector (1개 특성) - 기본값 0
        sector_features = np.array([[0]])
        
        # 4. le_purpose (1개 특성) - 기본값 0  
        purpose_features = np.array([[0]])
        
        # 모든 특성 결합
        combined_features = hstack([title_features, desc_features, sector_features, purpose_features])
        
        return combined_features
    
    def preprocess_text(self, text):
        """텍스트 전처리"""
        if not text:
            return ""
        
        text = str(text).lower()
        text = re.sub(r'[^\w\s]', ' ', text)
        text = re.sub(r'\s+', ' ', text).strip()
        return text
    
    def detect_greenwashing(self, project_data):
        """
        ML 기반 그린워싱 탐지 로직
        - 키워드와 마커의 불일치 탐지
        - ML 모델을 통한 정확한 예측
        """
        title = project_data.get('ProjectTitle', '')
        climate_mitigation = project_data.get('ClimateMitigation', 0)
        climate_adaptation = project_data.get('ClimateAdaptation', 0)
        environment = project_data.get('Environment', 0)
        biodiversity = project_data.get('Biodiversity', 0)
        
        total_markers = climate_mitigation + climate_adaptation + environment + biodiversity
        
        # 환경 키워드 탐지
        title_clean = self.preprocess_text(title)
        detected_keywords = []
        for keyword in self.environmental_keywords:
            if keyword.lower() in title_clean:
                detected_keywords.append(keyword)
        
        # ML 모델 예측 (순수 ML만 사용)
        ml_probability = 0.0
        if detected_keywords:  # 환경 키워드가 있을 때만 ML 예측
            # 모든 특성을 결합하여 예측 (3002개 특성 맞추기)
            features = self._create_full_features(title_clean)
            # ML 모델로 예측
            ml_probability = float(self.ml_model.predict_proba(features)[0][1])
        
        # 그린워싱 판정 로직 개선 (규칙 기반 + ML)
        has_keywords = len(detected_keywords) > 0
        has_zero_markers = total_markers == 0
        
        # 키워드가 있지만 마커가 0인 경우만 그린워싱으로 의심
        is_inconsistent = has_keywords and has_zero_markers
        
        # 불일치 점수 계산 (ML 중심)
        inconsistency_score = 0
        risk_level = 'MINIMAL'
        
        if is_inconsistent:
            # ML 확률을 기반으로 점수 계산 (0-100)
            inconsistency_score = int(ml_probability * 100)
            
            # 키워드 개수에 따른 보정
            keyword_bonus = min(len(detected_keywords) * 5, 20)  # 최대 20점 추가
            inconsistency_score += keyword_bonus
        
        inconsistency_score = min(100, inconsistency_score)
        
        # 위험도 레벨 (더 보수적으로)
        if inconsistency_score >= 70:
            risk_level = 'CRITICAL'
        elif inconsistency_score >= 50:
            risk_level = 'HIGH'
        elif inconsistency_score >= 30:
            risk_level = 'MEDIUM'
        elif inconsistency_score >= 10:
            risk_level = 'LOW'
        else:
            risk_level = 'MINIMAL'
        
        return {
            'title': title,
            'detected_keywords': detected_keywords,
            'total_markers': total_markers,
            'is_inconsistent': is_inconsistent,
            'inconsistency_score': inconsistency_score,
            'risk_level': risk_level,
            'keyword_count': len(detected_keywords),
            'keyword_density': len(detected_keywords) / max(len(title_clean.split()), 1) if title_clean else 0,
            'ml_probability': ml_probability,
            'has_ml_model': self.ml_model is not None
        }
    
    def batch_analyze(self, projects_df):
        """
        여러 프로젝트 일괄 분석
        """
        results = []
        
        for idx, row in projects_df.iterrows():
            project_data = {
                'ProjectTitle': row['ProjectTitle'],
                'ClimateMitigation': row.get('ClimateMitigation', 0),
                'ClimateAdaptation': row.get('ClimateAdaptation', 0),
                'Environment': row.get('Environment', 0),
                'Biodiversity': row.get('Biodiversity', 0)
            }
            
            result = self.detect_greenwashing(project_data)
            result['project_index'] = idx
            results.append(result)
        
        # 요약 통계
        total_projects = len(results)
        inconsistent_projects = len([r for r in results if r['is_inconsistent']])
        high_risk_projects = len([r for r in results if r['risk_level'] in ['CRITICAL', 'HIGH']])
        
        summary = {
            'total_projects': total_projects,
            'inconsistent_projects': inconsistent_projects,
            'inconsistency_rate': inconsistent_projects / total_projects * 100 if total_projects > 0 else 0,
            'high_risk_projects': high_risk_projects,
            'high_risk_rate': high_risk_projects / total_projects * 100 if total_projects > 0 else 0
        }
        
        return {
            'project_results': results,
            'summary': summary
        }

# 도구 초기화
detector = IndependentGreenwashingDetector()

# 샘플 프로젝트 데이터 생성 (더 다양한 케이스)
sample_projects = pd.DataFrame([
    {
        'ProjectTitle': 'Green Energy Development Project',
        'ClimateMitigation': 2,
        'ClimateAdaptation': 1,
        'Environment': 0,
        'Biodiversity': 0,
        'DonorName': 'Germany',
        'USD_Commitment': 2000000
    },
    {
        'ProjectTitle': 'Sustainable Climate Initiative',  # 그린워싱 의심: 키워드 O, 마커 X
        'ClimateMitigation': 0,
        'ClimateAdaptation': 0,
        'Environment': 0,
        'Biodiversity': 0,
        'DonorName': 'Korea',
        'USD_Commitment': 1500000
    },
    {
        'ProjectTitle': 'Solar Power Infrastructure',  # 정상: 키워드 O, 마커 O
        'ClimateMitigation': 2,
        'ClimateAdaptation': 0,
        'Environment': 1,
        'Biodiversity': 0,
        'DonorName': 'Japan',
        'USD_Commitment': 3000000
    },
    {
        'ProjectTitle': 'Green Forest Conservation Program',  # 그린워싱 의심: 키워드 O, 마커 X
        'ClimateMitigation': 0,
        'ClimateAdaptation': 0,
        'Environment': 0,
        'Biodiversity': 0,
        'DonorName': 'USA',
        'USD_Commitment': 800000
    },
    {
        'ProjectTitle': 'Road Construction Project',  # 정상: 키워드 X, 마커 X
        'ClimateMitigation': 0,
        'ClimateAdaptation': 0,
        'Environment': 0,
        'Biodiversity': 0,
        'DonorName': 'France',
        'USD_Commitment': 5000000
    },
    {
        'ProjectTitle': 'Green Sustainable Clean Energy Climate Initiative',  # 고위험: 키워드 많음, 마커 X
        'ClimateMitigation': 0,
        'ClimateAdaptation': 0,
        'Environment': 0,
        'Biodiversity': 0,
        'DonorName': 'Netherlands',
        'USD_Commitment': 2500000
    },
    {
        'ProjectTitle': 'Education Infrastructure Development',  # 정상: 키워드 X, 마커 X
        'ClimateMitigation': 0,
        'ClimateAdaptation': 0,
        'Environment': 0,
        'Biodiversity': 0,
        'DonorName': 'UK',
        'USD_Commitment': 1800000
    },
    {
        'ProjectTitle': 'Renewable Wind Farm Project',  # 정상: 키워드 O, 마커 O
        'ClimateMitigation': 2,
        'ClimateAdaptation': 0,
        'Environment': 2,
        'Biodiversity': 1,
        'DonorName': 'Denmark',
        'USD_Commitment': 4000000
    }
])

print("=== 샘플 프로젝트 데이터 ===")
print(f"총 {len(sample_projects)}개 프로젝트")
for idx, row in sample_projects.iterrows():
    print(f"{idx+1}. {row['ProjectTitle']}")

print("\n=== 대량 분석 실행 중... ===")

# 5. 대량 프로젝트 일괄 분석
batch_result = detector.batch_analyze(sample_projects)

print("\n=== 대량 분석 결과 ===")
print(f"총 프로젝트: {batch_result['summary']['total_projects']}개")
print(f"불일치 프로젝트: {batch_result['summary']['inconsistent_projects']}개")
print(f"불일치율: {batch_result['summary']['inconsistency_rate']:.1f}%")
print(f"고위험 프로젝트: {batch_result['summary']['high_risk_projects']}개")
print(f"고위험률: {batch_result['summary']['high_risk_rate']:.1f}%")

print("\n=== 개별 프로젝트 상세 결과 ===")
for i, result in enumerate(batch_result['project_results']):
    status_icon = "⚠️" if result['is_inconsistent'] else "✅"
    print(f"\n{status_icon} [프로젝트 {i+1}] {result['title']}")
    print(f"  - 탐지된 키워드: {result['detected_keywords']} ({result['keyword_count']}개)")
    print(f"  - 키워드 밀도: {result['keyword_density']:.2%}")
    print(f"  - 환경 마커 총합: {result['total_markers']}")
    print(f"  - 그린워싱 의심: {'예' if result['is_inconsistent'] else '아니오'}")
    print(f"  - 불일치 점수: {result['inconsistency_score']}/100")
    print(f"  - 위험도: {result['risk_level']}")
    print(f"  - ML 예측 확률: {result['ml_probability']:.3f}")

print("\n=== 분석 요약 ===")
print("🔍 분류 결과:")
risk_counts = {}
for result in batch_result['project_results']:
    risk_level = result['risk_level']
    risk_counts[risk_level] = risk_counts.get(risk_level, 0) + 1

for risk_level in ['CRITICAL', 'HIGH', 'MEDIUM', 'LOW', 'MINIMAL']:
    count = risk_counts.get(risk_level, 0)
    if count > 0:
        print(f"  - {risk_level}: {count}개")

print("\n=== 그린워싱 의심 프로젝트 ===")
suspicious_projects = [r for r in batch_result['project_results'] if r['is_inconsistent']]
if suspicious_projects:
    print(f"총 {len(suspicious_projects)}개 프로젝트가 그린워싱으로 의심됩니다:")
    for result in suspicious_projects:
        print(f"⚠️  {result['title']}")
        print(f"    키워드: {result['detected_keywords']}")
        print(f"    위험도: {result['risk_level']} ({result['inconsistency_score']}/100)")
        print(f"    ML 확률: {result['ml_probability']:.3f}")
        print(f"    문제: 환경 키워드는 있지만 환경 마커가 0")
else:
    print("✅ 그린워싱 의심 프로젝트가 없습니다.")

