#!/usr/bin/env python3
"""
그린워싱 탐지 모델 학습 스크립트
UNDP Data Dive 해커톤 2025

이 스크립트는 3가지 ML 모델을 훈련합니다:
1. 텍스트-투자 불일치 탐지 (TF-IDF + Logistic Regression)
2. 투자 패턴 이상 탐지 (Isolation Forest)
3. 기후 마커 인플레이션 탐지 (Gradient Boosting)
"""

import duckdb
import pandas as pd
import numpy as np
import joblib
import json
import os
import re
import warnings
from datetime import datetime
from typing import Dict, List, Tuple, Any

# ML 라이브러리
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import IsolationForest, GradientBoostingClassifier
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.metrics import classification_report, roc_auc_score, confusion_matrix

warnings.filterwarnings('ignore')

class GreenwashingModelTrainer:
    """그린워싱 탐지 모델 훈련 클래스"""
    
    def __init__(self, data_path: str = 'crs_processed.csv'):
        """
        초기화
        
        Args:
            data_path: 전처리된 CRS 데이터 경로
        """
        self.data_path = data_path
        self.models = {}
        self.config = {
            "environmental_keywords": [
                "climate", "green", "renewable", "environment", "environmental",
                "sustainable", "sustainability", "clean", "carbon", "emission",
                "solar", "wind", "energy efficiency", "biodiversity", "forest",
                "reforestation", "conservation", "ecosystem", "pollution"
            ],
            "anomaly_features": [
                "log_commitment", "log_disbursement", "climate_score",
                "marker_investment_ratio", "commitment_zscore", "climate_zscore",
                "year_normalized", "high_marker_low_investment"
            ],
            "inflation_features": [
                "total_projects", "mitigation_rate", "adaptation_rate",
                "environment_rate", "biodiversity_rate", "actual_green_investment_pct",
                "avg_investment", "high_marker_rate", "marker_reality_gap"
            ],
            "model_version": "1.0.0",
            "created_date": datetime.now().isoformat(),
            "performance_metrics": {}
        }
        
    def load_data(self) -> pd.DataFrame:
        """데이터 로딩"""
        print("📊 데이터 로딩 중...")
        
        if not os.path.exists(self.data_path):
            print(f"❌ 데이터 파일을 찾을 수 없습니다: {self.data_path}")
            print("💡 먼저 preprocess_crs_data.py를 실행해주세요!")
            raise FileNotFoundError(f"Data file not found: {self.data_path}")
        
        # DuckDB로 데이터 로딩
        conn = duckdb.connect()
        df = conn.execute(f"""
            SELECT * FROM read_csv_auto('{self.data_path}', 
                                      ignore_errors=true, 
                                      null_padding=true)
        """).df()
        conn.close()
        
        print(f"✅ 데이터 로딩 완료: {len(df):,}개 레코드, {len(df.columns)}개 컬럼")
        return df
    
    def create_text_features(self, df: pd.DataFrame) -> Tuple[np.ndarray, np.ndarray]:
        """텍스트 불일치 탐지를 위한 피처 생성"""
        print("🔤 텍스트 피처 생성 중...")
        
        # 텍스트 정리
        df['ProjectTitle_clean'] = df['ProjectTitle'].fillna('').astype(str)
        df['ShortDescription_clean'] = df['ShortDescription'].fillna('').astype(str)
        df['combined_text'] = df['ProjectTitle_clean'] + ' ' + df['ShortDescription_clean']
        
        # 환경 키워드 점수 계산
        def calculate_env_score(text):
            if pd.isna(text) or text == '':
                return 0
            text_lower = str(text).lower()
            return sum(1 for keyword in self.config["environmental_keywords"] 
                      if keyword in text_lower)
        
        df['env_keyword_count'] = df['combined_text'].apply(calculate_env_score)
        
        # 기후 마커 총합
        climate_cols = ['ClimateMitigation', 'ClimateAdaptation', 'Environment', 'Biodiversity']
        for col in climate_cols:
            df[col] = pd.to_numeric(df[col], errors='coerce').fillna(0)
        df['total_climate_markers'] = df[climate_cols].sum(axis=1)
        
        # 불일치 레이블 생성 (환경 키워드 많지만 기후 마커 없음)
        text_inconsistency = ((df['env_keyword_count'] >= 2) & 
                             (df['total_climate_markers'] == 0)).astype(int)
        
        # TF-IDF 벡터화
        tfidf = TfidfVectorizer(
            max_features=5000,
            stop_words='english',
            ngram_range=(1, 2),
            min_df=2,
            max_df=0.95
        )
        
        X_text = tfidf.fit_transform(df['combined_text'])
        self.models['tfidf_vectorizer'] = tfidf
        
        print(f"✅ 텍스트 피처 생성 완료: {X_text.shape}")
        print(f"   - 불일치 케이스: {text_inconsistency.sum():,}개 ({text_inconsistency.mean():.1%})")
        
        return X_text, text_inconsistency
    
    def create_anomaly_features(self, df: pd.DataFrame) -> Tuple[np.ndarray, np.ndarray]:
        """투자 이상 탐지를 위한 피처 생성"""
        print("📈 투자 이상 피처 생성 중...")
        
        # 수치형 변환
        df['USD_Commitment'] = pd.to_numeric(df['USD_Commitment'], errors='coerce').fillna(0)
        df['USD_Disbursement'] = pd.to_numeric(df['USD_Disbursement'], errors='coerce').fillna(0)
        
        # 로그 변환 (0값 처리)
        df['log_commitment'] = np.log1p(df['USD_Commitment'])
        df['log_disbursement'] = np.log1p(df['USD_Disbursement'])
        
        # 기후 점수
        df['climate_score'] = df['total_climate_markers'] / 4.0  # 0-1 정규화
        
        # 마커-투자 비율
        df['marker_investment_ratio'] = np.where(
            df['USD_Commitment'] > 0,
            df['total_climate_markers'] / (df['USD_Commitment'] / 1000000),  # 백만달러당
            0
        )
        
        # Z-score 정규화
        df['commitment_zscore'] = (df['log_commitment'] - df['log_commitment'].mean()) / df['log_commitment'].std()
        df['climate_zscore'] = (df['climate_score'] - df['climate_score'].mean()) / df['climate_score'].std()
        
        # 연도 정규화 (Year 컬럼이 없으므로 기본값 사용)
        df['year_normalized'] = 0.5  # 기본값 (2018-2019년 정도로 가정)
        
        # 높은 마커 + 낮은 투자 플래그
        df['high_marker_low_investment'] = (
            (df['total_climate_markers'] >= 2) & 
            (df['USD_Commitment'] < df['USD_Commitment'].quantile(0.25))
        ).astype(int)
        
        # 피처 선택
        feature_cols = [col for col in self.config["anomaly_features"] if col in df.columns]
        X_anomaly = df[feature_cols].fillna(0).values
        
        # 스케일링
        scaler = StandardScaler()
        X_anomaly_scaled = scaler.fit_transform(X_anomaly)
        self.models['anomaly_scaler'] = scaler
        
        # 이상치 레이블 (실제로는 비지도 학습이지만 평가용)
        anomaly_labels = np.zeros(len(df))  # 실제 구현에서는 도메인 지식 기반으로 생성
        
        print(f"✅ 투자 이상 피처 생성 완료: {X_anomaly_scaled.shape}")
        
        return X_anomaly_scaled, anomaly_labels
    
    def create_inflation_features(self, df: pd.DataFrame) -> Tuple[np.ndarray, np.ndarray]:
        """마커 인플레이션 탐지를 위한 피처 생성"""
        print("🎯 마커 인플레이션 피처 생성 중...")
        
        # 국가별 집계
        country_stats = df.groupby('DonorName').agg({
            'ProjectTitle': 'count',
            'ClimateMitigation': ['mean', 'sum'],
            'ClimateAdaptation': ['mean', 'sum'],
            'Environment': ['mean', 'sum'],
            'Biodiversity': ['mean', 'sum'],
            'USD_Commitment': ['mean', 'sum'],
            'total_climate_markers': 'mean'
        }).round(4)
        
        # 컬럼명 정리
        country_stats.columns = [
            'total_projects', 'mitigation_rate', 'mitigation_sum',
            'adaptation_rate', 'adaptation_sum', 'environment_rate', 'environment_sum',
            'biodiversity_rate', 'biodiversity_sum', 'avg_investment', 'total_investment',
            'avg_climate_score'
        ]
        
        # 추가 피처 계산
        country_stats['actual_green_investment_pct'] = (
            country_stats['total_investment'] * country_stats['avg_climate_score'] / 
            country_stats['total_investment'].clip(lower=1)
        ) * 100
        
        country_stats['high_marker_rate'] = (
            (country_stats['mitigation_rate'] + country_stats['adaptation_rate'] + 
             country_stats['environment_rate'] + country_stats['biodiversity_rate']) >= 2
        ).astype(float)
        
        country_stats['marker_reality_gap'] = (
            country_stats['avg_climate_score'] - 
            country_stats['actual_green_investment_pct'] / 100
        )
        
        # 피처 선택
        feature_cols = [col for col in self.config["inflation_features"] 
                       if col in country_stats.columns]
        X_inflation = country_stats[feature_cols].fillna(0).values
        
        # 인플레이션 레이블 생성 (높은 마커 + 낮은 실제 투자)
        inflation_labels = (
            (country_stats['high_marker_rate'] > 0.3) & 
            (country_stats['actual_green_investment_pct'] < 20)
        ).astype(int)
        
        print(f"✅ 마커 인플레이션 피처 생성 완료: {X_inflation.shape}")
        print(f"   - 인플레이션 의심 국가: {inflation_labels.sum()}개 ({inflation_labels.mean():.1%})")
        
        return X_inflation, inflation_labels
    
    def train_text_model(self, X_text: np.ndarray, y_text: np.ndarray) -> float:
        """텍스트 불일치 모델 훈련"""
        print("🔤 텍스트 불일치 모델 훈련 중...")
        
        # 데이터 분할
        X_train, X_test, y_train, y_test = train_test_split(
            X_text, y_text, test_size=0.2, random_state=42, stratify=y_text
        )
        
        # 모델 훈련
        model = LogisticRegression(
            random_state=42,
            max_iter=1000,
            class_weight='balanced'
        )
        model.fit(X_train, y_train)
        
        # 평가
        y_pred_proba = model.predict_proba(X_test)[:, 1]
        auc_score = roc_auc_score(y_test, y_pred_proba)
        
        print(f"✅ 텍스트 모델 훈련 완료 - AUC: {auc_score:.4f}")
        
        # 모델 저장
        self.models['text_inconsistency_model'] = model
        
        return auc_score
    
    def train_anomaly_model(self, X_anomaly: np.ndarray, y_anomaly: np.ndarray) -> float:
        """투자 이상 모델 훈련"""
        print("📈 투자 이상 모델 훈련 중...")
        
        # Isolation Forest 훈련
        model = IsolationForest(
            contamination=0.05,  # 5% 이상치 가정
            random_state=42,
            n_estimators=100
        )
        model.fit(X_anomaly)
        
        # 이상치 탐지
        anomaly_scores = model.decision_function(X_anomaly)
        anomaly_predictions = model.predict(X_anomaly)
        
        # 이상치 비율 계산
        anomaly_rate = (anomaly_predictions == -1).mean()
        
        print(f"✅ 투자 이상 모델 훈련 완료 - 이상치 비율: {anomaly_rate:.1%}")
        
        # 모델 저장
        self.models['investment_anomaly_model'] = model
        
        return anomaly_rate
    
    def train_inflation_model(self, X_inflation: np.ndarray, y_inflation: np.ndarray) -> float:
        """마커 인플레이션 모델 훈련"""
        print("🎯 마커 인플레이션 모델 훈련 중...")
        
        if len(np.unique(y_inflation)) < 2:
            print("⚠️  인플레이션 레이블이 단일 클래스입니다. 임의 레이블 생성...")
            # 상위 20% 국가를 인플레이션으로 설정
            threshold = np.percentile(X_inflation[:, -1], 80)  # marker_reality_gap 기준
            y_inflation = (X_inflation[:, -1] > threshold).astype(int)
        
        # 데이터 분할
        X_train, X_test, y_train, y_test = train_test_split(
            X_inflation, y_inflation, test_size=0.2, random_state=42, stratify=y_inflation
        )
        
        # 모델 훈련
        model = GradientBoostingClassifier(
            n_estimators=100,
            learning_rate=0.1,
            max_depth=3,
            random_state=42
        )
        model.fit(X_train, y_train)
        
        # 평가
        y_pred_proba = model.predict_proba(X_test)[:, 1]
        auc_score = roc_auc_score(y_test, y_pred_proba)
        
        print(f"✅ 마커 인플레이션 모델 훈련 완료 - AUC: {auc_score:.4f}")
        
        # 모델 저장
        self.models['marker_inflation_model'] = model
        
        return auc_score
    
    def save_models(self):
        """모델 및 설정 저장"""
        print("💾 모델 저장 중...")
        
        models_dir = 'models'
        os.makedirs(models_dir, exist_ok=True)
        
        # 개별 모델 저장
        for model_name, model in self.models.items():
            model_path = os.path.join(models_dir, f'{model_name}.pkl')
            joblib.dump(model, model_path)
            print(f"   ✅ {model_name}.pkl 저장 완료")
        
        # 설정 저장
        config_path = os.path.join(models_dir, 'model_config.json')
        with open(config_path, 'w', encoding='utf-8') as f:
            json.dump(self.config, f, indent=2, ensure_ascii=False)
        print(f"   ✅ model_config.json 저장 완료")
        
        print("🎉 모든 모델 저장 완료!")
    
    def train_all_models(self):
        """전체 모델 훈련 파이프라인"""
        print("🚀 그린워싱 탐지 모델 훈련 시작!")
        print("=" * 50)
        
        # 1. 데이터 로딩
        df = self.load_data()
        
        # 2. 텍스트 모델 훈련
        X_text, y_text = self.create_text_features(df)
        text_auc = self.train_text_model(X_text, y_text)
        self.config['performance_metrics']['text_model_auc'] = text_auc
        
        print("-" * 50)
        
        # 3. 투자 이상 모델 훈련
        X_anomaly, y_anomaly = self.create_anomaly_features(df)
        anomaly_rate = self.train_anomaly_model(X_anomaly, y_anomaly)
        self.config['performance_metrics']['anomaly_detection_rate'] = anomaly_rate
        
        print("-" * 50)
        
        # 4. 마커 인플레이션 모델 훈련
        X_inflation, y_inflation = self.create_inflation_features(df)
        inflation_auc = self.train_inflation_model(X_inflation, y_inflation)
        self.config['performance_metrics']['inflation_model_auc'] = inflation_auc
        
        print("-" * 50)
        
        # 5. 모델 저장
        self.save_models()
        
        print("\n🎉 훈련 완료! 성능 요약:")
        print(f"   📝 텍스트 불일치 AUC: {text_auc:.4f}")
        print(f"   📊 투자 이상 탐지율: {anomaly_rate:.1%}")
        print(f"   🎯 마커 인플레이션 AUC: {inflation_auc:.4f}")
        print("\n💡 이제 predict_greenwashing.py로 예측을 실행할 수 있습니다!")

def main():
    """메인 실행 함수"""
    print("🤖 UNDP 그린워싱 탐지 모델 훈련기")
    print("=" * 50)
    
    # 데이터 파일 확인
    data_path = 'crs_processed.csv'
    if not os.path.exists(data_path):
        print("❌ 전처리된 데이터를 찾을 수 없습니다!")
        print("💡 다음 명령어를 먼저 실행해주세요:")
        print("   python scripts/preprocess_crs_data.py")
        return
    
    try:
        # 훈련 시작
        trainer = GreenwashingModelTrainer(data_path)
        trainer.train_all_models()
        
    except Exception as e:
        print(f"❌ 훈련 중 오류 발생: {str(e)}")
        print("💡 문제가 지속되면 데이터 전처리를 다시 실행해보세요.")
        raise

if __name__ == "__main__":
    main() 