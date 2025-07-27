#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
그린워싱 탐지 예측 스크립트
개발자가 바로 사용할 수 있는 완전한 예시
"""

import duckdb
import pandas as pd
import numpy as np
import joblib
import json
import re
import warnings
import os
warnings.filterwarnings('ignore')

def load_models():
    """모델 및 설정 로드"""
    print("🔄 모델 로딩 중...")
    
    models = {
        'text_model': joblib.load('../models/text_inconsistency_model.pkl'),
        'tfidf': joblib.load('../models/tfidf_vectorizer.pkl'),
        'anomaly_model': joblib.load('../models/investment_anomaly_model.pkl'),
        'scaler': joblib.load('../models/anomaly_scaler.pkl'),
        'inflation_model': joblib.load('../models/marker_inflation_model.pkl')
    }
    
    with open('../models/model_config.json', 'r', encoding='utf-8') as f:
        config = json.load(f)
    
    print("✅ 모델 로드 완료!")
    return models, config

def preprocess_text(text):
    """텍스트 전처리"""
    if pd.isna(text) or text == '':
        return ''
    text = str(text).lower()
    text = re.sub(r'[^a-zA-Z\s]', ' ', text)
    text = re.sub(r'\s+', ' ', text).strip()
    return text

def predict_single_project(models, config, project_data):
    """
    단일 프로젝트 그린워싱 예측
    
    Args:
        project_data (dict): {
            'ProjectTitle': str,
            'USD_Commitment': float,
            'ClimateMitigation': int (0,1,2),
            'ClimateAdaptation': int (0,1,2),
            'Environment': int (0,1,2),
            'Biodiversity': int (0,1,2)
        }
    
    Returns:
        dict: 예측 결과
    """
    
    # 텍스트 전처리
    title_clean = preprocess_text(project_data['ProjectTitle'])
    
    # 환경 키워드 체크
    env_keywords = config['environmental_keywords']
    has_env_keywords = any(keyword in title_clean for keyword in env_keywords)
    
    # 기후 마커 총합
    total_climate_score = (project_data['ClimateMitigation'] + 
                          project_data['ClimateAdaptation'] + 
                          project_data['Environment'] + 
                          project_data['Biodiversity'])
    
    all_markers_zero = (total_climate_score == 0)
    
    # 위험 플래그들
    text_inconsistency = has_env_keywords and all_markers_zero
    investment_inconsistency = (total_climate_score >= 2) and (project_data['USD_Commitment'] < 100000)
    excessive_markers = (total_climate_score >= 4)
    
    # 위험도 점수 계산
    risk_score = 0
    if text_inconsistency:
        risk_score += 40
    if investment_inconsistency:
        risk_score += 35
    if excessive_markers:
        risk_score += 25
    
    # 위험도 0.9 이상 = "위험", 0.5 이상 = "주의", 그 외 = "정상"
    risk_ratio = risk_score / 100.0
    
    if risk_ratio >= 0.9:
        risk_level = "위험"
        risk_color = "🔴"
    elif risk_ratio >= 0.5:
        risk_level = "주의"
        risk_color = "🟡"
    else:
        risk_level = "정상"
        risk_color = "🟢"
    
    return {
        "project_title": project_data['ProjectTitle'],
        "risk_score": risk_score,
        "risk_ratio": round(risk_ratio, 3),
        "risk_level": risk_level,
        "risk_color": risk_color,
        "flags": {
            "text_inconsistency": text_inconsistency,
            "investment_inconsistency": investment_inconsistency,
            "excessive_markers": excessive_markers
        },
        "details": {
            "has_environmental_keywords": has_env_keywords,
            "total_climate_score": total_climate_score,
            "usd_commitment": project_data['USD_Commitment']
        }
    }

def predict_csv_file(models, config, input_csv, output_csv):
    """
    CSV 파일 일괄 예측
    
    Args:
        input_csv (str): 입력 CSV 파일명 (예: 'crs_processed.csv')
        output_csv (str): 출력 CSV 파일명 (예: 'log_commitment.csv')
    """
    print(f"📊 CSV 파일 처리: {input_csv}")
    
    # DuckDB로 CSV 파일 읽기 (빠른 성능)
    conn = duckdb.connect()
    
    try:
        # 필수 컬럼 확인 및 데이터 로드
        query = f"""
        SELECT 
            ProjectTitle,
            USD_Commitment,
            ClimateMitigation,
            ClimateAdaptation,
            Environment,
            Biodiversity
        FROM read_csv_auto('{input_csv}', null_padding=true, ignore_errors=true)
        WHERE 
            ProjectTitle IS NOT NULL
            AND USD_Commitment IS NOT NULL
            AND ClimateMitigation IS NOT NULL
            AND ClimateAdaptation IS NOT NULL
            AND Environment IS NOT NULL
            AND Biodiversity IS NOT NULL
        """
        
        df = conn.execute(query).df()
        print(f"✅ DuckDB 데이터 로드: {len(df)}개 프로젝트")
        
        if len(df) == 0:
            raise ValueError("유효한 데이터가 없습니다.")
            
    except Exception as e:
        conn.close()
        raise ValueError(f"데이터 로드 오류: {e}")
    finally:
        conn.close()
    
    # 각 프로젝트 예측
    results = []
    for idx, row in df.iterrows():
        project_data = {
            'ProjectTitle': row['ProjectTitle'],
            'USD_Commitment': row['USD_Commitment'],
            'ClimateMitigation': row['ClimateMitigation'],
            'ClimateAdaptation': row['ClimateAdaptation'],
            'Environment': row['Environment'],
            'Biodiversity': row['Biodiversity']
        }
        
        result = predict_single_project(models, config, project_data)
        results.append(result)
        
        if (idx + 1) % 100 == 0:
            print(f"   처리 완료: {idx + 1}/{len(df)}")
    
    # 결과를 DataFrame으로 변환 (log_commitment 형식)
    results_df = pd.DataFrame([
        {
            'ProjectTitle': r['project_title'],
            'USD_Commitment': df.iloc[i]['USD_Commitment'],
            'log_commitment': np.log1p(df.iloc[i]['USD_Commitment']),  # log_commitment 컬럼 추가
            'risk_score': r['risk_score'],
            'risk_ratio': r['risk_ratio'],
            'risk_level': r['risk_level'],
            'text_inconsistency': r['flags']['text_inconsistency'],
            'investment_inconsistency': r['flags']['investment_inconsistency'],
            'excessive_markers': r['flags']['excessive_markers'],
            'total_climate_score': r['details']['total_climate_score']
        }
        for i, r in enumerate(results)
    ])
    
    # CSV 파일로 저장
    results_df.to_csv(output_csv, index=False, encoding='utf-8-sig')
    print(f"✅ 결과 저장: {output_csv}")
    
    # 통계 출력
    risk_stats = results_df['risk_level'].value_counts()
    print(f"\n📊 위험도 분포:")
    for level, count in risk_stats.items():
        color = "🔴" if level == "위험" else "🟡" if level == "주의" else "🟢"
        print(f"   {color} {level}: {count}개 ({count/len(results_df)*100:.1f}%)")
    
    return results_df

def main():
    """메인 실행 함수"""
    print("🤖 그린워싱 탐지 예측 시작!")
    print("=" * 50)
    
    # 모델 로드
    models, config = load_models()
    
    # 예시 1: 단일 프로젝트 예측
    print("\n🧪 예시 1: 단일 프로젝트 예측")
    sample_project = {
        'ProjectTitle': 'Green Climate Sustainable Development Initiative',
        'USD_Commitment': 50000,  # 낮은 투자액
        'ClimateMitigation': 0,   # 모든 마커 0
        'ClimateAdaptation': 0,
        'Environment': 0,
        'Biodiversity': 0
    }
    
    result = predict_single_project(models, config, sample_project)
    print(f"프로젝트: {result['project_title']}")
    print(f"위험도: {result['risk_color']} {result['risk_ratio']} ({result['risk_level']})")
    print(f"플래그: 텍스트불일치={result['flags']['text_inconsistency']}, "
          f"투자불일치={result['flags']['investment_inconsistency']}, "
          f"마커과다={result['flags']['excessive_markers']}")
    
    # 예시 2: 전체 CRS 데이터 처리
    if os.path.exists('../crs_processed.csv'):
        print(f"\n📊 예시 2: 전체 CRS 데이터 일괄 처리")
        results_df = predict_csv_file(models, config, '../crs_processed.csv', '../log_commitment.csv')
        print("✅ CSV 처리 완료!")
    else:
        print("\n💡 CRS 데이터 전처리 필요:")
        print("   python scripts/preprocess_crs_data.py 실행하여 crs_processed.csv 생성")

if __name__ == '__main__':
    main() 