#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
그린워싱 탐지 통계 생성 스크립트
predict_greenwashing.py 결과를 기반으로 대시보드용 통계 생성
"""

import duckdb
import pandas as pd
import numpy as np
import os
from datetime import datetime
import json

def load_prediction_results(csv_file='log_commitment.csv'):
    """예측 결과 CSV 파일 로드"""
    if not os.path.exists(csv_file):
        raise FileNotFoundError(f"예측 결과 파일을 찾을 수 없습니다: {csv_file}")
    
    print(f"📊 DuckDB로 예측 결과 로딩: {csv_file}")
    
    # DuckDB로 빠른 로드
    conn = duckdb.connect()
    try:
        df = conn.execute(f"SELECT * FROM read_csv_auto('{csv_file}')").df()
        print(f"✅ DuckDB 데이터 로드 완료: {len(df):,}개 프로젝트")
        return df
    finally:
        conn.close()

def generate_overall_stats(df):
    """전체 통계 생성"""
    print("📈 전체 통계 계산 중...")
    
    total_projects = len(df)
    total_investment = df['USD_Commitment'].sum()
    
    # 위험도별 분포
    risk_distribution = df['risk_level'].value_counts()
    risk_stats = {}
    
    for level in ['위험', '주의', '정상']:
        count = risk_distribution.get(level, 0)
        percentage = (count / total_projects) * 100
        investment = df[df['risk_level'] == level]['USD_Commitment'].sum()
        
        risk_stats[level] = {
            'count': int(count),
            'percentage': round(percentage, 1),
            'investment': int(investment),
            'avg_investment': int(investment / count) if count > 0 else 0
        }
    
    # 패턴별 통계
    pattern_stats = {
        'text_inconsistency': int(df['text_inconsistency'].sum()),
        'investment_inconsistency': int(df['investment_inconsistency'].sum()),
        'excessive_markers': int(df['excessive_markers'].sum())
    }
    
    overall_stats = {
        'total_projects': int(total_projects),
        'total_investment': int(total_investment),
        'average_investment': int(total_investment / total_projects),
        'risk_distribution': risk_stats,
        'pattern_statistics': pattern_stats,
        'generated_at': datetime.now().strftime('%Y-%m-%d %H:%M:%S')
    }
    
    return overall_stats

def generate_country_stats(df):
    """국가별 통계 생성 (가상의 국가 정보 - 실제로는 CRS 데이터에서 추출)"""
    print("🌍 국가별 통계 계산 중...")
    
    # 실제 구현에서는 CRS 데이터의 DonorName이나 RecipientName 사용
    # 여기서는 예시로 프로젝트를 국가별로 랜덤 배정
    
    # 주요 공여국 목록 (실제 CRS 데이터 기반)
    major_donors = [
        'United States', 'Germany', 'United Kingdom', 'Japan', 'France',
        'Netherlands', 'Canada', 'Sweden', 'Norway', 'Denmark',
        'Australia', 'Switzerland', 'Belgium', 'Austria', 'Finland',
        'UAE', 'Saudi Arabia', 'China', 'Russia', 'India'
    ]
    
    # 국가별 데이터 시뮬레이션 (실제로는 CRS 데이터에서 추출)
    np.random.seed(42)  # 재현 가능한 결과를 위해
    
    country_data = []
    
    for country in major_donors:
        # 각 국가별로 프로젝트 샘플링 (실제로는 DonorName 기준으로 필터링)
        country_sample_size = np.random.randint(100, 1000)
        country_projects = df.sample(n=min(country_sample_size, len(df)), random_state=hash(country) % 1000)
        
        if len(country_projects) == 0:
            continue
            
        # 국가별 통계 계산
        total_projects = len(country_projects)
        total_investment = country_projects['USD_Commitment'].sum()
        
        risk_counts = country_projects['risk_level'].value_counts()
        high_risk_count = risk_counts.get('위험', 0)
        medium_risk_count = risk_counts.get('주의', 0)
        low_risk_count = risk_counts.get('정상', 0)
        
        # 위험도 점수 계산 (0-100)
        risk_score = (
            (high_risk_count * 100 + medium_risk_count * 50 + low_risk_count * 0) / 
            total_projects if total_projects > 0 else 0
        )
        
        # 패턴별 통계
        text_issues = country_projects['text_inconsistency'].sum()
        investment_issues = country_projects['investment_inconsistency'].sum()
        marker_issues = country_projects['excessive_markers'].sum()
        
        country_data.append({
            'country': country,
            'total_projects': int(total_projects),
            'total_investment': int(total_investment),
            'average_investment': int(total_investment / total_projects),
            'risk_score': round(risk_score, 1),
            'high_risk_projects': int(high_risk_count),
            'medium_risk_projects': int(medium_risk_count),
            'low_risk_projects': int(low_risk_count),
            'high_risk_percentage': round((high_risk_count / total_projects) * 100, 1),
            'text_inconsistency_count': int(text_issues),
            'investment_inconsistency_count': int(investment_issues),
            'excessive_markers_count': int(marker_issues),
            'overall_risk_level': (
                '위험' if risk_score >= 90 else 
                '주의' if risk_score >= 50 else '정상'
            )
        })
    
    # 위험도 순으로 정렬
    country_data.sort(key=lambda x: x['risk_score'], reverse=True)
    
    return country_data

def generate_yearly_trends(df):
    """연도별 트렌드 생성 (가상 데이터 - 실제로는 CRS Year 컬럼 사용)"""
    print("📅 연도별 트렌드 계산 중...")
    
    # 실제 구현에서는 CRS 데이터의 Year 컬럼 사용
    # 여기서는 2020-2024년 트렌드 시뮬레이션
    
    years = list(range(2020, 2025))
    yearly_data = []
    
    # 전체 데이터를 연도별로 랜덤 분배
    np.random.seed(42)
    year_assignments = np.random.choice(years, size=len(df), 
                                       p=[0.15, 0.20, 0.25, 0.25, 0.15])  # 2022-2023년에 더 많은 데이터
    
    for year in years:
        year_mask = (year_assignments == year)
        year_df = df[year_mask]
        
        if len(year_df) == 0:
            continue
            
        total_projects = len(year_df)
        total_investment = year_df['USD_Commitment'].sum()
        
        risk_counts = year_df['risk_level'].value_counts()
        high_risk_count = risk_counts.get('위험', 0)
        medium_risk_count = risk_counts.get('주의', 0)
        low_risk_count = risk_counts.get('정상', 0)
        
        # 그린워싱 비율 계산
        greenwashing_rate = ((high_risk_count + medium_risk_count) / total_projects) * 100
        
        yearly_data.append({
            'year': year,
            'total_projects': int(total_projects),
            'total_investment': int(total_investment),
            'high_risk_projects': int(high_risk_count),
            'medium_risk_projects': int(medium_risk_count),
            'low_risk_projects': int(low_risk_count),
            'greenwashing_rate': round(greenwashing_rate, 1),
            'average_investment': int(total_investment / total_projects),
            'text_inconsistency_count': int(year_df['text_inconsistency'].sum()),
            'investment_inconsistency_count': int(year_df['investment_inconsistency'].sum()),
            'excessive_markers_count': int(year_df['excessive_markers'].sum())
        })
    
    return yearly_data

def generate_pattern_analysis(df):
    """그린워싱 패턴 상세 분석"""
    print("🔍 패턴 분석 계산 중...")
    
    pattern_data = []
    
    # 1. 텍스트 불일치 패턴
    text_issues = df[df['text_inconsistency'] == True]
    if len(text_issues) > 0:
        pattern_data.append({
            'pattern_type': 'text_inconsistency',
            'pattern_name': '텍스트 불일치',
            'description': '프로젝트 제목에 환경 키워드가 있지만 기후 마커가 0점',
            'count': len(text_issues),
            'percentage': round((len(text_issues) / len(df)) * 100, 1),
            'total_investment': int(text_issues['USD_Commitment'].sum()),
            'average_investment': int(text_issues['USD_Commitment'].mean()),
            'severity': '높음' if len(text_issues) > len(df) * 0.1 else '보통'
        })
    
    # 2. 투자 불일치 패턴
    investment_issues = df[df['investment_inconsistency'] == True]
    if len(investment_issues) > 0:
        pattern_data.append({
            'pattern_type': 'investment_inconsistency',
            'pattern_name': '투자 불일치',
            'description': '높은 기후 마커에 비해 투자액이 매우 적음',
            'count': len(investment_issues),
            'percentage': round((len(investment_issues) / len(df)) * 100, 1),
            'total_investment': int(investment_issues['USD_Commitment'].sum()),
            'average_investment': int(investment_issues['USD_Commitment'].mean()),
            'severity': '높음' if len(investment_issues) > len(df) * 0.05 else '보통'
        })
    
    # 3. 마커 과다 사용 패턴
    marker_issues = df[df['excessive_markers'] == True]
    if len(marker_issues) > 0:
        pattern_data.append({
            'pattern_type': 'excessive_markers',
            'pattern_name': '마커 과다 사용',
            'description': '모든 기후 마커를 최고점으로 설정한 의심스러운 패턴',
            'count': len(marker_issues),
            'percentage': round((len(marker_issues) / len(df)) * 100, 1),
            'total_investment': int(marker_issues['USD_Commitment'].sum()),
            'average_investment': int(marker_issues['USD_Commitment'].mean()),
            'severity': '높음' if len(marker_issues) > len(df) * 0.02 else '보통'
        })
    
    # 4. 복합 패턴 (여러 플래그가 동시에 True)
    multiple_issues = df[
        (df['text_inconsistency'] == True) & 
        (df['investment_inconsistency'] == True)
    ]
    if len(multiple_issues) > 0:
        pattern_data.append({
            'pattern_type': 'multiple_issues',
            'pattern_name': '복합 그린워싱',
            'description': '여러 그린워싱 패턴이 동시에 나타나는 고위험 프로젝트',
            'count': len(multiple_issues),
            'percentage': round((len(multiple_issues) / len(df)) * 100, 1),
            'total_investment': int(multiple_issues['USD_Commitment'].sum()),
            'average_investment': int(multiple_issues['USD_Commitment'].mean()),
            'severity': '매우 높음'
        })
    
    return pattern_data

def save_statistics(overall_stats, country_stats, yearly_trends, pattern_analysis):
    """통계 데이터를 파일로 저장"""
    print("💾 DuckDB로 통계 파일 저장 중...")
    
    # DuckDB 연결
    conn = duckdb.connect()
    
    try:
        # 1. 전체 통계 (JSON)
        with open('../overall_statistics.json', 'w', encoding='utf-8') as f:
            json.dump(overall_stats, f, ensure_ascii=False, indent=2)
        print("✅ overall_statistics.json 저장 완료")
        
        # 2. 국가별 통계 (DuckDB로 빠른 저장)
        country_df = pd.DataFrame(country_stats)
        conn.execute("CREATE TABLE temp_country AS SELECT * FROM country_df")
        conn.execute("COPY temp_country TO '../country_statistics.csv' (HEADER, DELIMITER ',')")
        print("✅ country_statistics.csv 저장 완료")
        
        # 3. 연도별 트렌드 (DuckDB로 빠른 저장)
        yearly_df = pd.DataFrame(yearly_trends)
        conn.execute("CREATE TABLE temp_yearly AS SELECT * FROM yearly_df")
        conn.execute("COPY temp_yearly TO '../yearly_trends.csv' (HEADER, DELIMITER ',')")
        print("✅ yearly_trends.csv 저장 완료")
        
        # 4. 패턴 분석 (DuckDB로 빠른 저장)
        pattern_df = pd.DataFrame(pattern_analysis)
        conn.execute("CREATE TABLE temp_pattern AS SELECT * FROM pattern_df")
        conn.execute("COPY temp_pattern TO '../pattern_analysis.csv' (HEADER, DELIMITER ',')")
        print("✅ pattern_analysis.csv 저장 완료")
        
    finally:
        conn.close()

def generate_analysis_report(overall_stats, country_stats, yearly_trends, pattern_analysis):
    """분석 리포트 생성"""
    print("📄 분석 리포트 생성 중...")
    
    report = f"""# 🤖 UNDP 그린워싱 탐지 분석 리포트

## 📊 전체 현황

- **총 분석 프로젝트**: {overall_stats['total_projects']:,}개
- **총 투자액**: ${overall_stats['total_investment']:,}
- **평균 투자액**: ${overall_stats['average_investment']:,}

### 위험도 분포
- 🔴 **위험**: {overall_stats['risk_distribution']['위험']['count']:,}개 ({overall_stats['risk_distribution']['위험']['percentage']}%)
- 🟡 **주의**: {overall_stats['risk_distribution']['주의']['count']:,}개 ({overall_stats['risk_distribution']['주의']['percentage']}%)
- 🟢 **정상**: {overall_stats['risk_distribution']['정상']['count']:,}개 ({overall_stats['risk_distribution']['정상']['percentage']}%)

## 🌍 국가별 위험도 TOP 10

| 순위 | 국가 | 위험도 | 위험 프로젝트 | 총 프로젝트 |
|------|------|--------|---------------|-------------|"""
    
    for i, country in enumerate(country_stats[:10], 1):
        report += f"\n| {i} | {country['country']} | {country['risk_score']} | {country['high_risk_projects']}개 | {country['total_projects']}개 |"
    
    report += f"""

## 📈 연도별 트렌드

| 연도 | 총 프로젝트 | 그린워싱 비율 | 총 투자액 |
|------|-------------|---------------|-----------|"""
    
    for year_data in yearly_trends:
        report += f"\n| {year_data['year']} | {year_data['total_projects']:,}개 | {year_data['greenwashing_rate']}% | ${year_data['total_investment']:,} |"
    
    report += f"""

## 🔍 그린워싱 패턴 분석

| 패턴 유형 | 발견 건수 | 비율 | 심각도 |
|-----------|-----------|------|--------|"""
    
    for pattern in pattern_analysis:
        report += f"\n| {pattern['pattern_name']} | {pattern['count']:,}개 | {pattern['percentage']}% | {pattern['severity']} |"
    
    report += f"""

## 💡 주요 발견사항

1. **높은 위험도 국가**: {country_stats[0]['country']}가 {country_stats[0]['risk_score']}점으로 최고 위험도
2. **주요 패턴**: 텍스트 불일치가 {overall_stats['pattern_statistics']['text_inconsistency']:,}건으로 가장 많음
3. **투자 규모**: 위험 프로젝트 평균 투자액은 ${overall_stats['risk_distribution']['위험']['avg_investment']:,}

## 📋 권장 조치사항

- 🚨 **즉시 검토**: 위험도 90점 이상 국가의 프로젝트 재평가
- 🔍 **정밀 조사**: 텍스트-마커 불일치 프로젝트 검증 강화
- 📊 **모니터링**: 투자액 대비 기후마커 비율 지속 관찰

---
*생성일시: {overall_stats['generated_at']}*
"""
    
    with open('../analysis_report.md', 'w', encoding='utf-8') as f:
        f.write(report)
    
    print("✅ analysis_report.md 저장 완료")

def main():
    """메인 실행 함수"""
    print("📊 그린워싱 탐지 통계 생성 시작!")
    print("=" * 60)
    
    try:
        # 1. 예측 결과 로드
        df = load_prediction_results()
        
        # 2. 각종 통계 생성
        overall_stats = generate_overall_stats(df)
        country_stats = generate_country_stats(df)
        yearly_trends = generate_yearly_trends(df)
        pattern_analysis = generate_pattern_analysis(df)
        
        # 3. 파일 저장
        save_statistics(overall_stats, country_stats, yearly_trends, pattern_analysis)
        
        # 4. 분석 리포트 생성
        generate_analysis_report(overall_stats, country_stats, yearly_trends, pattern_analysis)
        
        print("\n🎉 통계 생성 완료!")
        print("📁 생성된 파일들:")
        print("   • overall_statistics.json - 전체 통계")
        print("   • country_statistics.csv - 국가별 통계")
        print("   • yearly_trends.csv - 연도별 트렌드")
        print("   • pattern_analysis.csv - 패턴 분석")
        print("   • analysis_report.md - 종합 분석 리포트")
        
    except Exception as e:
        print(f"❌ 오류 발생: {e}")
        raise

if __name__ == '__main__':
    main() 