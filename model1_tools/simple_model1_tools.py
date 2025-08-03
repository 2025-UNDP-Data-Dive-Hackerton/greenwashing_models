#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Model 1 텍스트 불일치 탐지 - 간단한 사용 도구
개발자가 결과를 직접 print하거나 처리할 수 있도록 데이터만 반환
"""

import joblib
import json
import pandas as pd
import numpy as np
from datetime import datetime
import re
from collections import Counter

class SimpleModel1Tools:
    def __init__(self, model_path='scripts/tuning/models/text_inconsistency/best_model_20250731_191849.pkl',
                 config_path='scripts/tuning/models/config/model_config.json'):
        """Model 1 도구 초기화 (출력 없음)"""
        
        # 모델 로드
        self.model_data = joblib.load(model_path)
        self.text_model = self.model_data['model']
        self.tfidf = self.model_data.get('tfidf_vectorizer')
        
        # 설정 로드
        with open(config_path, 'r', encoding='utf-8') as f:
            self.config = json.load(f)
        
        self.environmental_keywords = self.config['environmental_keywords']
    
    def preprocess_text(self, text):
        """텍스트 전처리"""
        if not text:
            return ""
        
        text = str(text).lower()
        text = re.sub(r'[^\w\s]', ' ', text)
        text = re.sub(r'\s+', ' ', text).strip()
        return text
    
    # ===== 1. 프로젝트 제목 분석 =====
    def analyze_title(self, title):
        """
        프로젝트 제목 분석 - 환경 키워드 탐지
        
        Returns:
            dict: {
                'title': str,
                'detected_keywords': list,
                'keyword_count': int,
                'ml_probability': float,
                'risk_score': int,
                'risk_level': str
            }
        """
        title_clean = self.preprocess_text(title)
        detected_keywords = []
        
        for keyword in self.environmental_keywords:
            if keyword.lower() in title_clean:
                detected_keywords.append(keyword)
        
        # ML 모델 예측
        ml_probability = 0.0
        if self.tfidf and detected_keywords:
            try:
                title_tfidf = self.tfidf.transform([title_clean])
                ml_probability = float(self.text_model.predict_proba(title_tfidf)[0][1])
            except:
                pass
        
        # 위험도 계산
        risk_score = 0
        if len(detected_keywords) >= 3:
            risk_score += 30
        elif len(detected_keywords) >= 2:
            risk_score += 20
        elif len(detected_keywords) >= 1:
            risk_score += 10
            
        if len(detected_keywords) / len(title_clean.split()) > 0.3:
            risk_score += 20
        
        risk_level = 'HIGH' if risk_score >= 40 else 'MEDIUM' if risk_score >= 20 else 'LOW'
        
        return {
            'title': title,
            'detected_keywords': detected_keywords,
            'keyword_count': len(detected_keywords),
            'ml_probability': ml_probability,
            'risk_score': risk_score,
            'risk_level': risk_level
        }
    
    # ===== 2. 텍스트-마커 불일치 탐지 =====
    def detect_greenwashing(self, project_data):
        """
        텍스트-마커 불일치 탐지
        
        Args:
            project_data (dict): {
                'ProjectTitle': str,
                'ClimateMitigation': int,
                'ClimateAdaptation': int,
                'Environment': int,
                'Biodiversity': int
            }
        
        Returns:
            dict: {
                'title': str,
                'detected_keywords': list,
                'total_markers': int,
                'is_inconsistent': bool,
                'inconsistency_score': int,
                'risk_level': str,
                'ml_probability': float
            }
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
        
        # ML 모델 예측
        ml_probability = 0.0
        if self.tfidf and detected_keywords:
            try:
                title_tfidf = self.tfidf.transform([title_clean])
                ml_probability = float(self.text_model.predict_proba(title_tfidf)[0][1])
            except:
                pass
        
        # 불일치 판정
        has_keywords = len(detected_keywords) > 0
        has_zero_markers = total_markers == 0
        is_inconsistent = has_keywords and has_zero_markers
        
        # 불일치 점수 계산
        inconsistency_score = 0
        if is_inconsistent:
            inconsistency_score += 40
            if len(detected_keywords) >= 3:
                inconsistency_score += 20
            elif len(detected_keywords) >= 2:
                inconsistency_score += 10
            
            if ml_probability > 0.7:
                inconsistency_score += 20
            elif ml_probability > 0.5:
                inconsistency_score += 10
        
        inconsistency_score = min(100, inconsistency_score)
        
        # 위험도 레벨
        if inconsistency_score >= 80:
            risk_level = 'CRITICAL'
        elif inconsistency_score >= 60:
            risk_level = 'HIGH'
        elif inconsistency_score >= 40:
            risk_level = 'MEDIUM'
        elif inconsistency_score >= 20:
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
            'ml_probability': ml_probability
        }
    
    # ===== 3. 키워드 패턴 분석 =====
    def analyze_keyword_patterns(self, projects_df):
        """
        키워드 사용 패턴 분석
        
        Args:
            projects_df: DataFrame with columns ['ProjectTitle', 'ClimateMitigation', 'ClimateAdaptation', 'Environment', 'Biodiversity']
        
        Returns:
            dict: {
                'keyword_stats': dict,  # 키워드별 통계
                'abuse_ranking': list,  # 악용 순위
                'summary': dict        # 전체 요약
            }
        """
        projects_df = projects_df.copy()
        projects_df['total_markers'] = (
            projects_df['ClimateMitigation'] + 
            projects_df['ClimateAdaptation'] + 
            projects_df['Environment'] + 
            projects_df['Biodiversity']
        )
        
        keyword_stats = {}
        total_projects = len(projects_df)
        
        for keyword in self.environmental_keywords:
            keyword_projects = projects_df[
                projects_df['ProjectTitle'].str.contains(keyword, case=False, na=False)
            ]
            
            if len(keyword_projects) == 0:
                continue
            
            greenwashing_projects = keyword_projects[keyword_projects['total_markers'] == 0]
            
            keyword_stats[keyword] = {
                'total_usage': len(keyword_projects),
                'usage_percentage': len(keyword_projects) / total_projects * 100,
                'greenwashing_count': len(greenwashing_projects),
                'greenwashing_rate': len(greenwashing_projects) / len(keyword_projects) * 100,
                'avg_investment': keyword_projects['USD_Commitment'].mean() if 'USD_Commitment' in keyword_projects.columns else 0
            }
        
        # 악용 순위 생성
        abuse_ranking = []
        for keyword, stats in keyword_stats.items():
            abuse_ranking.append({
                'keyword': keyword,
                'total_usage': stats['total_usage'],
                'greenwashing_rate': stats['greenwashing_rate'],
                'usage_percentage': stats['usage_percentage']
            })
        
        abuse_ranking.sort(key=lambda x: (x['greenwashing_rate'], x['total_usage']), reverse=True)
        
        # 요약 통계
        projects_with_keywords = len(projects_df[
            projects_df['ProjectTitle'].str.contains('|'.join(self.environmental_keywords), case=False, na=False)
        ])
        
        summary = {
            'total_projects': total_projects,
            'projects_with_keywords': projects_with_keywords,
            'keyword_usage_rate': projects_with_keywords / total_projects * 100,
            'analyzed_keywords': len(keyword_stats),
            'most_abused_keyword': abuse_ranking[0]['keyword'] if abuse_ranking else None,
            'highest_abuse_rate': abuse_ranking[0]['greenwashing_rate'] if abuse_ranking else 0
        }
        
        return {
            'keyword_stats': keyword_stats,
            'abuse_ranking': abuse_ranking,
            'summary': summary
        }
    
    # ===== 4. 기관별 언어 패턴 분석 =====
    def analyze_organization_patterns(self, projects_df, org_column='DonorName', min_projects=10):
        """
        기관별 언어 패턴 분석
        
        Args:
            projects_df: DataFrame
            org_column: 기관명 컬럼
            min_projects: 최소 프로젝트 수
        
        Returns:
            dict: {
                'organization_stats': dict,  # 기관별 통계
                'ranking': list,            # 의심도 순위
                'summary': dict            # 전체 요약
            }
        """
        projects_df = projects_df.copy()
        projects_df['total_markers'] = (
            projects_df['ClimateMitigation'] + 
            projects_df['ClimateAdaptation'] + 
            projects_df['Environment'] + 
            projects_df['Biodiversity']
        )
        
        org_groups = projects_df.groupby(org_column)
        organization_stats = {}
        
        for org_name, org_projects in org_groups:
            if len(org_projects) < min_projects:
                continue
            
            # 환경 키워드 사용 프로젝트
            env_projects = org_projects[
                org_projects['ProjectTitle'].str.contains('|'.join(self.environmental_keywords), case=False, na=False)
            ]
            
            # 그린워싱 프로젝트 (환경 키워드 있지만 마커 0)
            greenwashing_projects = env_projects[env_projects['total_markers'] == 0]
            
            # 의심도 점수 계산
            env_usage_rate = len(env_projects) / len(org_projects) * 100
            greenwashing_rate = len(greenwashing_projects) / len(env_projects) * 100 if len(env_projects) > 0 else 0
            
            suspicion_score = 0
            if env_usage_rate > 50:
                suspicion_score += 30
            elif env_usage_rate > 30:
                suspicion_score += 20
            elif env_usage_rate > 15:
                suspicion_score += 10
            
            if greenwashing_rate > 70:
                suspicion_score += 40
            elif greenwashing_rate > 50:
                suspicion_score += 30
            elif greenwashing_rate > 30:
                suspicion_score += 20
            
            # 위험도 레벨
            if suspicion_score >= 60:
                risk_level = 'HIGH'
            elif suspicion_score >= 40:
                risk_level = 'MEDIUM'
            elif suspicion_score >= 20:
                risk_level = 'LOW'
            else:
                risk_level = 'MINIMAL'
            
            organization_stats[org_name] = {
                'total_projects': len(org_projects),
                'env_projects': len(env_projects),
                'env_usage_rate': env_usage_rate,
                'greenwashing_projects': len(greenwashing_projects),
                'greenwashing_rate': greenwashing_rate,
                'suspicion_score': suspicion_score,
                'risk_level': risk_level,
                'avg_investment': org_projects['USD_Commitment'].mean() if 'USD_Commitment' in org_projects.columns else 0
            }
        
        # 순위 생성
        ranking = []
        for org_name, stats in organization_stats.items():
            ranking.append({
                'organization': org_name,
                'total_projects': stats['total_projects'],
                'env_usage_rate': stats['env_usage_rate'],
                'greenwashing_rate': stats['greenwashing_rate'],
                'suspicion_score': stats['suspicion_score'],
                'risk_level': stats['risk_level']
            })
        
        ranking.sort(key=lambda x: x['suspicion_score'], reverse=True)
        
        # 순위 번호 추가
        for i, org in enumerate(ranking, 1):
            org['rank'] = i
        
        # 요약 통계
        summary = {
            'total_organizations': len(organization_stats),
            'high_risk_organizations': len([org for org in ranking if org['risk_level'] == 'HIGH']),
            'most_suspicious_org': ranking[0]['organization'] if ranking else None,
            'highest_suspicion_score': ranking[0]['suspicion_score'] if ranking else 0
        }
        
        return {
            'organization_stats': organization_stats,
            'ranking': ranking,
            'summary': summary
        }
    
    # ===== 5. 배치 분석 (여러 프로젝트 한번에) =====
    def batch_analyze(self, projects_df):
        """
        여러 프로젝트 일괄 분석
        
        Returns:
            dict: {
                'project_results': list,  # 각 프로젝트별 결과
                'summary': dict          # 전체 요약
            }
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
            'inconsistency_rate': inconsistent_projects / total_projects * 100,
            'high_risk_projects': high_risk_projects,
            'high_risk_rate': high_risk_projects / total_projects * 100
        }
        
        return {
            'project_results': results,
            'summary': summary
        }

# ===== 사용 예시 =====
def example_usage():
    """사용 예시 - 개발자가 원하는 대로 print 가능"""
    
    # 도구 초기화
    tools = SimpleModel1Tools()
    
    # 1. 단일 제목 분석
    title_result = tools.analyze_title("Green Sustainable Climate Development Initiative")
    print("=== 제목 분석 결과 ===")
    print(f"제목: {title_result['title']}")
    print(f"키워드: {title_result['detected_keywords']}")
    print(f"위험도: {title_result['risk_level']} ({title_result['risk_score']}점)")
    print(f"ML 확률: {title_result['ml_probability']:.3f}")
    
    # 2. 그린워싱 탐지
    project = {
        'ProjectTitle': 'Green Climate Sustainable Development',
        'ClimateMitigation': 0,
        'ClimateAdaptation': 0,
        'Environment': 0,
        'Biodiversity': 0
    }
    
    greenwashing_result = tools.detect_greenwashing(project)
    print("\n=== 그린워싱 탐지 결과 ===")
    print(f"불일치 여부: {greenwashing_result['is_inconsistent']}")
    print(f"불일치 점수: {greenwashing_result['inconsistency_score']}/100")
    print(f"위험도: {greenwashing_result['risk_level']}")
    
    # 3. CSV 파일이 있다면 배치 분석
    try:
        import os
        if os.path.exists('crs_processed.csv'):
            df = pd.read_csv('crs_processed.csv').head(100)  # 샘플 100개만
            
            batch_result = tools.batch_analyze(df)
            print(f"\n=== 배치 분석 결과 ===")
            print(f"총 프로젝트: {batch_result['summary']['total_projects']}개")
            print(f"불일치 프로젝트: {batch_result['summary']['inconsistent_projects']}개")
            print(f"불일치율: {batch_result['summary']['inconsistency_rate']:.1f}%")
            
            # 키워드 패턴 분석
            keyword_result = tools.analyze_keyword_patterns(df)
            print(f"\n=== 키워드 패턴 분석 ===")
            print(f"분석된 키워드: {keyword_result['summary']['analyzed_keywords']}개")
            print(f"가장 악용되는 키워드: {keyword_result['summary']['most_abused_keyword']}")
            print(f"최고 악용률: {keyword_result['summary']['highest_abuse_rate']:.1f}%")
            
            # 기관별 분석 (DonorName 컬럼이 있다면)
            if 'DonorName' in df.columns:
                org_result = tools.analyze_organization_patterns(df)
                print(f"\n=== 기관별 분석 결과 ===")
                print(f"분석된 기관: {org_result['summary']['total_organizations']}개")
                print(f"고위험 기관: {org_result['summary']['high_risk_organizations']}개")
                if org_result['summary']['most_suspicious_org']:
                    print(f"가장 의심스러운 기관: {org_result['summary']['most_suspicious_org']}")
    
    except Exception as e:
        print(f"CSV 분석 중 오류: {e}")

if __name__ == '__main__':
    example_usage()