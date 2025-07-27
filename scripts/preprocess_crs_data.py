#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
CRS 데이터 전처리 스크립트
crs_data.csv를 그린워싱 탐지 모델 입력 형식으로 변환
"""

import duckdb
import pandas as pd
import numpy as np
import os

def preprocess_crs_data(input_file='../crs_data.csv', output_file='crs_processed.csv', sample_size=None):
    """
    CRS 데이터를 모델 입력 형식으로 전처리
    
    Args:
        input_file (str): 원본 CRS 데이터 파일 경로
        output_file (str): 처리된 데이터 출력 파일명
        sample_size (int): 샘플링할 행 수 (None이면 전체 처리)
    """
    
    print("🔄 CRS 데이터 전처리 시작...")
    print(f"📂 입력 파일: {input_file}")
    
    # 파일 존재 확인
    if not os.path.exists(input_file):
        raise FileNotFoundError(f"CRS 데이터 파일을 찾을 수 없습니다: {input_file}")
    
    # DuckDB 연결 생성
    conn = duckdb.connect()
    
    print("📊 DuckDB로 데이터 로딩 및 처리 중...")
    
    try:
        # DuckDB로 전체 데이터 한번에 처리 (훨씬 빠름!)
        query = f"""
                 SELECT 
             ProjectTitle,
             COALESCE(ShortDescription, '') as ShortDescription,
             COALESCE(DonorName, 'Unknown') as DonorName,
             CAST(USD_Commitment AS DOUBLE) as USD_Commitment,
             CAST(COALESCE(USD_Disbursement, 0) AS DOUBLE) as USD_Disbursement,
             CAST(COALESCE(ClimateMitigation, 0) AS INTEGER) as ClimateMitigation,
             CAST(COALESCE(ClimateAdaptation, 0) AS INTEGER) as ClimateAdaptation,
             CAST(COALESCE(Environment, 0) AS INTEGER) as Environment,
             CAST(COALESCE(Biodiversity, 0) AS INTEGER) as Biodiversity
        FROM read_csv_auto('{input_file}', null_padding=true, ignore_errors=true)
        WHERE 
            ProjectTitle IS NOT NULL
            AND ProjectTitle != ''
            AND LENGTH(TRIM(ProjectTitle)) > 5
            AND LENGTH(TRIM(ProjectTitle)) < 200
            AND USD_Commitment IS NOT NULL
            AND USD_Commitment > 0
            AND ClimateMitigation BETWEEN 0 AND 2
            AND ClimateAdaptation BETWEEN 0 AND 2  
            AND Environment BETWEEN 0 AND 2
            AND Biodiversity BETWEEN 0 AND 2
        """
        
        # 샘플링이 설정된 경우 LIMIT 추가
        if sample_size:
            query += f" LIMIT {sample_size}"
            
        print(f"🚀 DuckDB 쿼리 실행 중...")
        final_df = conn.execute(query).df()
        
        if len(final_df) == 0:
            raise ValueError("처리할 수 있는 유효한 데이터가 없습니다.")
            
        print(f"✅ DuckDB 처리 완료: {len(final_df):,}개 프로젝트")
        
    except Exception as e:
        print(f"❌ DuckDB 처리 중 오류: {e}")
        raise
    finally:
        conn.close()
    
    # 샘플링은 이미 SQL에서 처리됨
    
    # 중복 제거
    print("🧹 중복 데이터 제거...")
    initial_count = len(final_df)
    final_df = final_df.drop_duplicates(subset=['ProjectTitle', 'USD_Commitment'])
    final_df = final_df.reset_index(drop=True)
    removed_duplicates = initial_count - len(final_df)
    
    # 결과 저장
    final_df.to_csv(output_file, index=False, encoding='utf-8-sig')
    
    print("✅ CRS 데이터 전처리 완료!")
    print(f"📊 최종 결과:")
    print(f"   • 총 프로젝트: {len(final_df):,}개")
    print(f"   • 중복 제거: {removed_duplicates:,}개")
    print(f"   • 출력 파일: {output_file}")
    print(f"   • 파일 크기: {os.path.getsize(output_file) / 1024 / 1024:.1f} MB")
    
    # 기본 통계 출력
    print(f"\n📈 데이터 통계:")
    print(f"   • 평균 투자액: ${final_df['USD_Commitment'].mean():,.0f}")
    print(f"   • 최대 투자액: ${final_df['USD_Commitment'].max():,.0f}")
    print(f"   • 기후완화 마커 평균: {final_df['ClimateMitigation'].mean():.2f}")
    print(f"   • 환경 마커 평균: {final_df['Environment'].mean():.2f}")
    
    return final_df

def main():
    """메인 실행 함수"""
    print("🚀 CRS 데이터 전처리 도구")
    print("=" * 50)
    print("📊 전체 CRS 데이터를 처리합니다...")
    
    try:
        # 전체 데이터 처리만 수행
        df = preprocess_crs_data()
        print(f"\n🎉 처리 완료! 이제 scripts/predict_greenwashing.py에서 사용할 수 있습니다.")
        
    except KeyboardInterrupt:
        print("\n⏹️  사용자에 의해 중단되었습니다.")
    except Exception as e:
        print(f"❌ 오류 발생: {e}")

if __name__ == '__main__':
    main() 