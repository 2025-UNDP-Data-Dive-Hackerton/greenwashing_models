# 2. **`detect_greenwashing(project_data)`** - 그린워싱 탐지  

import sys
import os

# 현재 파일의 디렉토리를 Python 경로에 추가
current_dir = os.path.dirname(os.path.abspath(__file__))
sys.path.append(current_dir)

# 직접 import
from simple_model1_tools import SimpleModel1Tools


# 도구 초기화
tools = SimpleModel1Tools()

# 딕셔너리 형태로 프로젝트 정보 입력
project_data = {
    'ProjectTitle': 'Green Climate Initiative',
    'ClimateMitigation': 0,      # 기후완화 마커 (0,1,2)
    'ClimateAdaptation': 0,      # 기후적응 마커 (0,1,2)
    'Environment': 0,            # 환경 마커 (0,1,2)
    'Biodiversity': 0            # 생물다양성 마커 (0,1,2)
}
greenwashing_result = tools.detect_greenwashing(project_data)
print("\n=== 그린워싱 탐지 결과 ===")
print("그린워싱 탐지 결과:", greenwashing_result)
print(f"불일치 여부: {greenwashing_result['is_inconsistent']}")
print(f"불일치 점수: {greenwashing_result['inconsistency_score']}/100")
print(f"위험도: {greenwashing_result['risk_level']}")
