# 1. **`analyze_title(title)`** - 단일 제목 분석
import sys
import os

# 현재 파일의 디렉토리를 Python 경로에 추가
current_dir = os.path.dirname(os.path.abspath(__file__))
sys.path.append(current_dir)

# 직접 import
from simple_model1_tools import SimpleModel1Tools

# 도구 초기화
tools = SimpleModel1Tools()

# 1. 단일 제목 분석
result = tools.analyze_title("Green Climate Initiative")
print(f"키워드: {result['detected_keywords']}")
print(f"위험도: {result['risk_level']}")