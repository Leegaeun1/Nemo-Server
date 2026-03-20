import os
os.environ['PROTOCOL_BUFFERS_PYTHON_IMPLEMENTATION'] = 'python'

try:
    import numpy as np
    import cv2
    import tensorflow as tf
    import mediapipe as mp
    import insightface
    import onnxruntime
    
    print("--- ✅ 모든 라이브러리 로드 성공! ---")
    print(f"NumPy 버전: {np.__version__}")
    print(f"Protobuf 버전: {tf.sysconfig.get_build_info().get('protobuf_version', 'Unknown')}")
except ImportError as e:
    print(f"--- ❌ 설치 실패: {e} ---")
except Exception as e:
    print(f"--- ⚠️ 런타임 에러 (버전 충돌 가능성): {e} ---")