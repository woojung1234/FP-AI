"""
FlavorPairing 프로젝트 - Google Colab용 실행 스크립트

이 스크립트는 Google Colab에서 GPU를 사용해 FlavorPairing 모델을 학습하는 데 사용됩니다.
"""

import os
import sys
import torch

def check_gpu():
    """GPU 상태를 확인하고 출력합니다."""
    print("\n===== GPU 정보 =====")
    if torch.cuda.is_available():
        print(f"GPU 이용 가능: {torch.cuda.is_available()}")
        print(f"GPU 개수: {torch.cuda.device_count()}")
        print(f"GPU 이름: {torch.cuda.get_device_name(0)}")
        print(f"CUDA 버전: {torch.version.cuda}")
        
        # GPU 메모리 정보
        if hasattr(torch.cuda, 'get_device_properties'):
            print(f"GPU 총 메모리: {torch.cuda.get_device_properties(0).total_memory / 1e9:.2f} GB")
            print(f"현재 할당된 메모리: {torch.cuda.memory_allocated(0) / 1e9:.2f} GB")
            print(f"캐시된 메모리: {torch.cuda.memory_reserved(0) / 1e9:.2f} GB")
    else:
        print("GPU를 사용할 수 없습니다. CPU로 실행합니다.")
    print("=====================\n")

def main():
    # 현재 디렉토리 확인
    print(f"현재 작업 디렉토리: {os.getcwd()}")
    
    # GPU 상태 확인
    check_gpu()
    
    # 필요한 패키지 확인 및 설치
    try:
        import torch_geometric
        print("torch_geometric 패키지가 이미 설치되어 있습니다.")
    except ImportError:
        print("torch_geometric 패키지를 설치합니다...")
        # PyG 설치
        os.system("pip install torch-scatter torch-sparse torch-cluster torch-spline-conv -f https://data.pyg.org/whl/torch-2.0.0+cu118.html")
        os.system("pip install torch-geometric")
    
    # model 디렉토리로 이동하여 학습 스크립트 실행
    if os.path.exists("model"):
        os.chdir("model")
        print("model 디렉토리로 이동했습니다.")
    else:
        print("model 디렉토리를 찾을 수 없습니다!")
        return
    
    # 필요한 디렉토리 생성
    os.makedirs("checkpoint", exist_ok=True)
    
    # 학습 스크립트 실행
    print("\n===== 모델 학습 시작 =====")
    try:
        import train
        # 이미 main() 함수가 있어 실행됨
    except Exception as e:
        print(f"학습 중 오류가 발생했습니다: {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    main()
