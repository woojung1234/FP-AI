from dataset import map_graph_nodes, edges_index
import torch
from models import NeuralCF
import os
import glob

def predict(user_ids, item_ids, edges_indexes, edges_weights, edge_type):
    model = NeuralCF(num_users=155, num_items=6498, emb_size=128)
    
    # 모델 로드 로직 수정 - 최신 체크포인트 사용
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    
    # 모델 로드 로직 업데이트
    model_loaded = False
    
    # 1. 체크포인트 경로 확인 및 조정
    checkpoint_dirs = ["./model/checkpoint/", "./checkpoint/"]
    
    for checkpoint_dir in checkpoint_dirs:
        if not model_loaded:
            # 2. best_model.pth 먼저 시도
            best_model_path = os.path.join(checkpoint_dir, "best_model.pth")
            if os.path.exists(best_model_path):
                try:
                    if torch.cuda.is_available():
                        model.load_state_dict(torch.load(best_model_path))
                    else:
                        model.load_state_dict(torch.load(best_model_path, map_location=torch.device('cpu')))
                    print(f"Loaded best model from: {best_model_path}")
                    model_loaded = True
                except Exception as e:
                    print(f"Error loading best model: {e}")
            
            # 3. 최신 에포크 체크포인트 시도
            if not model_loaded:
                checkpoints = glob.glob(os.path.join(checkpoint_dir, "epoch_*.pth"))
                if checkpoints:
                    latest_checkpoint = max(checkpoints, key=lambda x: int(x.split('_')[-1].split('.')[0]))
                    try:
                        if torch.cuda.is_available():
                            model.load_state_dict(torch.load(latest_checkpoint))
                        else:
                            model.load_state_dict(torch.load(latest_checkpoint, map_location=torch.device('cpu')))
                        print(f"Loaded latest checkpoint from: {latest_checkpoint}")
                        model_loaded = True
                    except Exception as e:
                        print(f"Error loading checkpoint: {e}")
    
    if not model_loaded:
        print("WARNING: Could not load any model! Using untrained model.")
    
    model = model.to(device)
    model.eval()

    with torch.no_grad():
        user_tensor = torch.tensor(user_ids, device=device)
        item_tensor = torch.tensor(item_ids, device=device)
        output = model(user_tensor, item_tensor, edges_indexes, edge_type, edges_weights)
        return output

mapping = map_graph_nodes()
    
lid_to_idx = mapping['liquor']
iid_to_idx = mapping['ingredient']

edge_type_map ={
        'liqr-ingr': 0,
        'ingr-ingr': 1,
        'liqr-liqr': 1,
        'ingr-fcomp': 2,
        'ingr-dcomp': 2
    }

edges_indexes, edges_weights, edge_type = edges_index(edge_type_map)

# GPU 사용 시 텐서를 적절한 디바이스로 이동
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
edges_indexes = edges_indexes.to(device)
edges_weights = edges_weights.to(device)
edge_type = edge_type.to(device)

print(f"Test script running on: {device}")
print("모델 로드 중...")

while True:
    try:
        user_input = input("술과 재료를 입력 (예: '1 100') 또는 'q'로 종료: ")
        if user_input.lower() == 'q':
            break
            
        liquor, ingredient = user_input.split()
        liquor_idx = lid_to_idx[int(liquor)]
        ingredient_idx = iid_to_idx[int(ingredient)]
        
        score = predict(liquor_idx, ingredient_idx, edges_indexes, edges_weights, edge_type)
        print(f"예측 점수: {score.item():.4f}")
    except KeyboardInterrupt:
        print("\n프로그램을 종료합니다.")
        break
    except Exception as e:
        print(f"오류 발생: {e}")
        print("입력 형식을 확인하세요. 예: '1 100'")
