import torch
from torch.utils.data import DataLoader
import torch.nn as nn
import torch.optim as optim
from tqdm import tqdm
import pandas as pd
from sklearn.model_selection import train_test_split
import numpy as np
import random
import os

from dataset import map_graph_nodes, edges_index, BPRDataset
from plot import test_visualization, all_score_visualization
from models import NeuralCF

# 강제로 CUDA 사용 설정
FORCE_CUDA = True

def set_seed(seed=123):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed) 
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

class EarlyStopping:
    def __init__(self, patience=10, delta=0):
        self.patience = patience
        self.delta = delta
        self.best_score = None
        self.counter = 0
        self.early_stop = False

    def __call__(self, val_loss):
        if self.best_score is None:
            self.best_score = val_loss
        elif val_loss > self.best_score - self.delta:
            self.counter += 1
            if self.counter >= self.patience:
                self.early_stop = True
        else:
            self.best_score = val_loss
            self.counter = 0

def bpr_loss(pos_scores, neg_scores):
    return -torch.mean(torch.log(torch.sigmoid(pos_scores - neg_scores) + 1e-10))

def train_model(model, train_loader, val_loader, edges_index, edges_weights, edges_type, num_epochs=10, lr=0.0002, weight_decay=1e-5):
    # CUDA 강제 설정 확인
    if FORCE_CUDA:
        device = torch.device('cuda')
        print("FORCE_CUDA is enabled. Using GPU for training.")
    else:
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    
    # CUDA 가용성 확인 및 디버그 정보 출력
    print(f"CUDA available: {torch.cuda.is_available()}")
    print(f"CUDA device count: {torch.cuda.device_count()}")
    
    # GPU 정보 출력
    if torch.cuda.is_available():
        print(f"Using GPU: {torch.cuda.get_device_name(0)}")
        print(f"GPU Memory: {torch.cuda.get_device_properties(0).total_memory / 1e9:.2f} GB")
    else:
        print("No GPU available, using CPU instead")
    
    # 모델을 디바이스로 이동
    print(f"Moving model to {device}")
    model = model.to(device)
    
    # 데이터를 디바이스로 이동
    print(f"Moving graph data to {device}")
    edges_index = edges_index.to(device)
    edges_weights = edges_weights.to(device)
    edges_type = edges_type.to(device).long()
    
    # 학습 모드 설정
    model.train()

    #criterion = bpr_loss()
    optimizer = optim.Adam(model.parameters(), lr=lr, weight_decay=weight_decay)
    early_stopping = EarlyStopping(patience=10, delta=0.001)

    best_model = None
    best_val_loss = float('inf')

    topk = 5

    print(f"Training on {device}")
    for epoch in range(num_epochs):
        total_loss = 0
        correct = 0
        total = 0

        for user, pos, neg in tqdm(train_loader, desc=f"Epoch {epoch+1}/{num_epochs}"):
            user = user.long()   
            pos = pos.long()   
            neg = neg.long()

            # 명시적으로 디바이스로 이동
            user, pos, neg = user.to(device), pos.to(device), neg.to(device)

            optimizer.zero_grad()

            pos_output = model(user, pos, edges_index, edges_type, edges_weights)

            num_neg_candidates = 10
            neg_candidates = torch.randint(0, 6498, (user.size(0), num_neg_candidates), device=device)

            user_expand = user.unsqueeze(1).expand_as(neg_candidates)
            user_flat = user_expand.reshape(-1)
            neg_flat = neg_candidates.reshape(-1)

            neg_scores = model(user_flat, neg_flat, edges_index, edges_type, edges_weights)
            neg_scores = neg_scores.view(user.size(0), num_neg_candidates)

            hard_neg_scores, hard_neg_indices = torch.topk(neg_scores, k=topk, dim=1)

            random_idx = torch.randint(0, topk, (user.size(0),), device=device)
            hard_neg = neg_candidates[torch.arange(user.size(0)), hard_neg_indices[torch.arange(user.size(0)), random_idx]]

            neg_output = model(user, hard_neg, edges_index, edges_type, edges_weights)
            loss = bpr_loss(pos_output, neg_output)

            loss.backward()
            optimizer.step()

            total_loss += loss.item() * pos.size(0)
            # Calculate ranking accuracy for BPR
            correct += (pos_output > neg_output).sum().item()  # Count correct rankings
            total += pos.size(0)  # Total number of positive samples

        avg_loss = total_loss / total
        acc = correct / total
        print(f"[Epoch {epoch+1}] Loss: {avg_loss:.4f} | Accuracy: {acc:.4f}")
        
        # Validation
        model.eval()
        val_loss = 0
        val_correct = 0
        val_total = 0
        
        with torch.no_grad():
            for user, pos, neg in val_loader:
                user = user.long()
                pos = pos.long()
                neg = neg.long()

                user, pos, neg = user.to(device), pos.to(device), neg.to(device)

                pos_output = model(user, pos, edges_index, edges_type, edges_weights)
                neg_output = model(user, neg, edges_index, edges_type, edges_weights)
                loss = bpr_loss(pos_output, neg_output)
                
                val_loss += loss.item() * pos.size(0)
                # Calculate ranking accuracy for BPR
                val_correct += (pos_output > neg_output).sum().item()  # Count correct rankings
                val_total += pos.size(0)  # Total number of positive samples
        
        avg_val_loss = val_loss / val_total
        val_acc = val_correct / val_total
        
        print(f"[Validation] Loss: {avg_val_loss:.4f} | Accuracy: {val_acc:.4f}")
        
        # 체크포인트 디렉토리 존재 확인 및 생성
        os.makedirs("./model/checkpoint", exist_ok=True)
        torch.save(model.state_dict(), f"./model/checkpoint/epoch_{epoch}.pth")
        
        if avg_val_loss < best_val_loss:
            best_val_loss = avg_val_loss
            best_model = model.state_dict()
            print(f"Best model saved at epoch {epoch+1} with validation loss {best_val_loss:.4f}")
            
        # Check Early Stopping
        early_stopping(avg_val_loss)
        if early_stopping.early_stop:
            print("Early stopping triggered.")
            torch.save(best_model, "./model/checkpoint/best_model.pth")
            break

if __name__ == "__main__":
    # CUDA 초기 설정 확인
    print(f"Initial CUDA check - Available: {torch.cuda.is_available()}")
    if torch.cuda.is_available():
        print(f"Initial GPU name: {torch.cuda.get_device_name(0)}")
        # GPU 메모리 사용량 확인
        print(f"Initial GPU memory: {torch.cuda.memory_allocated(0) / 1e9:.2f} GB used, {torch.cuda.memory_reserved(0) / 1e9:.2f} GB reserved")

    print("Loading data...")
    mapping = map_graph_nodes()
    
    lid_to_idx = mapping['liquor']
    iid_to_idx = mapping['ingredient']

    print("Loading graph data...")
    
    edge_type_map ={
        'liqr-ingr': 0,
        'ingr-ingr': 1,
        'liqr-liqr': 1,
        'ingr-fcomp': 2,
        'ingr-dcomp': 2
    }
    
    edges_indexes, edges_weights, edges_type = edges_index(edge_type_map)
    
    print("Loading dataset...")
    positive_pairs = pd.read_csv("./liquor_good_ingredients.csv")
    positive_pairs = positive_pairs[['liquor_id', 'ingredient_id']]

    negative_pairs = pd.read_csv("./liquor_bad_ingredients.csv")
    negative_pairs = negative_pairs[['liquor_id', 'ingredient_id']]

    print("Mapping liquor and ingredient IDs to indices...")
    positive_pairs['liquor_id'] = positive_pairs['liquor_id'].map(lid_to_idx)
    positive_pairs['ingredient_id'] = positive_pairs['ingredient_id'].map(iid_to_idx)

    negative_pairs['liquor_id'] = negative_pairs['liquor_id'].map(lid_to_idx)
    negative_pairs['ingredient_id'] = negative_pairs['ingredient_id'].map(iid_to_idx)
    
    print("Creating dataset...")
    
    train_val_pairs, test_pairs = train_test_split(positive_pairs, test_size=0.2, random_state=42)
    train_pairs, val_pairs = train_test_split(train_val_pairs, test_size=0.2, random_state=42)
    
    train_dataset = BPRDataset(positive_pairs=train_pairs, hard_negatives=negative_pairs, num_users=155, num_items=6498)
    val_dataset = BPRDataset(positive_pairs=val_pairs, hard_negatives=negative_pairs, num_users=155, num_items=6498)
    test_dataset = BPRDataset(positive_pairs=test_pairs, hard_negatives=negative_pairs, num_users=155, num_items=6498)
    
    # GPU 메모리 최적화 설정을 위한 DataLoader 파라미터 추가
    # Colab 환경에서는 num_workers=0이 더 안정적일 수 있음
    num_workers = 0 
    train_loader = DataLoader(train_dataset, batch_size=32, shuffle=True, pin_memory=True, num_workers=num_workers)
    val_loader = DataLoader(val_dataset, batch_size=32, shuffle=False, pin_memory=True, num_workers=num_workers)
    test_loader = DataLoader(test_dataset, batch_size=32, shuffle=False, pin_memory=True, num_workers=num_workers)

    print("Creating model...")
    model = NeuralCF(num_users=155, num_items=6498, emb_size=128)

    print("Training model...")
    train_model(model=model, train_loader=train_loader, val_loader=val_loader, edges_type=edges_type, edges_index=edges_indexes, edges_weights=edges_weights, num_epochs=200)

    # checkpoint 디렉토리 존재 확인 
    os.makedirs("./model/checkpoint", exist_ok=True)

    # 최종 모델 로드 및 평가  
    try:
        # GPU가 있으면 GPU에 로드, 아니면 CPU에 로드
        if FORCE_CUDA or torch.cuda.is_available():
            model.load_state_dict(torch.load("./model/checkpoint/best_model.pth"))
        else:
            model.load_state_dict(torch.load("./model/checkpoint/best_model.pth", map_location=torch.device('cpu')))
        test_visualization(model, test_loader, edges_indexes, edges_weights, edges_type)
    except Exception as e:
        print(f"Error loading model or visualizing results: {e}")
