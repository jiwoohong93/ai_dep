import torch
import torch.nn as nn
import torch.nn.functional as F

class TwoVectorClassifier(nn.Module):
    def __init__(self, input_dim=768, reduced_dim=256, hidden_dim=128):
        super(TwoVectorClassifier, self).__init__()
        
        # 입력 벡터 차원 축소 (각각 독립적으로 처리)
        self.fc1_a = nn.Linear(input_dim * 3, reduced_dim)  # 첫 번째 벡터 처리
        self.fc1_b = nn.Linear(input_dim, reduced_dim)  # 두 번째 벡터 처리
        
        # 결합된 벡터를 위한 추가 레이어
        self.fc2 = nn.Linear(reduced_dim * 2, hidden_dim)  # Concat 이후 처리
        self.fc3 = nn.Linear(hidden_dim, 1)  # 최종 출력 레이어
        
        # 활성화 함수
        self.relu = nn.ReLU()
        self.sigmoid = nn.Sigmoid()
        
    def forward(self, x1, x2):
        # 입력 벡터 각각 처리 (차원 축소)
        x1 = self.relu(self.fc1_a(x1))  # 첫 번째 벡터
        x2 = self.relu(self.fc1_b(x2))  # 두 번째 벡터
        
        # 벡터 결합 (Concat)
        x = torch.cat((x1, x2), dim=1)
        
        # 분류를 위한 추가 레이어
        x = self.relu(self.fc2(x))  # 중간 레이어
        x = self.fc3(x)  # 최종 출력 레이어

        return x
