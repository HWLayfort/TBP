import torch
import torch.nn as nn
import torch.optim as optim
import pandas as pd
import numpy as np
from torch.utils.data import TensorDataset, DataLoader, random_split

# -------------------------------
# 1) DeepONet 정의 (위 예제 그대로)
# -------------------------------
class DeepONet(nn.Module):
    def __init__(self,
                 branch_input_dim,   # 입력 함수 차원 (초기 상태: r,v,a = 27)
                 trunk_input_dim,    # 위치/시간 차원 (t=1)
                 hidden_dim=128,
                 branch_layers=3,
                 trunk_layers=3,
                 out_dim=9,          # 출력: r (9차원)
                 activation=nn.Tanh
                 ):
        super().__init__()
        # Branch net
        layers = []
        in_dim = branch_input_dim
        for _ in range(branch_layers):
            layers += [nn.Linear(in_dim, hidden_dim), activation()]
            in_dim = hidden_dim
        self.branch_net = nn.Sequential(*layers)
        # Trunk net
        layers = []
        in_dim = trunk_input_dim
        for _ in range(trunk_layers):
            layers += [nn.Linear(in_dim, hidden_dim), activation()]
            in_dim = hidden_dim
        self.trunk_net = nn.Sequential(*layers)
        # 최종 합성
        self.fc = nn.Linear(hidden_dim, out_dim)

    def forward(self, branch_input, trunk_input):
        b = self.branch_net(branch_input)  # (B, hidden_dim)
        t = self.trunk_net(trunk_input)    # (B, hidden_dim)
        f = b * t                          # elementwise
        return self.fc(f)                 # (B, out_dim)

# --------------------------------------------------
# 2) 데이터 로드 및 (branch, trunk, target) 생성
# --------------------------------------------------
def load_three_body_data(csv_path, num_conditions=500, time_steps=2000):
    df = pd.read_csv(csv_path, header=None)
    arr = df.to_numpy(dtype=np.float32)           # (500*2000, 27)
    arr = arr.reshape(num_conditions, time_steps, 27)
    # 초기 상태 (각 trajectory마다 첫 타임스텝)
    init = arr[:, 0, :]                           # (500, 27)
    # 시간값 [0, T]
    T = 10.0                                      # 실제 관측 시간에 맞게 수정
    t_lin = np.linspace(0, T, time_steps, dtype=np.float32)  # (2000,)
    # Flatten: 총 500*2000 샘플
    B = num_conditions * time_steps
    # branch 입력: 초기 상태 반복
    branch = np.repeat(init, time_steps, axis=0)           # (B, 27)
    # trunk 입력: 시간값을 조건별로 tile
    trunk = np.tile(t_lin.reshape(1, -1), (num_conditions, 1)).reshape(-1, 1)  # (B,1)
    # target 출력: 위치 r (앞 9열)
    target = arr[:, :, 0:9].reshape(B, 9)                  # (B,9)
    return branch, trunk, target

# ---------------------------------
# 3) Dataset / DataLoader 준비
# ---------------------------------
def prepare_dataloaders(branch, trunk, target, batch_size=1024):
    tensor_branch = torch.from_numpy(branch)
    tensor_trunk  = torch.from_numpy(trunk)
    tensor_target = torch.from_numpy(target)
    dataset = TensorDataset(tensor_branch, tensor_trunk, tensor_target)
    N = len(dataset)
    n_train = int(0.8 * N)
    n_val   = int(0.1 * N)
    n_test  = N - n_train - n_val
    train_ds, val_ds, test_ds = random_split(dataset, [n_train, n_val, n_test])
    train_loader = DataLoader(train_ds, batch_size=batch_size, shuffle=True)
    val_loader   = DataLoader(val_ds,   batch_size=batch_size, shuffle=False)
    test_loader  = DataLoader(test_ds,  batch_size=batch_size, shuffle=False)
    return train_loader, val_loader, test_loader

# ----------------------------
# 4) 학습 / 검증 / 테스트 루프
# ----------------------------
def train_deeponet(csv_path="data.csv"):
    # 1) 데이터 준비
    branch, trunk, target = load_three_body_data(csv_path)
    train_loader, val_loader, test_loader = prepare_dataloaders(branch, trunk, target)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = DeepONet(branch_input_dim=27, trunk_input_dim=1,
                     hidden_dim=128, branch_layers=4, trunk_layers=4,
                     out_dim=9).to(device)
    optimizer = optim.Adam(model.parameters(), lr=1e-3)
    loss_fn = nn.MSELoss()

    num_epochs = 200
    best_val = float("inf")

    for epoch in range(1, num_epochs+1):
        # — train
        model.train()
        train_loss = 0.0
        for b, t, y in train_loader:
            b, t, y = b.to(device), t.to(device), y.to(device)
            optimizer.zero_grad()
            y_pred = model(b, t)
            loss = loss_fn(y_pred, y)
            loss.backward()
            optimizer.step()
            train_loss += loss.item() * b.size(0)
        train_loss /= len(train_loader.dataset)

        # — valid
        model.eval()
        val_loss = 0.0
        with torch.no_grad():
            for b, t, y in val_loader:
                b, t, y = b.to(device), t.to(device), y.to(device)
                y_pred = model(b, t)
                val_loss += loss_fn(y_pred, y).item() * b.size(0)
        val_loss /= len(val_loader.dataset)

        if val_loss < best_val:
            best_val = val_loss
            torch.save(model.state_dict(), "deeponet_threebody.pth")

        if epoch % 10 == 0 or epoch == 1:
            print(f"[Epoch {epoch:03d}] Train MSE: {train_loss:.4e}, Val MSE: {val_loss:.4e}")

    # — test
    model.load_state_dict(torch.load("deeponet_threebody.pth"))
    model.eval()
    test_loss = 0.0
    with torch.no_grad():
        for b, t, y in test_loader:
            b, t, y = b.to(device), t.to(device), y.to(device)
            y_pred = model(b, t)
            test_loss += loss_fn(y_pred, y).item() * b.size(0)
    test_loss /= len(test_loader.dataset)
    print(f">>> Test MSE: {test_loss:.4e}")
    print("Saved best model → deeponet_threebody.pth")

if __name__ == "__main__":
    train_deeponet("data.csv")
