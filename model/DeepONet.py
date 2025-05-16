import torch
import torch.nn as nn
import torch.optim as optim
import pandas as pd
import numpy as np
from torch.utils.data import TensorDataset, DataLoader, random_split
from tqdm import tqdm
import random
import os

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
def set_seed(seed=42):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)

def load_three_body_data(csv_path, num_conditions=500, time_steps=2000, T=10.0):
    df = pd.read_csv(csv_path, header=None)
    arr = df.to_numpy(dtype=np.float32)           # (500*2000, 27)
    arr = arr.reshape(num_conditions, time_steps, 27)
    # 초기 상태 (각 trajectory마다 첫 타임스텝)
    init = arr[:, 0, :]                           # (500, 27)
    # 시간값 [0, T]
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
def prepare_dataloaders(branch, trunk, target, batch_size=1024, seed=42):
    tensor_branch = torch.from_numpy(branch)
    tensor_trunk  = torch.from_numpy(trunk)
    tensor_target = torch.from_numpy(target)
    dataset = TensorDataset(tensor_branch, tensor_trunk, tensor_target)
    N = len(dataset)
    n_train = int(0.8 * N)
    n_val   = int(0.1 * N)
    n_test  = N - n_train - n_val
    train_ds, val_ds, test_ds = random_split(dataset, [n_train, n_val, n_test], generator=torch.Generator().manual_seed(seed))
    train_loader = DataLoader(train_ds, batch_size=batch_size, shuffle=True)
    val_loader   = DataLoader(val_ds,   batch_size=batch_size, shuffle=False)
    test_loader  = DataLoader(test_ds,  batch_size=batch_size, shuffle=False)
    return train_loader, val_loader, test_loader

# ----------------------------
# 4) 학습 / 검증 / 테스트 루프
# ----------------------------
def train_deeponet(
    csv_path="data.csv",
    batch_size=1024,
    hidden_dim=128,
    branch_layers=4,
    trunk_layers=4,
    out_dim=9,
    activation=nn.Tanh,
    num_conditions=500,
    time_steps=2000,
    T=10.0,
    lr=1e-3,
    num_epochs=200,
    patience=20,
    seed=42,
    model_save_path="deeponet_threebody.pth",
    log_path="deeponet_train_log.csv"
):
    set_seed(seed)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device: {device}")

    # 1) 데이터 준비
    branch, trunk, target = load_three_body_data(csv_path, num_conditions, time_steps, T)
    train_loader, val_loader, test_loader = prepare_dataloaders(branch, trunk, target, batch_size, seed=seed)

    model = DeepONet(branch_input_dim=branch.shape[1], trunk_input_dim=trunk.shape[1],
                     hidden_dim=hidden_dim, branch_layers=branch_layers, trunk_layers=trunk_layers,
                     out_dim=out_dim, activation=activation).to(device)
    optimizer = optim.Adam(model.parameters(), lr=lr)
    loss_fn = nn.MSELoss()

    best_val = float("inf")
    epochs_no_improve = 0

    # 로그 파일 초기화
    if log_path is not None:
        with open(log_path, "w") as f:
            f.write("epoch,train_mse,val_mse\n")

    for epoch in range(1, num_epochs+1):
        # — train
        model.train()
        total_train = 0.0
        for b, t, y in tqdm(train_loader, desc=f"Epoch {epoch} [Train]", leave=False):
            b, t, y = b.to(device), t.to(device), y.to(device)
            optimizer.zero_grad()
            y_pred = model(b, t)
            loss = loss_fn(y_pred, y)
            loss.backward()
            optimizer.step()
            total_train += loss.item() * b.size(0)
        train_loss = total_train / len(train_loader.dataset)

        # — valid
        model.eval()
        total_val = 0.0
        with torch.no_grad():
            for b, t, y in tqdm(val_loader, desc=f"Epoch {epoch} [Val]", leave=False):
                b, t, y = b.to(device), t.to(device), y.to(device)
                y_pred = model(b, t)
                total_val += loss_fn(y_pred, y).item() * b.size(0)
        val_loss = total_val / len(val_loader.dataset)

        # 로그 기록
        if log_path is not None:
            with open(log_path, "a") as f:
                f.write(f"{epoch},{train_loss},{val_loss}\n")

        # Early stopping
        if val_loss < best_val:
            best_val = val_loss
            epochs_no_improve = 0
            torch.save(model.state_dict(), model_save_path)
        else:
            epochs_no_improve += 1
            if patience > 0 and epochs_no_improve >= patience:
                print(f"Early stopping at epoch {epoch}")
                break

        if epoch % 10 == 0 or epoch == 1:
            print(f"[Epoch {epoch:03d}] Train MSE: {train_loss:.4e}, Val MSE: {val_loss:.4e}")

    # — test
    model.load_state_dict(torch.load(model_save_path))
    model.eval()
    total_test = 0.0
    with torch.no_grad():
        for b, t, y in tqdm(test_loader, desc="Test", leave=False):
            b, t, y = b.to(device), t.to(device), y.to(device)
            y_pred = model(b, t)
            total_test += loss_fn(y_pred, y).item() * b.size(0)
    test_loss = total_test / len(test_loader.dataset)
    print(f">>> Test MSE: {test_loss:.4e}")
    print(f"Saved best model → {model_save_path}")

if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--csv_path", type=str, default="data.csv")
    parser.add_argument("--batch_size", type=int, default=1024)
    parser.add_argument("--hidden_dim", type=int, default=128)
    parser.add_argument("--branch_layers", type=int, default=4)
    parser.add_argument("--trunk_layers", type=int, default=4)
    parser.add_argument("--out_dim", type=int, default=9)
    parser.add_argument("--num_conditions", type=int, default=500)
    parser.add_argument("--time_steps", type=int, default=2000)
    parser.add_argument("--T", type=float, default=10.0)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--num_epochs", type=int, default=200)
    parser.add_argument("--patience", type=int, default=20)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--model_save_path", type=str, default="deeponet_threebody.pth")
    parser.add_argument("--log_path", type=str, default="deeponet_train_log.csv")
    args = parser.parse_args()

    train_deeponet(
        csv_path=args.csv_path,
        batch_size=args.batch_size,
        hidden_dim=args.hidden_dim,
        branch_layers=args.branch_layers,
        trunk_layers=args.trunk_layers,
        out_dim=args.out_dim,
        num_conditions=args.num_conditions,
        time_steps=args.time_steps,
        T=args.T,
        lr=args.lr,
        num_epochs=args.num_epochs,
        patience=args.patience,
        seed=args.seed,
        model_save_path=args.model_save_path,
        log_path=args.log_path
    )
