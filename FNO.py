import os
import numpy as np
import pandas as pd
import torch
from torch.utils.data import Dataset, DataLoader, random_split
import torch.nn as nn
import torch.optim as optim
from tqdm import tqdm

# =======================
# SpectralConv1d/FNO1d 정의 (논문 구조)
# =======================

class SpectralConv1d(nn.Module):
    def __init__(self, in_channels, out_channels, modes1):
        super().__init__()
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.modes1 = modes1
        self.scale = 1 / (in_channels * out_channels)
        self.weights = nn.Parameter(self.scale * torch.randn(in_channels, out_channels, self.modes1, dtype=torch.cfloat))

    def compl_mul1d(self, input, weights):
        # input: (batch, in_c, modes1)
        # weights: (in_c, out_c, modes1)
        # output: (batch, out_c, modes1)
        return torch.einsum("bix, iox -> box", input, weights)

    def forward(self, x):
        batchsize = x.shape[0]
        x_ft = torch.fft.rfft(x, dim=-1)  # (batch, in_c, modes1_full)
        out_ft = torch.zeros(batchsize, self.out_channels, x_ft.shape[-1], device=x.device, dtype=torch.cfloat)
        out_ft[:, :, :self.modes1] = self.compl_mul1d(x_ft[:, :, :self.modes1], self.weights)
        x = torch.fft.irfft(out_ft, n=x.size(-1), dim=-1)
        return x


class FNO1d(nn.Module):
    def __init__(self, modes, width, input_dim=22, output_dim=9, depth=4):
        super().__init__()
        self.input_dim = input_dim
        self.width = width
        self.fc0 = nn.Linear(input_dim, width)  # input channel is (r0, v0, m, t)
        self.convs = nn.ModuleList([SpectralConv1d(width, width, modes) for _ in range(depth)])
        self.ws = nn.ModuleList([nn.Conv1d(width, width, 1) for _ in range(depth)])
        self.fc1 = nn.Linear(width, 128)
        self.fc2 = nn.Linear(128, output_dim)
        self.activation = nn.GELU()
    def forward(self, x):
        # x: (batch, input_dim, N)
        length = x.size(-1)
        x = x.permute(0, 2, 1)  # (batch, N, input_dim)
        x = self.fc0(x)         # (batch, N, width)
        x = x.permute(0, 2, 1)  # (batch, width, N)
        for conv, w in zip(self.convs, self.ws):
            x1 = conv(x)
            x2 = w(x)
            x = self.activation(x1 + x2)
        x = x.permute(0, 2, 1)  # (batch, N, width)
        x = self.activation(self.fc1(x))
        x = self.fc2(x)         # (batch, N, output_dim)
        return x.permute(0, 2, 1)  # (batch, output_dim, N)

# =======================
# Three-body FNO 데이터셋
# =======================

class ThreeBodyFNOTimeSeriesDataset(Dataset):
    def __init__(self, filelist):
        self.data = []
        for fname in filelist:
            filename = os.path.basename(fname)
            base = filename.split('_')[0]
            mass_str = base.replace('m', '')
            masses = np.array([float(x) for x in mass_str.split('-')])
            df = pd.read_csv(fname, header=None)
            t = df.iloc[:, 0].values          # (N,)
            N = t.shape[0]                    # 각 파일별 시계열 길이
            r0 = df.iloc[0, 1:10].values      # (9,)
            v0 = df.iloc[0, 10:19].values     # (9,)
            m = masses
            r = df.iloc[:, 1:10].values.T     # (9, N)
            # 입력 준비 (broadcast)
            r0_mat = np.tile(r0[:, None], (1, N))     # (9, N)
            v0_mat = np.tile(v0[:, None], (1, N))     # (9, N)
            m_mat  = np.tile(m[:, None], (1, N))      # (3, N)
            t_mat  = t[None, :]                       # (1, N)
            x = np.concatenate([r0_mat, v0_mat, m_mat, t_mat], axis=0)   # (22, N)
            self.data.append((x, r))
    def __len__(self):
        return len(self.data)
    def __getitem__(self, idx):
        x, y = self.data[idx]
        return (
            torch.tensor(x, dtype=torch.float32),     # (22, N)
            torch.tensor(y, dtype=torch.float32)      # (9, N)
        )

# =======================
# FNO 학습 및 평가
# =======================

def train_fno(
    model, train_loader, val_loader, num_epochs=100, lr=1e-3, device='cuda',
    model_save_path='fno_best.pth', log_path='fno.log', early_stop_patience=20
):
    print(f"Training FNO on device: {device}")
    model = model.to(device)
    optimizer = optim.Adam(model.parameters(), lr=lr)
    loss_fn = nn.MSELoss()
    best_val = float('inf')
    patience = 0
    if log_path is not None:
        with open(log_path, "w") as f:
            f.write("epoch,train_loss,val_loss\n")
    for epoch in range(num_epochs):
        model.train()
        total_loss = 0.0
        n = 0
        for xb, yb in tqdm(train_loader, desc=f"Train {epoch+1}", leave=False):
            xb = xb.to(device)
            yb = yb.to(device)
            optimizer.zero_grad()
            y_pred = model(xb)
            loss = loss_fn(y_pred, yb)
            loss.backward()
            optimizer.step()
            total_loss += loss.item() * xb.size(0)
            n += xb.size(0)
        avg_loss = total_loss / n
        # Validation
        model.eval()
        val_loss = 0.0
        n_val = 0
        with torch.no_grad():
            for xb, yb in tqdm(val_loader, desc=f"Val {epoch+1}", leave=False):
                xb = xb.to(device)
                yb = yb.to(device)
                y_pred = model(xb)
                vloss = loss_fn(y_pred, yb)
                val_loss += vloss.item() * xb.size(0)
                n_val += xb.size(0)
        avg_val_loss = val_loss / n_val
        if log_path is not None:
            with open(log_path, "a") as f:
                f.write(f"{epoch},{avg_loss},{avg_val_loss}\n")
        if avg_val_loss < best_val:
            best_val = avg_val_loss
            patience = 0
            torch.save(model.state_dict(), model_save_path)
        else:
            patience += 1
            if patience >= early_stop_patience:
                print(f"Early stopping at epoch {epoch+1}: no improvement in {early_stop_patience} epochs.")
                break
        if epoch % 10 == 0:
            print(f"[Epoch {epoch+1}] Train Loss: {avg_loss:.6f}, Val Loss: {avg_val_loss:.6f}")
    print(f"Best validation loss: {best_val:.6f}")

def test_fno(model, test_loader, device='cuda', log_path='fno.log'):
    model.eval()
    loss_fn = nn.MSELoss()
    total_loss = 0.0
    n = 0
    with torch.no_grad():
        for xb, yb in tqdm(test_loader, desc="Test"):
            xb = xb.to(device)
            yb = yb.to(device)
            y_pred = model(xb)
            loss = loss_fn(y_pred, yb)
            total_loss += loss.item() * xb.size(0)
            n += xb.size(0)
    avg_loss = total_loss / n
    print(f"[Test] Loss: {avg_loss:.6f}")
    if log_path is not None:
        with open(log_path, "a") as f:
            f.write(f"test,{avg_loss}\n")
    return avg_loss

# =======================
# 전체 파이프라인
# =======================

def run_fno_pipeline():
    data_dir = os.path.join(os.path.dirname(__file__), "data", "output")
    file_list = [os.path.join(data_dir, fname) for fname in os.listdir(data_dir) if fname.endswith('.csv')]
    dataset = ThreeBodyFNOTimeSeriesDataset(file_list)
    print(f"Loaded dataset with {len(dataset)} samples.")
    # 데이터셋 분할
    total = len(dataset)
    n_train = int(0.7 * total)
    n_val = int(0.15 * total)
    n_test = total - n_train - n_val
    train_ds, val_ds, test_ds = random_split(dataset, [n_train, n_val, n_test], generator=torch.Generator().manual_seed(42))
    train_loader = DataLoader(train_ds, batch_size=8, shuffle=True)
    val_loader = DataLoader(val_ds, batch_size=8, shuffle=False)
    test_loader = DataLoader(test_ds, batch_size=8, shuffle=False)

    model = FNO1d(modes=32, width=64, input_dim=22, output_dim=9, depth=4)
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    train_fno(
        model, train_loader, val_loader,
        num_epochs=100, lr=1e-3, device=device, model_save_path='fno.pth', early_stop_patience=20
    )

    # 테스트 평가
    test_fno(model, test_loader, device=device)

if __name__ == "__main__":
    run_fno_pipeline()