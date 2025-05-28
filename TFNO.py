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

class TemporalFourierEmbedding(nn.Module):
    def __init__(self, n_steps, embed_dim=32, scale=10.0):
        super().__init__()
        self.B = nn.Parameter(torch.randn((1, embed_dim)) * scale, requires_grad=False)
        t_lin = torch.linspace(0, 1, n_steps).unsqueeze(1)  # (n_steps, 1)
        x_proj = 2.0 * np.pi * t_lin @ self.B  # (n_steps, embed_dim)
        self.register_buffer('t_embedding', torch.cat([torch.sin(x_proj), torch.cos(x_proj)], dim=1))  # (n_steps, 2*embed_dim)
    def forward(self, batch_size):
        # return (batch, n_steps, 2*embed_dim)
        n_steps, d = self.t_embedding.shape
        return self.t_embedding[None, :, :].expand(batch_size, n_steps, d)


class SpectralConv1dTFNO(nn.Module):
    def __init__(self, in_channels, out_channels, modes1):
        super().__init__()
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.modes1 = modes1
        self.weights = nn.Parameter(torch.randn(in_channels, out_channels, self.modes1, dtype=torch.cfloat) * 0.02)

    def compl_mul1d(self, input, weights):
        # input: (batch, in_c, modes1)
        # weights: (in_c, out_c, modes1)
        return torch.einsum("bix, iox -> box", input, weights)

    def forward(self, x):
        batchsize = x.shape[0]
        x_ft = torch.fft.rfft(x, dim=-1)
        out_ft = torch.zeros(batchsize, self.out_channels, x_ft.shape[-1], device=x.device, dtype=torch.cfloat)
        out_ft[:, :, :self.modes1] = self.compl_mul1d(x_ft[:, :, :self.modes1], self.weights)
        x = torch.fft.irfft(out_ft, n=x.size(-1), dim=-1)
        return x

class ChannelMixing(nn.Module):
    def __init__(self, width, hidden_mul=2):
        super().__init__()
        self.layer = nn.Sequential(
            nn.Conv1d(width, width * hidden_mul, 1),
            nn.GELU(),
            nn.Conv1d(width * hidden_mul, width, 1)
        )
    def forward(self, x):
        return self.layer(x)


class TFNO(nn.Module):
    def __init__(self, t_steps, input_dim=22, embed_dim=32, width=64, modes=32, depth=4, output_dim=9):
        super().__init__()
        self.t_steps = t_steps
        self.input_dim = input_dim
        self.temb = TemporalFourierEmbedding(t_steps, embed_dim)
        self.fc0 = nn.Linear(input_dim + 2*embed_dim, width)
        self.spectral_layers = nn.ModuleList([SpectralConv1dTFNO(width, width, modes) for _ in range(depth)])
        self.channel_mixings = nn.ModuleList([ChannelMixing(width) for _ in range(depth)])
        self.norms = nn.ModuleList([nn.LayerNorm(width) for _ in range(depth)])
        self.fc1 = nn.Linear(width, 128)
        self.fc2 = nn.Linear(128, output_dim)
        self.activation = nn.GELU()
    def forward(self, x):
        # x: (batch, input_dim, N)
        batch_size, input_dim, n_steps = x.shape
        # 시간 축 임베딩: (batch, n_steps, 2*embed_dim)
        t_emb = self.temb(batch_size)  # (batch, n_steps, 2*embed_dim)
        # 입력 채널 + 시간 임베딩 concat
        x_in = torch.cat([x.permute(0,2,1), t_emb], dim=-1)  # (batch, n_steps, input_dim+2*embed_dim)
        x = self.fc0(x_in)                                  # (batch, n_steps, width)
        x = x.permute(0,2,1)                                # (batch, width, n_steps)
        for spec, cmix, norm in zip(self.spectral_layers, self.channel_mixings, self.norms):
            x1 = spec(x)
            x2 = cmix(x)
            x = self.activation(x1 + x2)
            # LayerNorm over channel dimension: (batch, width, n_steps) -> (batch, n_steps, width)
            x = norm(x.permute(0,2,1)).permute(0,2,1)
        x = x.permute(0,2,1)              # (batch, n_steps, width)
        x = self.activation(self.fc1(x))
        x = self.fc2(x)                   # (batch, n_steps, output_dim)
        return x.permute(0,2,1)           # (batch, output_dim, n_steps)


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
            N = t.shape[0]
            r0 = df.iloc[0, 1:10].values      # (9,)
            v0 = df.iloc[0, 10:19].values     # (9,)
            m = masses
            r = df.iloc[:, 1:10].values.T     # (9, N)
            r0_mat = np.tile(r0[:, None], (1, N))
            v0_mat = np.tile(v0[:, None], (1, N))
            m_mat  = np.tile(m[:, None], (1, N))
            t_mat  = t[None, :]
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

def train_tfno(
    model, train_loader, val_loader, num_epochs=100, lr=1e-3, device='cuda',
    model_save_path='tfno.pth', log_path='tfno.log', early_stop_patience=20
):
    print(f"Training TFNO on device: {device}")
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

def test_tfno(model, test_loader, device='cuda', log_path='tfno.log'):
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

def run_tfno_pipeline():
    data_dir = os.path.join(os.path.dirname(__file__), "data", "output")
    file_list = [os.path.join(data_dir, fname) for fname in os.listdir(data_dir) if fname.endswith('.csv')]
    dataset = ThreeBodyFNOTimeSeriesDataset(file_list)
    print(f"Loaded dataset with {len(dataset)} samples.")
    total = len(dataset)
    n_train = int(0.7 * total)
    n_val = int(0.15 * total)
    n_test = total - n_train - n_val
    train_ds, val_ds, test_ds = random_split(dataset, [n_train, n_val, n_test], generator=torch.Generator().manual_seed(42))
    train_loader = DataLoader(train_ds, batch_size=8, shuffle=True)
    val_loader = DataLoader(val_ds, batch_size=8, shuffle=False)
    test_loader = DataLoader(test_ds, batch_size=8, shuffle=False)

    # n_steps는 시계열 길이(파일마다 동일해야 함, 예시: 2000)
    n_steps = dataset[0][0].shape[1]
    print(f"Number of time steps: {n_steps}")
    model = TFNO(
        t_steps=n_steps, input_dim=22, embed_dim=32, width=64, modes=32,
        depth=4, output_dim=9
    )
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    train_tfno(
        model, train_loader, val_loader,
        num_epochs=100, lr=1e-3, device=device, model_save_path='tfno.pth', early_stop_patience=20
    )
    test_tfno(model, test_loader, device=device)
    
if __name__ == "__main__":
    run_tfno_pipeline()