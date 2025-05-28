import os
import numpy as np
import pandas as pd
import torch
from torch.utils.data import Dataset, DataLoader, random_split
import torch.nn as nn
import torch.optim as optim
from tqdm import tqdm

class DeepONet(nn.Module):
    def __init__(self, branch_dim=21, trunk_dim=1, hidden_branch=128, hidden_trunk=128, p=128, output_dim=9, num_branch_layers=4, num_trunk_layers=4):
        super().__init__()
        # Branch: (r0(9), v0(9), m(3)) → latent p
        layers = [nn.Linear(branch_dim, hidden_branch), nn.ReLU()]
        for _ in range(num_branch_layers-1):
            layers += [nn.Linear(hidden_branch, hidden_branch), nn.ReLU()]
        layers += [nn.Linear(hidden_branch, p)]
        self.branch_net = nn.Sequential(*layers)
        # Trunk: (t) → latent p
        layers = [nn.Linear(trunk_dim, hidden_trunk), nn.ReLU()]
        for _ in range(num_trunk_layers-1):
            layers += [nn.Linear(hidden_trunk, hidden_trunk), nn.ReLU()]
        layers += [nn.Linear(hidden_trunk, p)]
        self.trunk_net = nn.Sequential(*layers)
        # output map: (batch, p) → (batch, output_dim)
        self.output_map = nn.Linear(p, output_dim)
    def forward(self, branch_input, trunk_input):
        # branch_input: (batch, branch_dim)
        # trunk_input: (batch, N, trunk_dim)
        #   - N: number of eval points (time grid)
        B, N, d_trunk = trunk_input.shape
        branch_feat = self.branch_net(branch_input)          # (batch, p)
        trunk_feat  = self.trunk_net(trunk_input.view(-1, d_trunk)).view(B, N, -1) # (batch, N, p)
        # Elementwise product and sum over p (inner product)
        # (batch, N, p) * (batch, 1, p)
        y = (trunk_feat * branch_feat.unsqueeze(1)).sum(-1)  # (batch, N)
        # 확장: output_dim=9개 각기 다른 DeepONet을 두거나, FC 추가하여 다중출력 확장
        # 여기서는 FC를 씌워서 9차원 출력
        out = self.output_map(trunk_feat * branch_feat.unsqueeze(1)) # (batch, N, output_dim)
        return out.permute(0, 2, 1)  # (batch, output_dim, N)

class ThreeBodyDeepONetTimeSeriesDataset(torch.utils.data.Dataset):
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
            y = df.iloc[:, 1:10].values.T     # (9, N)
            branch_in = np.concatenate([r0, v0, m])   # (21,)
            trunk_in = t[:, None]                          # (N, 1)
            self.data.append((branch_in, trunk_in, y))
    def __len__(self):
        return len(self.data)
    def __getitem__(self, idx):
        branch_in, trunk_in, y = self.data[idx]
        return (
            torch.tensor(branch_in, dtype=torch.float32),      # (21,)
            torch.tensor(trunk_in, dtype=torch.float32),       # (N, 1)
            torch.tensor(y, dtype=torch.float32)               # (9, N)
        )

def train_deeponet(
    model, train_loader, val_loader, num_epochs=100, lr=1e-3, device='cuda',
    model_save_path='deeponet_best.pth', log_path='deeponet.log', early_stop_patience=20
):
    print(f"Training DeepONet on device: {device}")
    model = model.to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)
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
        for branch, trunk, y in tqdm(train_loader, desc=f"Train {epoch+1}", leave=False):
            branch = branch.to(device)
            trunk = trunk.to(device)
            y = y.to(device)
            optimizer.zero_grad()
            y_pred = model(branch, trunk)
            loss = loss_fn(y_pred, y)
            loss.backward()
            optimizer.step()
            total_loss += loss.item() * branch.size(0)
            n += branch.size(0)
        avg_loss = total_loss / n
        # Validation
        model.eval()
        val_loss = 0.0
        n_val = 0
        with torch.no_grad():
            for branch, trunk, y in tqdm(val_loader, desc=f"Val {epoch+1}", leave=False):
                branch = branch.to(device)
                trunk = trunk.to(device)
                y = y.to(device)
                y_pred = model(branch, trunk)
                vloss = loss_fn(y_pred, y)
                val_loss += vloss.item() * branch.size(0)
                n_val += branch.size(0)
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

def test_deeponet(model, test_loader, device='cuda', log_path='deeponet.log'):
    model.eval()
    loss_fn = nn.MSELoss()
    total_loss = 0.0
    n = 0
    with torch.no_grad():
        for branch, trunk, y in tqdm(test_loader, desc="Test"):
            branch = branch.to(device)
            trunk = trunk.to(device)
            y = y.to(device)
            y_pred = model(branch, trunk)
            loss = loss_fn(y_pred, y)
            total_loss += loss.item() * branch.size(0)
            n += branch.size(0)
    avg_loss = total_loss / n
    print(f"[Test] Loss: {avg_loss:.6f}")
    if log_path is not None:
        with open(log_path, "a") as f:
            f.write(f"test,{avg_loss}\n")
    return avg_loss

def run_deeponet_pipeline():
    import pandas as pd
    import os
    from torch.utils.data import DataLoader, random_split

    data_dir = os.path.join(os.path.dirname(__file__), "data", "output")
    file_list = [os.path.join(data_dir, fname) for fname in os.listdir(data_dir) if fname.endswith('.csv')]
    dataset = ThreeBodyDeepONetTimeSeriesDataset(file_list)
    print(f"Loaded dataset with {len(dataset)} samples.")
    total = len(dataset)
    n_train = int(0.7 * total)
    n_val = int(0.15 * total)
    n_test = total - n_train - n_val
    train_ds, val_ds, test_ds = random_split(dataset, [n_train, n_val, n_test], generator=torch.Generator().manual_seed(42))
    train_loader = DataLoader(train_ds, batch_size=8, shuffle=True)
    val_loader = DataLoader(val_ds, batch_size=8, shuffle=False)
    test_loader = DataLoader(test_ds, batch_size=8, shuffle=False)

    model = DeepONet(
        branch_dim=21, trunk_dim=1, hidden_branch=128, hidden_trunk=128,
        p=128, output_dim=9, num_branch_layers=4, num_trunk_layers=4
    )
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    train_deeponet(
        model, train_loader, val_loader,
        num_epochs=100, lr=1e-3, device=device, model_save_path='deeponet.pth', early_stop_patience=20
    )
    test_deeponet(model, test_loader, device=device)

if __name__ == "__main__":
    run_deeponet_pipeline()