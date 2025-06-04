import os
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, random_split
from tqdm import tqdm
from DataLoader import TBPDataset, compute_scalers
from DeepONet import DeepONet
from PINN import PINN, compute_second_derivative, compute_gravity_acceleration

class GeneralOLPINN(nn.Module):
    def __init__(self, 
                 base_type='DeepONet',  # 'DeepONet', 'FNO', 'TFNO'
                 correction_type='PINN',  # 'PINN', 'FPINN', 'ResPINN'
                 base_ckpt=None,
                 input_dim=22, output_dim=9, device='cuda',
                 **kwargs):
        super().__init__()
        self.device = device
        self.base_type = base_type.lower()
        self.correction_type = correction_type.lower()

        # --- Base Model ---
        if self.base_type == 'deeponet':
            from DeepONet import DeepONet
            self.base_model = DeepONet(
                branch_dim=21, trunk_dim=1, hidden_branch=128, hidden_trunk=128,
                p=128, output_dim=9, num_branch_layers=4, num_trunk_layers=4
            ).to(device)
            if base_ckpt:
                self.base_model.load_state_dict(torch.load(base_ckpt, map_location=device))
            self.base_model.eval()
            for param in self.base_model.parameters():
                param.requires_grad = False

        elif self.base_type == 'fno':
            from FNO import FNO
            self.base_model = FNO(modes=32, width=64, input_dim=22, output_dim=9, depth=4).to(device)
            if base_ckpt:
                self.base_model.load_state_dict(torch.load(base_ckpt, map_location=device))
            self.base_model.eval()
            for param in self.base_model.parameters():
                param.requires_grad = False

        elif self.base_type == 'tfno':
            from TFNO import TFNO
            self.base_model = TFNO(
                t_steps=100, input_dim=22, embed_dim=32, width=64, modes=32,
                depth=4, output_dim=9
            ).to(device)
            if base_ckpt:
                self.base_model.load_state_dict(torch.load(base_ckpt, map_location=device))
            self.base_model.eval()
            for param in self.base_model.parameters():
                param.requires_grad = False

        else:
            raise ValueError(f"Unknown base_type: {base_type}")

        # --- Correction Model ---
        if self.correction_type == 'pinn':
            from PINN import PINN
            self.correction = PINN(input_dim=22, hidden_dim=128, depth=4, output_dim=9).to(device)

        elif self.correction_type == 'fpinn':
            from FPINN import FPINN
            self.correction = FPINN(input_dim=21, embed_dim=32, hidden_dim=128, depth=4, output_dim=9).to(device)

        elif self.correction_type == 'respinn':
            from ResPINN import ResPINN
            self.correction = ResPINN(input_dim=21, embed_dim=32, hidden_dim=128, depth=6, output_dim=9).to(device)

        else:
            raise ValueError(f"Unknown correction_type: {correction_type}")

    def forward(self, xb):  # xb: (B, D, N)
        B, D, N = xb.shape

        # --- Base prediction ---
        with torch.no_grad():
            if self.base_type == 'deeponet':
                branch_input = xb[:, :21, :].mean(dim=2)  # (B, 21)
                trunk_input = xb[:, 21:, :].permute(0, 2, 1)  # (B, N, 1)
                base_pred = self.base_model(branch_input, trunk_input)  # (B, 9, N)
            else:
                base_pred = self.base_model(xb)

        # --- Residual correction ---
        if self.correction_type == 'pinn':
            residual = self.correction(xb)  # (B, 9, N)
        else:
            x_cond = xb[:, :21, 0]                 # (B, 21)
            t = xb[:, 21:, :].permute(0, 2, 1)     # (B, N, 1)
            residual = self.correction(x_cond, t)  # (B, 9, N)

        return base_pred + residual

def train_ol_pinn(model, train_loader, val_loader, x_scaler, y_scaler, 
                  model_save_path, log_path, num_epochs=10000, lr=1e-3, early_stop_patience=50,
                  device='cuda', phys_loss_weight=1e-5):
    model = model.to(device)
    optimizer = torch.optim.Adam(model.correction.parameters(), lr=lr)
    loss_fn = nn.MSELoss()
    best_val = float('inf')
    patience = 0

    with open(log_path, "w") as f:
        f.write("epoch,train_loss,train_phys,val_loss,val_phys\n")

    def compute_loss(xb, yb, mb, is_train=True):
        xb, yb, mb = xb.to(device), yb.to(device), mb.to(device)
        t = xb[:, -1, :]
        x_scaled, y_scaled = x_scaler.transform(xb), y_scaler.transform(yb)

        if is_train:
            optimizer.zero_grad()

        y_pred = model(x_scaled)
        loss_data = loss_fn(y_pred, y_scaled)

        y_pred_phys = y_scaler.inverse_transform(y_pred)
        from PINN import compute_second_derivative, compute_gravity_acceleration
        a_pred = compute_second_derivative(y_pred_phys, t)
        a_grav = compute_gravity_acceleration(y_pred_phys, mb)[:, :, 1:-1]
        loss_phys = ((a_pred - a_grav) ** 2).mean()
        loss_phys_scaled = loss_phys * (loss_data.detach() / (loss_phys.detach() + 1e-8))
        loss = loss_data + phys_loss_weight * loss_phys_scaled

        if is_train:
            loss.backward()
            optimizer.step()

        return loss_data.item(), loss_phys.item(), xb.size(0)

    for epoch in range(num_epochs):
        model.train()
        train_loss, train_phys, n = 0.0, 0.0, 0
        for xb, yb, _, mb in tqdm(train_loader, desc=f"Train {epoch+1}", leave=False):
            l_data, l_phys, bsz = compute_loss(xb, yb, mb, is_train=True)
            train_loss += l_data * bsz
            train_phys += l_phys * bsz
            n += bsz
        train_loss /= n
        train_phys /= n

        model.eval()
        val_loss, val_phys, nv = 0.0, 0.0, 0
        with torch.no_grad():
            for xb, yb, _, mb in tqdm(val_loader, desc=f"Val {epoch+1}", leave=False):
                l_data, l_phys, bsz = compute_loss(xb, yb, mb, is_train=False)
                val_loss += l_data * bsz
                val_phys += l_phys * bsz
                nv += bsz
        val_loss /= nv
        val_phys /= nv

        with open(log_path, "a") as f:
            f.write(f"{epoch},{train_loss},{train_phys},{val_loss},{val_phys}\n")

        if val_loss < best_val:
            best_val = val_loss
            patience = 0
            torch.save(model.state_dict(), model_save_path)
        else:
            patience += 1
            if patience >= early_stop_patience:
                print(f"Early stopping at epoch {epoch+1}")
                break

        if epoch % 10 == 0:
            print(f"[Epoch {epoch+1}] Train: {train_loss:.6f}, Phys: {train_phys:.6f} | Val: {val_loss:.6f}, Phys: {val_phys:.6f}")

def test_ol_pinn(model, test_loader, x_scaler, y_scaler, device='cuda', log_path=None):
    model.eval()
    loss_fn = nn.MSELoss()
    test_loss, n = 0.0, 0
    with torch.no_grad():
        for xb, yb, _, _ in tqdm(test_loader, desc="Test"):
            xb, yb = xb.to(device), yb.to(device)
            x_scaled, y_scaled = x_scaler.transform(xb), y_scaler.transform(yb)
            y_pred = model(x_scaled)
            loss = loss_fn(y_pred, y_scaled)
            test_loss += loss.item() * xb.size(0)
            n += xb.size(0)
    test_loss /= n
    print(f"[Test] Loss: {test_loss:.6f}")
    if log_path is not None:
        with open(log_path, "a") as f:
            f.write(f"test,{test_loss}\n")
    return test_loss

def run_ol_pinn_pipeline():
    train_dir = os.path.join(os.path.dirname(__file__), "data", "train")
    train_file_list = [os.path.join(train_dir, fname) for fname in os.listdir(train_dir) if fname.endswith('.pt')]
    train_dataset = TBPDataset(train_file_list[:], preload=False)  # For testing, limit to 100 files
    train_ds, val_ds, _ = random_split(train_dataset, [0.8, 0.2, 0], generator=torch.Generator().manual_seed(42))
    print(f"Loaded train dataset with {len(train_dataset)} samples.")
    
    test_dir = os.path.join(os.path.dirname(__file__), "data", "test")
    test_file_list = [os.path.join(test_dir, fname) for fname in os.listdir(test_dir) if fname.endswith('.pt')]
    test_ds = TBPDataset(test_file_list)
    print(f"Loaded test dataset with {len(test_ds)} samples.")
    
    train_loader = DataLoader(train_ds, batch_size=1024, shuffle=True, num_workers=8, pin_memory=True, persistent_workers=True)
    val_loader = DataLoader(val_ds, batch_size=1024, shuffle=False, num_workers=8, pin_memory=True, persistent_workers=True)
    test_loader = DataLoader(test_ds, batch_size=1024, shuffle=False, num_workers=8, pin_memory=True, persistent_workers=True)
    print (f"Created DataLoaders: train={len(train_loader)}, val={len(val_loader)}, test={len(test_loader)}")
    
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    x_scaler, y_scaler = compute_scalers(train_loader, device)

    model = GeneralOLPINN(
        base_type='FNO', 
        correction_type='PINN', 
        base_ckpt='models/fno.pth',
        input_dim=22, output_dim=9, t_steps=100, device='cuda'
    )

    model_path = os.path.join(os.path.dirname(__file__), 'models', 'residual_fno_pinn.pth')
    log_path = os.path.join(os.path.dirname(__file__), 'logs', 'residual_fno_pinn.log')

    train_ol_pinn(
        model, train_loader, val_loader, x_scaler, y_scaler,
        model_save_path=model_path,
        log_path=log_path,
        device=device
    )

    test_ol_pinn(model, test_loader, x_scaler, y_scaler, device=device, log_path=log_path)
    
    model = GeneralOLPINN(
        base_type='TFNO', 
        correction_type='PINN', 
        base_ckpt='models/tfno.pth',
        input_dim=22, output_dim=9, t_steps=100, device='cuda'
    )

    model_path = os.path.join(os.path.dirname(__file__), 'models', 'residual_tfno_pinn.pth')
    log_path = os.path.join(os.path.dirname(__file__), 'logs', 'residual_tfno_pinn.log')

    train_ol_pinn(
        model, train_loader, val_loader, x_scaler, y_scaler,
        model_save_path=model_path,
        log_path=log_path,
        device=device
    )

    test_ol_pinn(model, test_loader, x_scaler, y_scaler, device=device, log_path=log_path)
    
    model = GeneralOLPINN(
        base_type='DeepONet', 
        correction_type='PINN', 
        base_ckpt='models/deeponet.pth',
        input_dim=22, output_dim=9, t_steps=100, device='cuda'
    )
    
    model_path = os.path.join(os.path.dirname(__file__), 'models', 'residual_deeponet_pinn.pth')
    log_path = os.path.join(os.path.dirname(__file__), 'logs', 'residual_deeponet_pinn.log')
    
    train_ol_pinn(
        model, train_loader, val_loader, x_scaler, y_scaler,
        model_save_path=model_path,
        log_path=log_path,
        device=device
    )
    
    test_ol_pinn(model, test_loader, x_scaler, y_scaler, device=device, log_path=log_path)
    
    model = GeneralOLPINN(
        base_type='FNO', 
        correction_type='FPINN', 
        base_ckpt='models/fno.pth',
        input_dim=22, output_dim=9, t_steps=100, device='cuda'
    )

    model_path = os.path.join(os.path.dirname(__file__), 'models', 'residual_fno_fpinn.pth')
    log_path = os.path.join(os.path.dirname(__file__), 'logs', 'residual_fno_fpinn.log')

    train_ol_pinn(
        model, train_loader, val_loader, x_scaler, y_scaler,
        model_save_path=model_path,
        log_path=log_path,
        device=device
    )

    test_ol_pinn(model, test_loader, x_scaler, y_scaler, device=device, log_path=log_path)
    
    model = GeneralOLPINN(
        base_type='TFNO', 
        correction_type='FPINN', 
        base_ckpt='models/tfno.pth',
        input_dim=22, output_dim=9, t_steps=100, device='cuda'
    )
    
    model_path = os.path.join(os.path.dirname(__file__), 'models', 'residual_tfno_fpinn.pth')
    log_path = os.path.join(os.path.dirname(__file__), 'logs', 'residual_tfno_fpinn.log')
    
    train_ol_pinn(
        model, train_loader, val_loader, x_scaler, y_scaler,
        model_save_path=model_path,
        log_path=log_path,
        device=device
    )

    test_ol_pinn(model, test_loader, x_scaler, y_scaler, device=device, log_path=log_path)
    
    model = GeneralOLPINN(
        base_type='DeepONet', 
        correction_type='FPINN', 
        base_ckpt='models/deeponet.pth',
        input_dim=22, output_dim=9, t_steps=100, device='cuda'
    )
    
    model_path = os.path.join(os.path.dirname(__file__), 'models', 'residual_deeponet_fpinn.pth')
    log_path = os.path.join(os.path.dirname(__file__), 'logs', 'residual_deeponet_fpinn.log')
    train_ol_pinn(
        model, train_loader, val_loader, x_scaler, y_scaler,
        model_save_path=model_path,
        log_path=log_path,
        device=device
    )
    
    test_ol_pinn(model, test_loader, x_scaler, y_scaler, device=device, log_path=log_path)
    
    model = GeneralOLPINN(
        base_type='FNO', 
        correction_type='ResPINN', 
        base_ckpt='models/fno.pth',
        input_dim=22, output_dim=9, t_steps=100, device='cuda'
    )

    model_path = os.path.join(os.path.dirname(__file__), 'models', 'residual_fno_respinn.pth')
    log_path = os.path.join(os.path.dirname(__file__), 'logs', 'residual_fno_respinn.log')

    train_ol_pinn(
        model, train_loader, val_loader, x_scaler, y_scaler,
        model_save_path=model_path,
        log_path=log_path,
        device=device
    )

    test_ol_pinn(model, test_loader, x_scaler, y_scaler, device=device, log_path=log_path)
    
    model = GeneralOLPINN(
        base_type='TFNO', 
        correction_type='ResPINN', 
        base_ckpt='models/tfno.pth',
        input_dim=22, output_dim=9, t_steps=100, device='cuda'
    )

    model_path = os.path.join(os.path.dirname(__file__), 'models', 'residual_tfno_respinn.pth')
    log_path = os.path.join(os.path.dirname(__file__), 'logs', 'residual_tfno_respinn.log')

    train_ol_pinn(
        model, train_loader, val_loader, x_scaler, y_scaler,
        model_save_path=model_path,
        log_path=log_path,
        device=device
    )

    test_ol_pinn(model, test_loader, x_scaler, y_scaler, device=device, log_path=log_path)
    
    model = GeneralOLPINN(
        base_type='DeepONet', 
        correction_type='ResPINN', 
        base_ckpt='models/deeponet.pth',
        input_dim=22, output_dim=9, t_steps=100, device='cuda'
    )

    model_path = os.path.join(os.path.dirname(__file__), 'models', 'residual_deeponet_respinn.pth')
    log_path = os.path.join(os.path.dirname(__file__), 'logs', 'residual_deeponet_respinn.log')

    train_ol_pinn(
        model, train_loader, val_loader, x_scaler, y_scaler,
        model_save_path=model_path,
        log_path=log_path,
        device=device
    )

    test_ol_pinn(model, test_loader, x_scaler, y_scaler, device=device, log_path=log_path)

if __name__ == '__main__':
    run_ol_pinn_pipeline()
