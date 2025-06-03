import os
import torch
import numpy as np
import pandas as pd
from PINN import PINN
from FPINN import FPINN
from ResPINN import ResPINN
from FNO import FNO
from TFNO import TFNO
from DeepONet import DeepONet
from tqdm import tqdm

class TBPTestDataset(torch.utils.data.Dataset):
    def __init__(self, data_path):
        data = pd.read_csv(data_path, header=None)
        self.input = []
        self.output = []
        self.length = len(data)
        for i in range(self.length):
            row = data.iloc[i]
            t = row[0]
            r0 = row[1:10].values
            v0 = row[10:19].values
            masses = row[19:22].values
            r_n = row[22:31].values
            self.input.append(np.concatenate(([t], r0, v0, masses), axis=0))
            self.output.append(r_n)

    def __len__(self):
        return self.length

    def __getitem__(self, idx):
        input_data = self.input[idx]
        output_data = self.output[idx]
        return (
            torch.tensor(input_data, dtype=torch.float32),  # t
            torch.tensor(output_data, dtype=torch.float32),  # r0
        )

def import_models(model_path, source='PINN'):
    if not os.path.exists(model_path):
        raise FileNotFoundError(f"Model file not found: {model_path}")
    
    if source == 'PINN':
        model = PINN(input_dim=22, hidden_dim=128, num_hidden_layers=8, output_dim=9)
        model.load_state_dict(torch.load(model_path, map_location='cuda'))
    elif source == 'FPINN':
        model = FPINN(mapping_size=32, scale=10.0, hidden_dim=128, num_hidden_layers=8, output_dim=9)
        model.load_state_dict(torch.load(model_path, map_location='cuda'))
    elif source == 'ResPINN':
        model = ResPINN(input_dim=22, hidden_dim=128, num_hidden_layers=8, output_dim=9)
        model.load_state_dict(torch.load(model_path, map_location='cuda'))
    elif source == 'FNO':
        model = FNO(modes=32, width=64, input_dim=22, output_dim=9, depth=4)
        model.load_state_dict(torch.load(model_path, map_location='cuda'))
    elif source == 'TFNO':
        model =  TFNO(t_steps=10000, input_dim=22, embed_dim=32, width=64, modes=32, depth=4, output_dim=9)
        model.load_state_dict(torch.load(model_path, map_location='cuda'))
    elif source == 'DeepONet':
        model = DeepONet(branch_dim=21, trunk_dim=1, hidden_branch=128, hidden_trunk=128, p=128, output_dim=9, num_branch_layers=4, num_trunk_layers=4)
        model.load_state_dict(torch.load(model_path, map_location='cuda'))
    
    return model

def load_test_data(data_path):
    dataset = TBPTestDataset(data_path)
    dataloader = torch.utils.data.DataLoader(dataset, batch_size=1, shuffle=False)
    return dataloader

def test_model(model, dataloader):
    # Check loss using MSE
    criterion = torch.nn.MSELoss()
    total_loss = 0.0
    with torch.no_grad():
        for xb, yb in tqdm(dataloader, desc="Testing", leave=False):
            xb = xb.to('cuda')
            yb = yb.to('cuda')
            output = model(xb)
            total_loss += criterion(output, yb).item()
    avg_loss = total_loss / len(dataloader)
    print(f"Average MSE Loss: {avg_loss:.4f}")
    return avg_loss

if __name__ == "__main__":
    model_path = os.path.join(os.path.dirname(__file__), 'models', 'deeponet_to_fpinn.pth')  # 모델 경로
    data_path = os.path.join(os.path.dirname(__file__), 'data', 'output', 'test.csv')  # 테스트 데이터 경로
    
    model = import_models(model_path, source='FPINN')  # 모델 불러오기
    model.to('cuda')

    dataloader = load_test_data(data_path)

    test_loss = test_model(model, dataloader)
    print(f"Test Loss: {test_loss:.4f}")