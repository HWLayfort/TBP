import configparser
import sys
import os
from model import DeepONet, FNO, FPINN, PINN, ResPINN, TFNO

# 각 모델별 실행 함수 정의
def run_deeponet(cfg):
    params = cfg['DeepONet']
    DeepONet.train_deeponet(
        csv_path=os.path.join(os.path.dirname(__file__), os.path.join("data", params.get('csv_path'))),
        batch_size=cfg.getint('Common', 'batch_size'),
        hidden_dim=cfg.getint('Common', 'hidden_dim'),
        branch_layers=params.getint('branch_layers'),
        trunk_layers=params.getint('trunk_layers'),
        out_dim=cfg.getint('Common', 'output_dim'),
        num_conditions=cfg.getint('Common', 'num_conditions'),
        time_steps=cfg.getint('Common', 'time_steps'),
        T=cfg.getfloat('Common', 'T'),
        lr=cfg.getfloat('Common', 'lr'),
        num_epochs=cfg.getint('Common', 'num_epochs'),
        patience=cfg.getint('Common', 'patience'),
        seed=cfg.getint('Common', 'seed'),
        model_save_path=params.get('model_save_path'),
        log_path=params.get('log_path')
    )

def run_fno(cfg):
    params = cfg['FNO']
    FNO.train_fno(
        csv_path=os.path.join(os.path.dirname(__file__), os.path.join("data", params.get('csv_path'))),
        batch_size=cfg.getint('Common', 'batch_size'),
        width=params.getint('width'),
        modes1=params.getint('modes1'),
        depth=params.getint('depth'),
        lr=cfg.getfloat('Common', 'lr'),
        num_epochs=cfg.getint('Common', 'num_epochs'),
        patience=cfg.getint('Common', 'patience'),
        seed=cfg.getint('Common', 'seed'),
        num_conditions=cfg.getint('Common', 'num_conditions'),
        time_steps=cfg.getint('Common', 'time_steps'),
        model_save_path=params.get('model_save_path'),
        log_path=params.get('log_path')
    )

def run_fpinn(cfg):
    params = cfg['FPINN']
    FPINN.train_fpinn(
        csv_path=os.path.join(os.path.dirname(__file__), os.path.join("data", params.get('csv_path'))),
        T=cfg.getfloat('Common', 'T'),
        hidden_dim=cfg.getint('Common', 'hidden_dim'),
        num_hidden_layers=cfg.getint('Common', 'num_hidden_layers'),
        input_dim=cfg.getint('Common', 'input_dim'),
        output_dim=cfg.getint('Common', 'output_dim'),
        lr=cfg.getfloat('Common', 'lr'),
        num_epochs=cfg.getint('Common', 'num_epochs'),
        batch_size=cfg.getint('Common', 'batch_size'),
        seed=cfg.getint('Common', 'seed'),
        fourier_mapping_size=params.getint('fourier_mapping_size'),
        fourier_scale=params.getfloat('fourier_scale'),
        model_save_path=params.get('model_save_path'),
        log_path=params.get('log_path'),
        early_stopping_patience=cfg.getint('Common', 'patience')
    )

def run_pinn(cfg):
    params = cfg['PINN']
    PINN.train_pinn(
        csv_path=os.path.join(os.path.dirname(__file__), os.path.join("data", params.get('csv_path'))),
        T=cfg.getfloat('Common', 'T'),
        hidden_dim=cfg.getint('Common', 'hidden_dim'),
        num_hidden_layers=cfg.getint('Common', 'num_hidden_layers'),
        input_dim=cfg.getint('Common', 'input_dim'),
        output_dim=cfg.getint('Common', 'output_dim'),
        lr=cfg.getfloat('Common', 'lr'),
        num_epochs=cfg.getint('Common', 'num_epochs'),
        batch_size=cfg.getint('Common', 'batch_size'),
        seed=cfg.getint('Common', 'seed'),
        model_save_path=params.get('model_save_path'),
        log_path=params.get('log_path'),
        early_stopping_patience=cfg.getint('Common', 'patience')
    )

def run_respinn(cfg):
    params = cfg['ResPINN']
    ResPINN.train_respinn(
        csv_path=os.path.join(os.path.dirname(__file__), os.path.join("data", params.get('csv_path'))),
        T=cfg.getfloat('Common', 'T'),
        hidden_dim=cfg.getint('Common', 'hidden_dim'),
        num_hidden_layers=cfg.getint('Common', 'num_hidden_layers'),
        input_dim=cfg.getint('Common', 'input_dim'),
        output_dim=cfg.getint('Common', 'output_dim'),
        lr=cfg.getfloat('Common', 'lr'),
        num_epochs=cfg.getint('Common', 'num_epochs'),
        batch_size=cfg.getint('Common', 'batch_size'),
        seed=cfg.getint('Common', 'seed'),
        model_save_path=params.get('model_save_path'),
        log_path=params.get('log_path'),
        early_stopping_patience=cfg.getint('Common', 'patience')
    )

def run_tfno(cfg):
    params = cfg['TFNO']
    TFNO.train_tfno(
        csv_path=os.path.join(os.path.dirname(__file__), os.path.join("data", params.get('csv_path'))),
        batch_size=cfg.getint('Common', 'batch_size'),
        width=params.getint('width'),
        modes1=params.getint('modes1'),
        depth=params.getint('depth'),
        t_embed_dim=params.getint('t_embed_dim'),
        t_embed_scale=params.getfloat('t_embed_scale'),
        lr=cfg.getfloat('Common', 'lr'),
        num_epochs=cfg.getint('Common', 'num_epochs'),
        patience=cfg.getint('Common', 'patience'),
        seed=cfg.getint('Common', 'seed'),
        num_conditions=cfg.getint('Common', 'num_conditions'),
        time_steps=cfg.getint('Common', 'time_steps'),
        model_save_path=params.get('model_save_path'),
        log_path=params.get('log_path')
    )

def main():
    if len(sys.argv) < 2:
        print("사용법: python controller.py <config_file.cfg>")
        sys.exit(1)

    config_path = sys.argv[1]
    if not os.path.isfile(config_path):
        print(f"설정 파일을 찾을 수 없음: {config_path}")
        sys.exit(1)

    cfg = configparser.ConfigParser()
    cfg.read(config_path)

    model = cfg.get('Common', 'model')
    dispatch = {
        'deeponet': run_deeponet,
        'fno': run_fno,
        'fpinn': run_fpinn,
        'pinn': run_pinn,
        'respinn': run_respinn,
        'tfno': run_tfno
    }

    if model == 'all':
        for fn in dispatch.values():
            fn(cfg)
    elif model in dispatch:
        dispatch[model](cfg)
    else:
        print(f"알 수 없는 모델: {model}")
        sys.exit(1)

if __name__ == '__main__':
    main()
