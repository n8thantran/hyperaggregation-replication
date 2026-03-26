#!/usr/bin/env python3
"""Run ZINC graph regression experiment with 3 seeds."""
import torch
import json
import os
import time
import numpy as np
import copy
import torch.nn.functional as F
from torch_geometric.loader import DataLoader
from models import GHC
from datasets import load_dataset

device = 'cuda' if torch.cuda.is_available() else 'cpu'
os.makedirs('results', exist_ok=True)

# Load dataset once
cached = load_dataset('zinc')
datasets_tuple, num_features, num_classes, task_str, num_splits = cached
train_dataset, val_dataset, test_dataset = datasets_tuple

print(f"ZINC: Train={len(train_dataset)}, Val={len(val_dataset)}, Test={len(test_dataset)}")

def train_epoch(model, loader, optimizer):
    model.train()
    total_loss = 0
    total_samples = 0
    for batch in loader:
        batch = batch.to(device)
        optimizer.zero_grad()
        out = model(batch.x, batch.edge_index, batch.batch)
        loss = F.l1_loss(out.squeeze(-1), batch.y.float())
        loss.backward()
        optimizer.step()
        total_loss += loss.item() * batch.num_graphs
        total_samples += batch.num_graphs
    return total_loss / total_samples

@torch.no_grad()
def eval_model(model, loader):
    model.eval()
    total_loss = 0
    total_samples = 0
    for batch in loader:
        batch = batch.to(device)
        out = model(batch.x, batch.edge_index, batch.batch)
        loss = F.l1_loss(out.squeeze(-1), batch.y.float(), reduction='sum')
        total_loss += loss.item()
        total_samples += batch.num_graphs
    return total_loss / total_samples

# Config
configs = [
    {
        'name': 'GHC_v1',
        'hidden_dim': 256, 'mix_dim': 64, 'num_blocks': 4,
        'dropout': 0.0, 'mix_dropout': 0.0,
        'mean_agg': True, 'root_conn': True, 'residual': True,
        'trans_input': False, 'trans_output': True,
        'add_self_loop': True, 'make_undirected': False,
        'lr': 0.001, 'weight_decay': 0.0,
        'epochs': 500, 'patience': 100, 'batch_size': 128,
    },
    {
        'name': 'GHC_v2',
        'hidden_dim': 256, 'mix_dim': 64, 'num_blocks': 4,
        'dropout': 0.1, 'mix_dropout': 0.0,
        'mean_agg': True, 'root_conn': True, 'residual': True,
        'trans_input': True, 'trans_output': True,
        'add_self_loop': True, 'make_undirected': False,
        'lr': 0.001, 'weight_decay': 1e-5,
        'epochs': 500, 'patience': 100, 'batch_size': 128,
    },
    {
        'name': 'GHC_v3',
        'hidden_dim': 256, 'mix_dim': 32, 'num_blocks': 4,
        'dropout': 0.0, 'mix_dropout': 0.0,
        'mean_agg': True, 'root_conn': True, 'residual': True,
        'trans_input': False, 'trans_output': True,
        'add_self_loop': True, 'make_undirected': False,
        'lr': 0.001, 'weight_decay': 0.0,
        'epochs': 500, 'patience': 100, 'batch_size': 128,
    },
]

# First, find best config with 1 seed
best_config = None
best_val = float('inf')

for cfg in configs:
    print(f"\n{'='*60}")
    print(f"Testing config: {cfg['name']}")
    print(f"{'='*60}")
    
    torch.manual_seed(42)
    np.random.seed(42)
    torch.cuda.manual_seed(42)
    
    model = GHC(
        in_dim=1, hidden_dim=cfg['hidden_dim'], out_dim=1,
        num_blocks=cfg['num_blocks'], mix_dim=cfg['mix_dim'],
        mix_dropout=cfg['mix_dropout'], dropout=cfg['dropout'],
        trans_input=cfg['trans_input'], trans_output=cfg['trans_output'],
        mean_agg=cfg['mean_agg'], root_conn=cfg['root_conn'],
        residual=cfg['residual'], add_self_loop=cfg['add_self_loop'],
        make_undirected=cfg['make_undirected'],
        task='graph_regression', use_embedding=True, num_embeddings=28,
    ).to(device)
    
    n_params = sum(p.numel() for p in model.parameters())
    print(f"Parameters: {n_params:,}")
    
    train_loader = DataLoader(train_dataset, batch_size=cfg['batch_size'], shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=256)
    test_loader = DataLoader(test_dataset, batch_size=256)
    
    optimizer = torch.optim.Adam(model.parameters(), lr=cfg['lr'], weight_decay=cfg['weight_decay'])
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, mode='min', factor=0.5, patience=25, min_lr=1e-5
    )
    
    best_val_mae = float('inf')
    best_state = None
    patience_counter = 0
    
    t0 = time.time()
    for epoch in range(cfg['epochs']):
        train_mae = train_epoch(model, train_loader, optimizer)
        val_mae = eval_model(model, val_loader)
        scheduler.step(val_mae)
        
        if val_mae < best_val_mae:
            best_val_mae = val_mae
            best_state = copy.deepcopy(model.state_dict())
            patience_counter = 0
        else:
            patience_counter += 1
        
        if (epoch + 1) % 50 == 0:
            lr = optimizer.param_groups[0]['lr']
            print(f"  Epoch {epoch+1}: train={train_mae:.4f}, val={val_mae:.4f}, best_val={best_val_mae:.4f}, lr={lr:.6f}")
        
        if patience_counter >= cfg['patience']:
            print(f"  Early stopping at epoch {epoch+1}")
            break
    
    model.load_state_dict(best_state)
    test_mae = eval_model(model, test_loader)
    dt = time.time() - t0
    print(f"  Config {cfg['name']}: test_mae={test_mae:.4f}, best_val={best_val_mae:.4f}, time={dt:.1f}s")
    
    if best_val_mae < best_val:
        best_val = best_val_mae
        best_config = cfg
        print(f"  ** New best config! **")

print(f"\n{'='*60}")
print(f"Best config: {best_config['name']}")
print(f"{'='*60}")

# Now run best config with 3 seeds
all_test = []
for seed in range(3):
    torch.manual_seed(seed)
    np.random.seed(seed)
    torch.cuda.manual_seed(seed)
    
    model = GHC(
        in_dim=1, hidden_dim=best_config['hidden_dim'], out_dim=1,
        num_blocks=best_config['num_blocks'], mix_dim=best_config['mix_dim'],
        mix_dropout=best_config['mix_dropout'], dropout=best_config['dropout'],
        trans_input=best_config['trans_input'], trans_output=best_config['trans_output'],
        mean_agg=best_config['mean_agg'], root_conn=best_config['root_conn'],
        residual=best_config['residual'], add_self_loop=best_config['add_self_loop'],
        make_undirected=best_config['make_undirected'],
        task='graph_regression', use_embedding=True, num_embeddings=28,
    ).to(device)
    
    train_loader = DataLoader(train_dataset, batch_size=best_config['batch_size'], shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=256)
    test_loader = DataLoader(test_dataset, batch_size=256)
    
    optimizer = torch.optim.Adam(model.parameters(), lr=best_config['lr'], weight_decay=best_config['weight_decay'])
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, mode='min', factor=0.5, patience=25, min_lr=1e-5
    )
    
    best_val_mae = float('inf')
    best_state = None
    patience_counter = 0
    
    t0 = time.time()
    for epoch in range(best_config['epochs']):
        train_mae = train_epoch(model, train_loader, optimizer)
        val_mae = eval_model(model, val_loader)
        scheduler.step(val_mae)
        
        if val_mae < best_val_mae:
            best_val_mae = val_mae
            best_state = copy.deepcopy(model.state_dict())
            patience_counter = 0
        else:
            patience_counter += 1
        
        if patience_counter >= best_config['patience']:
            break
    
    model.load_state_dict(best_state)
    test_mae = eval_model(model, test_loader)
    dt = time.time() - t0
    all_test.append(test_mae)
    print(f"  Seed {seed}: test_mae={test_mae:.4f}, epochs={epoch+1}, time={dt:.1f}s")

mean_mae = np.mean(all_test)
std_mae = np.std(all_test)
print(f"\nFinal ZINC GHC: MAE = {mean_mae:.4f} ± {std_mae:.4f}")
print(f"Paper target: 0.337 ± 0.020")

# Save results
result_data = {
    'test_mean': float(mean_mae),
    'test_std': float(std_mae),
    'all_test': [float(x) for x in all_test],
    'config': best_config,
    'paper_target': '0.337 ± 0.020',
}
with open('results/GHC_ZINC_graph.json', 'w') as f:
    json.dump(result_data, f, indent=2)
print("Saved to results/GHC_ZINC_graph.json")
