"""
Training pipeline for HyperAggregation experiments.

Handles:
- Vertex-level classification (transductive and inductive)
- Graph-level classification and regression
- Multiple seeds and splits
- Early stopping on validation
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import time
import copy
import json
import os
from torch_geometric.loader import DataLoader
from torch_geometric.utils import subgraph

from models import GHC, GHM, GCN, MLP
from datasets import load_dataset


def train_vertex_epoch(model, data, optimizer, train_mask):
    """Train one epoch for vertex-level task."""
    model.train()
    optimizer.zero_grad()
    
    out = model(data.x, data.edge_index)
    loss = F.cross_entropy(out[train_mask], data.y[train_mask])
    loss.backward()
    optimizer.step()
    
    return loss.item()


@torch.no_grad()
def eval_vertex(model, data, mask):
    """Evaluate vertex-level task. Returns accuracy."""
    model.eval()
    out = model(data.x, data.edge_index)
    pred = out[mask].argmax(dim=1)
    correct = (pred == data.y[mask]).sum().item()
    total = mask.sum().item()
    return correct / total if total > 0 else 0.0


def train_graph_epoch(model, loader, optimizer, task='graph'):
    """Train one epoch for graph-level task."""
    model.train()
    total_loss = 0
    total_samples = 0
    
    for batch in loader:
        batch = batch.to(next(model.parameters()).device)
        optimizer.zero_grad()
        out = model(batch.x, batch.edge_index, batch.batch)
        
        if task == 'graph_regression':
            loss = F.l1_loss(out.squeeze(-1), batch.y.float())
        else:
            loss = F.cross_entropy(out, batch.y)
        
        loss.backward()
        optimizer.step()
        
        total_loss += loss.item() * batch.num_graphs
        total_samples += batch.num_graphs
    
    return total_loss / total_samples


@torch.no_grad()
def eval_graph(model, loader, task='graph'):
    """Evaluate graph-level task. Returns accuracy or MAE."""
    model.eval()
    total_correct = 0
    total_samples = 0
    total_loss = 0
    
    for batch in loader:
        batch = batch.to(next(model.parameters()).device)
        out = model(batch.x, batch.edge_index, batch.batch)
        
        if task == 'graph_regression':
            loss = F.l1_loss(out.squeeze(-1), batch.y.float(), reduction='sum')
            total_loss += loss.item()
        else:
            pred = out.argmax(dim=1)
            total_correct += (pred == batch.y).sum().item()
        
        total_samples += batch.num_graphs
    
    if task == 'graph_regression':
        return total_loss / total_samples  # MAE (lower is better)
    else:
        return total_correct / total_samples  # Accuracy (higher is better)


def make_inductive_data(data, train_mask, device):
    """Create inductive training graph (only train nodes and their edges)."""
    train_nodes = train_mask.nonzero(as_tuple=True)[0]
    edge_index, _ = subgraph(train_nodes, data.edge_index, relabel_nodes=False, 
                              num_nodes=data.num_nodes)
    
    inductive_data = data.clone()
    inductive_data.edge_index = edge_index
    return inductive_data.to(device)


def create_model(config, in_dim, out_dim, task, device):
    """Create model from config."""
    model_name = config['model']
    hidden_dim = config.get('hidden_dim', 256)
    
    if model_name == 'GHC':
        model = GHC(
            in_dim=in_dim, hidden_dim=hidden_dim, out_dim=out_dim,
            num_blocks=config.get('num_blocks', 2),
            mix_dim=config.get('mix_dim', 64),
            mix_dropout=config.get('mix_dropout', 0.0),
            dropout=config.get('dropout', 0.5),
            input_dropout=config.get('input_dropout', 0.0),
            trans_input=config.get('trans_input', False),
            trans_output=config.get('trans_output', False),
            input_activation=config.get('input_activation', False),
            mean_agg=config.get('mean_agg', True),
            root_conn=config.get('root_conn', True),
            residual=config.get('residual', False),
            add_self_loop=config.get('add_self_loop', True),
            make_undirected=config.get('make_undirected', False),
            normalize_input=config.get('normalize_input', False),
            task=task,
            use_embedding=config.get('use_embedding', False),
            num_embeddings=config.get('num_embeddings', 28),
        )
    elif model_name == 'GHM':
        model = GHM(
            in_dim=in_dim, hidden_dim=hidden_dim, out_dim=out_dim,
            num_blocks=config.get('num_blocks', 2),
            mix_dim=config.get('mix_dim', 64),
            mix_dropout=config.get('mix_dropout', 0.0),
            dropout=config.get('dropout', 0.5),
            input_dropout=config.get('input_dropout', 0.0),
            trans_input=config.get('trans_input', False),
            trans_output=config.get('trans_output', False),
            input_activation=config.get('input_activation', False),
            root_conn=config.get('root_conn', True),
            residual=config.get('residual', False),
            k_hop=config.get('k_hop', 2),
            normalize_input=config.get('normalize_input', False),
            task=task,
            use_embedding=config.get('use_embedding', False),
            num_embeddings=config.get('num_embeddings', 28),
        )
    elif model_name == 'GCN':
        model = GCN(
            in_dim=in_dim, hidden_dim=hidden_dim, out_dim=out_dim,
            num_layers=config.get('num_layers', 2),
            dropout=config.get('dropout', 0.5),
            input_dropout=config.get('input_dropout', 0.0),
            task=task,
        )
    elif model_name == 'MLP':
        model = MLP(
            in_dim=in_dim, hidden_dim=hidden_dim, out_dim=out_dim,
            num_layers=config.get('num_layers', 2),
            dropout=config.get('dropout', 0.5),
            input_dropout=config.get('input_dropout', 0.0),
            task=task,
        )
    else:
        raise ValueError(f"Unknown model: {model_name}")
    
    return model.to(device)


def run_single_experiment(config, seed, split_idx=0, device='cuda',
                          data_cache=None):
    """
    Run a single experiment with given config, seed, and split.
    
    data_cache: optional dict to cache loaded data across runs
    
    Returns: dict with train, val, test metrics
    """
    torch.manual_seed(seed)
    np.random.seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
    
    dataset_name = config['dataset']
    model_name = config['model']
    task_type = config.get('task', 'vertex')
    setting = config.get('setting', 'transductive')
    
    # Load dataset (use cache if available)
    if data_cache is not None and dataset_name in data_cache:
        cached = data_cache[dataset_name]
    else:
        cached = load_dataset(dataset_name)
        if data_cache is not None:
            data_cache[dataset_name] = cached
    
    if task_type in ['graph', 'graph_regression']:
        datasets_tuple, num_features, num_classes, task_str, num_dataset_splits = cached
        train_dataset, val_dataset, test_dataset = datasets_tuple
        
        in_dim = num_features
        out_dim = num_classes if task_type == 'graph' else 1
        
        train_loader = DataLoader(train_dataset, batch_size=config.get('batch_size', 128), 
                                  shuffle=True)
        val_loader = DataLoader(val_dataset, batch_size=config.get('batch_size', 256))
        test_loader = DataLoader(test_dataset, batch_size=config.get('batch_size', 256))
    else:
        data, num_features, num_classes, task_str, num_dataset_splits = cached
        in_dim = num_features
        out_dim = num_classes
        
        data = data.to(device)
        
        # Get split masks
        actual_split = split_idx % len(data.train_masks)
        train_mask = data.train_masks[actual_split].to(device)
        val_mask = data.val_masks[actual_split].to(device)
        test_mask = data.test_masks[actual_split].to(device)
        
        if setting == 'inductive':
            train_data = make_inductive_data(data, train_mask, device)
        else:
            train_data = data
    
    # Create model
    model = create_model(config, in_dim, out_dim, task_type, device)
    
    # Optimizer
    lr = config.get('lr', 0.001)
    weight_decay = config.get('weight_decay', 0.0)
    optimizer = torch.optim.Adam(model.parameters(), lr=lr, weight_decay=weight_decay)
    
    # Optional LR scheduler
    scheduler = None
    if config.get('use_scheduler', False):
        scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
            optimizer, mode='min' if task_type == 'graph_regression' else 'max',
            factor=0.5, patience=25, min_lr=1e-5
        )
    
    # Training loop
    epochs = config.get('epochs', 500)
    patience = config.get('patience', 100)
    
    is_regression = (task_type == 'graph_regression')
    best_val_metric = float('inf') if is_regression else -float('inf')
    best_model_state = None
    patience_counter = 0
    
    for epoch in range(epochs):
        if task_type in ['graph', 'graph_regression']:
            train_loss = train_graph_epoch(model, train_loader, optimizer, task_type)
            val_metric = eval_graph(model, val_loader, task_type)
        else:
            if setting == 'inductive':
                train_loss = train_vertex_epoch(model, train_data, optimizer, train_mask)
            else:
                train_loss = train_vertex_epoch(model, data, optimizer, train_mask)
            val_metric = eval_vertex(model, data, val_mask)
        
        if scheduler is not None:
            scheduler.step(val_metric)
        
        # Check improvement
        if is_regression:
            improved = val_metric < best_val_metric
        else:
            improved = val_metric > best_val_metric
        
        if improved:
            best_val_metric = val_metric
            best_model_state = copy.deepcopy(model.state_dict())
            patience_counter = 0
        else:
            patience_counter += 1
        
        if patience_counter >= patience:
            break
    
    # Load best model
    if best_model_state is not None:
        model.load_state_dict(best_model_state)
    
    # Final evaluation
    if task_type in ['graph', 'graph_regression']:
        test_metric = eval_graph(model, test_loader, task_type)
        val_metric = eval_graph(model, val_loader, task_type)
        train_metric = eval_graph(model, train_loader, task_type)
    else:
        test_metric = eval_vertex(model, data, test_mask)
        val_metric = eval_vertex(model, data, val_mask)
        train_metric = eval_vertex(model, data, train_mask)
    
    return {
        'train': train_metric,
        'val': val_metric,
        'test': test_metric,
        'epochs_trained': epoch + 1,
    }


def run_experiment(config, device='cuda'):
    """
    Run experiment with multiple seeds and splits as specified in config.
    
    Returns: dict with mean and std of test metrics
    """
    num_seeds = config.get('num_seeds', 10)
    num_splits = config.get('num_splits', 1)
    
    all_results = []
    data_cache = {}
    
    for split_idx in range(num_splits):
        for seed_idx in range(num_seeds):
            seed = seed_idx + config.get('seed_offset', 0)
            print(f"  Split {split_idx}/{num_splits}, Seed {seed_idx}/{num_seeds}...", 
                  end=' ', flush=True)
            t0 = time.time()
            result = run_single_experiment(config, seed=seed, split_idx=split_idx, 
                                           device=device, data_cache=data_cache)
            dt = time.time() - t0
            all_results.append(result)
            print(f"Test: {result['test']:.4f} ({dt:.1f}s, {result['epochs_trained']} epochs)")
    
    test_metrics = [r['test'] for r in all_results]
    val_metrics = [r['val'] for r in all_results]
    
    return {
        'test_mean': np.mean(test_metrics),
        'test_std': np.std(test_metrics),
        'val_mean': np.mean(val_metrics),
        'val_std': np.std(val_metrics),
        'all_results': all_results,
        'config': {k: v for k, v in config.items() if k != 'all_results'},
    }


if __name__ == '__main__':
    # Quick test with Cora + GHC
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")
    
    config = {
        'dataset': 'Cora',
        'model': 'GHC',
        'task': 'vertex',
        'setting': 'transductive',
        'hidden_dim': 256,
        'mix_dim': 64,
        'dropout': 0.6,
        'input_dropout': 0.0,
        'mix_dropout': 0.0,
        'num_blocks': 2,
        'mean_agg': True,
        'root_conn': True,
        'residual': False,
        'trans_input': False,
        'trans_output': True,
        'add_self_loop': True,
        'make_undirected': False,
        'lr': 0.001,
        'weight_decay': 5e-4,
        'epochs': 200,
        'patience': 50,
        'num_seeds': 1,
        'num_splits': 1,
    }
    
    print("Testing GHC on Cora (1 seed, 1 split)...")
    result = run_experiment(config, device=str(device))
    print(f"\nResult: test_mean={result['test_mean']:.4f}")
