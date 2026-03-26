"""
Dataset loading and split generation for HyperAggregation experiments.

Datasets:
- Homophilic: Cora, CiteSeer, PubMed, OGB arXiv, Computers, Photo
- Heterophilic: Actor, Chameleon, Squirrel, Minesweeper, Roman-Empire
- Graph-level: MNIST, CIFAR10, ZINC (10k/1k/1k subset)

Split generation:
- Cora/CiteSeer/PubMed/Computers/Photo: 10 random splits, 20 per class train, 30 val
- OGB arXiv: provided year-based splits
- Actor: 48%/32%/20%
- Chameleon/Squirrel/Minesweeper/Roman-Empire: provided splits (10 splits)
- MNIST/CIFAR10/ZINC: provided splits
"""

import os
import torch
import numpy as np
from torch_geometric.datasets import (
    Planetoid, Amazon, Actor as ActorDataset, WikipediaNetwork,
    GNNBenchmarkDataset, ZINC, HeterophilousGraphDataset
)
from torch_geometric.data import Data
from torch_geometric.utils import to_undirected, add_self_loops
from ogb.nodeproppred import PygNodePropPredDataset
import torch_geometric.transforms as T


DATA_ROOT = os.path.join(os.path.dirname(__file__), 'data')


def generate_random_splits(data, num_classes, num_splits=10, 
                           train_per_class=20, val_per_class=30, seed_base=42):
    """Generate random splits with fixed number of vertices per class for train/val."""
    n = data.num_nodes
    labels = data.y.numpy()
    
    train_masks = []
    val_masks = []
    test_masks = []
    
    for i in range(num_splits):
        rng = np.random.RandomState(seed_base + i)
        train_mask = np.zeros(n, dtype=bool)
        val_mask = np.zeros(n, dtype=bool)
        test_mask = np.zeros(n, dtype=bool)
        
        for c in range(num_classes):
            idx = np.where(labels == c)[0]
            rng.shuffle(idx)
            
            n_train = min(train_per_class, len(idx))
            n_val = min(val_per_class, len(idx) - n_train)
            
            train_mask[idx[:n_train]] = True
            val_mask[idx[n_train:n_train + n_val]] = True
            test_mask[idx[n_train + n_val:]] = True
        
        train_masks.append(torch.tensor(train_mask))
        val_masks.append(torch.tensor(val_mask))
        test_masks.append(torch.tensor(test_mask))
    
    return train_masks, val_masks, test_masks


def generate_actor_splits(data, num_splits=10, seed_base=42,
                          train_ratio=0.48, val_ratio=0.32):
    """Generate Actor splits: 48%/32%/20%."""
    n = data.num_nodes
    train_masks = []
    val_masks = []
    test_masks = []
    
    for i in range(num_splits):
        rng = np.random.RandomState(seed_base + i)
        perm = rng.permutation(n)
        
        n_train = int(n * train_ratio)
        n_val = int(n * val_ratio)
        
        train_mask = np.zeros(n, dtype=bool)
        val_mask = np.zeros(n, dtype=bool)
        test_mask = np.zeros(n, dtype=bool)
        
        train_mask[perm[:n_train]] = True
        val_mask[perm[n_train:n_train + n_val]] = True
        test_mask[perm[n_train + n_val:]] = True
        
        train_masks.append(torch.tensor(train_mask))
        val_masks.append(torch.tensor(val_mask))
        test_masks.append(torch.tensor(test_mask))
    
    return train_masks, val_masks, test_masks


def make_inductive_split(data, train_mask, test_mask, unlabeled_ratio=0.8, seed=42):
    """
    Create inductive setting:
    - Only training vertices and edges between training vertices are available
    - For datasets with few training vertices per class (NOSMOG setup):
      Split test data into 80% unlabeled (used during training) and 20% actual test
    """
    rng = np.random.RandomState(seed)
    
    # Split test vertices into unlabeled and actual test
    test_indices = torch.where(test_mask)[0].numpy()
    rng.shuffle(test_indices)
    
    n_unlabeled = int(len(test_indices) * unlabeled_ratio)
    unlabeled_indices = test_indices[:n_unlabeled]
    actual_test_indices = test_indices[n_unlabeled:]
    
    # Create masks
    n = data.num_nodes
    inductive_train_mask = train_mask.clone()
    
    # Vertices available during training = train + unlabeled (but without labels)
    available_mask = torch.zeros(n, dtype=torch.bool)
    available_mask[torch.where(train_mask)[0]] = True
    available_mask[unlabeled_indices] = True
    
    # New test mask = only the 20% actual test vertices
    inductive_test_mask = torch.zeros(n, dtype=torch.bool)
    inductive_test_mask[actual_test_indices] = True
    
    # Filter edges: only keep edges where both endpoints are available
    edge_index = data.edge_index
    src, dst = edge_index[0], edge_index[1]
    mask = available_mask[src] & available_mask[dst]
    inductive_edge_index = edge_index[:, mask]
    
    return inductive_train_mask, inductive_test_mask, inductive_edge_index, available_mask


def load_dataset(name, root=DATA_ROOT):
    """Load a dataset by name and return data + metadata."""
    name_lower = name.lower()
    
    if name_lower in ['cora', 'citeseer', 'pubmed']:
        dataset = Planetoid(root=root, name=name, transform=None)
        data = dataset[0]
        num_classes = dataset.num_classes
        num_features = dataset.num_features
        # Generate custom splits
        train_masks, val_masks, test_masks = generate_random_splits(
            data, num_classes, num_splits=10, train_per_class=20, val_per_class=30
        )
        data.train_masks = train_masks
        data.val_masks = val_masks
        data.test_masks = test_masks
        return data, num_features, num_classes, 'vertex', 10
    
    elif name_lower in ['computers', 'photo']:
        dataset = Amazon(root=root, name=name)
        data = dataset[0]
        num_classes = dataset.num_classes
        num_features = dataset.num_features
        train_masks, val_masks, test_masks = generate_random_splits(
            data, num_classes, num_splits=10, train_per_class=20, val_per_class=30
        )
        data.train_masks = train_masks
        data.val_masks = val_masks
        data.test_masks = test_masks
        return data, num_features, num_classes, 'vertex', 10
    
    elif name_lower == 'ogb-arxiv' or name_lower == 'arxiv':
        dataset = PygNodePropPredDataset(name='ogbn-arxiv', root=root)
        data = dataset[0]
        data.y = data.y.squeeze()
        num_classes = dataset.num_classes
        num_features = data.x.shape[1]
        split_idx = dataset.get_idx_split()
        
        train_mask = torch.zeros(data.num_nodes, dtype=torch.bool)
        val_mask = torch.zeros(data.num_nodes, dtype=torch.bool)
        test_mask = torch.zeros(data.num_nodes, dtype=torch.bool)
        train_mask[split_idx['train']] = True
        val_mask[split_idx['valid']] = True
        test_mask[split_idx['test']] = True
        
        data.train_masks = [train_mask]
        data.val_masks = [val_mask]
        data.test_masks = [test_mask]
        return data, num_features, num_classes, 'vertex', 1
    
    elif name_lower == 'actor':
        dataset = ActorDataset(root=os.path.join(root, 'actor'))
        data = dataset[0]
        num_classes = dataset.num_classes
        num_features = dataset.num_features
        train_masks, val_masks, test_masks = generate_actor_splits(
            data, num_splits=10
        )
        data.train_masks = train_masks
        data.val_masks = val_masks
        data.test_masks = test_masks
        return data, num_features, num_classes, 'vertex', 10
    
    elif name_lower in ['chameleon', 'squirrel']:
        dataset = WikipediaNetwork(root=root, name=name_lower, 
                                   geom_gcn_preprocess=True)
        data = dataset[0]
        num_classes = dataset.num_classes
        num_features = dataset.num_features
        # Use provided splits (10 splits stored in data.train_mask etc.)
        if hasattr(data, 'train_mask') and data.train_mask.dim() == 2:
            n_splits = data.train_mask.shape[1]
            data.train_masks = [data.train_mask[:, i] for i in range(n_splits)]
            data.val_masks = [data.val_mask[:, i] for i in range(n_splits)]
            data.test_masks = [data.test_mask[:, i] for i in range(n_splits)]
        else:
            # Generate splits if not provided
            train_masks, val_masks, test_masks = generate_random_splits(
                data, num_classes, num_splits=10, 
                train_per_class=int(data.num_nodes * 0.48 / num_classes),
                val_per_class=int(data.num_nodes * 0.32 / num_classes)
            )
            data.train_masks = train_masks
            data.val_masks = val_masks
            data.test_masks = test_masks
        return data, num_features, num_classes, 'vertex', 10
    
    elif name_lower in ['minesweeper', 'roman-empire', 'roman_empire']:
        ds_name = 'Roman-empire' if 'roman' in name_lower else 'Minesweeper'
        dataset = HeterophilousGraphDataset(root=root, name=ds_name)
        data = dataset[0]
        num_classes = dataset.num_classes
        num_features = dataset.num_features
        # These have 10 provided splits
        if hasattr(data, 'train_mask') and data.train_mask.dim() == 2:
            n_splits = data.train_mask.shape[1]
            data.train_masks = [data.train_mask[:, i] for i in range(n_splits)]
            data.val_masks = [data.val_mask[:, i] for i in range(n_splits)]
            data.test_masks = [data.test_mask[:, i] for i in range(n_splits)]
        else:
            raise ValueError(f"Expected provided splits for {name}")
        return data, num_features, num_classes, 'vertex', 10
    
    elif name_lower == 'mnist':
        train_dataset = GNNBenchmarkDataset(root=root, name='MNIST', split='train')
        val_dataset = GNNBenchmarkDataset(root=root, name='MNIST', split='val')
        test_dataset = GNNBenchmarkDataset(root=root, name='MNIST', split='test')
        num_classes = train_dataset.num_classes
        num_features = train_dataset.num_features
        return (train_dataset, val_dataset, test_dataset), num_features, num_classes, 'graph', 1
    
    elif name_lower == 'cifar10':
        train_dataset = GNNBenchmarkDataset(root=root, name='CIFAR10', split='train')
        val_dataset = GNNBenchmarkDataset(root=root, name='CIFAR10', split='val')
        test_dataset = GNNBenchmarkDataset(root=root, name='CIFAR10', split='test')
        num_classes = train_dataset.num_classes
        num_features = train_dataset.num_features
        return (train_dataset, val_dataset, test_dataset), num_features, num_classes, 'graph', 1
    
    elif name_lower == 'zinc':
        train_dataset = ZINC(root=os.path.join(root, 'zinc'), subset=True, split='train')
        val_dataset = ZINC(root=os.path.join(root, 'zinc'), subset=True, split='val')
        test_dataset = ZINC(root=os.path.join(root, 'zinc'), subset=True, split='test')
        # ZINC: regression, 28 features (one-hot atom types)
        num_features = 28  # Will need to one-hot encode
        num_classes = 1  # regression
        return (train_dataset, val_dataset, test_dataset), num_features, num_classes, 'graph_regression', 1
    
    else:
        raise ValueError(f"Unknown dataset: {name}")


def preprocess_data(data, add_self_loop=True, make_undirected=True):
    """Preprocess graph data: add self-loops, make undirected."""
    if isinstance(data, tuple):
        # Graph-level datasets
        return data
    
    edge_index = data.edge_index
    
    if make_undirected:
        edge_index = to_undirected(edge_index)
    
    if add_self_loop:
        edge_index, _ = add_self_loops(edge_index, num_nodes=data.num_nodes)
    
    data.edge_index = edge_index
    return data


if __name__ == '__main__':
    # Quick test of dataset loading
    for name in ['Cora', 'CiteSeer', 'PubMed']:
        data, nf, nc, task, ns = load_dataset(name)
        print(f"{name}: {data.num_nodes} nodes, {data.num_edges} edges, "
              f"{nf} features, {nc} classes, {ns} splits")
    
    print("\nLoading Amazon datasets...")
    for name in ['Computers', 'Photo']:
        data, nf, nc, task, ns = load_dataset(name)
        print(f"{name}: {data.num_nodes} nodes, {data.num_edges} edges, "
              f"{nf} features, {nc} classes")
    
    print("\nLoading heterophilic datasets...")
    for name in ['Actor', 'Chameleon', 'Squirrel']:
        try:
            data, nf, nc, task, ns = load_dataset(name)
            print(f"{name}: {data.num_nodes} nodes, {data.num_edges} edges, "
                  f"{nf} features, {nc} classes")
        except Exception as e:
            print(f"{name}: Error - {e}")
    
    for name in ['Minesweeper', 'Roman-Empire']:
        try:
            data, nf, nc, task, ns = load_dataset(name)
            print(f"{name}: {data.num_nodes} nodes, {data.num_edges} edges, "
                  f"{nf} features, {nc} classes")
        except Exception as e:
            print(f"{name}: Error - {e}")
    
    print("\nLoading graph-level datasets...")
    for name in ['ZINC']:
        try:
            datasets, nf, nc, task, ns = load_dataset(name)
            train_ds, val_ds, test_ds = datasets
            print(f"{name}: {len(train_ds)}/{len(val_ds)}/{len(test_ds)} graphs, "
                  f"{nf} features, task={task}")
        except Exception as e:
            print(f"{name}: Error - {e}")
