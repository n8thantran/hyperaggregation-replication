"""
HyperAggregation models: GHC (GraphHyperConv), GHM (GraphHyperMixer), and baselines.

Based on: "HyperAggregation: Aggregating over Graph Edges with Hypernetworks"

Key equations:
  W_tar = GeLU(X @ W_A) @ W_B          # Hypernetwork predicts target weights
  HA(X) = (GeLU(X^T @ W_tar) @ W_tar^T)^T  # Target network does channel mixing
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.nn import MessagePassing, global_mean_pool
from torch_geometric.utils import add_self_loops, to_undirected
from torch_scatter import scatter_mean, scatter_add


class HyperAggregation(nn.Module):
    """
    HyperAggregation module (non-batched, for GHM).
    
    Hypernetwork: W_tar = GeLU(X @ W_A) @ W_B
    Target network: HA(X) = (GeLU(X^T @ W_tar) @ W_tar^T)^T
    """
    def __init__(self, hidden_dim, mix_dim, mix_dropout=0.0,
                 trans_input=False, trans_output=False,
                 input_activation=False, dropout=0.0):
        super().__init__()
        self.hidden_dim = hidden_dim
        self.mix_dim = mix_dim
        self.mix_dropout = mix_dropout
        self.trans_input = trans_input
        self.trans_output = trans_output
        self.input_activation = input_activation
        
        # Hypernetwork weights
        self.W_A = nn.Linear(hidden_dim, hidden_dim, bias=False)
        self.W_B = nn.Linear(hidden_dim, mix_dim, bias=False)
        
        # Optional transformations
        if trans_input:
            self.ln_input = nn.LayerNorm(hidden_dim)
            self.drop_input = nn.Dropout(dropout)
        if trans_output:
            self.ln_output = nn.LayerNorm(hidden_dim)
            self.drop_output = nn.Dropout(dropout)
    
    def forward(self, X):
        """
        X: (N, h) - neighborhood features for one vertex, or batched
        Returns: (N, h) - aggregated features
        """
        # Optional input activation
        if self.input_activation:
            X_hyper = F.gelu(X)
        else:
            X_hyper = X
        
        # Hypernetwork: predict target weights
        # W_tar = GeLU(X @ W_A) @ W_B -> (N, m)
        W_tar = self.W_B(F.gelu(self.W_A(X_hyper)))
        
        # Optional: transform before target network
        if self.trans_input:
            X_target = self.drop_input(self.ln_input(X))
        else:
            X_target = X
        
        # Apply mixing dropout
        if self.training and self.mix_dropout > 0:
            X_target = F.dropout(X_target, p=self.mix_dropout, training=True)
        
        # Target network: HA(X) = (GeLU(X^T @ W_tar) @ W_tar^T)^T
        # X^T: (h, N), W_tar: (N, m) -> (h, m)
        out = F.gelu(X_target.t() @ W_tar)  # (h, m)
        out = out @ W_tar.t()  # (h, N)
        out = out.t()  # (N, h)
        
        # Optional: transform after target network
        if self.trans_output:
            out = self.drop_output(self.ln_output(out))
        
        return out


class HyperAggregationBatched(nn.Module):
    """
    Batched HyperAggregation for GHC - operates on neighborhoods defined by edge_index.
    Uses scatter operations for efficient batched computation.
    Memory-efficient: processes mix_dim in chunks to avoid (E, h, m) tensors.
    """
    def __init__(self, hidden_dim, mix_dim, mix_dropout=0.0,
                 trans_input=False, trans_output=False,
                 input_activation=False, dropout=0.0):
        super().__init__()
        self.hidden_dim = hidden_dim
        self.mix_dim = mix_dim
        self.mix_dropout = mix_dropout
        self.trans_input = trans_input
        self.trans_output = trans_output
        self.input_activation = input_activation
        
        # Hypernetwork weights
        self.W_A = nn.Linear(hidden_dim, hidden_dim, bias=False)
        self.W_B = nn.Linear(hidden_dim, mix_dim, bias=False)
        
        # Optional transformations
        if trans_input:
            self.ln_input = nn.LayerNorm(hidden_dim)
            self.drop_input = nn.Dropout(dropout)
        if trans_output:
            self.ln_output = nn.LayerNorm(hidden_dim)
            self.drop_output = nn.Dropout(dropout)
    
    def compute_W_tar(self, X):
        """Compute target weights from features."""
        if self.input_activation:
            X_hyper = F.gelu(X)
        else:
            X_hyper = X
        return self.W_B(F.gelu(self.W_A(X_hyper)))  # (N, m)
    
    def forward(self, X, edge_index, num_nodes, mean_agg=True):
        """
        X: (num_nodes, h) - all node features
        edge_index: (2, E) - edges (src -> dst means src is neighbor of dst)
        num_nodes: number of nodes
        mean_agg: if True, return mean of per-edge outputs; if False, return root readout
        
        Returns: (num_nodes, h) - per-node output
        """
        src, dst = edge_index[0], edge_index[1]
        E = src.size(0)
        h = self.hidden_dim
        m = self.mix_dim
        
        # Get source node features (neighbors)
        X_src = X[src]  # (E, h)
        
        # Hypernetwork: W_tar per edge
        W_tar_edge = self.compute_W_tar(X_src)  # (E, m)
        
        # For target network input
        if self.trans_input:
            X_target = self.drop_input(self.ln_input(X_src))
        else:
            X_target = X_src
        
        if self.training and self.mix_dropout > 0:
            X_target = F.dropout(X_target, p=self.mix_dropout, training=True)
        
        # Memory-efficient: compute agg[:, :, i] = scatter_add(X_target * W_tar_edge[:, i:i+1], dst)
        # Process in chunks to manage memory
        # For small E*h*m, use the fast path; for large, use chunked
        mem_estimate = E * h * m * 4  # bytes for float32
        
        dst_idx = dst.unsqueeze(1).expand(-1, h)  # (E, h) - reused for all chunks
        
        if mem_estimate < 2e9:  # Less than 2GB - use fast full materialization
            cross = X_target.unsqueeze(2) * W_tar_edge.unsqueeze(1)  # (E, h, m)
            agg = torch.zeros(num_nodes, h, m, device=X.device, dtype=X.dtype)
            agg.scatter_add_(0, dst.unsqueeze(1).unsqueeze(2).expand_as(cross), cross)
        else:
            # Chunked computation: process chunk_size mix dims at a time
            chunk_size = max(1, int(1.5e9 / (E * h * 4)))  # target ~1.5GB per chunk
            agg = torch.zeros(num_nodes, h, m, device=X.device, dtype=X.dtype)
            for start in range(0, m, chunk_size):
                end = min(start + chunk_size, m)
                cs = end - start
                W_chunk = W_tar_edge[:, start:end]  # (E, cs)
                cross_chunk = X_target.unsqueeze(2) * W_chunk.unsqueeze(1)  # (E, h, cs)
                agg[:, :, start:end].scatter_add_(
                    0, dst.unsqueeze(1).unsqueeze(2).expand(-1, h, cs), cross_chunk
                )
                del cross_chunk, W_chunk
        
        # Apply GeLU: GeLU(X^T @ W_tar) 
        agg = F.gelu(agg)  # (num_nodes, h, m)
        
        if mean_agg:
            # Step 2: For each edge, compute agg[dst] @ W_tar[edge] -> per-edge output
            # Memory-efficient: chunk over edges
            if E * h * m * 4 < 2e9:
                agg_dst = agg[dst]  # (E, h, m)
                out_edge = (agg_dst * W_tar_edge.unsqueeze(1)).sum(dim=2)  # (E, h)
            else:
                # Process in edge chunks
                edge_chunk = max(1, int(1.5e9 / (h * m * 4)))
                out_edge = torch.zeros(E, h, device=X.device, dtype=X.dtype)
                for start in range(0, E, edge_chunk):
                    end = min(start + edge_chunk, E)
                    agg_chunk = agg[dst[start:end]]  # (chunk, h, m)
                    w_chunk = W_tar_edge[start:end]  # (chunk, m)
                    out_edge[start:end] = (agg_chunk * w_chunk.unsqueeze(1)).sum(dim=2)
                    del agg_chunk, w_chunk
            
            # Optional: transform after target network
            if self.trans_output:
                out_edge = self.drop_output(self.ln_output(out_edge))
            
            # Mean pool per-edge outputs
            out = scatter_mean(out_edge, dst, dim=0, dim_size=num_nodes)  # (num_nodes, h)
        else:
            # Root vertex readout: use the ROOT vertex's own W_tar to read from agg
            W_tar_root = self.compute_W_tar(X)  # (num_nodes, m)
            
            # out[v] = agg[v] @ W_tar_root[v] -> (h,)
            out = (agg * W_tar_root.unsqueeze(1)).sum(dim=2)  # (num_nodes, h)
            
            # Optional: transform after target network
            if self.trans_output:
                out = self.drop_output(self.ln_output(out))
        
        return out


class GHCBlock(nn.Module):
    """
    GraphHyperConv block: FF -> HA(A) -> FF
    
    After HA, either use root vertex embedding or mean-pool over neighborhood.
    Optional root connection: concat root embedding with HA output.
    Optional residual connection.
    """
    def __init__(self, in_dim, hidden_dim, out_dim, mix_dim,
                 mix_dropout=0.0, dropout=0.0,
                 trans_input=False, trans_output=False,
                 input_activation=False,
                 mean_agg=True, root_conn=True, residual=False):
        super().__init__()
        self.mean_agg = mean_agg
        self.root_conn = root_conn
        self.residual = residual
        self.dropout = dropout
        
        # First FF layer
        self.ff1 = nn.Linear(in_dim, hidden_dim)
        
        # HyperAggregation
        self.ha = HyperAggregationBatched(
            hidden_dim, mix_dim, mix_dropout=mix_dropout,
            trans_input=trans_input, trans_output=trans_output,
            input_activation=input_activation, dropout=dropout
        )
        
        # Second FF layer
        ff2_in = hidden_dim * 2 if root_conn else hidden_dim
        self.ff2 = nn.Linear(ff2_in, out_dim)
        
        self.drop = nn.Dropout(dropout)
        
        # Residual projection if dimensions don't match
        if residual and in_dim != out_dim:
            self.res_proj = nn.Linear(in_dim, out_dim)
        else:
            self.res_proj = None
    
    def forward(self, X, edge_index):
        """
        X: (num_nodes, in_dim)
        edge_index: (2, E)
        Returns: (num_nodes, out_dim)
        """
        num_nodes = X.size(0)
        residual = X
        
        # First FF
        H = self.drop(F.gelu(self.ff1(X)))  # (num_nodes, hidden_dim)
        
        # HyperAggregation - returns per-node output directly
        agg = self.ha(H, edge_index, num_nodes, mean_agg=self.mean_agg)  # (num_nodes, hidden_dim)
        
        # Root connection: concat root embedding (from H) with aggregated
        if self.root_conn:
            agg = torch.cat([agg, H], dim=1)  # (num_nodes, hidden_dim * 2)
            agg = F.gelu(agg)
        
        # Second FF
        out = self.drop(self.ff2(agg))  # (num_nodes, out_dim)
        
        # Residual
        if self.residual:
            if self.res_proj is not None:
                residual = self.res_proj(residual)
            out = out + residual
        
        return out


class GHC(nn.Module):
    """
    GraphHyperConv model.
    
    Multiple GHC blocks followed by a classification head.
    For graph-level tasks, adds mean pooling before the head.
    """
    def __init__(self, in_dim, hidden_dim, out_dim, num_blocks=2,
                 mix_dim=64, mix_dropout=0.0, dropout=0.0, input_dropout=0.0,
                 trans_input=False, trans_output=False,
                 input_activation=False,
                 mean_agg=True, root_conn=True, residual=False,
                 add_self_loop=True, make_undirected=False,
                 task='vertex', normalize_input=False,
                 use_embedding=False, num_embeddings=28):
        super().__init__()
        self.add_self_loop = add_self_loop
        self.make_undirected = make_undirected
        self.task = task
        self.normalize_input = normalize_input
        self.use_embedding = use_embedding
        self.input_dropout = nn.Dropout(input_dropout)
        
        if use_embedding:
            self.embedding = nn.Embedding(num_embeddings, hidden_dim)
            in_dim = hidden_dim
        
        if normalize_input:
            self.input_norm = nn.LayerNorm(in_dim)
        
        self.blocks = nn.ModuleList()
        for i in range(num_blocks):
            block_in = in_dim if i == 0 else hidden_dim
            block_out = hidden_dim
            self.blocks.append(GHCBlock(
                block_in, hidden_dim, block_out, mix_dim,
                mix_dropout=mix_dropout, dropout=dropout,
                trans_input=trans_input, trans_output=trans_output,
                input_activation=input_activation,
                mean_agg=mean_agg, root_conn=root_conn, residual=residual
            ))
        
        self.head = nn.Linear(hidden_dim, out_dim)
    
    def forward(self, x, edge_index, batch=None):
        # Embedding for integer features (e.g., ZINC atom types)
        if self.use_embedding:
            x = self.embedding(x.squeeze(-1).long())
        
        # Preprocess edges
        if self.make_undirected:
            edge_index = to_undirected(edge_index)
        if self.add_self_loop:
            edge_index, _ = add_self_loops(edge_index, num_nodes=x.size(0))
        
        if self.normalize_input:
            x = self.input_norm(x)
        x = self.input_dropout(x)
        
        for block in self.blocks:
            x = block(x, edge_index)
        
        if self.task in ['graph', 'graph_regression']:
            x = global_mean_pool(x, batch)
        
        return self.head(x)


class GHMBlock(nn.Module):
    """
    GraphHyperMixer block: FF -> HA -> FF
    Treats neighborhood as fully connected.
    """
    def __init__(self, in_dim, hidden_dim, out_dim, mix_dim,
                 mix_dropout=0.0, dropout=0.0,
                 trans_input=False, trans_output=False,
                 input_activation=False,
                 root_conn=True, residual=False):
        super().__init__()
        self.root_conn = root_conn
        self.residual = residual
        self.dropout = dropout
        
        self.ff1 = nn.Linear(in_dim, hidden_dim)
        
        self.ha = HyperAggregation(
            hidden_dim, mix_dim, mix_dropout=mix_dropout,
            trans_input=trans_input, trans_output=trans_output,
            input_activation=input_activation, dropout=dropout
        )
        
        ff2_in = hidden_dim * 2 if root_conn else hidden_dim
        self.ff2 = nn.Linear(ff2_in, out_dim)
        
        self.drop = nn.Dropout(dropout)
        
        if residual and in_dim != out_dim:
            self.res_proj = nn.Linear(in_dim, out_dim)
        else:
            self.res_proj = None
    
    def forward(self, X):
        """
        X: (N, in_dim) - neighborhood features (fully connected assumption)
        Returns: (N, out_dim)
        """
        residual = X
        
        H = self.drop(F.gelu(self.ff1(X)))  # (N, hidden_dim)
        
        # HA on fully connected neighborhood
        agg = self.ha(H)  # (N, hidden_dim)
        
        if self.root_conn:
            # Concat root (original H) with aggregated
            agg = torch.cat([agg, H], dim=1)
            agg = F.gelu(agg)
        
        out = self.drop(self.ff2(agg))
        
        if self.residual:
            if self.res_proj is not None:
                residual = self.res_proj(residual)
            out = out + residual
        
        return out


class GHM(nn.Module):
    """
    GraphHyperMixer model.
    
    Samples k-hop neighborhood, applies GHM blocks, then classifies root vertex.
    For vertex-level tasks in transductive setting.
    """
    def __init__(self, in_dim, hidden_dim, out_dim, num_blocks=2,
                 mix_dim=64, mix_dropout=0.0, dropout=0.0, input_dropout=0.0,
                 trans_input=False, trans_output=False,
                 input_activation=False,
                 root_conn=True, residual=False,
                 k_hop=2, task='vertex', normalize_input=False):
        super().__init__()
        self.k_hop = k_hop
        self.task = task
        self.normalize_input = normalize_input
        self.input_dropout = nn.Dropout(input_dropout)
        
        if normalize_input:
            self.input_norm = nn.LayerNorm(in_dim)
        
        self.blocks = nn.ModuleList()
        for i in range(num_blocks):
            block_in = in_dim if i == 0 else hidden_dim
            self.blocks.append(GHMBlock(
                block_in, hidden_dim, hidden_dim, mix_dim,
                mix_dropout=mix_dropout, dropout=dropout,
                trans_input=trans_input, trans_output=trans_output,
                input_activation=input_activation,
                root_conn=root_conn, residual=residual
            ))
        
        self.head = nn.Linear(hidden_dim, out_dim)
    
    def get_k_hop_neighbors(self, node_idx, edge_index, num_nodes, k):
        """Get k-hop neighborhood of a node."""
        neighbors = {node_idx}
        current = {node_idx}
        
        src, dst = edge_index[0].cpu().numpy(), edge_index[1].cpu().numpy()
        
        # Build adjacency list
        from collections import defaultdict
        adj = defaultdict(set)
        for s, d in zip(src, dst):
            adj[d].add(s)  # neighbors of d include s
            adj[s].add(d)  # undirected
        
        for _ in range(k):
            next_nodes = set()
            for n in current:
                next_nodes.update(adj[n])
            current = next_nodes - neighbors
            neighbors.update(current)
        
        return sorted(neighbors)
    
    def forward(self, x, edge_index, batch=None, node_indices=None):
        """
        For vertex-level tasks:
            Process each node's k-hop neighborhood independently.
            node_indices: which nodes to classify (if None, all nodes)
        For graph-level tasks:
            Process each graph as a fully connected neighborhood.
        """
        if self.normalize_input:
            x = self.input_norm(x)
        x = self.input_dropout(x)
        
        if self.task in ['graph', 'graph_regression']:
            # Each graph is treated as a fully connected neighborhood
            if batch is None:
                # Single graph
                for block in self.blocks:
                    x = block(x)
                x = x.mean(dim=0, keepdim=True)
                return self.head(x)
            else:
                # Batched graphs
                outputs = []
                unique_batches = batch.unique()
                for b in unique_batches:
                    mask = batch == b
                    x_b = x[mask]
                    for block in self.blocks:
                        x_b = block(x_b)
                    outputs.append(x_b.mean(dim=0))
                x = torch.stack(outputs)
                return self.head(x)
        else:
            # Vertex-level: process neighborhoods
            if node_indices is None:
                node_indices = torch.arange(x.size(0), device=x.device)
            
            # Build k-hop neighborhoods and process
            all_outputs = torch.zeros(x.size(0), self.head.in_features, device=x.device)
            
            for idx in node_indices:
                neighbors = self.get_k_hop_neighbors(idx.item(), edge_index, x.size(0), self.k_hop)
                x_neigh = x[neighbors]  # (|N|, in_dim)
                
                for block in self.blocks:
                    x_neigh = block(x_neigh)
                
                # Find root vertex position in neighborhood
                root_pos = neighbors.index(idx.item())
                all_outputs[idx] = x_neigh[root_pos]
            
            return self.head(all_outputs[node_indices])


class GCNLayer(MessagePassing):
    """Simple GCN layer."""
    def __init__(self, in_dim, out_dim):
        super().__init__(aggr='add')
        self.linear = nn.Linear(in_dim, out_dim)
    
    def forward(self, x, edge_index):
        # Add self-loops
        edge_index, _ = add_self_loops(edge_index, num_nodes=x.size(0))
        
        # Compute normalization
        row, col = edge_index
        deg = torch.zeros(x.size(0), device=x.device)
        deg.scatter_add_(0, row, torch.ones(row.size(0), device=x.device))
        deg_inv_sqrt = deg.pow(-0.5)
        deg_inv_sqrt[deg_inv_sqrt == float('inf')] = 0
        norm = deg_inv_sqrt[row] * deg_inv_sqrt[col]
        
        x = self.linear(x)
        return self.propagate(edge_index, x=x, norm=norm)
    
    def message(self, x_j, norm):
        return norm.view(-1, 1) * x_j


class GCN(nn.Module):
    """GCN baseline model."""
    def __init__(self, in_dim, hidden_dim, out_dim, num_layers=2,
                 dropout=0.5, input_dropout=0.0, task='vertex'):
        super().__init__()
        self.task = task
        self.dropout = dropout
        self.input_dropout = nn.Dropout(input_dropout)
        
        self.layers = nn.ModuleList()
        for i in range(num_layers):
            layer_in = in_dim if i == 0 else hidden_dim
            layer_out = hidden_dim
            self.layers.append(GCNLayer(layer_in, layer_out))
        
        self.head = nn.Linear(hidden_dim, out_dim)
    
    def forward(self, x, edge_index, batch=None):
        x = self.input_dropout(x)
        
        for layer in self.layers:
            x = F.relu(layer(x, edge_index))
            x = F.dropout(x, p=self.dropout, training=self.training)
        
        if self.task in ['graph', 'graph_regression']:
            x = global_mean_pool(x, batch)
        
        return self.head(x)


class MLP(nn.Module):
    """MLP baseline model."""
    def __init__(self, in_dim, hidden_dim, out_dim, num_layers=2,
                 dropout=0.5, input_dropout=0.0, task='vertex'):
        super().__init__()
        self.task = task
        self.dropout = dropout
        self.input_dropout = nn.Dropout(input_dropout)
        
        layers = []
        for i in range(num_layers):
            layer_in = in_dim if i == 0 else hidden_dim
            layers.append(nn.Linear(layer_in, hidden_dim))
            layers.append(nn.ReLU())
            layers.append(nn.Dropout(dropout))
        self.layers = nn.Sequential(*layers)
        self.head = nn.Linear(hidden_dim, out_dim)
    
    def forward(self, x, edge_index=None, batch=None):
        x = self.input_dropout(x)
        x = self.layers(x)
        
        if self.task in ['graph', 'graph_regression']:
            x = global_mean_pool(x, batch)
        
        return self.head(x)


if __name__ == '__main__':
    # Quick sanity tests
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    
    # Test HyperAggregation
    print("Testing HyperAggregation...")
    ha = HyperAggregation(64, 32).to(device)
    X = torch.randn(10, 64).to(device)
    out = ha(X)
    print(f"  Input: {X.shape}, Output: {out.shape}")
    assert out.shape == X.shape
    
    # Test HyperAggregationBatched (mean_agg=True)
    print("Testing HyperAggregationBatched (mean_agg)...")
    hab = HyperAggregationBatched(64, 32).to(device)
    edge_index = torch.tensor([[0,1,2,1,0,1,2,3,4], 
                                [1,0,1,2,0,1,2,3,4]], dtype=torch.long).to(device)
    X = torch.randn(5, 64).to(device)
    out = hab(X, edge_index, 5, mean_agg=True)
    print(f"  Input: {X.shape}, Edges: {edge_index.shape}, Output: {out.shape}")
    assert out.shape == (5, 64)
    
    # Test HyperAggregationBatched (root readout, mean_agg=False)
    print("Testing HyperAggregationBatched (root readout)...")
    out = hab(X, edge_index, 5, mean_agg=False)
    print(f"  Input: {X.shape}, Edges: {edge_index.shape}, Output: {out.shape}")
    assert out.shape == (5, 64)
    
    # Test GHC (mean_agg=True, with self-loops)
    print("Testing GHC (mean_agg=True)...")
    model = GHC(100, 64, 7, num_blocks=2, mix_dim=32, mean_agg=True, add_self_loop=True).to(device)
    X = torch.randn(50, 100).to(device)
    edge_index = torch.randint(0, 50, (2, 200)).to(device)
    out = model(X, edge_index)
    print(f"  Input: {X.shape}, Output: {out.shape}")
    assert out.shape == (50, 7)
    
    # Test memory-efficient path with large graph
    print("Testing HyperAggregationBatched memory-efficient path...")
    hab2 = HyperAggregationBatched(256, 128).to(device)
    X2 = torch.randn(5000, 256).to(device)
    edge_index2 = torch.randint(0, 5000, (2, 300000)).to(device)
    out2 = hab2(X2, edge_index2, 5000, mean_agg=True)
    print(f"  Large graph test passed: {out2.shape}")
    assert out2.shape == (5000, 256)
    del hab2, X2, edge_index2, out2
    torch.cuda.empty_cache()
    
    # Test GHC for graph-level
    print("Testing GHC graph-level...")
    model = GHC(28, 64, 1, num_blocks=2, mix_dim=32, task='graph_regression').to(device)
    X = torch.randn(23, 28).to(device)
    edge_index = torch.randint(0, 23, (2, 50)).to(device)
    batch = torch.zeros(23, dtype=torch.long).to(device)
    out = model(X, edge_index, batch)
    print(f"  Input: {X.shape}, Output: {out.shape}")
    assert out.shape == (1, 1)
    
    # Test GCN
    print("Testing GCN...")
    model = GCN(100, 64, 7).to(device)
    X = torch.randn(50, 100).to(device)
    edge_index = torch.randint(0, 50, (2, 200)).to(device)
    out = model(X, edge_index)
    print(f"  Input: {X.shape}, Output: {out.shape}")
    assert out.shape == (50, 7)
    
    # Test MLP
    print("Testing MLP...")
    model = MLP(100, 64, 7).to(device)
    X = torch.randn(50, 100).to(device)
    out = model(X)
    print(f"  Input: {X.shape}, Output: {out.shape}")
    assert out.shape == (50, 7)
    
    print("\nAll tests passed!")
