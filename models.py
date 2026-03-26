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
    HyperAggregation module.
    
    Hypernetwork: W_tar = GeLU(X @ W_A) @ W_B
    Target network: HA(X) = (GeLU(X^T @ W_tar) @ W_tar^T)^T
    
    Args:
        hidden_dim: embedding dimension h
        mix_dim: mixing dimension m
        mix_dropout: dropout on mixing (applied to embeddings input to target network)
        trans_input: apply LayerNorm + dropout before target network
        trans_output: apply LayerNorm + dropout after target network
        input_activation: apply GeLU to input before hypernetwork
        dropout: dropout rate for trans_input/trans_output
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
    
    def forward(self, X, edge_index, num_nodes):
        """
        X: (num_nodes, h) - all node features
        edge_index: (2, E) - edges (src -> dst means src is neighbor of dst)
        num_nodes: number of nodes
        
        For each target node v, we need to:
        1. Gather neighborhood features X_N(v)
        2. Compute W_tar for each neighborhood
        3. Apply target network per neighborhood
        
        This is done efficiently using scatter operations.
        """
        src, dst = edge_index[0], edge_index[1]
        
        # Get source node features (neighbors)
        X_src = X[src]  # (E, h)
        
        # Optional input activation
        if self.input_activation:
            X_hyper = F.gelu(X_src)
        else:
            X_hyper = X_src
        
        # Hypernetwork: W_tar per edge
        # W_tar_edge = GeLU(X_src @ W_A) @ W_B -> (E, m)
        W_tar_edge = self.W_B(F.gelu(self.W_A(X_hyper)))  # (E, m)
        
        # For target network input
        if self.trans_input:
            X_target = self.drop_input(self.ln_input(X_src))
        else:
            X_target = X_src
        
        if self.training and self.mix_dropout > 0:
            X_target = F.dropout(X_target, p=self.mix_dropout, training=True)
        
        # Target network per neighborhood:
        # For each dst node v:
        #   step1: X_N(v)^T @ W_tar_N(v) -> (h, m)  [sum over neighbors]
        #   step2: GeLU(step1) @ W_tar_N(v)^T -> (h, |N(v)|)
        #   But we need per-edge outputs, so we do it differently.
        
        # Step 1: For each dst, compute sum_j X_j * W_tar_j^T for each mix dim
        # This is: for each dst v, sum over src j in N(v): X_target[j] outer W_tar[j]
        # = scatter_add of X_target[e] * W_tar[e] grouped by dst
        # Result shape: (num_nodes, h, m)
        
        # Efficient: compute X_target (E,h,1) * W_tar (E,1,m) -> (E, h, m)
        # Then scatter_add by dst -> (num_nodes, h, m)
        cross = X_target.unsqueeze(2) * W_tar_edge.unsqueeze(1)  # (E, h, m)
        agg = torch.zeros(num_nodes, self.hidden_dim, self.mix_dim, 
                         device=X.device, dtype=X.dtype)
        agg.scatter_add_(0, dst.unsqueeze(1).unsqueeze(2).expand_as(cross), cross)
        
        # Apply GeLU
        agg = F.gelu(agg)  # (num_nodes, h, m)
        
        # Step 2: For each edge (src->dst), compute agg[dst] @ W_tar[edge]
        # agg[dst]: (h, m), W_tar[edge]: (m,) -> (h,)
        # This gives per-edge output
        agg_dst = agg[dst]  # (E, h, m)
        out_edge = (agg_dst * W_tar_edge.unsqueeze(1)).sum(dim=2)  # (E, h)
        
        # Optional: transform after target network
        if self.trans_output:
            out_edge = self.drop_output(self.ln_output(out_edge))
        
        return out_edge, src, dst


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
        
        # HyperAggregation
        out_edge, src, dst = self.ha(H, edge_index, num_nodes)  # (E, hidden_dim)
        
        # Aggregate: mean or root
        if self.mean_agg:
            # Mean pool over neighborhood for each dst node
            agg = scatter_mean(out_edge, dst, dim=0, dim_size=num_nodes)  # (num_nodes, hidden_dim)
        else:
            # Use root vertex embedding (self-loop edge)
            # The root vertex is the one where src == dst
            # But with self-loops, we can just use the output for self-loop edges
            # Simpler: just use mean (which includes self-loop)
            # Actually, "root vertex embedding" means we pick the embedding of v itself after HA
            # With self-loops, the self-loop edge gives us the root's contribution
            # Let's use scatter to get the self-loop contribution
            # But it's mixed with neighbor contributions in the target network
            # The paper says "either the embedding of the root vertex or the mean"
            # I think "root vertex embedding" means: from the HA output, take only the 
            # row corresponding to the root vertex (which is the self-loop edge output)
            
            # Find self-loop edges
            self_loop_mask = (src == dst)
            if self_loop_mask.any():
                # For nodes with self-loops, use the self-loop edge output
                self_loop_dst = dst[self_loop_mask]
                self_loop_out = out_edge[self_loop_mask]
                agg = torch.zeros(num_nodes, out_edge.size(1), device=X.device, dtype=X.dtype)
                agg[self_loop_dst] = self_loop_out
            else:
                # Fallback to mean if no self-loops
                agg = scatter_mean(out_edge, dst, dim=0, dim_size=num_nodes)
        
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
                 task='vertex', normalize_input=False):
        super().__init__()
        self.add_self_loop = add_self_loop
        self.make_undirected = make_undirected
        self.task = task
        self.normalize_input = normalize_input
        self.input_dropout = nn.Dropout(input_dropout)
        
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
            # For efficiency, we process all nodes' neighborhoods
            # Using the GHC-style batched approach but treating as fully connected
            # Actually, for GHM we need to sample k-hop neighborhoods
            # This is expensive per-node. Let's use a simpler approach:
            # Process the whole graph through blocks (like GHC but fully connected within k-hop)
            
            # For simplicity and efficiency, use the same approach as GHC
            # but with k-hop expanded edge_index (fully connected within k-hop)
            if node_indices is None:
                node_indices = torch.arange(x.size(0), device=x.device)
            
            # Build k-hop neighborhoods and process
            # For efficiency, process all at once using batched neighborhoods
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
    
    # Test HyperAggregationBatched
    print("Testing HyperAggregationBatched...")
    hab = HyperAggregationBatched(64, 32).to(device)
    edge_index = torch.tensor([[0,1,2,1,0,1,2,3,4], 
                                [1,0,1,2,0,1,2,3,4]], dtype=torch.long).to(device)
    X = torch.randn(5, 64).to(device)
    out_edge, src, dst = hab(X, edge_index, 5)
    print(f"  Input: {X.shape}, Edges: {edge_index.shape}, Output: {out_edge.shape}")
    
    # Test GHC
    print("Testing GHC...")
    model = GHC(100, 64, 7, num_blocks=2, mix_dim=32).to(device)
    X = torch.randn(50, 100).to(device)
    edge_index = torch.randint(0, 50, (2, 200)).to(device)
    out = model(X, edge_index)
    print(f"  Input: {X.shape}, Output: {out.shape}")
    assert out.shape == (50, 7)
    
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
