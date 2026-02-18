"""
Graph Neural Networks - Advanced relational reasoning and network analysis
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Dict, List, Any, Tuple, Optional
import numpy as np
import logging

logger = logging.getLogger(__name__)


class GraphAttentionLayer(nn.Module):
    """
    Graph Attention Network (GAT) layer
    Learns importance of neighboring nodes dynamically
    """
    
    def __init__(self, in_features: int, out_features: int, n_heads: int = 8, dropout: float = 0.2):
        super().__init__()
        
        self.n_heads = n_heads
        self.out_features = out_features
        
        # Multi-head attention
        self.W = nn.Parameter(torch.zeros(n_heads, in_features, out_features))
        self.a = nn.Parameter(torch.zeros(n_heads, 2 * out_features, 1))
        
        self.leakyrelu = nn.LeakyReLU(0.2)
        self.dropout = nn.Dropout(dropout)
        
        nn.init.xavier_uniform_(self.W)
        nn.init.xavier_uniform_(self.a)
    
    def forward(self, x: torch.Tensor, adj: torch.Tensor) -> torch.Tensor:
        """
        x: Node features (N, in_features)
        adj: Adjacency matrix (N, N)
        """
        N = x.size(0)
        
        # Multi-head transformation
        h = torch.stack([torch.mm(x, self.W[i]) for i in range(self.n_heads)])  # (n_heads, N, out_features)
        
        # Attention mechanism
        attention_scores = []
        for i in range(self.n_heads):
            # Concatenate features for all pairs
            h_i = h[i]  # (N, out_features)
            a_input = torch.cat([
                h_i.repeat(1, N).view(N * N, -1),
                h_i.repeat(N, 1)
            ], dim=1).view(N, N, 2 * self.out_features)
            
            # Compute attention scores
            e = self.leakyrelu(torch.matmul(a_input, self.a[i]).squeeze(-1))
            
            # Mask with adjacency
            e = e.masked_fill(adj == 0, float('-inf'))
            
            # Softmax
            alpha = F.softmax(e, dim=1)
            alpha = self.dropout(alpha)
            
            attention_scores.append(alpha)
        
        # Aggregate
        attention_scores = torch.stack(attention_scores)  # (n_heads, N, N)
        h_prime = torch.stack([
            torch.mm(attention_scores[i], h[i]) for i in range(self.n_heads)
        ])  # (n_heads, N, out_features)
        
        # Concatenate or average heads
        h_prime = h_prime.mean(dim=0)  # (N, out_features)
        
        return F.elu(h_prime)


class GraphConvolutionalNetwork(nn.Module):
    """
    Graph Convolutional Network for relational reasoning
    """
    
    def __init__(self, input_dim: int, hidden_dims: List[int], output_dim: int):
        super().__init__()
        
        self.layers = nn.ModuleList()
        
        prev_dim = input_dim
        for hidden_dim in hidden_dims:
            self.layers.append(GraphAttentionLayer(prev_dim, hidden_dim))
            prev_dim = hidden_dim
        
        self.output_layer = nn.Linear(prev_dim, output_dim)
    
    def forward(self, x: torch.Tensor, adj: torch.Tensor) -> torch.Tensor:
        """
        x: Node features (N, input_dim)
        adj: Adjacency matrix (N, N)
        """
        for layer in self.layers:
            x = layer(x, adj)
        
        return self.output_layer(x)


class TemporalGraphNetwork(nn.Module):
    """
    Temporal Graph Network for dynamic relationship modeling
    """
    
    def __init__(self, node_dim: int, edge_dim: int, hidden_dim: int = 128):
        super().__init__()
        
        self.node_encoder = nn.Sequential(
            nn.Linear(node_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim)
        )
        
        self.edge_encoder = nn.Sequential(
            nn.Linear(edge_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim)
        )
        
        self.temporal_encoder = nn.LSTM(hidden_dim, hidden_dim, batch_first=True)
        
        self.message_passing = nn.Sequential(
            nn.Linear(hidden_dim * 3, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim)
        )
    
    def forward(
        self, 
        node_features: torch.Tensor,
        edge_features: torch.Tensor,
        edge_index: torch.Tensor,
        timestamps: torch.Tensor
    ) -> torch.Tensor:
        """
        Dynamic graph processing with temporal evolution
        """
        # Encode nodes and edges
        node_emb = self.node_encoder(node_features)
        edge_emb = self.edge_encoder(edge_features)
        
        # Temporal encoding
        temporal_emb, _ = self.temporal_encoder(node_emb.unsqueeze(1))
        temporal_emb = temporal_emb.squeeze(1)
        
        # Message passing
        messages = []
        for i in range(edge_index.size(1)):
            src, dst = edge_index[:, i]
            message = torch.cat([
                node_emb[src],
                edge_emb[i],
                node_emb[dst]
            ])
            messages.append(self.message_passing(message))
        
        return torch.stack(messages) if messages else node_emb


class KnowledgeGraphReasoner:
    """
    Knowledge graph reasoning for complex relationship inference
    """
    
    def __init__(self, entity_dim: int = 128, relation_dim: int = 64):
        self.entity_embeddings = {}
        self.relation_embeddings = {}
        self.entity_dim = entity_dim
        self.relation_dim = relation_dim
        
        # TransE-style reasoning
        self.scoring_fn = self._translational_scoring
    
    def add_entity(self, entity_id: str, features: Optional[np.ndarray] = None):
        """Add entity to knowledge graph"""
        if features is None:
            features = np.random.randn(self.entity_dim)
        self.entity_embeddings[entity_id] = features
    
    def add_relation(self, relation_id: str, features: Optional[np.ndarray] = None):
        """Add relation type to knowledge graph"""
        if features is None:
            features = np.random.randn(self.relation_dim)
        self.relation_embeddings[relation_id] = features
    
    def add_triple(self, head: str, relation: str, tail: str):
        """Add knowledge triple (head, relation, tail)"""
        if head not in self.entity_embeddings:
            self.add_entity(head)
        if tail not in self.entity_embeddings:
            self.add_entity(tail)
        if relation not in self.relation_embeddings:
            self.add_relation(relation)
    
    def query(self, head: str, relation: str, top_k: int = 5) -> List[Tuple[str, float]]:
        """
        Query knowledge graph: (head, relation, ?)
        Returns top-k most likely tail entities
        """
        if head not in self.entity_embeddings or relation not in self.relation_embeddings:
            return []
        
        head_emb = self.entity_embeddings[head]
        rel_emb = self.relation_embeddings[relation]
        
        # Score all possible tails
        scores = []
        for tail_id, tail_emb in self.entity_embeddings.items():
            if tail_id != head:
                score = self.scoring_fn(head_emb, rel_emb, tail_emb)
                scores.append((tail_id, score))
        
        # Sort by score
        scores.sort(key=lambda x: x[1], reverse=True)
        
        return scores[:top_k]
    
    def _translational_scoring(
        self, head: np.ndarray, relation: np.ndarray, tail: np.ndarray
    ) -> float:
        """TransE scoring: h + r ≈ t"""
        # Pad relation to match entity dimension
        if len(relation) < len(head):
            relation = np.pad(relation, (0, len(head) - len(relation)))
        
        distance = np.linalg.norm(head + relation - tail)
        return float(-distance)  # Negative distance (higher is better)
    
    def infer_missing_links(self, threshold: float = 0.5) -> List[Tuple[str, str, str, float]]:
        """
        Infer missing links in knowledge graph
        Returns: List of (head, relation, tail, confidence)
        """
        inferred = []
        
        for head_id in self.entity_embeddings.keys():
            for rel_id in self.relation_embeddings.keys():
                results = self.query(head_id, rel_id, top_k=3)
                
                for tail_id, score in results:
                    # Normalize score to confidence
                    confidence = 1.0 / (1.0 + np.exp(-score))
                    
                    if confidence > threshold:
                        inferred.append((head_id, rel_id, tail_id, confidence))
        
        return inferred


class HypergraphNetwork(nn.Module):
    """
    Hypergraph Neural Network for modeling complex multi-way relationships
    """
    
    def __init__(self, node_dim: int, hyperedge_dim: int, hidden_dim: int = 128):
        super().__init__()
        
        self.node_transform = nn.Linear(node_dim, hidden_dim)
        self.hyperedge_transform = nn.Linear(hyperedge_dim, hidden_dim)
        
        self.node_to_edge = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim)
        )
        
        self.edge_to_node = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim)
        )
    
    def forward(
        self,
        node_features: torch.Tensor,
        hyperedge_features: torch.Tensor,
        incidence_matrix: torch.Tensor
    ) -> torch.Tensor:
        """
        node_features: (N, node_dim)
        hyperedge_features: (E, hyperedge_dim)
        incidence_matrix: (N, E) - binary matrix indicating node-hyperedge connections
        """
        # Transform features
        node_emb = self.node_transform(node_features)
        edge_emb = self.hyperedge_transform(hyperedge_features)
        
        # Node to hyperedge aggregation
        edge_agg = torch.mm(incidence_matrix.t(), node_emb)  # (E, hidden_dim)
        edge_updated = self.node_to_edge(edge_agg) + edge_emb
        
        # Hyperedge to node aggregation
        node_agg = torch.mm(incidence_matrix, edge_updated)  # (N, hidden_dim)
        node_updated = self.edge_to_node(node_agg) + node_emb
        
        return node_updated


class RelationalReasoningEngine:
    """
    Comprehensive relational reasoning combining multiple graph techniques
    """
    
    def __init__(self, node_dim: int = 128):
        self.gcn = GraphConvolutionalNetwork(
            input_dim=node_dim,
            hidden_dims=[256, 128],
            output_dim=64
        )
        
        self.knowledge_graph = KnowledgeGraphReasoner(entity_dim=node_dim)
        
        self.temporal_graph = TemporalGraphNetwork(
            node_dim=node_dim,
            edge_dim=32
        )
    
    def analyze_relationships(
        self,
        entities: List[Dict[str, Any]],
        relationships: List[Tuple[str, str, str]]
    ) -> Dict[str, Any]:
        """
        Comprehensive relationship analysis
        
        Args:
            entities: List of entity dictionaries with features
            relationships: List of (head, relation, tail) triples
        
        Returns:
            Analysis results with inferred relationships and patterns
        """
        # Build knowledge graph
        for head, relation, tail in relationships:
            self.knowledge_graph.add_triple(head, relation, tail)
        
        # Infer missing links
        inferred = self.knowledge_graph.infer_missing_links(threshold=0.6)
        
        # Analyze patterns
        patterns = self._identify_patterns(relationships, inferred)
        
        return {
            "explicit_relationships": len(relationships),
            "inferred_relationships": len(inferred),
            "patterns": patterns,
            "network_density": self._calculate_density(entities, relationships),
            "key_entities": self._identify_key_entities(entities, relationships)
        }
    
    def _identify_patterns(
        self,
        explicit: List[Tuple[str, str, str]],
        inferred: List[Tuple[str, str, str, float]]
    ) -> List[str]:
        """Identify common patterns in relationships"""
        patterns = []
        
        # Count relation types
        relation_counts = {}
        for _, rel, _ in explicit:
            relation_counts[rel] = relation_counts.get(rel, 0) + 1
        
        # Identify dominant patterns
        for rel, count in relation_counts.items():
            if count >= 3:
                patterns.append(f"Frequent {rel} relationships ({count} instances)")
        
        return patterns
    
    def _calculate_density(
        self,
        entities: List[Dict[str, Any]],
        relationships: List[Tuple[str, str, str]]
    ) -> float:
        """Calculate network density"""
        n = len(entities)
        if n <= 1:
            return 0.0
        
        max_edges = n * (n - 1)
        actual_edges = len(relationships)
        
        return actual_edges / max_edges if max_edges > 0 else 0.0
    
    def _identify_key_entities(
        self,
        entities: List[Dict[str, Any]],
        relationships: List[Tuple[str, str, str]]
    ) -> List[str]:
        """Identify most connected entities"""
        degree = {}
        
        for head, _, tail in relationships:
            degree[head] = degree.get(head, 0) + 1
            degree[tail] = degree.get(tail, 0) + 1
        
        # Sort by degree
        sorted_entities = sorted(degree.items(), key=lambda x: x[1], reverse=True)
        
        return [entity for entity, _ in sorted_entities[:5]]
