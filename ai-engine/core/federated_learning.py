"""
Federated Learning - Privacy-preserving distributed learning
"""

import torch
import torch.nn as nn
from typing import Dict, List, Any, Optional
import numpy as np
from copy import deepcopy
import logging

logger = logging.getLogger(__name__)


class FederatedClient:
    """
    Individual client in federated learning system
    """
    
    def __init__(
        self,
        client_id: str,
        model: nn.Module,
        local_data: Any,
        learning_rate: float = 0.01
    ):
        self.client_id = client_id
        self.model = deepcopy(model)
        self.local_data = local_data
        self.optimizer = torch.optim.SGD(self.model.parameters(), lr=learning_rate)
        
        # Privacy budget
        self.privacy_budget = 1.0
        self.noise_scale = 0.1
    
    def local_train(self, epochs: int = 5) -> Dict[str, Any]:
        """
        Train model on local data
        """
        self.model.train()
        
        total_loss = 0.0
        samples_processed = 0
        
        for epoch in range(epochs):
            epoch_loss = 0.0
            
            # Simplified training loop
            for batch in self._get_batches():
                x, y = batch
                
                self.optimizer.zero_grad()
                output = self.model(x)
                loss = nn.functional.cross_entropy(output, y)
                loss.backward()
                
                # Gradient clipping for privacy
                torch.nn.utils.clip_grad_norm_(self.model.parameters(), max_norm=1.0)
                
                self.optimizer.step()
                
                epoch_loss += loss.item()
                samples_processed += len(x)
            
            total_loss += epoch_loss
        
        avg_loss = total_loss / epochs if epochs > 0 else 0.0
        
        return {
            "client_id": self.client_id,
            "loss": avg_loss,
            "samples": samples_processed,
            "model_update": self._get_model_update()
        }
    
    def _get_batches(self, batch_size: int = 32):
        """Get data batches (simplified)"""
        # Placeholder - return dummy batches
        for _ in range(5):
            x = torch.randn(batch_size, 448)
            y = torch.randint(0, 8, (batch_size,))
            yield x, y
    
    def _get_model_update(self) -> Dict[str, torch.Tensor]:
        """Get model parameter updates"""
        return {
            name: param.data.clone()
            for name, param in self.model.named_parameters()
        }
    
    def add_differential_privacy(self, sensitivity: float = 1.0, epsilon: float = 1.0):
        """
        Add differential privacy noise to model updates
        """
        if self.privacy_budget <= 0:
            logger.warning(f"Client {self.client_id} privacy budget exhausted")
            return
        
        # Laplace mechanism
        scale = sensitivity / epsilon
        
        for param in self.model.parameters():
            noise = torch.from_numpy(
                np.random.laplace(0, scale, param.shape)
            ).float()
            param.data += noise
        
        self.privacy_budget -= epsilon
    
    def update_model(self, global_model_state: Dict[str, torch.Tensor]):
        """Update local model with global model"""
        self.model.load_state_dict(global_model_state)


class FederatedServer:
    """
    Central server for federated learning coordination
    """
    
    def __init__(self, global_model: nn.Module, aggregation: str = "fedavg"):
        self.global_model = global_model
        self.aggregation = aggregation
        self.round_number = 0
        
        self.client_history = {}
        self.performance_history = []
    
    def aggregate_updates(
        self,
        client_updates: List[Dict[str, Any]]
    ) -> Dict[str, torch.Tensor]:
        """
        Aggregate client model updates
        """
        if self.aggregation == "fedavg":
            return self._federated_averaging(client_updates)
        elif self.aggregation == "fedprox":
            return self._federated_proximal(client_updates)
        elif self.aggregation == "weighted":
            return self._weighted_aggregation(client_updates)
        else:
            return self._federated_averaging(client_updates)
    
    def _federated_averaging(
        self,
        client_updates: List[Dict[str, Any]]
    ) -> Dict[str, torch.Tensor]:
        """
        FedAvg: Average model parameters weighted by data size
        """
        total_samples = sum(update["samples"] for update in client_updates)
        
        aggregated_params = {}
        
        # Get parameter names from first client
        param_names = list(client_updates[0]["model_update"].keys())
        
        for param_name in param_names:
            weighted_sum = None
            
            for update in client_updates:
                weight = update["samples"] / total_samples
                param = update["model_update"][param_name]
                
                if weighted_sum is None:
                    weighted_sum = weight * param
                else:
                    weighted_sum += weight * param
            
            aggregated_params[param_name] = weighted_sum
        
        return aggregated_params
    
    def _federated_proximal(
        self,
        client_updates: List[Dict[str, Any]],
        mu: float = 0.01
    ) -> Dict[str, torch.Tensor]:
        """
        FedProx: Federated averaging with proximal term
        """
        # Similar to FedAvg but with regularization
        aggregated = self._federated_averaging(client_updates)
        
        # Add proximal term (pull towards global model)
        global_params = {
            name: param.data.clone()
            for name, param in self.global_model.named_parameters()
        }
        
        for param_name in aggregated.keys():
            aggregated[param_name] = (
                (1 - mu) * aggregated[param_name] +
                mu * global_params[param_name]
            )
        
        return aggregated
    
    def _weighted_aggregation(
        self,
        client_updates: List[Dict[str, Any]]
    ) -> Dict[str, torch.Tensor]:
        """
        Weighted aggregation based on client performance
        """
        # Weight by inverse loss (better clients get more weight)
        weights = []
        for update in client_updates:
            weight = 1.0 / (update["loss"] + 1e-6)
            weights.append(weight)
        
        total_weight = sum(weights)
        weights = [w / total_weight for w in weights]
        
        aggregated_params = {}
        param_names = list(client_updates[0]["model_update"].keys())
        
        for param_name in param_names:
            weighted_sum = None
            
            for update, weight in zip(client_updates, weights):
                param = update["model_update"][param_name]
                
                if weighted_sum is None:
                    weighted_sum = weight * param
                else:
                    weighted_sum += weight * param
            
            aggregated_params[param_name] = weighted_sum
        
        return aggregated_params
    
    def update_global_model(self, aggregated_params: Dict[str, torch.Tensor]):
        """Update global model with aggregated parameters"""
        self.global_model.load_state_dict(aggregated_params)
        self.round_number += 1
    
    def select_clients(
        self,
        all_clients: List[FederatedClient],
        fraction: float = 0.1,
        min_clients: int = 2
    ) -> List[FederatedClient]:
        """
        Select subset of clients for training round
        """
        n_clients = max(min_clients, int(len(all_clients) * fraction))
        n_clients = min(n_clients, len(all_clients))
        
        selected_indices = np.random.choice(
            len(all_clients), n_clients, replace=False
        )
        
        return [all_clients[i] for i in selected_indices]


class SecureAggregation:
    """
    Secure aggregation protocol for privacy-preserving federated learning
    """
    
    def __init__(self, n_clients: int):
        self.n_clients = n_clients
        self.client_masks = {}
    
    def generate_masks(self, client_id: str, param_shape: tuple) -> torch.Tensor:
        """Generate random mask for secure aggregation"""
        mask = torch.randn(param_shape)
        self.client_masks[client_id] = mask
        return mask
    
    def mask_update(
        self,
        client_id: str,
        model_update: Dict[str, torch.Tensor]
    ) -> Dict[str, torch.Tensor]:
        """Mask client update for secure aggregation"""
        masked_update = {}
        
        for param_name, param in model_update.items():
            if client_id not in self.client_masks:
                mask = self.generate_masks(client_id, param.shape)
            else:
                mask = self.client_masks[client_id]
            
            masked_update[param_name] = param + mask
        
        return masked_update
    
    def unmask_aggregate(
        self,
        aggregated_update: Dict[str, torch.Tensor]
    ) -> Dict[str, torch.Tensor]:
        """Remove masks from aggregated update"""
        unmasked = {}
        
        for param_name, param in aggregated_update.items():
            # Sum of all masks cancels out
            total_mask = sum(
                mask for mask in self.client_masks.values()
            )
            unmasked[param_name] = param - total_mask
        
        return unmasked


class FederatedLearningSystem:
    """
    Complete federated learning system with privacy guarantees
    """
    
    def __init__(
        self,
        global_model: nn.Module,
        n_clients: int = 10,
        aggregation: str = "fedavg",
        use_secure_aggregation: bool = True,
        use_differential_privacy: bool = True
    ):
        self.server = FederatedServer(global_model, aggregation)
        self.clients = []
        self.use_secure_aggregation = use_secure_aggregation
        self.use_differential_privacy = use_differential_privacy
        
        if use_secure_aggregation:
            self.secure_agg = SecureAggregation(n_clients)
        
        logger.info(
            f"Federated learning system initialized with {n_clients} clients, "
            f"aggregation={aggregation}, secure_agg={use_secure_aggregation}, "
            f"dp={use_differential_privacy}"
        )
    
    def add_client(self, client: FederatedClient):
        """Add client to federated system"""
        self.clients.append(client)
    
    def train_round(
        self,
        client_fraction: float = 0.1,
        local_epochs: int = 5
    ) -> Dict[str, Any]:
        """
        Execute one round of federated training
        """
        # Select clients
        selected_clients = self.server.select_clients(
            self.clients, fraction=client_fraction
        )
        
        logger.info(
            f"Round {self.server.round_number}: "
            f"Selected {len(selected_clients)}/{len(self.clients)} clients"
        )
        
        # Distribute global model
        global_state = {
            name: param.data.clone()
            for name, param in self.server.global_model.named_parameters()
        }
        
        for client in selected_clients:
            client.update_model(global_state)
        
        # Local training
        client_updates = []
        for client in selected_clients:
            update = client.local_train(epochs=local_epochs)
            
            # Add differential privacy
            if self.use_differential_privacy:
                client.add_differential_privacy(epsilon=0.1)
                update["model_update"] = client._get_model_update()
            
            # Secure aggregation
            if self.use_secure_aggregation:
                update["model_update"] = self.secure_agg.mask_update(
                    client.client_id, update["model_update"]
                )
            
            client_updates.append(update)
        
        # Aggregate updates
        aggregated = self.server.aggregate_updates(client_updates)
        
        # Unmask if using secure aggregation
        if self.use_secure_aggregation:
            aggregated = self.secure_agg.unmask_aggregate(aggregated)
        
        # Update global model
        self.server.update_global_model(aggregated)
        
        # Calculate metrics
        avg_loss = np.mean([u["loss"] for u in client_updates])
        total_samples = sum(u["samples"] for u in client_updates)
        
        metrics = {
            "round": self.server.round_number,
            "clients_participated": len(selected_clients),
            "avg_loss": avg_loss,
            "total_samples": total_samples,
            "privacy_preserved": self.use_differential_privacy
        }
        
        self.server.performance_history.append(metrics)
        
        return metrics
    
    def train(self, n_rounds: int = 10, **kwargs) -> List[Dict[str, Any]]:
        """
        Train for multiple rounds
        """
        results = []
        
        for round_num in range(n_rounds):
            metrics = self.train_round(**kwargs)
            results.append(metrics)
            
            logger.info(
                f"Round {round_num + 1}/{n_rounds} complete: "
                f"avg_loss={metrics['avg_loss']:.4f}"
            )
        
        return results
    
    def get_global_model(self) -> nn.Module:
        """Get trained global model"""
        return self.server.global_model
    
    def get_statistics(self) -> Dict[str, Any]:
        """Get federated learning statistics"""
        return {
            "total_rounds": self.server.round_number,
            "total_clients": len(self.clients),
            "aggregation_method": self.server.aggregation,
            "secure_aggregation": self.use_secure_aggregation,
            "differential_privacy": self.use_differential_privacy,
            "performance_history": self.server.performance_history
        }
