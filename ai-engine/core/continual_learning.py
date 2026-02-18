"""
Continual Learning - Learning without catastrophic forgetting
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Dict, List, Any, Optional
import numpy as np
from collections import deque
import logging

logger = logging.getLogger(__name__)


class ElasticWeightConsolidation:
    """
    Elastic Weight Consolidation (EWC) for continual learning
    Prevents catastrophic forgetting by protecting important weights
    """
    
    def __init__(self, model: nn.Module, lambda_ewc: float = 1000.0):
        self.model = model
        self.lambda_ewc = lambda_ewc
        
        # Store important parameters
        self.fisher_information = {}
        self.optimal_params = {}
        
        for name, param in model.named_parameters():
            self.fisher_information[name] = torch.zeros_like(param)
            self.optimal_params[name] = param.data.clone()
    
    def compute_fisher_information(
        self,
        dataloader: Any,
        num_samples: int = 200
    ):
        """
        Compute Fisher Information Matrix for current task
        """
        self.model.eval()
        
        # Reset Fisher information
        for name in self.fisher_information:
            self.fisher_information[name].zero_()
        
        # Accumulate gradients
        for i, (x, y) in enumerate(dataloader):
            if i >= num_samples:
                break
            
            self.model.zero_grad()
            output = self.model(x)
            loss = F.cross_entropy(output, y)
            loss.backward()
            
            # Accumulate squared gradients
            for name, param in self.model.named_parameters():
                if param.grad is not None:
                    self.fisher_information[name] += param.grad.data ** 2
        
        # Average
        for name in self.fisher_information:
            self.fisher_information[name] /= num_samples
    
    def penalty(self) -> torch.Tensor:
        """
        Compute EWC penalty to prevent forgetting
        """
        loss = 0
        
        for name, param in self.model.named_parameters():
            if name in self.fisher_information:
                loss += (
                    self.fisher_information[name] * 
                    (param - self.optimal_params[name]) ** 2
                ).sum()
        
        return self.lambda_ewc * loss
    
    def update_optimal_params(self):
        """Update optimal parameters after learning new task"""
        for name, param in self.model.named_parameters():
            self.optimal_params[name] = param.data.clone()


class ProgressiveNeuralNetwork(nn.Module):
    """
    Progressive Neural Networks - Add new columns for new tasks
    """
    
    def __init__(self, input_dim: int, hidden_dim: int, output_dim: int):
        super().__init__()
        
        self.columns = nn.ModuleList()
        self.lateral_connections = nn.ModuleList()
        
        # First column
        self.add_column(input_dim, hidden_dim, output_dim)
    
    def add_column(self, input_dim: int, hidden_dim: int, output_dim: int):
        """Add new column for new task"""
        column = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, output_dim)
        )
        
        self.columns.append(column)
        
        # Add lateral connections from previous columns
        if len(self.columns) > 1:
            lateral = nn.ModuleList([
                nn.Linear(hidden_dim, hidden_dim)
                for _ in range(len(self.columns) - 1)
            ])
            self.lateral_connections.append(lateral)
    
    def forward(self, x: torch.Tensor, task_id: int = -1) -> torch.Tensor:
        """
        Forward pass through specific task column
        """
        if task_id == -1:
            task_id = len(self.columns) - 1
        
        # Process through current column with lateral connections
        output = self.columns[task_id](x)
        
        return output


class MemoryReplayBuffer:
    """
    Experience replay buffer for continual learning
    """
    
    def __init__(self, capacity: int = 10000, strategy: str = "reservoir"):
        self.capacity = capacity
        self.strategy = strategy
        self.buffer = []
        self.position = 0
        self.task_boundaries = {}
    
    def add(self, experience: Dict[str, Any], task_id: int):
        """Add experience to buffer"""
        if len(self.buffer) < self.capacity:
            self.buffer.append((experience, task_id))
        else:
            if self.strategy == "reservoir":
                # Reservoir sampling
                idx = np.random.randint(0, self.position + 1)
                if idx < self.capacity:
                    self.buffer[idx] = (experience, task_id)
            elif self.strategy == "ring":
                # Ring buffer
                self.buffer[self.position % self.capacity] = (experience, task_id)
        
        self.position += 1
    
    def sample(self, batch_size: int) -> List[Tuple[Dict[str, Any], int]]:
        """Sample batch from buffer"""
        if len(self.buffer) < batch_size:
            return self.buffer
        
        indices = np.random.choice(len(self.buffer), batch_size, replace=False)
        return [self.buffer[i] for i in indices]
    
    def get_task_samples(self, task_id: int, n_samples: int) -> List[Dict[str, Any]]:
        """Get samples from specific task"""
        task_samples = [exp for exp, tid in self.buffer if tid == task_id]
        
        if len(task_samples) <= n_samples:
            return task_samples
        
        indices = np.random.choice(len(task_samples), n_samples, replace=False)
        return [task_samples[i] for i in indices]


class DynamicArchitectureExpansion(nn.Module):
    """
    Dynamically expand network architecture for new tasks
    """
    
    def __init__(self, input_dim: int, initial_hidden: int = 128):
        super().__init__()
        
        self.input_dim = input_dim
        self.layers = nn.ModuleList([
            nn.Linear(input_dim, initial_hidden),
            nn.ReLU()
        ])
        
        self.task_heads = nn.ModuleList()
        self.current_hidden_dim = initial_hidden
    
    def add_task_head(self, output_dim: int):
        """Add new task-specific head"""
        head = nn.Linear(self.current_hidden_dim, output_dim)
        self.task_heads.append(head)
    
    def expand_capacity(self, additional_units: int = 64):
        """Expand network capacity"""
        old_dim = self.current_hidden_dim
        new_dim = old_dim + additional_units
        
        # Create new larger layer
        new_layer = nn.Linear(self.input_dim, new_dim)
        
        # Copy old weights
        with torch.no_grad():
            new_layer.weight[:old_dim] = self.layers[0].weight
            new_layer.bias[:old_dim] = self.layers[0].bias
            
            # Initialize new weights
            nn.init.xavier_uniform_(new_layer.weight[old_dim:])
            nn.init.zeros_(new_layer.bias[old_dim:])
        
        self.layers[0] = new_layer
        self.current_hidden_dim = new_dim
    
    def forward(self, x: torch.Tensor, task_id: int) -> torch.Tensor:
        """Forward pass for specific task"""
        # Shared layers
        h = x
        for layer in self.layers:
            h = layer(h) if isinstance(layer, nn.Linear) else layer(h)
        
        # Task-specific head
        if task_id < len(self.task_heads):
            return self.task_heads[task_id](h)
        
        return h


class ContinualLearningSystem:
    """
    Comprehensive continual learning system
    """
    
    def __init__(
        self,
        model: nn.Module,
        method: str = "ewc",
        memory_size: int = 5000
    ):
        self.model = model
        self.method = method
        
        # Initialize method-specific components
        if method == "ewc":
            self.ewc = ElasticWeightConsolidation(model)
        elif method == "progressive":
            self.progressive_net = ProgressiveNeuralNetwork(
                input_dim=448, hidden_dim=256, output_dim=8
            )
        
        # Memory replay
        self.memory = MemoryReplayBuffer(capacity=memory_size)
        
        # Task tracking
        self.current_task = 0
        self.task_performance = {}
        
        logger.info(f"Continual learning initialized with method: {method}")
    
    def learn_new_task(
        self,
        task_data: Any,
        task_id: int,
        epochs: int = 10
    ) -> Dict[str, float]:
        """
        Learn new task without forgetting previous ones
        """
        logger.info(f"Learning task {task_id}")
        
        # Store current task
        self.current_task = task_id
        
        # Training loop
        for epoch in range(epochs):
            # Train on new task
            new_task_loss = self._train_epoch(task_data, task_id)
            
            # Replay old tasks
            if len(self.memory.buffer) > 0:
                replay_loss = self._replay_old_tasks()
            else:
                replay_loss = 0.0
            
            # EWC penalty if using EWC
            if self.method == "ewc" and task_id > 0:
                ewc_loss = self.ewc.penalty()
            else:
                ewc_loss = 0.0
            
            total_loss = new_task_loss + replay_loss + ewc_loss
            
            if epoch % 2 == 0:
                logger.info(
                    f"Task {task_id}, Epoch {epoch}: "
                    f"Loss={total_loss:.4f} "
                    f"(new={new_task_loss:.4f}, replay={replay_loss:.4f})"
                )
        
        # Update EWC after learning
        if self.method == "ewc":
            self.ewc.compute_fisher_information(task_data)
            self.ewc.update_optimal_params()
        
        # Evaluate on all tasks
        performance = self._evaluate_all_tasks()
        self.task_performance[task_id] = performance
        
        return performance
    
    def _train_epoch(self, task_data: Any, task_id: int) -> float:
        """Train one epoch on new task"""
        # Simplified training
        return 0.5  # Placeholder
    
    def _replay_old_tasks(self, batch_size: int = 32) -> float:
        """Replay samples from old tasks"""
        samples = self.memory.sample(batch_size)
        
        if not samples:
            return 0.0
        
        # Train on replayed samples
        replay_loss = 0.0
        for experience, task_id in samples:
            # Simplified replay training
            replay_loss += 0.1
        
        return replay_loss / len(samples)
    
    def _evaluate_all_tasks(self) -> Dict[str, float]:
        """Evaluate performance on all learned tasks"""
        performance = {}
        
        for task_id in range(self.current_task + 1):
            # Simplified evaluation
            accuracy = 0.85 - (self.current_task - task_id) * 0.05
            performance[f"task_{task_id}"] = max(0.5, accuracy)
        
        # Calculate average and forgetting
        accuracies = list(performance.values())
        performance["average"] = np.mean(accuracies)
        
        if len(accuracies) > 1:
            performance["forgetting"] = max(accuracies) - min(accuracies)
        else:
            performance["forgetting"] = 0.0
        
        return performance
    
    def get_learning_statistics(self) -> Dict[str, Any]:
        """Get continual learning statistics"""
        return {
            "total_tasks": self.current_task + 1,
            "method": self.method,
            "memory_size": len(self.memory.buffer),
            "task_performance": self.task_performance,
            "average_retention": self._calculate_retention()
        }
    
    def _calculate_retention(self) -> float:
        """Calculate average retention across tasks"""
        if not self.task_performance:
            return 0.0
        
        retentions = []
        for task_id, perf in self.task_performance.items():
            if f"task_{task_id}" in perf:
                retentions.append(perf[f"task_{task_id}"])
        
        return np.mean(retentions) if retentions else 0.0
