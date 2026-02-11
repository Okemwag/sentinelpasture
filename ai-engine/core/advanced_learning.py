"""
Advanced Learning Systems - Meta-learning, Reinforcement Learning, and Adaptive Intelligence
"""

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Dict, List, Any, Tuple, Optional
from collections import deque
import logging

logger = logging.getLogger(__name__)


class MetaLearner(nn.Module):
    """
    Meta-learning system for rapid adaptation to new threat patterns
    Implements MAML (Model-Agnostic Meta-Learning) for few-shot learning
    """
    
    def __init__(self, input_dim: int = 448, hidden_dim: int = 256):
        super().__init__()
        
        self.feature_extractor = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(hidden_dim, hidden_dim // 2),
            nn.LayerNorm(hidden_dim // 2),
            nn.ReLU()
        )
        
        self.task_adapter = nn.Sequential(
            nn.Linear(hidden_dim // 2, 64),
            nn.ReLU(),
            nn.Linear(64, 32)
        )
        
        self.meta_parameters = nn.ParameterDict({
            'alpha': nn.Parameter(torch.tensor(0.01)),  # Inner loop learning rate
            'beta': nn.Parameter(torch.tensor(0.001))   # Meta learning rate
        })
        
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        features = self.feature_extractor(x)
        return self.task_adapter(features)
    
    def adapt(self, support_set: torch.Tensor, support_labels: torch.Tensor, 
              steps: int = 5) -> nn.Module:
        """
        Adapt to new task using support set
        """
        adapted_model = type(self)(
            input_dim=support_set.shape[-1],
            hidden_dim=256
        )
        adapted_model.load_state_dict(self.state_dict())
        
        optimizer = torch.optim.SGD(
            adapted_model.parameters(), 
            lr=self.meta_parameters['alpha'].item()
        )
        
        for _ in range(steps):
            predictions = adapted_model(support_set)
            loss = F.mse_loss(predictions, support_labels)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
        
        return adapted_model


class ReinforcementLearningAgent:
    """
    Multi-agent reinforcement learning for decision optimization
    Implements PPO (Proximal Policy Optimization) for stable learning
    """
    
    def __init__(self, state_dim: int, action_dim: int, hidden_dim: int = 256):
        self.state_dim = state_dim
        self.action_dim = action_dim
        
        # Actor network (policy)
        self.actor = nn.Sequential(
            nn.Linear(state_dim, hidden_dim),
            nn.Tanh(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.Tanh(),
            nn.Linear(hidden_dim, action_dim),
            nn.Softmax(dim=-1)
        )
        
        # Critic network (value function)
        self.critic = nn.Sequential(
            nn.Linear(state_dim, hidden_dim),
            nn.Tanh(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.Tanh(),
            nn.Linear(hidden_dim, 1)
        )
        
        self.actor_optimizer = torch.optim.Adam(self.actor.parameters(), lr=3e-4)
        self.critic_optimizer = torch.optim.Adam(self.critic.parameters(), lr=1e-3)
        
        self.memory = deque(maxlen=10000)
        self.gamma = 0.99  # Discount factor
        self.epsilon = 0.2  # PPO clip parameter
        
    def select_action(self, state: torch.Tensor) -> Tuple[int, float]:
        """Select action using current policy"""
        with torch.no_grad():
            action_probs = self.actor(state)
            action = torch.multinomial(action_probs, 1).item()
            log_prob = torch.log(action_probs[action])
        
        return action, log_prob.item()
    
    def store_transition(self, state, action, reward, next_state, done):
        """Store experience in replay buffer"""
        self.memory.append((state, action, reward, next_state, done))
    
    def update(self, batch_size: int = 64, epochs: int = 10):
        """Update policy and value networks using PPO"""
        if len(self.memory) < batch_size:
            return
        
        # Sample batch
        indices = np.random.choice(len(self.memory), batch_size, replace=False)
        batch = [self.memory[i] for i in indices]
        
        states = torch.stack([b[0] for b in batch])
        actions = torch.tensor([b[1] for b in batch])
        rewards = torch.tensor([b[2] for b in batch], dtype=torch.float32)
        next_states = torch.stack([b[3] for b in batch])
        dones = torch.tensor([b[4] for b in batch], dtype=torch.float32)
        
        # Calculate advantages
        with torch.no_grad():
            values = self.critic(states).squeeze()
            next_values = self.critic(next_states).squeeze()
            td_targets = rewards + self.gamma * next_values * (1 - dones)
            advantages = td_targets - values
            advantages = (advantages - advantages.mean()) / (advantages.std() + 1e-8)
        
        # PPO update
        for _ in range(epochs):
            # Actor loss
            action_probs = self.actor(states)
            old_action_probs = action_probs.detach()
            
            ratio = action_probs.gather(1, actions.unsqueeze(1)) / \
                    (old_action_probs.gather(1, actions.unsqueeze(1)) + 1e-8)
            
            surr1 = ratio.squeeze() * advantages
            surr2 = torch.clamp(ratio.squeeze(), 1 - self.epsilon, 1 + self.epsilon) * advantages
            actor_loss = -torch.min(surr1, surr2).mean()
            
            # Critic loss
            values = self.critic(states).squeeze()
            critic_loss = F.mse_loss(values, td_targets)
            
            # Update networks
            self.actor_optimizer.zero_grad()
            actor_loss.backward()
            self.actor_optimizer.step()
            
            self.critic_optimizer.zero_grad()
            critic_loss.backward()
            self.critic_optimizer.step()


class AdaptiveLearningSystem:
    """
    Adaptive learning system that continuously improves from feedback
    Implements online learning with concept drift detection
    """
    
    def __init__(self, input_dim: int = 448, output_dim: int = 8):
        self.model = nn.Sequential(
            nn.Linear(input_dim, 256),
            nn.BatchNorm1d(256),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(256, 128),
            nn.BatchNorm1d(128),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(128, output_dim)
        )
        
        self.optimizer = torch.optim.Adam(self.model.parameters(), lr=1e-4)
        self.performance_history = deque(maxlen=100)
        self.drift_detector = ConceptDriftDetector()
        
    def predict(self, x: torch.Tensor) -> torch.Tensor:
        """Make prediction"""
        self.model.eval()
        with torch.no_grad():
            return self.model(x)
    
    def update_online(self, x: torch.Tensor, y: torch.Tensor) -> float:
        """Update model with new data point"""
        self.model.train()
        
        # Forward pass
        predictions = self.model(x)
        loss = F.cross_entropy(predictions, y)
        
        # Backward pass
        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()
        
        # Track performance
        accuracy = (predictions.argmax(dim=1) == y).float().mean().item()
        self.performance_history.append(accuracy)
        
        # Detect concept drift
        if self.drift_detector.detect_drift(accuracy):
            logger.warning("Concept drift detected - adapting learning rate")
            self._adapt_to_drift()
        
        return loss.item()
    
    def _adapt_to_drift(self):
        """Adapt learning strategy when drift is detected"""
        # Increase learning rate temporarily
        for param_group in self.optimizer.param_groups:
            param_group['lr'] *= 1.5


class ConceptDriftDetector:
    """
    Detects concept drift in data streams using ADWIN algorithm
    """
    
    def __init__(self, delta: float = 0.002):
        self.delta = delta
        self.window = deque(maxlen=1000)
        self.mean_history = deque(maxlen=50)
        
    def detect_drift(self, value: float) -> bool:
        """Detect if concept drift has occurred"""
        self.window.append(value)
        
        if len(self.window) < 30:
            return False
        
        current_mean = np.mean(list(self.window)[-30:])
        self.mean_history.append(current_mean)
        
        if len(self.mean_history) < 2:
            return False
        
        # Check for significant change in mean
        recent_mean = np.mean(list(self.mean_history)[-10:])
        historical_mean = np.mean(list(self.mean_history)[:-10])
        
        drift = abs(recent_mean - historical_mean) > 0.1
        
        return drift


class EnsembleLearner:
    """
    Advanced ensemble learning with dynamic model selection
    """
    
    def __init__(self, n_models: int = 5, input_dim: int = 448, output_dim: int = 8):
        self.models = [
            self._create_diverse_model(i, input_dim, output_dim) 
            for i in range(n_models)
        ]
        self.model_weights = torch.ones(n_models) / n_models
        self.performance_tracker = {i: deque(maxlen=100) for i in range(n_models)}
        
    def _create_diverse_model(self, idx: int, input_dim: int, output_dim: int) -> nn.Module:
        """Create diverse model architectures"""
        architectures = [
            # Deep narrow
            nn.Sequential(
                nn.Linear(input_dim, 128),
                nn.ReLU(),
                nn.Linear(128, 64),
                nn.ReLU(),
                nn.Linear(64, 32),
                nn.ReLU(),
                nn.Linear(32, output_dim)
            ),
            # Wide shallow
            nn.Sequential(
                nn.Linear(input_dim, 512),
                nn.ReLU(),
                nn.Dropout(0.3),
                nn.Linear(512, output_dim)
            ),
            # Residual
            ResidualNetwork(input_dim, output_dim),
            # Attention-based
            AttentionNetwork(input_dim, output_dim),
            # Mixture of Experts
            MixtureOfExperts(input_dim, output_dim, n_experts=4)
        ]
        
        return architectures[idx % len(architectures)]
    
    def predict(self, x: torch.Tensor) -> torch.Tensor:
        """Ensemble prediction with dynamic weighting"""
        predictions = []
        
        for model in self.models:
            model.eval()
            with torch.no_grad():
                pred = model(x)
                predictions.append(pred)
        
        # Weighted average
        predictions = torch.stack(predictions)
        weights = self.model_weights.view(-1, 1, 1)
        ensemble_pred = (predictions * weights).sum(dim=0)
        
        return ensemble_pred
    
    def update_weights(self, model_idx: int, performance: float):
        """Update model weights based on performance"""
        self.performance_tracker[model_idx].append(performance)
        
        # Recalculate weights based on recent performance
        avg_performances = []
        for i in range(len(self.models)):
            if len(self.performance_tracker[i]) > 0:
                avg_perf = np.mean(list(self.performance_tracker[i]))
                avg_performances.append(avg_perf)
            else:
                avg_performances.append(0.5)
        
        # Softmax weighting
        performances = torch.tensor(avg_performances, dtype=torch.float32)
        self.model_weights = F.softmax(performances * 5, dim=0)  # Temperature = 5


class ResidualNetwork(nn.Module):
    """Residual network architecture"""
    
    def __init__(self, input_dim: int, output_dim: int):
        super().__init__()
        self.input_proj = nn.Linear(input_dim, 256)
        
        self.res_blocks = nn.ModuleList([
            ResidualBlock(256) for _ in range(3)
        ])
        
        self.output = nn.Linear(256, output_dim)
    
    def forward(self, x):
        x = self.input_proj(x)
        for block in self.res_blocks:
            x = block(x)
        return self.output(x)


class ResidualBlock(nn.Module):
    """Single residual block"""
    
    def __init__(self, dim: int):
        super().__init__()
        self.layers = nn.Sequential(
            nn.Linear(dim, dim),
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(dim, dim)
        )
        self.norm = nn.LayerNorm(dim)
    
    def forward(self, x):
        return self.norm(x + self.layers(x))


class AttentionNetwork(nn.Module):
    """Attention-based network"""
    
    def __init__(self, input_dim: int, output_dim: int):
        super().__init__()
        self.embedding = nn.Linear(input_dim, 256)
        self.attention = nn.MultiheadAttention(256, num_heads=8, batch_first=True)
        self.output = nn.Sequential(
            nn.Linear(256, 128),
            nn.ReLU(),
            nn.Linear(128, output_dim)
        )
    
    def forward(self, x):
        # Add sequence dimension
        x = self.embedding(x).unsqueeze(1)
        x, _ = self.attention(x, x, x)
        x = x.squeeze(1)
        return self.output(x)


class MixtureOfExperts(nn.Module):
    """Mixture of Experts architecture"""
    
    def __init__(self, input_dim: int, output_dim: int, n_experts: int = 4):
        super().__init__()
        
        # Gating network
        self.gate = nn.Sequential(
            nn.Linear(input_dim, 128),
            nn.ReLU(),
            nn.Linear(128, n_experts),
            nn.Softmax(dim=-1)
        )
        
        # Expert networks
        self.experts = nn.ModuleList([
            nn.Sequential(
                nn.Linear(input_dim, 256),
                nn.ReLU(),
                nn.Dropout(0.2),
                nn.Linear(256, output_dim)
            )
            for _ in range(n_experts)
        ])
    
    def forward(self, x):
        # Get gating weights
        gate_weights = self.gate(x)
        
        # Get expert outputs
        expert_outputs = torch.stack([expert(x) for expert in self.experts], dim=1)
        
        # Weighted combination
        output = (expert_outputs * gate_weights.unsqueeze(-1)).sum(dim=1)
        
        return output
