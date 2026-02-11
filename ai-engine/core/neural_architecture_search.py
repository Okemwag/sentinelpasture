"""
Neural Architecture Search - Automated discovery of optimal architectures
"""

import torch
import torch.nn as nn
import numpy as np
from typing import List, Dict, Any, Tuple
import logging

logger = logging.getLogger(__name__)


class NeuralArchitectureSearch:
    """
    Automated neural architecture search using evolutionary algorithms
    """
    
    def __init__(
        self, 
        input_dim: int, 
        output_dim: int,
        population_size: int = 20,
        generations: int = 50
    ):
        self.input_dim = input_dim
        self.output_dim = output_dim
        self.population_size = population_size
        self.generations = generations
        
        self.search_space = {
            'n_layers': [2, 3, 4, 5, 6],
            'hidden_dims': [64, 128, 256, 512],
            'activations': ['relu', 'gelu', 'silu', 'tanh'],
            'dropout_rates': [0.0, 0.1, 0.2, 0.3],
            'use_batch_norm': [True, False],
            'use_residual': [True, False]
        }
    
    def search(
        self, 
        train_data: torch.Tensor,
        train_labels: torch.Tensor,
        val_data: torch.Tensor,
        val_labels: torch.Tensor
    ) -> Dict[str, Any]:
        """
        Search for optimal architecture
        """
        # Initialize population
        population = [self._random_architecture() for _ in range(self.population_size)]
        
        best_architecture = None
        best_fitness = float('-inf')
        
        for generation in range(self.generations):
            # Evaluate fitness
            fitness_scores = []
            for arch in population:
                fitness = self._evaluate_architecture(
                    arch, train_data, train_labels, val_data, val_labels
                )
                fitness_scores.append(fitness)
                
                if fitness > best_fitness:
                    best_fitness = fitness
                    best_architecture = arch
            
            logger.info(f"Generation {generation}: Best fitness = {best_fitness:.4f}")
            
            # Selection and reproduction
            population = self._evolve_population(population, fitness_scores)
        
        return {
            'architecture': best_architecture,
            'fitness': best_fitness,
            'model': self._build_model(best_architecture)
        }

    def _random_architecture(self) -> Dict[str, Any]:
        """Generate random architecture"""
        return {
            'n_layers': np.random.choice(self.search_space['n_layers']),
            'hidden_dims': [
                np.random.choice(self.search_space['hidden_dims']) 
                for _ in range(np.random.choice(self.search_space['n_layers']))
            ],
            'activations': [
                np.random.choice(self.search_space['activations'])
                for _ in range(np.random.choice(self.search_space['n_layers']))
            ],
            'dropout_rate': np.random.choice(self.search_space['dropout_rates']),
            'use_batch_norm': np.random.choice(self.search_space['use_batch_norm']),
            'use_residual': np.random.choice(self.search_space['use_residual'])
        }
    
    def _build_model(self, architecture: Dict[str, Any]) -> nn.Module:
        """Build model from architecture specification"""
        layers = []
        prev_dim = self.input_dim
        
        for i in range(architecture['n_layers']):
            hidden_dim = architecture['hidden_dims'][i]
            
            # Linear layer
            layers.append(nn.Linear(prev_dim, hidden_dim))
            
            # Batch norm
            if architecture['use_batch_norm']:
                layers.append(nn.BatchNorm1d(hidden_dim))
            
            # Activation
            activation = architecture['activations'][i]
            if activation == 'relu':
                layers.append(nn.ReLU())
            elif activation == 'gelu':
                layers.append(nn.GELU())
            elif activation == 'silu':
                layers.append(nn.SiLU())
            elif activation == 'tanh':
                layers.append(nn.Tanh())
            
            # Dropout
            if architecture['dropout_rate'] > 0:
                layers.append(nn.Dropout(architecture['dropout_rate']))
            
            prev_dim = hidden_dim
        
        # Output layer
        layers.append(nn.Linear(prev_dim, self.output_dim))
        
        return nn.Sequential(*layers)
    
    def _evaluate_architecture(
        self,
        architecture: Dict[str, Any],
        train_data: torch.Tensor,
        train_labels: torch.Tensor,
        val_data: torch.Tensor,
        val_labels: torch.Tensor
    ) -> float:
        """Evaluate architecture fitness"""
        model = self._build_model(architecture)
        optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)
        
        # Quick training
        model.train()
        for _ in range(10):  # Limited epochs for speed
            optimizer.zero_grad()
            outputs = model(train_data)
            loss = nn.functional.cross_entropy(outputs, train_labels)
            loss.backward()
            optimizer.step()
        
        # Validation
        model.eval()
        with torch.no_grad():
            val_outputs = model(val_data)
            val_loss = nn.functional.cross_entropy(val_outputs, val_labels)
            accuracy = (val_outputs.argmax(dim=1) == val_labels).float().mean()
        
        # Fitness = accuracy - complexity penalty
        complexity = sum(p.numel() for p in model.parameters())
        fitness = accuracy.item() - 0.0001 * complexity / 1e6
        
        return fitness
    
    def _evolve_population(
        self, 
        population: List[Dict[str, Any]], 
        fitness_scores: List[float]
    ) -> List[Dict[str, Any]]:
        """Evolve population using selection, crossover, and mutation"""
        # Selection (tournament)
        selected = []
        for _ in range(self.population_size):
            tournament = np.random.choice(len(population), 3, replace=False)
            winner = tournament[np.argmax([fitness_scores[i] for i in tournament])]
            selected.append(population[winner])
        
        # Crossover and mutation
        new_population = []
        for i in range(0, len(selected), 2):
            parent1 = selected[i]
            parent2 = selected[i + 1] if i + 1 < len(selected) else selected[0]
            
            # Crossover
            child = self._crossover(parent1, parent2)
            
            # Mutation
            if np.random.random() < 0.2:
                child = self._mutate(child)
            
            new_population.append(child)
        
        return new_population
    
    def _crossover(
        self, parent1: Dict[str, Any], parent2: Dict[str, Any]
    ) -> Dict[str, Any]:
        """Crossover two architectures"""
        child = {}
        for key in parent1.keys():
            if np.random.random() < 0.5:
                child[key] = parent1[key]
            else:
                child[key] = parent2[key]
        return child
    
    def _mutate(self, architecture: Dict[str, Any]) -> Dict[str, Any]:
        """Mutate architecture"""
        mutated = architecture.copy()
        
        # Randomly mutate one parameter
        key = np.random.choice(list(self.search_space.keys()))
        if key in ['hidden_dims', 'activations']:
            # Regenerate list
            mutated[key] = [
                np.random.choice(self.search_space[key.replace('s', '')])
                for _ in range(mutated['n_layers'])
            ]
        else:
            mutated[key] = np.random.choice(self.search_space[key])
        
        return mutated
