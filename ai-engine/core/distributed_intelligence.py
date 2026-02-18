"""
Distributed Intelligence - Multi-agent coordination and swarm intelligence
"""

import numpy as np
import torch
import torch.nn as nn
from typing import Dict, List, Any, Tuple, Optional
from dataclasses import dataclass
import logging

logger = logging.getLogger(__name__)


@dataclass
class Agent:
    """Individual intelligent agent"""
    id: str
    position: np.ndarray
    velocity: np.ndarray
    best_position: np.ndarray
    best_fitness: float
    specialization: str


class SwarmIntelligence:
    """
    Swarm intelligence for distributed problem solving
    Implements Particle Swarm Optimization with cognitive agents
    """
    
    def __init__(
        self, 
        n_agents: int = 50,
        dimensions: int = 10,
        specializations: Optional[List[str]] = None
    ):
        self.n_agents = n_agents
        self.dimensions = dimensions
        self.specializations = specializations or [
            "exploration", "exploitation", "coordination", "analysis"
        ]
        
        # Initialize swarm
        self.agents = self._initialize_swarm()
        self.global_best_position = None
        self.global_best_fitness = float('inf')
        
        # Swarm parameters
        self.w = 0.7  # Inertia weight
        self.c1 = 1.5  # Cognitive parameter
        self.c2 = 1.5  # Social parameter
        
    def _initialize_swarm(self) -> List[Agent]:
        """Initialize swarm of agents"""
        agents = []
        
        for i in range(self.n_agents):
            position = np.random.uniform(-10, 10, self.dimensions)
            velocity = np.random.uniform(-1, 1, self.dimensions)
            
            agent = Agent(
                id=f"agent_{i}",
                position=position,
                velocity=velocity,
                best_position=position.copy(),
                best_fitness=float('inf'),
                specialization=self.specializations[i % len(self.specializations)]
            )
            agents.append(agent)
        
        return agents
    
    def optimize(
        self,
        objective_function: callable,
        max_iterations: int = 100
    ) -> Dict[str, Any]:
        """
        Optimize using swarm intelligence
        """
        for iteration in range(max_iterations):
            for agent in self.agents:
                # Evaluate fitness
                fitness = objective_function(agent.position)
                
                # Update personal best
                if fitness < agent.best_fitness:
                    agent.best_fitness = fitness
                    agent.best_position = agent.position.copy()
                
                # Update global best
                if fitness < self.global_best_fitness:
                    self.global_best_fitness = fitness
                    self.global_best_position = agent.position.copy()
            
            # Update velocities and positions
            for agent in self._update_swarm():
                pass
            
            # Adaptive parameters
            self.w = 0.9 - (0.5 * iteration / max_iterations)
        
        return {
            "best_solution": self.global_best_position,
            "best_fitness": self.global_best_fitness,
            "iterations": max_iterations,
            "convergence": self._calculate_convergence()
        }
    
    def _update_swarm(self) -> List[Agent]:
        """Update swarm positions and velocities"""
        for agent in self.agents:
            r1, r2 = np.random.random(2)
            
            # Velocity update
            cognitive = self.c1 * r1 * (agent.best_position - agent.position)
            social = self.c2 * r2 * (self.global_best_position - agent.position)
            
            agent.velocity = self.w * agent.velocity + cognitive + social
            
            # Position update
            agent.position = agent.position + agent.velocity
            
            # Boundary handling
            agent.position = np.clip(agent.position, -10, 10)
        
        return self.agents
    
    def _calculate_convergence(self) -> float:
        """Calculate swarm convergence"""
        positions = np.array([agent.position for agent in self.agents])
        variance = np.var(positions, axis=0).mean()
        return float(1.0 / (1.0 + variance))


class MultiAgentCoordination:
    """
    Multi-agent coordination for complex decision making
    """
    
    def __init__(self, n_agents: int = 5):
        self.n_agents = n_agents
        self.agents = [
            CoordinatedAgent(i, specialization) 
            for i, specialization in enumerate([
                "threat_detection",
                "risk_assessment", 
                "resource_allocation",
                "communication",
                "coordination"
            ][:n_agents])
        ]
        
        self.communication_network = self._build_network()
    
    def _build_network(self) -> np.ndarray:
        """Build communication network between agents"""
        # Fully connected network
        network = np.ones((self.n_agents, self.n_agents))
        np.fill_diagonal(network, 0)
        return network
    
    def coordinate_decision(
        self, 
        situation: Dict[str, Any]
    ) -> Dict[str, Any]:
        """
        Coordinate decision making across agents
        """
        # Each agent analyzes situation
        agent_assessments = []
        for agent in self.agents:
            assessment = agent.analyze(situation)
            agent_assessments.append(assessment)
        
        # Share information through network
        shared_knowledge = self._share_information(agent_assessments)
        
        # Consensus building
        consensus = self._build_consensus(shared_knowledge)
        
        # Final decision
        decision = self._make_collective_decision(consensus)
        
        return {
            "decision": decision,
            "consensus_level": consensus["agreement"],
            "agent_contributions": agent_assessments,
            "coordination_quality": self._evaluate_coordination()
        }
    
    def _share_information(
        self, assessments: List[Dict[str, Any]]
    ) -> List[Dict[str, Any]]:
        """Share information between agents"""
        shared = []
        
        for i, agent in enumerate(self.agents):
            # Aggregate information from connected agents
            connected_info = []
            for j, other_assessment in enumerate(assessments):
                if self.communication_network[i, j] > 0:
                    connected_info.append(other_assessment)
            
            # Integrate information
            integrated = agent.integrate_information(connected_info)
            shared.append(integrated)
        
        return shared
    
    def _build_consensus(
        self, shared_knowledge: List[Dict[str, Any]]
    ) -> Dict[str, Any]:
        """Build consensus from shared knowledge"""
        # Voting mechanism
        votes = [k.get("recommendation", 0) for k in shared_knowledge]
        
        # Calculate agreement
        if len(votes) > 0:
            agreement = 1.0 - (np.std(votes) / (np.mean(votes) + 1e-8))
        else:
            agreement = 0.0
        
        return {
            "recommendation": np.mean(votes) if votes else 0,
            "agreement": float(np.clip(agreement, 0, 1)),
            "diversity": float(np.std(votes)) if votes else 0
        }
    
    def _make_collective_decision(
        self, consensus: Dict[str, Any]
    ) -> str:
        """Make final collective decision"""
        recommendation = consensus["recommendation"]
        agreement = consensus["agreement"]
        
        if agreement > 0.8:
            confidence = "high"
        elif agreement > 0.5:
            confidence = "moderate"
        else:
            confidence = "low"
        
        return f"Action level {int(recommendation)} (confidence: {confidence})"
    
    def _evaluate_coordination(self) -> float:
        """Evaluate coordination quality"""
        return 0.85  # Simplified


class CoordinatedAgent:
    """Individual agent in multi-agent system"""
    
    def __init__(self, agent_id: int, specialization: str):
        self.id = agent_id
        self.specialization = specialization
        self.knowledge_base = {}
        
    def analyze(self, situation: Dict[str, Any]) -> Dict[str, Any]:
        """Analyze situation from agent's perspective"""
        # Specialized analysis based on role
        if self.specialization == "threat_detection":
            score = situation.get("threat_indicators", 0) * 2
        elif self.specialization == "risk_assessment":
            score = situation.get("risk_level", 0) * 1.5
        elif self.specialization == "resource_allocation":
            score = situation.get("resource_needs", 0)
        else:
            score = situation.get("general_severity", 0)
        
        return {
            "agent_id": self.id,
            "specialization": self.specialization,
            "recommendation": score,
            "confidence": 0.8
        }
    
    def integrate_information(
        self, other_assessments: List[Dict[str, Any]]
    ) -> Dict[str, Any]:
        """Integrate information from other agents"""
        if not other_assessments:
            return {"recommendation": 0, "confidence": 0.5}
        
        # Weighted average based on confidence
        total_weight = sum(a.get("confidence", 0.5) for a in other_assessments)
        weighted_sum = sum(
            a.get("recommendation", 0) * a.get("confidence", 0.5)
            for a in other_assessments
        )
        
        integrated_recommendation = weighted_sum / (total_weight + 1e-8)
        
        return {
            "recommendation": integrated_recommendation,
            "confidence": total_weight / len(other_assessments),
            "sources": len(other_assessments)
        }
