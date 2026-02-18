"""
Neuro-Symbolic AI - Combining neural networks with symbolic reasoning
"""

import torch
import torch.nn as nn
from typing import Dict, List, Any, Tuple, Optional, Set
import numpy as np
from dataclasses import dataclass
import logging

logger = logging.getLogger(__name__)


@dataclass
class LogicalRule:
    """Symbolic logical rule"""
    premise: List[str]
    conclusion: str
    confidence: float
    
    def __str__(self):
        premise_str = " AND ".join(self.premise)
        return f"IF {premise_str} THEN {conclusion} (conf: {self.confidence:.2f})"


class SymbolicKnowledgeBase:
    """
    Symbolic knowledge base with logical rules and facts
    """
    
    def __init__(self):
        self.facts = set()
        self.rules = []
        self.ontology = {}
    
    def add_fact(self, fact: str):
        """Add fact to knowledge base"""
        self.facts.add(fact)
    
    def add_rule(self, rule: LogicalRule):
        """Add logical rule"""
        self.rules.append(rule)
    
    def add_ontology_relation(self, entity: str, relation: str, target: str):
        """Add ontological relationship"""
        if entity not in self.ontology:
            self.ontology[entity] = {}
        if relation not in self.ontology[entity]:
            self.ontology[entity][relation] = []
        self.ontology[entity][relation].append(target)
    
    def query(self, query: str) -> Tuple[bool, float]:
        """
        Query knowledge base
        Returns: (is_true, confidence)
        """
        # Direct fact lookup
        if query in self.facts:
            return True, 1.0
        
        # Rule-based inference
        for rule in self.rules:
            if rule.conclusion == query:
                # Check if all premises are satisfied
                premises_satisfied = all(
                    premise in self.facts for premise in rule.premise
                )
                if premises_satisfied:
                    return True, rule.confidence
        
        return False, 0.0
    
    def forward_chain(self, max_iterations: int = 10) -> Set[str]:
        """
        Forward chaining inference
        Derive all possible conclusions from facts and rules
        """
        derived_facts = self.facts.copy()
        
        for _ in range(max_iterations):
            new_facts = set()
            
            for rule in self.rules:
                # Check if all premises are in derived facts
                if all(premise in derived_facts for premise in rule.premise):
                    if rule.conclusion not in derived_facts:
                        new_facts.add(rule.conclusion)
            
            if not new_facts:
                break
            
            derived_facts.update(new_facts)
        
        return derived_facts
    
    def backward_chain(self, goal: str, depth: int = 0, max_depth: int = 5) -> bool:
        """
        Backward chaining inference
        Try to prove goal from facts and rules
        """
        if depth > max_depth:
            return False
        
        # Check if goal is a fact
        if goal in self.facts:
            return True
        
        # Try to prove goal using rules
        for rule in self.rules:
            if rule.conclusion == goal:
                # Try to prove all premises
                all_proved = all(
                    self.backward_chain(premise, depth + 1, max_depth)
                    for premise in rule.premise
                )
                if all_proved:
                    return True
        
        return False


class NeuralSymbolicBridge(nn.Module):
    """
    Bridge between neural and symbolic representations
    """
    
    def __init__(self, neural_dim: int, n_symbols: int):
        super().__init__()
        
        # Neural to symbolic
        self.neural_to_symbolic = nn.Sequential(
            nn.Linear(neural_dim, 256),
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(256, n_symbols),
            nn.Sigmoid()
        )
        
        # Symbolic to neural
        self.symbolic_to_neural = nn.Sequential(
            nn.Linear(n_symbols, 256),
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(256, neural_dim)
        )
        
        self.symbol_embeddings = nn.Embedding(n_symbols, neural_dim)
    
    def extract_symbols(
        self, neural_features: torch.Tensor, threshold: float = 0.5
    ) -> torch.Tensor:
        """Extract symbolic representation from neural features"""
        symbol_probs = self.neural_to_symbolic(neural_features)
        symbols = (symbol_probs > threshold).float()
        return symbols
    
    def ground_symbols(self, symbols: torch.Tensor) -> torch.Tensor:
        """Ground symbolic representation back to neural space"""
        return self.symbolic_to_neural(symbols)


class LogicTensorNetwork(nn.Module):
    """
    Logic Tensor Network - Differentiable first-order logic
    """
    
    def __init__(self, n_predicates: int, n_entities: int, embedding_dim: int = 64):
        super().__init__()
        
        self.entity_embeddings = nn.Embedding(n_entities, embedding_dim)
        self.predicate_embeddings = nn.Embedding(n_predicates, embedding_dim)
        
        # Fuzzy logic operators
        self.and_operator = self._fuzzy_and
        self.or_operator = self._fuzzy_or
        self.not_operator = self._fuzzy_not
    
    def predicate(self, predicate_id: int, entity_id: int) -> torch.Tensor:
        """
        Evaluate predicate on entity
        Returns fuzzy truth value [0, 1]
        """
        entity_emb = self.entity_embeddings(torch.tensor(entity_id))
        predicate_emb = self.predicate_embeddings(torch.tensor(predicate_id))
        
        # Similarity as truth value
        similarity = torch.cosine_similarity(
            entity_emb.unsqueeze(0), predicate_emb.unsqueeze(0)
        )
        
        # Map to [0, 1]
        truth_value = (similarity + 1) / 2
        
        return truth_value
    
    def _fuzzy_and(self, a: torch.Tensor, b: torch.Tensor) -> torch.Tensor:
        """Fuzzy AND (product t-norm)"""
        return a * b
    
    def _fuzzy_or(self, a: torch.Tensor, b: torch.Tensor) -> torch.Tensor:
        """Fuzzy OR (probabilistic sum)"""
        return a + b - a * b
    
    def _fuzzy_not(self, a: torch.Tensor) -> torch.Tensor:
        """Fuzzy NOT"""
        return 1 - a
    
    def implies(self, a: torch.Tensor, b: torch.Tensor) -> torch.Tensor:
        """Fuzzy implication (Lukasiewicz)"""
        return torch.min(torch.ones_like(a), 1 - a + b)
    
    def forall(self, predicate_fn, entities: List[int]) -> torch.Tensor:
        """Universal quantifier (fuzzy)"""
        truth_values = [predicate_fn(e) for e in entities]
        return torch.min(torch.stack(truth_values))
    
    def exists(self, predicate_fn, entities: List[int]) -> torch.Tensor:
        """Existential quantifier (fuzzy)"""
        truth_values = [predicate_fn(e) for e in entities]
        return torch.max(torch.stack(truth_values))


class NeuralTheoremProver(nn.Module):
    """
    Neural theorem prover for automated reasoning
    """
    
    def __init__(self, statement_dim: int = 128, hidden_dim: int = 256):
        super().__init__()
        
        self.statement_encoder = nn.Sequential(
            nn.Linear(statement_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim)
        )
        
        self.proof_step_generator = nn.LSTM(
            hidden_dim, hidden_dim, num_layers=2, batch_first=True
        )
        
        self.validity_checker = nn.Sequential(
            nn.Linear(hidden_dim, 64),
            nn.ReLU(),
            nn.Linear(64, 1),
            nn.Sigmoid()
        )
    
    def prove(
        self,
        premises: torch.Tensor,
        goal: torch.Tensor,
        max_steps: int = 10
    ) -> Tuple[bool, List[torch.Tensor]]:
        """
        Attempt to prove goal from premises
        Returns: (is_provable, proof_steps)
        """
        # Encode premises and goal
        premise_encodings = self.statement_encoder(premises)
        goal_encoding = self.statement_encoder(goal)
        
        # Generate proof steps
        proof_steps = []
        current_state = premise_encodings.mean(dim=0, keepdim=True).unsqueeze(0)
        
        for step in range(max_steps):
            # Generate next proof step
            output, (h, c) = self.proof_step_generator(current_state)
            proof_steps.append(output.squeeze(0))
            
            # Check if goal is reached
            similarity = torch.cosine_similarity(
                output.squeeze(0), goal_encoding.unsqueeze(0)
            )
            
            if similarity > 0.9:
                validity = self.validity_checker(output.squeeze(0))
                return validity.item() > 0.5, proof_steps
            
            current_state = output
        
        return False, proof_steps


class NeuroSymbolicReasoner:
    """
    Comprehensive neuro-symbolic reasoning system
    """
    
    def __init__(self, neural_dim: int = 448, n_symbols: int = 100):
        self.knowledge_base = SymbolicKnowledgeBase()
        self.bridge = NeuralSymbolicBridge(neural_dim, n_symbols)
        self.logic_network = LogicTensorNetwork(
            n_predicates=20, n_entities=100, embedding_dim=64
        )
        self.theorem_prover = NeuralTheoremProver(statement_dim=128)
        
        self.symbol_to_name = {}
        self.name_to_symbol = {}
    
    def add_knowledge(self, facts: List[str], rules: List[LogicalRule]):
        """Add symbolic knowledge"""
        for fact in facts:
            self.knowledge_base.add_fact(fact)
        
        for rule in rules:
            self.knowledge_base.add_rule(rule)
    
    def reason(
        self,
        neural_input: torch.Tensor,
        query: str
    ) -> Dict[str, Any]:
        """
        Perform neuro-symbolic reasoning
        
        Args:
            neural_input: Neural network features
            query: Symbolic query
        
        Returns:
            Reasoning result with explanation
        """
        # Extract symbolic representation from neural features
        symbols = self.bridge.extract_symbols(neural_input)
        
        # Convert to symbolic facts
        active_symbols = (symbols > 0.5).nonzero(as_tuple=True)[0]
        derived_facts = set()
        
        for symbol_id in active_symbols:
            symbol_name = self.symbol_to_name.get(symbol_id.item(), f"symbol_{symbol_id}")
            derived_facts.add(symbol_name)
            self.knowledge_base.add_fact(symbol_name)
        
        # Perform symbolic reasoning
        is_true, confidence = self.knowledge_base.query(query)
        
        # Forward chaining for additional inferences
        all_derived = self.knowledge_base.forward_chain()
        
        # Backward chaining to find proof
        has_proof = self.knowledge_base.backward_chain(query)
        
        # Generate explanation
        explanation = self._generate_explanation(
            query, is_true, confidence, derived_facts, all_derived
        )
        
        return {
            "query": query,
            "result": is_true,
            "confidence": confidence,
            "has_proof": has_proof,
            "derived_facts": list(derived_facts),
            "all_inferences": list(all_derived),
            "explanation": explanation
        }
    
    def learn_rules(
        self,
        examples: List[Tuple[List[str], str]],
        min_confidence: float = 0.7
    ) -> List[LogicalRule]:
        """
        Learn logical rules from examples
        
        Args:
            examples: List of (premises, conclusion) pairs
        
        Returns:
            Learned rules
        """
        learned_rules = []
        
        # Group by conclusion
        conclusion_to_premises = {}
        for premises, conclusion in examples:
            if conclusion not in conclusion_to_premises:
                conclusion_to_premises[conclusion] = []
            conclusion_to_premises[conclusion].append(premises)
        
        # Extract common patterns
        for conclusion, premise_lists in conclusion_to_premises.items():
            # Find common premises
            if len(premise_lists) > 1:
                common = set(premise_lists[0])
                for premises in premise_lists[1:]:
                    common &= set(premises)
                
                if common:
                    confidence = len(premise_lists) / (len(premise_lists) + 1)
                    if confidence >= min_confidence:
                        rule = LogicalRule(
                            premise=list(common),
                            conclusion=conclusion,
                            confidence=confidence
                        )
                        learned_rules.append(rule)
                        self.knowledge_base.add_rule(rule)
        
        return learned_rules
    
    def _generate_explanation(
        self,
        query: str,
        result: bool,
        confidence: float,
        derived_facts: Set[str],
        all_inferences: Set[str]
    ) -> str:
        """Generate human-readable explanation"""
        if result:
            explanation = f"Query '{query}' is TRUE (confidence: {confidence:.2f})\n"
            explanation += f"Derived from facts: {', '.join(list(derived_facts)[:5])}\n"
            explanation += f"Total inferences made: {len(all_inferences)}"
        else:
            explanation = f"Query '{query}' is FALSE or UNKNOWN\n"
            explanation += f"Could not be derived from available facts and rules"
        
        return explanation
    
    def integrate_neural_symbolic(
        self,
        neural_prediction: torch.Tensor,
        symbolic_constraints: List[str]
    ) -> torch.Tensor:
        """
        Integrate neural predictions with symbolic constraints
        """
        # Check constraints
        violations = []
        for constraint in symbolic_constraints:
            is_satisfied, _ = self.knowledge_base.query(constraint)
            if not is_satisfied:
                violations.append(constraint)
        
        # Adjust neural prediction based on violations
        if violations:
            # Penalize prediction
            penalty = len(violations) * 0.1
            neural_prediction = neural_prediction * (1 - penalty)
        
        return neural_prediction
