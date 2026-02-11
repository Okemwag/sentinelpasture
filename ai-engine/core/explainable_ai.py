"""
Explainable AI - Interpretability and transparency for AI decisions
"""

import torch
import torch.nn as nn
import numpy as np
from typing import Dict, List, Any, Tuple, Optional
import logging

logger = logging.getLogger(__name__)


class ExplainableAI:
    """
    Comprehensive explainability system for AI decisions
    Implements SHAP, attention visualization, and counterfactual explanations
    """
    
    def __init__(self, model: nn.Module):
        self.model = model
        self.baseline = None
        
    def explain_prediction(
        self,
        input_data: torch.Tensor,
        feature_names: Optional[List[str]] = None
    ) -> Dict[str, Any]:
        """
        Generate comprehensive explanation for a prediction
        """
        # Get prediction
        self.model.eval()
        with torch.no_grad():
            prediction = self.model(input_data)
        
        # Feature importance (SHAP-like)
        feature_importance = self._compute_feature_importance(input_data)
        
        # Attention weights (if model has attention)
        attention_weights = self._extract_attention_weights(input_data)
        
        # Counterfactual explanations
        counterfactuals = self._generate_counterfactuals(input_data, prediction)
        
        # Decision path
        decision_path = self._trace_decision_path(input_data)
        
        explanation = {
            "prediction": prediction.cpu().numpy(),
            "feature_importance": feature_importance,
            "attention_weights": attention_weights,
            "counterfactuals": counterfactuals,
            "decision_path": decision_path,
            "confidence": self._compute_confidence(prediction),
            "explanation_text": self._generate_text_explanation(
                feature_importance, feature_names
            )
        }
        
        return explanation
    
    def _compute_feature_importance(
        self, input_data: torch.Tensor
    ) -> Dict[str, float]:
        """
        Compute feature importance using integrated gradients
        """
        input_data.requires_grad = True
        
        # Set baseline (zeros or mean)
        if self.baseline is None:
            baseline = torch.zeros_like(input_data)
        else:
            baseline = self.baseline
        
        # Integrated gradients
        n_steps = 50
        alphas = torch.linspace(0, 1, n_steps)
        
        gradients = []
        for alpha in alphas:
            interpolated = baseline + alpha * (input_data - baseline)
            interpolated.requires_grad = True
            
            output = self.model(interpolated)
            output.sum().backward()
            
            gradients.append(interpolated.grad.detach())
            interpolated.grad.zero_()
        
        # Average gradients
        avg_gradients = torch.stack(gradients).mean(dim=0)
        
        # Integrated gradients
        integrated_grads = (input_data - baseline) * avg_gradients
        
        # Feature importance scores
        importance = integrated_grads.abs().mean(dim=0).cpu().numpy()
        
        return {
            f"feature_{i}": float(imp) 
            for i, imp in enumerate(importance)
        }
    
    def _extract_attention_weights(
        self, input_data: torch.Tensor
    ) -> Optional[Dict[str, Any]]:
        """
        Extract attention weights if model has attention layers
        """
        attention_weights = {}
        
        # Hook to capture attention weights
        def attention_hook(module, input, output):
            if isinstance(output, tuple) and len(output) > 1:
                attention_weights['weights'] = output[1].detach().cpu().numpy()
        
        # Register hooks
        hooks = []
        for name, module in self.model.named_modules():
            if isinstance(module, nn.MultiheadAttention):
                hooks.append(module.register_forward_hook(attention_hook))
        
        # Forward pass
        with torch.no_grad():
            self.model(input_data)
        
        # Remove hooks
        for hook in hooks:
            hook.remove()
        
        return attention_weights if attention_weights else None
    
    def _generate_counterfactuals(
        self,
        input_data: torch.Tensor,
        original_prediction: torch.Tensor,
        n_counterfactuals: int = 3
    ) -> List[Dict[str, Any]]:
        """
        Generate counterfactual explanations
        """
        counterfactuals = []
        
        for _ in range(n_counterfactuals):
            # Perturb input
            perturbation = torch.randn_like(input_data) * 0.1
            counterfactual_input = input_data + perturbation
            
            # Get new prediction
            with torch.no_grad():
                counterfactual_pred = self.model(counterfactual_input)
            
            # Calculate change
            change = (counterfactual_input - input_data).abs().mean().item()
            pred_change = (counterfactual_pred - original_prediction).abs().mean().item()
            
            counterfactuals.append({
                "input_change": change,
                "prediction_change": pred_change,
                "new_prediction": counterfactual_pred.cpu().numpy()
            })
        
        return counterfactuals
    
    def _trace_decision_path(
        self, input_data: torch.Tensor
    ) -> List[Dict[str, Any]]:
        """
        Trace decision path through network layers
        """
        activations = []
        
        def activation_hook(module, input, output):
            activations.append({
                "layer": module.__class__.__name__,
                "output_shape": output.shape,
                "mean_activation": output.mean().item(),
                "std_activation": output.std().item()
            })
        
        # Register hooks
        hooks = []
        for module in self.model.modules():
            if isinstance(module, (nn.Linear, nn.Conv1d, nn.Conv2d)):
                hooks.append(module.register_forward_hook(activation_hook))
        
        # Forward pass
        with torch.no_grad():
            self.model(input_data)
        
        # Remove hooks
        for hook in hooks:
            hook.remove()
        
        return activations
    
    def _compute_confidence(self, prediction: torch.Tensor) -> float:
        """
        Compute prediction confidence
        """
        probs = torch.softmax(prediction, dim=-1)
        confidence = probs.max().item()
        return confidence
    
    def _generate_text_explanation(
        self,
        feature_importance: Dict[str, float],
        feature_names: Optional[List[str]] = None
    ) -> str:
        """
        Generate human-readable explanation
        """
        # Sort features by importance
        sorted_features = sorted(
            feature_importance.items(),
            key=lambda x: x[1],
            reverse=True
        )
        
        # Top 3 features
        top_features = sorted_features[:3]
        
        explanation = "The prediction is primarily influenced by: "
        
        for i, (feature, importance) in enumerate(top_features):
            if feature_names and i < len(feature_names):
                feature_name = feature_names[i]
            else:
                feature_name = feature
            
            explanation += f"{feature_name} (importance: {importance:.3f})"
            
            if i < len(top_features) - 1:
                explanation += ", "
        
        return explanation


class SaliencyMapper:
    """
    Generate saliency maps for input attribution
    """
    
    def __init__(self, model: nn.Module):
        self.model = model
    
    def generate_saliency_map(
        self, input_data: torch.Tensor, target_class: Optional[int] = None
    ) -> np.ndarray:
        """
        Generate saliency map using gradients
        """
        input_data.requires_grad = True
        
        # Forward pass
        output = self.model(input_data)
        
        # Select target
        if target_class is None:
            target = output.max()
        else:
            target = output[0, target_class]
        
        # Backward pass
        target.backward()
        
        # Get gradients
        saliency = input_data.grad.abs().cpu().numpy()
        
        return saliency
