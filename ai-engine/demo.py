"""
Comprehensive Demo - Showcase AI Engine Capabilities
"""

import asyncio
import numpy as np
import torch
from datetime import datetime
from orchestrator import AIOrchestrator
import logging

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


async def demo_signal_processing():
    """Demo 1: Advanced Signal Processing"""
    print("\n" + "="*80)
    print("DEMO 1: ADVANCED SIGNAL PROCESSING")
    print("="*80)
    
    orchestrator = AIOrchestrator(config={"device": "cpu"})
    
    # Create diverse signals
    signals = [
        {
            "type": "social_media",
            "source": "verified_media",
            "data": {
                "text": "Large protest gathering downtown, tensions rising",
                "likes": 5000,
                "shares": 1200,
                "verified": True
            },
            "location": {"latitude": 40.7128, "longitude": -74.0060, "region": "NYC"},
            "temporal": {"timestamp": datetime.now().isoformat()}
        },
        {
            "type": "economic",
            "source": "official_government",
            "data": {
                "gdp": -2.5,
                "inflation": 8.2,
                "unemployment": 6.5
            },
            "location": {"latitude": 40.7128, "longitude": -74.0060, "region": "NYC"},
            "temporal": {"timestamp": datetime.now().isoformat()}
        },
        {
            "type": "infrastructure",
            "source": "official_government",
            "data": {
                "infrastructure_type": "power_grid",
                "status": "degraded",
                "current_capacity": 75,
                "max_capacity": 100
            },
            "location": {"latitude": 40.7128, "longitude": -74.0060, "region": "NYC"},
            "temporal": {"timestamp": datetime.now().isoformat()}
        }
    ]
    
    result = await orchestrator.process_intelligence_pipeline(signals)
    
    print(f"\n✓ Processed {result['metadata']['signals_processed']} signals")
    print(f"✓ Processing time: {result['metadata']['processing_time_seconds']:.3f}s")
    print(f"✓ Threat level: {result['assessment']['threat_level']}/6")
    print(f"✓ Confidence: {result['assessment']['confidence']:.1%}")
    print(f"✓ Indicators identified: {result['assessment']['indicator_count']}")
    
    return result


async def demo_threat_analysis():
    """Demo 2: Advanced Threat Analysis"""
    print("\n" + "="*80)
    print("DEMO 2: ADVANCED THREAT ANALYSIS WITH ENSEMBLE LEARNING")
    print("="*80)
    
    orchestrator = AIOrchestrator(config={"device": "cpu"})
    
    # Create high-risk scenario
    signals = []
    for i in range(20):
        signals.append({
            "type": np.random.choice(["social_media", "security", "political"]),
            "source": "verified_media",
            "data": {
                "text": f"Escalating situation report {i}",
                "severity": np.random.uniform(0.6, 0.9)
            },
            "location": {
                "latitude": 40.7128 + np.random.uniform(-0.1, 0.1),
                "longitude": -74.0060 + np.random.uniform(-0.1, 0.1),
                "region": "NYC"
            },
            "temporal": {"timestamp": datetime.now().isoformat()}
        })
    
    result = await orchestrator.process_intelligence_pipeline(signals)
    
    print(f"\n✓ Analyzed {len(signals)} signals")
    print(f"✓ Threat indicators: {len(result['indicators'])}")
    
    if result['indicators']:
        for i, indicator in enumerate(result['indicators'][:3], 1):
            print(f"\n  Indicator {i}:")
            print(f"    - Name: {indicator['name']}")
            print(f"    - Severity: {indicator['severity']}/6")
            print(f"    - Confidence: {indicator['confidence']:.1%}")
            print(f"    - Patterns: {len(indicator.get('patterns', []))}")
    
    return result


async def demo_causal_inference():
    """Demo 3: Causal Inference"""
    print("\n" + "="*80)
    print("DEMO 3: CAUSAL INFERENCE AND COUNTERFACTUAL REASONING")
    print("="*80)
    
    from core.causal_inference import CausalInferenceEngine
    
    engine = CausalInferenceEngine()
    
    # Generate synthetic data
    n_samples = 100
    economic_stress = np.random.randn(n_samples)
    social_unrest = 0.7 * economic_stress + np.random.randn(n_samples) * 0.3
    violence = 0.5 * social_unrest + np.random.randn(n_samples) * 0.2
    
    data = np.column_stack([economic_stress, social_unrest, violence])
    variables = ["economic_stress", "social_unrest", "violence"]
    
    # Discover causal structure
    causal_graph = engine.discover_causal_structure(data, variables)
    
    print("\n✓ Causal Structure Discovered:")
    for cause, effects in causal_graph.items():
        if effects:
            print(f"  {cause} → {', '.join(effects)}")
    
    # Estimate intervention effect
    effect = engine.estimate_intervention_effect(
        data, "economic_stress", "violence"
    )
    
    print(f"\n✓ Intervention Analysis:")
    print(f"  Effect of reducing economic stress on violence:")
    print(f"  Average Treatment Effect: {effect['average_treatment_effect']:.3f}")
    print(f"  Significance: {effect['significance']}")
    
    return causal_graph


async def demo_quantum_optimization():
    """Demo 4: Quantum-Inspired Optimization"""
    print("\n" + "="*80)
    print("DEMO 4: QUANTUM-INSPIRED OPTIMIZATION")
    print("="*80)
    
    from core.quantum_optimizer import QuantumInspiredOptimizer
    
    optimizer = QuantumInspiredOptimizer(n_qubits=8, n_iterations=500)
    
    # Complex optimization problem
    def objective(x):
        return sum((x - np.array([1, 2, 3, 4, 5]))**2)
    
    result = optimizer.optimize(
        objective_function=objective,
        bounds=[(-10, 10)] * 5
    )
    
    print(f"\n✓ Optimization Complete:")
    print(f"  Best solution: {result['solution']}")
    print(f"  Best fitness: {result['fitness']:.6f}")
    print(f"  Iterations: {result['iterations']}")
    
    return result


async def demo_swarm_intelligence():
    """Demo 5: Swarm Intelligence"""
    print("\n" + "="*80)
    print("DEMO 5: SWARM INTELLIGENCE FOR DISTRIBUTED PROBLEM SOLVING")
    print("="*80)
    
    from core.distributed_intelligence import SwarmIntelligence
    
    swarm = SwarmIntelligence(n_agents=30, dimensions=5)
    
    # Optimization problem
    def rastrigin(x):
        return 10 * len(x) + sum(x**2 - 10 * np.cos(2 * np.pi * x))
    
    result = swarm.optimize(rastrigin, max_iterations=100)
    
    print(f"\n✓ Swarm Optimization Complete:")
    print(f"  Best solution: {result['best_solution']}")
    print(f"  Best fitness: {result['best_fitness']:.6f}")
    print(f"  Convergence: {result['convergence']:.1%}")
    
    return result


async def demo_multiagent_coordination():
    """Demo 6: Multi-Agent Coordination"""
    print("\n" + "="*80)
    print("DEMO 6: MULTI-AGENT COORDINATION")
    print("="*80)
    
    from core.distributed_intelligence import MultiAgentCoordination
    
    coordinator = MultiAgentCoordination(n_agents=5)
    
    situation = {
        "threat_indicators": 7,
        "risk_level": 6,
        "resource_needs": 8,
        "general_severity": 7
    }
    
    decision = coordinator.coordinate_decision(situation)
    
    print(f"\n✓ Collective Decision:")
    print(f"  Decision: {decision['decision']}")
    print(f"  Consensus level: {decision['consensus_level']:.1%}")
    print(f"  Coordination quality: {decision['coordination_quality']:.1%}")
    print(f"\n  Agent Contributions:")
    for contrib in decision['agent_contributions']:
        print(f"    - {contrib['specialization']}: {contrib['recommendation']:.2f}")
    
    return decision


async def demo_explainable_ai():
    """Demo 7: Explainable AI"""
    print("\n" + "="*80)
    print("DEMO 7: EXPLAINABLE AI - TRANSPARENT DECISION MAKING")
    print("="*80)
    
    from core.explainable_ai import ExplainableAI
    from core.threat_analyzer import ThreatPatternRecognizer
    
    model = ThreatPatternRecognizer(input_dim=448)
    explainer = ExplainableAI(model)
    
    # Sample input
    input_data = torch.randn(1, 448)
    
    explanation = explainer.explain_prediction(
        input_data,
        feature_names=["signal_strength", "temporal_urgency", "spatial_density"]
    )
    
    print(f"\n✓ Prediction Explanation:")
    print(f"  {explanation['explanation_text']}")
    print(f"  Confidence: {explanation['confidence']:.1%}")
    print(f"\n  Top Features:")
    
    sorted_features = sorted(
        explanation['feature_importance'].items(),
        key=lambda x: x[1],
        reverse=True
    )[:5]
    
    for feature, importance in sorted_features:
        print(f"    - {feature}: {importance:.4f}")
    
    return explanation


async def demo_performance_metrics():
    """Demo 8: System Performance"""
    print("\n" + "="*80)
    print("DEMO 8: SYSTEM PERFORMANCE METRICS")
    print("="*80)
    
    orchestrator = AIOrchestrator(config={"device": "cpu"})
    
    # Run multiple iterations
    for i in range(5):
        signals = [{
            "type": "social_media",
            "source": "verified_media",
            "data": {"text": f"Test signal {i}"},
            "location": {"latitude": 40.7, "longitude": -74.0, "region": "NYC"},
            "temporal": {"timestamp": datetime.now().isoformat()}
        }]
        
        await orchestrator.process_intelligence_pipeline(signals)
    
    status = orchestrator.get_system_status()
    
    print(f"\n✓ System Status:")
    print(f"  Status: {status['status']}")
    print(f"  Active assessments: {status['active_assessments']}")
    print(f"  Monitoring: {'Active' if status['monitoring_active'] else 'Inactive'}")
    
    print(f"\n✓ Performance Metrics:")
    for key, value in status['performance'].items():
        print(f"  {key}: {value}")
    
    print(f"\n✓ Subsystems:")
    for subsystem, state in status['subsystems'].items():
        print(f"  {subsystem}: {state}")
    
    return status


async def run_all_demos():
    """Run all demonstrations"""
    print("\n" + "="*80)
    print("AI ENGINE - COMPREHENSIVE DEMONSTRATION")
    print("Advanced Intelligence System with State-of-the-Art Capabilities")
    print("="*80)
    
    demos = [
        ("Signal Processing", demo_signal_processing),
        ("Threat Analysis", demo_threat_analysis),
        ("Causal Inference", demo_causal_inference),
        ("Quantum Optimization", demo_quantum_optimization),
        ("Swarm Intelligence", demo_swarm_intelligence),
        ("Multi-Agent Coordination", demo_multiagent_coordination),
        ("Explainable AI", demo_explainable_ai),
        ("Performance Metrics", demo_performance_metrics)
    ]
    
    results = {}
    
    for name, demo_func in demos:
        try:
            result = await demo_func()
            results[name] = result
        except Exception as e:
            logger.error(f"Demo '{name}' failed: {e}")
            results[name] = None
    
    print("\n" + "="*80)
    print("DEMONSTRATION COMPLETE")
    print("="*80)
    print(f"\n✓ Successfully completed {sum(1 for r in results.values() if r is not None)}/{len(demos)} demos")
    print("\nThe AI Engine demonstrates:")
    print("  • Advanced signal processing and feature extraction")
    print("  • Multi-layered threat detection with ensemble learning")
    print("  • Causal inference and counterfactual reasoning")
    print("  • Quantum-inspired optimization algorithms")
    print("  • Swarm intelligence for distributed problem solving")
    print("  • Multi-agent coordination and consensus building")
    print("  • Full explainability and transparency")
    print("  • Real-time adaptive learning")
    print("\n" + "="*80 + "\n")
    
    return results


if __name__ == "__main__":
    asyncio.run(run_all_demos())
