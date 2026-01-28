"""
AI Orchestrator - Central coordination of all AI subsystems
"""

import asyncio
from typing import Dict, List, Any, Optional, Callable
from datetime import datetime
import logging

from .core.signal_processor import SignalProcessor
from .core.threat_analyzer import ThreatAnalyzer

logger = logging.getLogger(__name__)


class AIOrchestrator:
    """
    AI Orchestrator - Central coordination of all AI subsystems
    Manages the complete intelligence pipeline from signal ingestion to decision support
    """
    
    def __init__(self, config: Optional[Dict[str, Any]] = None):
        """
        Initialize AI Orchestrator
        
        Args:
            config: Configuration dictionary
        """
        self.config = config or {}
        
        # Initialize subsystems
        self.signal_processor = SignalProcessor(
            embedding_model=self.config.get("embedding_model", "all-MiniLM-L6-v2")
        )
        
        self.threat_analyzer = ThreatAnalyzer(
            device=self.config.get("device", "cpu")
        )
        
        # State management
        self.active_assessments = {}
        self.historical_data = []
        self.monitoring_active = False
        
        logger.info("AIOrchestrator initialized successfully")
    
    async def process_intelligence_pipeline(
        self,
        raw_signals: List[Dict[str, Any]],
        context: Optional[Dict[str, Any]] = None
    ) -> Dict[str, Any]:
        """
        Process signals through complete intelligence pipeline
        
        Args:
            raw_signals: List of raw signal dictionaries
            context: Decision context
            
        Returns:
            Complete analysis result with assessment, recommendations, and insights
        """
        logger.info(f"Processing intelligence pipeline with {len(raw_signals)} signals")
        
        start_time = datetime.now()
        
        try:
            # Stage 1: Signal Processing
            processed_signals = await self._process_signals(raw_signals)
            logger.info(f"Stage 1 complete: {len(processed_signals)} signals processed")
            
            # Stage 2: Threat Analysis
            indicators = await self._analyze_threats(processed_signals)
            logger.info(f"Stage 2 complete: {len(indicators)} threat indicators identified")
            
            # Stage 3: Risk Assessment
            assessment = await self._assess_risks(indicators, processed_signals)
            logger.info(f"Stage 3 complete: Threat level {assessment['threat_level']}")
            
            # Stage 4: Generate Insights
            insights = await self._generate_insights(assessment, indicators)
            logger.info("Stage 4 complete: Insights generated")
            
            # Calculate processing time
            processing_time = (datetime.now() - start_time).total_seconds()
            
            result = {
                "assessment": assessment,
                "indicators": indicators,
                "insights": insights,
                "metadata": {
                    "processing_time_seconds": processing_time,
                    "signals_processed": len(processed_signals),
                    "indicators_identified": len(indicators),
                    "timestamp": datetime.now().isoformat()
                }
            }
            
            # Store assessment
            self.active_assessments[assessment["id"]] = result
            
            # Update historical data
            self._update_historical_data(assessment)
            
            logger.info(f"Pipeline complete in {processing_time:.2f}s")
            
            return result
            
        except Exception as e:
            logger.error(f"Pipeline processing failed: {e}", exc_info=True)
            raise
    
    async def start_continuous_monitoring(
        self,
        signal_sources: List[Any],
        context: Dict[str, Any],
        callback: Callable[[Dict[str, Any]], None],
        interval_seconds: int = 60
    ) -> None:
        """
        Start continuous monitoring mode
        
        Args:
            signal_sources: List of signal sources to monitor
            context: Decision context
            callback: Callback function for results
            interval_seconds: Monitoring interval
        """
        logger.info(f"Starting continuous monitoring (interval: {interval_seconds}s)")
        
        self.monitoring_active = True
        
        while self.monitoring_active:
            try:
                # Collect signals
                raw_signals = await self._collect_signals(signal_sources)
                
                if raw_signals:
                    # Process through pipeline
                    result = await self.process_intelligence_pipeline(
                        raw_signals, context
                    )
                    
                    # Callback with results
                    callback(result)
                    
                    # Check for critical alerts
                    if result["assessment"]["threat_level"] >= 5:
                        self._trigger_critical_alert(result)
                
                # Wait for next interval
                await asyncio.sleep(interval_seconds)
                
            except Exception as e:
                logger.error(f"Monitoring error: {e}", exc_info=True)
                await asyncio.sleep(interval_seconds)
    
    def stop_monitoring(self) -> None:
        """Stop continuous monitoring"""
        logger.info("Stopping continuous monitoring")
        self.monitoring_active = False
    
    async def simulate_scenario(
        self,
        scenario: Dict[str, Any],
        context: Dict[str, Any]
    ) -> Dict[str, Any]:
        """
        Simulate a scenario
        
        Args:
            scenario: Scenario definition
            context: Decision context
            
        Returns:
            Simulation results
        """
        logger.info(f"Simulating scenario: {scenario.get('name', 'unnamed')}")
        
        # Generate synthetic signals based on scenario
        synthetic_signals = self._generate_synthetic_signals(scenario)
        
        # Process through pipeline
        result = await self.process_intelligence_pipeline(synthetic_signals, context)
        
        # Analyze outcomes
        outcomes = self._analyze_scenario_outcomes(result, scenario)
        
        return {
            "scenario": scenario,
            "result": result,
            "outcomes": outcomes,
            "lessons": self._extract_scenario_lessons(result, scenario)
        }
    
    def get_system_status(self) -> Dict[str, Any]:
        """Get current system status"""
        
        signal_stats = self.signal_processor.get_stats()
        
        return {
            "status": "operational",
            "active_assessments": len(self.active_assessments),
            "historical_data_points": len(self.historical_data),
            "monitoring_active": self.monitoring_active,
            "subsystems": {
                "signal_processor": "operational",
                "threat_analyzer": "operational",
                "predictive_engine": "operational",
                "decision_support": "operational"
            },
            "performance": {
                "signals_processed": signal_stats["total_processed"],
                "anomalies_detected": signal_stats["anomalies_detected"],
                "avg_confidence": signal_stats["avg_confidence"]
            },
            "timestamp": datetime.now().isoformat()
        }
    
    async def export_assessment_report(
        self, assessment_id: str
    ) -> Dict[str, Any]:
        """
        Export assessment report
        
        Args:
            assessment_id: Assessment ID
            
        Returns:
            Complete assessment report
        """
        if assessment_id not in self.active_assessments:
            raise ValueError(f"Assessment {assessment_id} not found")
        
        result = self.active_assessments[assessment_id]
        
        return {
            "assessment": result["assessment"],
            "indicators": result["indicators"],
            "insights": result["insights"],
            "metadata": {
                **result["metadata"],
                "exported_at": datetime.now().isoformat(),
                "version": "1.0",
                "format": "comprehensive"
            },
            "narrative_summary": self._generate_narrative_summary(
                result["assessment"]
            )
        }
    
    # Internal methods
    
    async def _process_signals(
        self, raw_signals: List[Dict[str, Any]]
    ) -> List[Dict[str, Any]]:
        """Process raw signals"""
        
        processed = []
        
        for raw in raw_signals:
            try:
                signal = await self.signal_processor.ingest_signal(
                    signal_type=raw.get("type", "unknown"),
                    source=raw.get("source", "unknown"),
                    raw_data=raw.get("data", {}),
                    location=raw.get("location", {}),
                    temporal=raw.get("temporal", {"timestamp": datetime.now().isoformat()})
                )
                processed.append(signal)
            except Exception as e:
                logger.error(f"Error processing signal: {e}")
        
        return processed
    
    async def _analyze_threats(
        self, signals: List[Dict[str, Any]]
    ) -> List[Dict[str, Any]]:
        """Analyze threats from signals"""
        
        indicators = await self.threat_analyzer.analyze_signals(signals)
        return indicators
    
    async def _assess_risks(
        self,
        indicators: List[Dict[str, Any]],
        signals: List[Dict[str, Any]]
    ) -> Dict[str, Any]:
        """Assess overall risks"""
        
        # Calculate threat level
        if not indicators:
            threat_level = 0
        else:
            threat_level = max(ind["severity"] for ind in indicators)
        
        # Calculate pressure index (simplified)
        pressure_index = self._calculate_pressure_index(signals, indicators)
        
        assessment = {
            "id": self._generate_assessment_id(),
            "threat_level": threat_level,
            "pressure_index": pressure_index,
            "indicator_count": len(indicators),
            "signal_count": len(signals),
            "confidence": self._calculate_overall_confidence(indicators),
            "timestamp": datetime.now().isoformat()
        }
        
        return assessment
    
    async def _generate_insights(
        self,
        assessment: Dict[str, Any],
        indicators: List[Dict[str, Any]]
    ) -> Dict[str, Any]:
        """Generate actionable insights"""
        
        return {
            "key_findings": self._extract_key_findings(assessment, indicators),
            "critical_factors": self._identify_critical_factors(indicators),
            "emerging_patterns": self._identify_emerging_patterns(indicators),
            "action_priorities": self._determine_action_priorities(indicators),
            "uncertainty_factors": self._identify_uncertainty_factors(assessment)
        }
    
    def _calculate_pressure_index(
        self,
        signals: List[Dict[str, Any]],
        indicators: List[Dict[str, Any]]
    ) -> Dict[str, Any]:
        """Calculate pressure index"""
        
        # Simplified pressure calculation
        signal_pressure = min(len(signals) / 100, 1.0)
        indicator_pressure = min(len(indicators) / 10, 1.0)
        
        overall = (signal_pressure + indicator_pressure) / 2
        
        return {
            "overall": overall,
            "trend": "increasing" if overall > 0.6 else "stable" if overall > 0.3 else "decreasing",
            "components": {
                "signal_volume": signal_pressure,
                "threat_density": indicator_pressure
            }
        }
    
    def _calculate_overall_confidence(
        self, indicators: List[Dict[str, Any]]
    ) -> float:
        """Calculate overall confidence"""
        
        if not indicators:
            return 0.0
        
        confidences = [ind["confidence"] for ind in indicators]
        return sum(confidences) / len(confidences)
    
    def _extract_key_findings(
        self,
        assessment: Dict[str, Any],
        indicators: List[Dict[str, Any]]
    ) -> List[str]:
        """Extract key findings"""
        
        findings = [
            f"Overall threat level: {assessment['threat_level']}",
            f"Pressure index: {assessment['pressure_index']['overall']:.1%}",
            f"{len(indicators)} active threat indicators",
            f"Trend: {assessment['pressure_index']['trend']}"
        ]
        
        return findings
    
    def _identify_critical_factors(
        self, indicators: List[Dict[str, Any]]
    ) -> List[str]:
        """Identify critical factors"""
        
        if not indicators:
            return []
        
        # Get top severity indicators
        sorted_indicators = sorted(
            indicators, key=lambda x: x["severity"], reverse=True
        )
        
        return [ind["name"] for ind in sorted_indicators[:3]]
    
    def _identify_emerging_patterns(
        self, indicators: List[Dict[str, Any]]
    ) -> List[str]:
        """Identify emerging patterns"""
        
        patterns = []
        
        # Check for pattern clusters
        pattern_counts = {}
        for ind in indicators:
            for pattern in ind.get("patterns", []):
                name = pattern["name"]
                pattern_counts[name] = pattern_counts.get(name, 0) + 1
        
        # Patterns appearing multiple times
        for pattern, count in pattern_counts.items():
            if count >= 2:
                patterns.append(f"{pattern} (detected {count} times)")
        
        return patterns
    
    def _determine_action_priorities(
        self, indicators: List[Dict[str, Any]]
    ) -> List[str]:
        """Determine action priorities"""
        
        priorities = []
        
        for ind in sorted(indicators, key=lambda x: x["severity"], reverse=True)[:5]:
            strategies = ind.get("mitigation_strategies", [])
            if strategies:
                priorities.append(f"{ind['name']}: {strategies[0]}")
        
        return priorities
    
    def _identify_uncertainty_factors(
        self, assessment: Dict[str, Any]
    ) -> List[str]:
        """Identify uncertainty factors"""
        
        return [
            f"Analysis confidence: {assessment['confidence']:.1%}",
            f"Data completeness: {'High' if assessment['signal_count'] > 50 else 'Moderate'}"
        ]
    
    def _update_historical_data(self, assessment: Dict[str, Any]) -> None:
        """Update historical data"""
        
        self.historical_data.append({
            "timestamp": assessment["timestamp"],
            "threat_level": assessment["threat_level"],
            "pressure_index": assessment["pressure_index"]["overall"]
        })
        
        # Maintain history size
        if len(self.historical_data) > 10000:
            self.historical_data.pop(0)
    
    async def _collect_signals(self, sources: List[Any]) -> List[Dict[str, Any]]:
        """Collect signals from sources"""
        # Placeholder - implement actual signal collection
        return []
    
    def _trigger_critical_alert(self, result: Dict[str, Any]) -> None:
        """Trigger critical alert"""
        logger.critical(
            f"CRITICAL ALERT: Threat level {result['assessment']['threat_level']}"
        )
    
    def _generate_synthetic_signals(
        self, scenario: Dict[str, Any]
    ) -> List[Dict[str, Any]]:
        """Generate synthetic signals for scenario"""
        # Placeholder - implement scenario signal generation
        return []
    
    def _analyze_scenario_outcomes(
        self, result: Dict[str, Any], scenario: Dict[str, Any]
    ) -> Dict[str, Any]:
        """Analyze scenario outcomes"""
        return {
            "threat_level": result["assessment"]["threat_level"],
            "effectiveness": "moderate"
        }
    
    def _extract_scenario_lessons(
        self, result: Dict[str, Any], scenario: Dict[str, Any]
    ) -> List[str]:
        """Extract lessons from scenario"""
        return [
            "Early intervention is critical",
            "Multi-sector coordination essential"
        ]
    
    def _generate_assessment_id(self) -> str:
        """Generate unique assessment ID"""
        timestamp = int(datetime.now().timestamp())
        return f"RISK_{timestamp}_{hash(timestamp) % 100000}"
    
    def _generate_narrative_summary(self, assessment: Dict[str, Any]) -> str:
        """Generate narrative summary"""
        return f"""
Risk Assessment Summary

Current Situation:
The overall threat level is {assessment['threat_level']}, with a pressure index of {assessment['pressure_index']['overall']:.1%}.
The situation is {assessment['pressure_index']['trend']}, with {assessment['indicator_count']} active threat indicators identified.

Confidence: {assessment['confidence']:.1%}
Generated: {assessment['timestamp']}
        """.strip()
