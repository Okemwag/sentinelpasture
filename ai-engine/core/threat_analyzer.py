"""
Threat Analyzer - Multi-layered threat detection and pattern recognition
"""

import numpy as np
import torch
import torch.nn as nn
from typing import Dict, List, Any, Tuple
from datetime import datetime
from sklearn.cluster import DBSCAN
from sklearn.metrics.pairwise import cosine_similarity
import logging

logger = logging.getLogger(__name__)


class ThreatPatternRecognizer(nn.Module):
    """Neural network for threat pattern recognition"""
    
    def __init__(self, input_dim: int = 448, hidden_dims: List[int] = [256, 128, 64]):
        super().__init__()
        
        layers = []
        prev_dim = input_dim
        
        for hidden_dim in hidden_dims:
            layers.extend([
                nn.Linear(prev_dim, hidden_dim),
                nn.BatchNorm1d(hidden_dim),
                nn.ReLU(),
                nn.Dropout(0.3)
            ])
            prev_dim = hidden_dim
        
        # Output layer for 8 threat patterns
        layers.append(nn.Linear(prev_dim, 8))
        layers.append(nn.Sigmoid())
        
        self.network = nn.Sequential(*layers)
    
    def forward(self, x):
        return self.network(x)


class EscalationPredictor(nn.Module):
    """LSTM-based escalation prediction model"""
    
    def __init__(self, input_dim: int = 448, hidden_dim: int = 128, num_layers: int = 2):
        super().__init__()
        
        self.lstm = nn.LSTM(
            input_dim,
            hidden_dim,
            num_layers,
            batch_first=True,
            dropout=0.3,
            bidirectional=True
        )
        
        self.fc = nn.Sequential(
            nn.Linear(hidden_dim * 2, 64),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(64, 3)  # probability, timeframe, severity
        )
    
    def forward(self, x):
        # x shape: (batch, seq_len, input_dim)
        lstm_out, _ = self.lstm(x)
        # Take last output
        last_out = lstm_out[:, -1, :]
        return self.fc(last_out)


class ThreatAnalyzer:
    """
    Advanced Threat Analysis Engine
    Multi-layered threat detection, pattern recognition, and escalation prediction
    """
    
    def __init__(self, device: str = "cpu"):
        """
        Initialize Threat Analyzer
        
        Args:
            device: Device for PyTorch models ('cpu' or 'cuda')
        """
        self.device = torch.device(device)
        
        # Initialize models
        self.pattern_recognizer = ThreatPatternRecognizer().to(self.device)
        self.escalation_predictor = EscalationPredictor().to(self.device)
        
        # Set to eval mode (in production, load trained weights)
        self.pattern_recognizer.eval()
        self.escalation_predictor.eval()
        
        # Clustering for signal grouping
        self.clusterer = DBSCAN(eps=0.3, min_samples=3, metric='cosine')
        
        # Threat patterns
        self.threat_patterns = [
            "rapid_mobilization",
            "communication_disruption",
            "economic_stress",
            "social_unrest_precursors",
            "infrastructure_vulnerability",
            "coordinated_activity",
            "resource_scarcity",
            "institutional_stress"
        ]
        
        self.historical_threats = []
        
        logger.info(f"ThreatAnalyzer initialized on device: {device}")
    
    async def analyze_signals(self, signals: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        """
        Analyze signals to identify threat indicators
        
        Args:
            signals: List of processed signals
            
        Returns:
            List of threat indicators
        """
        if len(signals) == 0:
            return []
        
        logger.info(f"Analyzing {len(signals)} signals for threats")
        
        # Stage 1: Cluster signals
        clustered_signals = await self._cluster_signals(signals)
        
        # Stage 2: Analyze each cluster
        indicators = []
        for cluster in clustered_signals:
            indicator = await self._analyze_cluster(cluster)
            if indicator:
                indicators.append(indicator)
        
        # Stage 3: Cross-reference indicators
        await self._cross_reference_indicators(indicators)
        
        # Stage 4: Assess cascade risks
        for indicator in indicators:
            indicator["cascade_risks"] = await self._assess_cascade_risks(
                indicator, indicators
            )
        
        logger.info(f"Identified {len(indicators)} threat indicators")
        
        return indicators
    
    async def _cluster_signals(
        self, signals: List[Dict[str, Any]]
    ) -> List[List[Dict[str, Any]]]:
        """Cluster signals by spatiotemporal and semantic similarity"""
        
        if len(signals) < 2:
            return [signals]
        
        # Extract embeddings
        embeddings = np.array([s["embeddings"] for s in signals])
        
        # Normalize embeddings
        norms = np.linalg.norm(embeddings, axis=1, keepdims=True)
        embeddings_normalized = embeddings / (norms + 1e-8)
        
        # Cluster
        try:
            labels = self.clusterer.fit_predict(embeddings_normalized)
        except Exception as e:
            logger.warning(f"Clustering failed: {e}, using single cluster")
            return [signals]
        
        # Group by cluster
        clusters = {}
        for idx, label in enumerate(labels):
            if label == -1:  # Noise points
                continue
            if label not in clusters:
                clusters[label] = []
            clusters[label].append(signals[idx])
        
        # Add noise points as individual clusters if significant
        for idx, label in enumerate(labels):
            if label == -1 and signals[idx]["confidence"] > 0.7:
                clusters[f"noise_{idx}"] = [signals[idx]]
        
        return list(clusters.values())
    
    async def _analyze_cluster(
        self, cluster: List[Dict[str, Any]]
    ) -> Optional[Dict[str, Any]]:
        """Analyze a cluster of signals to identify threat indicator"""
        
        if len(cluster) == 0:
            return None
        
        # Aggregate cluster information
        aggregated = self._aggregate_signals(cluster)
        
        # Pattern matching using neural network
        patterns = await self._match_threat_patterns(aggregated, cluster)
        
        if len(patterns) == 0:
            return None
        
        # Assess threat level
        threat_level = await self._assess_threat_level(aggregated, patterns)
        
        if threat_level < 1:  # Minimal threat
            return None
        
        # Calculate confidence
        confidence = await self._calculate_analysis_confidence(cluster, patterns)
        
        # Predict escalation
        escalation = await self._predict_escalation(aggregated, cluster, patterns)
        
        # Generate mitigation strategies
        mitigation_strategies = self._generate_mitigation_strategies(
            patterns, threat_level
        )
        
        # Create indicator
        indicator = {
            "id": self._generate_indicator_id(cluster),
            "name": self._generate_indicator_name(patterns),
            "description": self._generate_indicator_description(aggregated, patterns),
            "category": patterns[0]["name"] if patterns else "Unknown",
            "severity": threat_level,
            "confidence": float(confidence),
            "signals": cluster,
            "patterns": patterns,
            "location": self._calculate_centroid(cluster),
            "temporal": self._calculate_temporal_range(cluster),
            "predicted_escalation": escalation,
            "related_indicators": [],
            "mitigation_strategies": mitigation_strategies,
            "timestamp": datetime.now().isoformat()
        }
        
        self.historical_threats.append(indicator)
        
        return indicator
    
    def _aggregate_signals(self, signals: List[Dict[str, Any]]) -> Dict[str, Any]:
        """Aggregate multiple signals into unified representation"""
        
        signal_types = list(set(s["type"] for s in signals))
        sources = list(set(s["source"] for s in signals))
        
        avg_confidence = np.mean([s["confidence"] for s in signals])
        avg_anomaly = np.mean([s["anomaly_score"] for s in signals])
        
        # Combine embeddings (average)
        embeddings = np.array([s["embeddings"] for s in signals])
        combined_embeddings = np.mean(embeddings, axis=0)
        
        return {
            "signal_count": len(signals),
            "types": signal_types,
            "sources": sources,
            "avg_confidence": float(avg_confidence),
            "avg_anomaly": float(avg_anomaly),
            "combined_embeddings": combined_embeddings,
            "time_span": self._calculate_time_span(signals),
            "spatial_spread": self._calculate_spatial_spread(signals),
        }
    
    async def _match_threat_patterns(
        self, aggregated: Dict[str, Any], cluster: List[Dict[str, Any]]
    ) -> List[Dict[str, Any]]:
        """Match against known threat patterns using neural network"""
        
        # Prepare input tensor
        embeddings = torch.FloatTensor(aggregated["combined_embeddings"]).unsqueeze(0)
        embeddings = embeddings.to(self.device)
        
        # Run pattern recognition
        with torch.no_grad():
            pattern_scores = self.pattern_recognizer(embeddings)
            pattern_scores = pattern_scores.cpu().numpy()[0]
        
        # Filter patterns above threshold
        patterns = []
        for idx, score in enumerate(pattern_scores):
            if score > 0.5:  # Threshold
                patterns.append({
                    "name": self.threat_patterns[idx],
                    "confidence": float(score),
                    "indicators": self._get_pattern_indicators(
                        self.threat_patterns[idx], aggregated
                    )
                })
        
        # Sort by confidence
        patterns.sort(key=lambda x: x["confidence"], reverse=True)
        
        return patterns
    
    def _get_pattern_indicators(
        self, pattern_name: str, aggregated: Dict[str, Any]
    ) -> List[str]:
        """Get indicators for specific pattern"""
        
        indicators_map = {
            "rapid_mobilization": ["increased_movement", "resource_concentration"],
            "communication_disruption": ["network_anomalies", "service_degradation"],
            "economic_stress": ["price_volatility", "supply_disruption"],
            "social_unrest_precursors": ["sentiment_shift", "gathering_activity"],
            "infrastructure_vulnerability": ["capacity_strain", "maintenance_gaps"],
            "coordinated_activity": ["synchronized_events", "network_coordination"],
            "resource_scarcity": ["supply_shortage", "distribution_issues"],
            "institutional_stress": ["capacity_overload", "response_delays"],
        }
        
        return indicators_map.get(pattern_name, [])
    
    async def _assess_threat_level(
        self, aggregated: Dict[str, Any], patterns: List[Dict[str, Any]]
    ) -> int:
        """Assess overall threat level (0-6)"""
        
        score = 0.0
        
        # Factor 1: Number and confidence of patterns
        score += len(patterns) * 0.5
        score += sum(p["confidence"] for p in patterns) / len(patterns) if patterns else 0
        
        # Factor 2: Signal strength
        score += aggregated["signal_count"] * 0.1
        score += aggregated["avg_confidence"] * 0.5
        
        # Factor 3: Spatial concentration
        score += (1 - min(aggregated["spatial_spread"] / 1000, 1)) * 0.3
        
        # Factor 4: Temporal urgency
        score += (1 - min(aggregated["time_span"] / (24 * 7), 1)) * 0.4
        
        # Factor 5: Anomaly score
        score += aggregated["avg_anomaly"] * 0.6
        
        # Map to threat level (0-6)
        if score < 1.0:
            return 0  # MINIMAL
        elif score < 2.0:
            return 1  # LOW
        elif score < 3.0:
            return 2  # MODERATE
        elif score < 4.0:
            return 3  # ELEVATED
        elif score < 5.0:
            return 4  # HIGH
        elif score < 6.0:
            return 5  # SEVERE
        else:
            return 6  # CRITICAL
    
    async def _calculate_analysis_confidence(
        self, cluster: List[Dict[str, Any]], patterns: List[Dict[str, Any]]
    ) -> float:
        """Calculate confidence in the analysis"""
        
        # Signal quality
        avg_signal_confidence = np.mean([s["confidence"] for s in cluster])
        
        # Pattern match strength
        avg_pattern_confidence = np.mean([p["confidence"] for p in patterns]) if patterns else 0
        
        # Data completeness
        completeness = min(len(cluster) / 10, 1.0)
        
        # Weighted combination
        confidence = (
            avg_signal_confidence * 0.4 +
            avg_pattern_confidence * 0.4 +
            completeness * 0.2
        )
        
        return min(1.0, max(0.0, confidence))
    
    async def _predict_escalation(
        self,
        aggregated: Dict[str, Any],
        cluster: List[Dict[str, Any]],
        patterns: List[Dict[str, Any]]
    ) -> Dict[str, Any]:
        """Predict escalation trajectory using LSTM"""
        
        # Prepare sequence (use last 10 signals or pad)
        sequence_length = 10
        embeddings_list = [s["embeddings"] for s in cluster[-sequence_length:]]
        
        # Pad if necessary
        while len(embeddings_list) < sequence_length:
            embeddings_list.insert(0, np.zeros_like(embeddings_list[0] if embeddings_list else np.zeros(448)))
        
        # Create tensor
        sequence = torch.FloatTensor(embeddings_list).unsqueeze(0)  # (1, seq_len, input_dim)
        sequence = sequence.to(self.device)
        
        # Predict
        with torch.no_grad():
            prediction = self.escalation_predictor(sequence)
            prediction = prediction.cpu().numpy()[0]
        
        probability = float(torch.sigmoid(torch.tensor(prediction[0])).item())
        timeframe_hours = max(1, int(abs(prediction[1]) * 72))  # 0-72 hours
        severity = int(min(6, max(0, prediction[2])))
        
        return {
            "probability": probability,
            "timeframe": self._format_timeframe(timeframe_hours),
            "potential_impact": self._estimate_impact(severity),
            "severity": severity
        }
    
    def _generate_mitigation_strategies(
        self, patterns: List[Dict[str, Any]], threat_level: int
    ) -> List[str]:
        """Generate mitigation strategies based on patterns and threat level"""
        
        strategies = []
        
        # Pattern-specific strategies
        for pattern in patterns[:3]:  # Top 3 patterns
            pattern_strategies = {
                "rapid_mobilization": [
                    "Increase surveillance in affected areas",
                    "Deploy rapid response teams",
                    "Establish communication channels with local leaders"
                ],
                "communication_disruption": [
                    "Activate backup communication systems",
                    "Deploy mobile communication units",
                    "Establish alternative information channels"
                ],
                "economic_stress": [
                    "Release strategic reserves",
                    "Implement price stabilization measures",
                    "Provide targeted economic support"
                ],
                "social_unrest_precursors": [
                    "Engage community leaders",
                    "Address grievances through dialogue",
                    "Increase visible but non-confrontational presence"
                ],
                "infrastructure_vulnerability": [
                    "Conduct emergency infrastructure assessment",
                    "Deploy maintenance teams",
                    "Activate redundancy systems"
                ],
                "coordinated_activity": [
                    "Enhance intelligence gathering",
                    "Coordinate with regional authorities",
                    "Monitor communication networks"
                ],
                "resource_scarcity": [
                    "Activate distribution networks",
                    "Coordinate with suppliers",
                    "Implement rationing if necessary"
                ],
                "institutional_stress": [
                    "Mobilize additional personnel",
                    "Streamline decision-making processes",
                    "Activate mutual aid agreements"
                ]
            }
            
            strategies.extend(pattern_strategies.get(pattern["name"], []))
        
        # Threat-level strategies
        if threat_level >= 4:
            strategies.extend([
                "Activate crisis management protocols",
                "Mobilize emergency resources",
                "Coordinate with regional authorities"
            ])
        
        # Deduplicate and limit
        return list(set(strategies))[:10]
    
    async def _cross_reference_indicators(
        self, indicators: List[Dict[str, Any]]
    ) -> None:
        """Cross-reference indicators to find relationships"""
        
        for i, ind1 in enumerate(indicators):
            for j, ind2 in enumerate(indicators[i+1:], i+1):
                relationship = self._assess_indicator_relationship(ind1, ind2)
                
                if relationship["strength"] > 0.5:
                    ind1["related_indicators"].append(ind2["id"])
                    ind2["related_indicators"].append(ind1["id"])
    
    def _assess_indicator_relationship(
        self, ind1: Dict[str, Any], ind2: Dict[str, Any]
    ) -> Dict[str, Any]:
        """Assess relationship between two indicators"""
        
        # Spatial proximity
        spatial_sim = self._calculate_spatial_similarity(
            ind1["location"], ind2["location"]
        )
        
        # Temporal proximity
        temporal_sim = self._calculate_temporal_similarity(
            ind1["temporal"], ind2["temporal"]
        )
        
        # Category similarity
        category_sim = 1.0 if ind1["category"] == ind2["category"] else 0.3
        
        # Overall strength
        strength = (
            spatial_sim * 0.35 +
            temporal_sim * 0.35 +
            category_sim * 0.30
        )
        
        return {
            "strength": strength,
            "type": "causal" if strength > 0.7 else "correlated" if strength > 0.4 else "coincidental"
        }
    
    async def _assess_cascade_risks(
        self, indicator: Dict[str, Any], all_indicators: List[Dict[str, Any]]
    ) -> List[Dict[str, Any]]:
        """Assess cascade risks from this indicator"""
        
        cascade_risks = []
        
        # Find related indicators that could be triggered
        for other in all_indicators:
            if other["id"] == indicator["id"]:
                continue
            
            relationship = self._assess_indicator_relationship(indicator, other)
            
            if relationship["strength"] > 0.6:
                cascade_risks.append({
                    "trigger": indicator["name"],
                    "consequence": other["name"],
                    "probability": relationship["strength"],
                    "impact": {
                        "severity": other["severity"],
                        "scope": "regional",
                        "affected_population": 100000  # Simplified
                    }
                })
        
        return cascade_risks[:5]  # Top 5 risks
    
    # Helper methods
    
    def _generate_indicator_id(self, cluster: List[Dict[str, Any]]) -> str:
        """Generate unique indicator ID"""
        timestamp = int(datetime.now().timestamp())
        hash_val = hash(tuple(s["id"] for s in cluster[:5]))
        return f"IND_{timestamp}_{abs(hash_val) % 100000}"
    
    def _generate_indicator_name(self, patterns: List[Dict[str, Any]]) -> str:
        """Generate indicator name"""
        if not patterns:
            return "Unknown Threat"
        return patterns[0]["name"].replace("_", " ").title()
    
    def _generate_indicator_description(
        self, aggregated: Dict[str, Any], patterns: List[Dict[str, Any]]
    ) -> str:
        """Generate indicator description"""
        pattern_names = ", ".join(p["name"] for p in patterns[:3])
        return (
            f"Detected {pattern_names} based on {aggregated['signal_count']} signals "
            f"across {len(aggregated['types'])} signal types"
        )
    
    def _calculate_centroid(self, cluster: List[Dict[str, Any]]) -> Dict[str, Any]:
        """Calculate geographic centroid of cluster"""
        lats = [s["location"].get("latitude", 0) for s in cluster]
        lons = [s["location"].get("longitude", 0) for s in cluster]
        
        return {
            "latitude": np.mean(lats) if lats else 0,
            "longitude": np.mean(lons) if lons else 0,
            "region": cluster[0]["location"].get("region", "unknown") if cluster else "unknown"
        }
    
    def _calculate_temporal_range(self, cluster: List[Dict[str, Any]]) -> Dict[str, Any]:
        """Calculate temporal range of cluster"""
        timestamps = [
            datetime.fromisoformat(s["timestamp"]) for s in cluster
        ]
        
        return {
            "start": min(timestamps).isoformat(),
            "end": max(timestamps).isoformat(),
            "duration_hours": (max(timestamps) - min(timestamps)).total_seconds() / 3600
        }
    
    def _calculate_time_span(self, signals: List[Dict[str, Any]]) -> float:
        """Calculate time span in hours"""
        timestamps = [datetime.fromisoformat(s["timestamp"]) for s in signals]
        return (max(timestamps) - min(timestamps)).total_seconds() / 3600
    
    def _calculate_spatial_spread(self, signals: List[Dict[str, Any]]) -> float:
        """Calculate spatial spread in km"""
        lats = [s["location"].get("latitude", 0) for s in signals]
        lons = [s["location"].get("longitude", 0) for s in signals]
        
        if len(lats) < 2:
            return 0.0
        
        # Simplified distance calculation
        max_dist = 0.0
        for i in range(len(lats)):
            for j in range(i+1, len(lats)):
                dist = np.sqrt((lats[i] - lats[j])**2 + (lons[i] - lons[j])**2) * 111  # Rough km conversion
                max_dist = max(max_dist, dist)
        
        return max_dist
    
    def _calculate_spatial_similarity(
        self, loc1: Dict[str, Any], loc2: Dict[str, Any]
    ) -> float:
        """Calculate spatial similarity (0-1)"""
        lat1, lon1 = loc1.get("latitude", 0), loc1.get("longitude", 0)
        lat2, lon2 = loc2.get("latitude", 0), loc2.get("longitude", 0)
        
        # Haversine-like distance
        dist = np.sqrt((lat1 - lat2)**2 + (lon1 - lon2)**2) * 111  # km
        
        # Convert to similarity (closer = more similar)
        return np.exp(-dist / 100)  # 100km decay
    
    def _calculate_temporal_similarity(
        self, temp1: Dict[str, Any], temp2: Dict[str, Any]
    ) -> float:
        """Calculate temporal similarity (0-1)"""
        try:
            t1 = datetime.fromisoformat(temp1.get("start", temp1.get("timestamp", datetime.now().isoformat())))
            t2 = datetime.fromisoformat(temp2.get("start", temp2.get("timestamp", datetime.now().isoformat())))
            
            hours_diff = abs((t1 - t2).total_seconds()) / 3600
            
            # Exponential decay
            return np.exp(-hours_diff / 24)  # 24 hour decay
        except:
            return 0.5
    
    def _format_timeframe(self, hours: int) -> str:
        """Format timeframe string"""
        if hours < 24:
            return f"{hours} hours"
        elif hours < 168:
            return f"{hours // 24} days"
        else:
            return f"{hours // 168} weeks"
    
    def _estimate_impact(self, severity: int) -> str:
        """Estimate impact description"""
        impact_map = {
            0: "Minimal local impact",
            1: "Limited local impact",
            2: "Moderate local impact",
            3: "Significant regional impact",
            4: "Major regional disruption",
            5: "Severe regional crisis",
            6: "Critical widespread crisis"
        }
        return impact_map.get(severity, "Unknown impact")
