"""
Signal Processor - Multi-modal signal ingestion and processing
"""

import numpy as np
import torch
from typing import Dict, List, Any, Optional
from datetime import datetime
from collections import defaultdict
from sentence_transformers import SentenceTransformer
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import IsolationForest
import logging

logger = logging.getLogger(__name__)


class SignalProcessor:
    """
    Advanced Signal Processing Engine
    Handles multi-modal signal ingestion, normalization, and feature extraction
    """
    
    def __init__(self, embedding_model: str = "all-MiniLM-L6-v2"):
        """
        Initialize Signal Processor
        
        Args:
            embedding_model: Sentence transformer model for embeddings
        """
        self.embedding_model = SentenceTransformer(embedding_model)
        self.signal_buffer = defaultdict(list)
        self.scaler = StandardScaler()
        self.anomaly_detector = IsolationForest(contamination=0.1, random_state=42)
        self.processing_stats = {
            "total_processed": 0,
            "anomalies_detected": 0,
            "avg_confidence": 0.0
        }
        
        logger.info(f"SignalProcessor initialized with model: {embedding_model}")
    
    async def ingest_signal(
        self,
        signal_type: str,
        source: str,
        raw_data: Dict[str, Any],
        location: Dict[str, Any],
        temporal: Dict[str, Any]
    ) -> Dict[str, Any]:
        """
        Ingest and process a raw signal through the complete pipeline
        
        Args:
            signal_type: Type of signal (social_media, economic, etc.)
            source: Source of the signal
            raw_data: Raw signal data
            location: Geographic location information
            temporal: Temporal context
            
        Returns:
            Processed signal dictionary
        """
        signal_id = self._generate_signal_id(signal_type, source, temporal)
        
        # Stage 1: Normalization
        normalized_data = await self._normalize_data(raw_data, signal_type)
        
        # Stage 2: Feature Extraction
        features = await self._extract_features(normalized_data, location, temporal)
        
        # Stage 3: Embedding Generation
        embeddings = await self._generate_embeddings(normalized_data, features)
        
        # Stage 4: Confidence Scoring
        confidence = await self._calculate_confidence(
            source, normalized_data, features
        )
        
        # Stage 5: Anomaly Detection
        anomaly_score = await self._detect_anomalies(features, embeddings)
        
        # Stage 6: Contextual Enrichment
        context = await self._enrich_context(signal_type, location, temporal)
        
        signal = {
            "id": signal_id,
            "type": signal_type,
            "source": source,
            "raw_data": raw_data,
            "processed_data": normalized_data,
            "features": features,
            "embeddings": embeddings.tolist(),
            "confidence": float(confidence),
            "anomaly_score": float(anomaly_score),
            "context": context,
            "location": location,
            "temporal": temporal,
            "timestamp": datetime.now().isoformat()
        }
        
        # Add to buffer
        self._add_to_buffer(signal)
        
        # Update stats
        self.processing_stats["total_processed"] += 1
        if anomaly_score > 0.7:
            self.processing_stats["anomalies_detected"] += 1
        
        logger.debug(f"Processed signal {signal_id} with confidence {confidence:.3f}")
        
        return signal
    
    async def _normalize_data(
        self, raw_data: Dict[str, Any], signal_type: str
    ) -> Dict[str, Any]:
        """Normalize heterogeneous data formats"""
        
        normalizers = {
            "social_media": self._normalize_social_media,
            "economic": self._normalize_economic,
            "infrastructure": self._normalize_infrastructure,
            "demographic": self._normalize_demographic,
            "environmental": self._normalize_environmental,
            "political": self._normalize_political,
            "security": self._normalize_security,
            "health": self._normalize_health,
        }
        
        normalizer = normalizers.get(signal_type, lambda x: x)
        return normalizer(raw_data)
    
    def _normalize_social_media(self, data: Dict[str, Any]) -> Dict[str, Any]:
        """Normalize social media data"""
        return {
            "content": data.get("text", data.get("content", "")),
            "sentiment": self._analyze_sentiment(data.get("text", "")),
            "engagement": {
                "likes": data.get("likes", 0),
                "shares": data.get("shares", 0),
                "comments": data.get("comments", 0),
            },
            "reach": data.get("followers", data.get("reach", 0)),
            "verified": data.get("verified", False),
            "language": data.get("language", "unknown"),
            "hashtags": data.get("hashtags", []),
            "mentions": data.get("mentions", []),
        }
    
    def _normalize_economic(self, data: Dict[str, Any]) -> Dict[str, Any]:
        """Normalize economic data"""
        return {
            "indicators": {
                "gdp": data.get("gdp"),
                "inflation": data.get("inflation"),
                "unemployment": data.get("unemployment"),
                "price_indices": data.get("price_indices", {}),
            },
            "market_data": {
                "commodity_prices": data.get("commodity_prices", {}),
                "exchange_rates": data.get("exchange_rates", {}),
                "stock_indices": data.get("stock_indices", {}),
            },
            "fiscal_data": {
                "revenue": data.get("revenue"),
                "expenditure": data.get("expenditure"),
                "deficit": data.get("deficit"),
            },
        }
    
    def _normalize_infrastructure(self, data: Dict[str, Any]) -> Dict[str, Any]:
        """Normalize infrastructure data"""
        current_capacity = data.get("current_capacity", 0)
        max_capacity = data.get("max_capacity", 1)
        
        return {
            "type": data.get("infrastructure_type"),
            "status": data.get("status", "operational"),
            "capacity": {
                "current": current_capacity,
                "maximum": max_capacity,
                "utilization": current_capacity / max_capacity if max_capacity > 0 else 0,
            },
            "condition": {
                "score": data.get("condition_score", 0),
                "last_maintenance": data.get("last_maintenance"),
                "next_maintenance": data.get("next_maintenance"),
            },
            "incidents": data.get("incidents", []),
        }
    
    def _normalize_demographic(self, data: Dict[str, Any]) -> Dict[str, Any]:
        """Normalize demographic data"""
        return {
            "population": {
                "total": data.get("population"),
                "density": data.get("density"),
                "growth": data.get("growth_rate"),
            },
            "distribution": {
                "age": data.get("age_distribution", {}),
                "gender": data.get("gender_distribution", {}),
                "ethnicity": data.get("ethnicity_distribution", {}),
            },
            "migration": {
                "inflow": data.get("immigration", 0),
                "outflow": data.get("emigration", 0),
                "internal": data.get("internal_migration", 0),
            },
            "socioeconomic": {
                "income": data.get("income_distribution", {}),
                "education": data.get("education_levels", {}),
                "employment": data.get("employment_status", {}),
            },
        }
    
    def _normalize_environmental(self, data: Dict[str, Any]) -> Dict[str, Any]:
        """Normalize environmental data"""
        return {
            "climate": {
                "temperature": data.get("temperature"),
                "precipitation": data.get("precipitation"),
                "humidity": data.get("humidity"),
            },
            "air_quality": {
                "aqi": data.get("aqi"),
                "pollutants": data.get("pollutants", {}),
            },
            "natural_events": {
                "type": data.get("event_type"),
                "severity": data.get("severity"),
                "affected": data.get("affected_area"),
            },
            "resources": {
                "water": data.get("water_availability"),
                "land": data.get("land_use"),
                "energy": data.get("energy_production"),
            },
        }
    
    def _normalize_political(self, data: Dict[str, Any]) -> Dict[str, Any]:
        """Normalize political data"""
        return {
            "events": {
                "type": data.get("event_type"),
                "participants": data.get("participants", []),
                "location": data.get("location"),
                "size": data.get("attendees", 0),
            },
            "sentiment": {
                "approval": data.get("approval_rating"),
                "polarization": data.get("polarization_index"),
                "trust": data.get("trust_level"),
            },
            "activity": {
                "protests": data.get("protest_count", 0),
                "meetings": data.get("meeting_count", 0),
                "statements": data.get("statements", []),
            },
        }
    
    def _normalize_security(self, data: Dict[str, Any]) -> Dict[str, Any]:
        """Normalize security data"""
        return {
            "incidents": {
                "type": data.get("incident_type"),
                "severity": data.get("severity"),
                "casualties": data.get("casualties", 0),
                "perpetrators": data.get("perpetrators", "unknown"),
            },
            "deployment": {
                "personnel": data.get("personnel_count"),
                "equipment": data.get("equipment", []),
                "location": data.get("deployment_location"),
            },
            "intelligence": {
                "source": data.get("intel_source"),
                "classification": data.get("classification"),
                "reliability": data.get("reliability"),
            },
        }
    
    def _normalize_health(self, data: Dict[str, Any]) -> Dict[str, Any]:
        """Normalize health data"""
        return {
            "disease": {
                "type": data.get("disease_type"),
                "cases": data.get("case_count"),
                "deaths": data.get("death_count"),
                "recoveries": data.get("recovery_count"),
            },
            "healthcare": {
                "capacity": data.get("hospital_capacity"),
                "utilization": data.get("bed_occupancy"),
                "resources": data.get("medical_resources", {}),
            },
            "surveillance": {
                "testing_rate": data.get("testing_rate"),
                "positivity_rate": data.get("positivity_rate"),
                "trends": data.get("trends", []),
            },
        }
    
    async def _extract_features(
        self,
        normalized_data: Dict[str, Any],
        location: Dict[str, Any],
        temporal: Dict[str, Any]
    ) -> Dict[str, Any]:
        """Extract multi-dimensional features"""
        
        return {
            "temporal": self._extract_temporal_features(temporal),
            "spatial": self._extract_spatial_features(location),
            "statistical": self._extract_statistical_features(normalized_data),
            "semantic": await self._extract_semantic_features(normalized_data),
        }
    
    def _extract_temporal_features(self, temporal: Dict[str, Any]) -> Dict[str, Any]:
        """Extract temporal features"""
        timestamp = datetime.fromisoformat(temporal.get("timestamp", datetime.now().isoformat()))
        
        return {
            "hour": timestamp.hour,
            "day_of_week": timestamp.weekday(),
            "day_of_month": timestamp.day,
            "month": timestamp.month,
            "quarter": (timestamp.month - 1) // 3 + 1,
            "year": timestamp.year,
            "is_weekend": timestamp.weekday() >= 5,
            "time_of_day": self._categorize_time_of_day(timestamp.hour),
        }
    
    def _extract_spatial_features(self, location: Dict[str, Any]) -> Dict[str, Any]:
        """Extract spatial features"""
        return {
            "coordinates": {
                "lat": location.get("latitude"),
                "lon": location.get("longitude"),
            },
            "region": location.get("region"),
            "urbanization": self._estimate_urbanization(location),
            "population_density": self._estimate_population_density(location),
        }
    
    def _extract_statistical_features(self, data: Dict[str, Any]) -> Dict[str, Any]:
        """Extract statistical features from data"""
        values = self._extract_numeric_values(data)
        
        if len(values) == 0:
            return {"mean": 0, "median": 0, "std": 0, "min": 0, "max": 0}
        
        return {
            "mean": float(np.mean(values)),
            "median": float(np.median(values)),
            "std": float(np.std(values)),
            "min": float(np.min(values)),
            "max": float(np.max(values)),
        }
    
    async def _extract_semantic_features(
        self, data: Dict[str, Any]
    ) -> Dict[str, Any]:
        """Extract semantic features from text content"""
        text_content = self._extract_text_content(data)
        
        return {
            "keywords": self._extract_keywords(text_content),
            "sentiment": self._analyze_sentiment(text_content),
            "urgency": self._assess_urgency(text_content),
            "complexity": self._assess_complexity(text_content),
        }
    
    async def _generate_embeddings(
        self, normalized_data: Dict[str, Any], features: Dict[str, Any]
    ) -> np.ndarray:
        """Generate high-dimensional embeddings"""
        
        # Text embeddings
        text_content = self._extract_text_content(normalized_data)
        if text_content:
            text_embeddings = self.embedding_model.encode(
                text_content, convert_to_numpy=True
            )
        else:
            text_embeddings = np.zeros(384)  # Default embedding size
        
        # Structural embeddings
        structural_features = self._flatten_dict(features)
        structural_embeddings = np.array(list(structural_features.values())[:64])
        
        # Pad if necessary
        if len(structural_embeddings) < 64:
            structural_embeddings = np.pad(
                structural_embeddings,
                (0, 64 - len(structural_embeddings)),
                mode='constant'
            )
        
        # Combine embeddings
        combined = np.concatenate([text_embeddings, structural_embeddings])
        
        return combined
    
    async def _calculate_confidence(
        self,
        source: str,
        normalized_data: Dict[str, Any],
        features: Dict[str, Any]
    ) -> float:
        """Calculate confidence score for signal"""
        
        # Source reliability
        source_reliability = self._get_source_reliability(source)
        
        # Data completeness
        completeness = self._assess_data_completeness(normalized_data)
        
        # Feature quality
        feature_quality = self._assess_feature_quality(features)
        
        # Weighted combination
        confidence = (
            source_reliability * 0.4 +
            completeness * 0.3 +
            feature_quality * 0.3
        )
        
        return min(1.0, max(0.0, confidence))
    
    async def _detect_anomalies(
        self, features: Dict[str, Any], embeddings: np.ndarray
    ) -> float:
        """Detect anomalies in signal"""
        
        # Use embeddings for anomaly detection
        try:
            # Reshape for sklearn
            X = embeddings.reshape(1, -1)
            
            # Fit and predict (in production, use pre-fitted model)
            score = self.anomaly_detector.fit_predict(X)[0]
            
            # Convert to probability (0-1)
            anomaly_score = 1.0 if score == -1 else 0.0
            
        except Exception as e:
            logger.warning(f"Anomaly detection failed: {e}")
            anomaly_score = 0.0
        
        return anomaly_score
    
    async def _enrich_context(
        self, signal_type: str, location: Dict[str, Any], temporal: Dict[str, Any]
    ) -> Dict[str, Any]:
        """Enrich signal with contextual information"""
        
        return {
            "historical_context": self._get_historical_context(signal_type, location),
            "spatial_context": self._get_spatial_context(location),
            "temporal_context": self._get_temporal_context(temporal),
        }
    
    # Helper methods
    
    def _generate_signal_id(
        self, signal_type: str, source: str, temporal: Dict[str, Any]
    ) -> str:
        """Generate unique signal ID"""
        timestamp = datetime.now().timestamp()
        return f"{signal_type}_{source}_{int(timestamp)}_{np.random.randint(10000)}"
    
    def _analyze_sentiment(self, text: str) -> float:
        """Simple sentiment analysis (-1 to 1)"""
        if not text:
            return 0.0
        
        positive_words = ["good", "great", "excellent", "positive", "success", "peace"]
        negative_words = ["bad", "terrible", "crisis", "negative", "failure", "violence"]
        
        text_lower = text.lower()
        pos_count = sum(1 for word in positive_words if word in text_lower)
        neg_count = sum(1 for word in negative_words if word in text_lower)
        
        total = pos_count + neg_count
        if total == 0:
            return 0.0
        
        return (pos_count - neg_count) / total
    
    def _categorize_time_of_day(self, hour: int) -> str:
        """Categorize time of day"""
        if 6 <= hour < 12:
            return "morning"
        elif 12 <= hour < 18:
            return "afternoon"
        elif 18 <= hour < 22:
            return "evening"
        else:
            return "night"
    
    def _estimate_urbanization(self, location: Dict[str, Any]) -> float:
        """Estimate urbanization level"""
        # Simplified - in production, use actual data
        return 0.5
    
    def _estimate_population_density(self, location: Dict[str, Any]) -> float:
        """Estimate population density"""
        # Simplified - in production, use actual data
        return 100.0
    
    def _extract_numeric_values(self, data: Dict[str, Any]) -> List[float]:
        """Extract all numeric values from nested dict"""
        values = []
        
        def extract(obj):
            if isinstance(obj, (int, float)):
                values.append(float(obj))
            elif isinstance(obj, dict):
                for v in obj.values():
                    extract(v)
            elif isinstance(obj, list):
                for item in obj:
                    extract(item)
        
        extract(data)
        return values
    
    def _extract_text_content(self, data: Dict[str, Any]) -> str:
        """Extract text content from data"""
        if isinstance(data, str):
            return data
        if isinstance(data, dict):
            if "content" in data:
                return str(data["content"])
            if "text" in data:
                return str(data["text"])
        return ""
    
    def _extract_keywords(self, text: str, top_n: int = 10) -> List[str]:
        """Extract keywords from text"""
        if not text:
            return []
        
        # Simple keyword extraction
        words = text.lower().split()
        stop_words = {"the", "is", "at", "which", "on", "a", "an", "and", "or"}
        keywords = [w for w in words if len(w) > 3 and w not in stop_words]
        
        return list(set(keywords))[:top_n]
    
    def _assess_urgency(self, text: str) -> float:
        """Assess urgency level from text"""
        if not text:
            return 0.0
        
        urgent_keywords = ["urgent", "immediate", "critical", "emergency", "crisis"]
        text_lower = text.lower()
        
        urgency_score = sum(0.2 for keyword in urgent_keywords if keyword in text_lower)
        return min(1.0, urgency_score)
    
    def _assess_complexity(self, text: str) -> float:
        """Assess text complexity"""
        if not text:
            return 0.0
        
        words = text.split()
        if len(words) == 0:
            return 0.0
        
        avg_word_length = sum(len(w) for w in words) / len(words)
        return min(1.0, avg_word_length / 10)
    
    def _flatten_dict(self, d: Dict[str, Any], parent_key: str = '') -> Dict[str, float]:
        """Flatten nested dictionary to numeric values"""
        items = {}
        for k, v in d.items():
            new_key = f"{parent_key}_{k}" if parent_key else k
            if isinstance(v, dict):
                items.update(self._flatten_dict(v, new_key))
            elif isinstance(v, (int, float)):
                items[new_key] = float(v)
        return items
    
    def _get_source_reliability(self, source: str) -> float:
        """Get source reliability score"""
        reliability_map = {
            "official_government": 0.95,
            "verified_media": 0.85,
            "academic": 0.90,
            "ngo": 0.80,
            "social_media": 0.50,
            "anonymous": 0.30,
        }
        return reliability_map.get(source, 0.50)
    
    def _assess_data_completeness(self, data: Dict[str, Any]) -> float:
        """Assess data completeness"""
        total_fields = len(self._flatten_dict(data))
        if total_fields == 0:
            return 0.0
        
        filled_fields = sum(1 for v in self._flatten_dict(data).values() if v != 0)
        return filled_fields / total_fields
    
    def _assess_feature_quality(self, features: Dict[str, Any]) -> float:
        """Assess feature quality"""
        # Simplified quality assessment
        return 0.8
    
    def _get_historical_context(
        self, signal_type: str, location: Dict[str, Any]
    ) -> Dict[str, Any]:
        """Get historical context"""
        return {"similar_events": [], "trends": []}
    
    def _get_spatial_context(self, location: Dict[str, Any]) -> Dict[str, Any]:
        """Get spatial context"""
        return {"nearby_events": [], "regional_factors": []}
    
    def _get_temporal_context(self, temporal: Dict[str, Any]) -> Dict[str, Any]:
        """Get temporal context"""
        return {"seasonal_patterns": [], "historical_periods": []}
    
    def _add_to_buffer(self, signal: Dict[str, Any]) -> None:
        """Add signal to buffer"""
        key = f"{signal['type']}_{signal['location'].get('region', 'unknown')}"
        self.signal_buffer[key].append(signal)
        
        # Maintain buffer size
        if len(self.signal_buffer[key]) > 1000:
            self.signal_buffer[key].pop(0)
    
    def get_stats(self) -> Dict[str, Any]:
        """Get processing statistics"""
        return self.processing_stats
