"""
Constants and Enumerations
"""

from enum import Enum, IntEnum


# Default configuration values
DEFAULT_EMBEDDING_MODEL = "all-MiniLM-L6-v2"
DEFAULT_DEVICE = "cpu"
MAX_SIGNAL_BUFFER_SIZE = 10000
DEFAULT_BATCH_SIZE = 32


class ThreatLevel(IntEnum):
    """Threat level enumeration"""
    MINIMAL = 0
    LOW = 1
    MODERATE = 2
    ELEVATED = 3
    HIGH = 4
    SEVERE = 5
    CRITICAL = 6


class SignalType(str, Enum):
    """Signal type enumeration"""
    SOCIAL_MEDIA = "social_media"
    ECONOMIC = "economic"
    INFRASTRUCTURE = "infrastructure"
    DEMOGRAPHIC = "demographic"
    ENVIRONMENTAL = "environmental"
    POLITICAL = "political"
    SECURITY = "security"
    HEALTH = "health"
    UNKNOWN = "unknown"


class SourceReliability(str, Enum):
    """Source reliability levels"""
    OFFICIAL_GOVERNMENT = "official_government"
    VERIFIED_MEDIA = "verified_media"
    ACADEMIC = "academic"
    NGO = "ngo"
    SOCIAL_MEDIA = "social_media"
    ANONYMOUS = "anonymous"


# Threat level mappings
THREAT_LEVELS = {
    ThreatLevel.MINIMAL: "Minimal local impact",
    ThreatLevel.LOW: "Limited local impact",
    ThreatLevel.MODERATE: "Moderate local impact",
    ThreatLevel.ELEVATED: "Significant regional impact",
    ThreatLevel.HIGH: "Major regional disruption",
    ThreatLevel.SEVERE: "Severe regional crisis",
    ThreatLevel.CRITICAL: "Critical widespread crisis",
}

# Signal types list
SIGNAL_TYPES = [e.value for e in SignalType]

# Source reliability scores
SOURCE_RELIABILITY_SCORES = {
    SourceReliability.OFFICIAL_GOVERNMENT: 0.95,
    SourceReliability.VERIFIED_MEDIA: 0.85,
    SourceReliability.ACADEMIC: 0.90,
    SourceReliability.NGO: 0.80,
    SourceReliability.SOCIAL_MEDIA: 0.50,
    SourceReliability.ANONYMOUS: 0.30,
}

# Feature extraction parameters
TEMPORAL_FEATURES = [
    "hour", "day_of_week", "day_of_month", "month", 
    "quarter", "year", "is_weekend", "time_of_day"
]

SPATIAL_FEATURES = [
    "latitude", "longitude", "region", 
    "urbanization", "population_density"
]

# Model parameters
META_LEARNING_ADAPTATION_STEPS = 5
RL_GAMMA = 0.99
RL_EPSILON = 0.2
ENSEMBLE_N_MODELS = 5

# Processing limits
MAX_SIGNALS_PER_BATCH = 1000
MAX_INDICATORS_PER_ASSESSMENT = 100
MAX_HISTORICAL_DATA_POINTS = 10000

# Timeouts (seconds)
SIGNAL_PROCESSING_TIMEOUT = 30
THREAT_ANALYSIS_TIMEOUT = 60
OPTIMIZATION_TIMEOUT = 120
