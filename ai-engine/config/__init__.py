"""
Configuration Management

Centralized configuration for the AI engine with validation and defaults.
"""

from .settings import Settings, get_settings
from .constants import (
    DEFAULT_EMBEDDING_MODEL,
    DEFAULT_DEVICE,
    THREAT_LEVELS,
    SIGNAL_TYPES,
    MAX_SIGNAL_BUFFER_SIZE
)

__all__ = [
    'Settings',
    'get_settings',
    'DEFAULT_EMBEDDING_MODEL',
    'DEFAULT_DEVICE',
    'THREAT_LEVELS',
    'SIGNAL_TYPES',
    'MAX_SIGNAL_BUFFER_SIZE',
]
