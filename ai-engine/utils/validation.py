"""
Input Validation Utilities
"""

from typing import Dict, Any, List
from datetime import datetime


class ValidationError(Exception):
    """Custom validation error"""
    pass


def validate_signal(signal: Dict[str, Any]) -> None:
    """
    Validate signal structure
    
    Args:
        signal: Signal dictionary
    
    Raises:
        ValidationError: If validation fails
    """
    required_fields = ['type', 'source', 'data', 'location', 'temporal']
    
    for field in required_fields:
        if field not in signal:
            raise ValidationError(f"Missing required field: {field}")
    
    validate_location(signal['location'])
    validate_temporal(signal['temporal'])


def validate_location(location: Dict[str, Any]) -> None:
    """
    Validate location data
    
    Args:
        location: Location dictionary
    
    Raises:
        ValidationError: If validation fails
    """
    if 'latitude' in location:
        lat = location['latitude']
        if not isinstance(lat, (int, float)) or not -90 <= lat <= 90:
            raise ValidationError(f"Invalid latitude: {lat}")
    
    if 'longitude' in location:
        lon = location['longitude']
        if not isinstance(lon, (int, float)) or not -180 <= lon <= 180:
            raise ValidationError(f"Invalid longitude: {lon}")


def validate_temporal(temporal: Dict[str, Any]) -> None:
    """
    Validate temporal data
    
    Args:
        temporal: Temporal dictionary
    
    Raises:
        ValidationError: If validation fails
    """
    if 'timestamp' not in temporal:
        raise ValidationError("Missing timestamp in temporal data")
    
    try:
        datetime.fromisoformat(temporal['timestamp'])
    except (ValueError, TypeError) as e:
        raise ValidationError(f"Invalid timestamp format: {e}")


def validate_config(config: Dict[str, Any]) -> None:
    """
    Validate configuration dictionary
    
    Args:
        config: Configuration dictionary
    
    Raises:
        ValidationError: If validation fails
    """
    if 'device' in config:
        if config['device'] not in ['cpu', 'cuda', 'mps']:
            raise ValidationError(f"Invalid device: {config['device']}")
    
    if 'batch_size' in config:
        if not isinstance(config['batch_size'], int) or config['batch_size'] <= 0:
            raise ValidationError(f"Invalid batch_size: {config['batch_size']}")
