"""
Utility Decorators
"""

import asyncio
import functools
import time
from typing import Any, Callable, Optional
import logging

logger = logging.getLogger(__name__)


def async_retry(
    max_attempts: int = 3,
    delay: float = 1.0,
    backoff: float = 2.0,
    exceptions: tuple = (Exception,)
):
    """
    Retry async function on failure
    
    Args:
        max_attempts: Maximum number of attempts
        delay: Initial delay between retries
        backoff: Backoff multiplier
        exceptions: Tuple of exceptions to catch
    """
    def decorator(func: Callable) -> Callable:
        @functools.wraps(func)
        async def wrapper(*args, **kwargs) -> Any:
            current_delay = delay
            
            for attempt in range(max_attempts):
                try:
                    return await func(*args, **kwargs)
                except exceptions as e:
                    if attempt == max_attempts - 1:
                        logger.error(f"Failed after {max_attempts} attempts: {e}")
                        raise
                    
                    logger.warning(
                        f"Attempt {attempt + 1} failed: {e}. "
                        f"Retrying in {current_delay}s..."
                    )
                    await asyncio.sleep(current_delay)
                    current_delay *= backoff
            
            return None
        
        return wrapper
    return decorator


def measure_time(func: Callable) -> Callable:
    """Measure execution time of function"""
    
    if asyncio.iscoroutinefunction(func):
        @functools.wraps(func)
        async def async_wrapper(*args, **kwargs) -> Any:
            start = time.time()
            result = await func(*args, **kwargs)
            elapsed = time.time() - start
            logger.debug(f"{func.__name__} took {elapsed:.3f}s")
            return result
        return async_wrapper
    else:
        @functools.wraps(func)
        def sync_wrapper(*args, **kwargs) -> Any:
            start = time.time()
            result = func(*args, **kwargs)
            elapsed = time.time() - start
            logger.debug(f"{func.__name__} took {elapsed:.3f}s")
            return result
        return sync_wrapper


def cache_result(maxsize: int = 128):
    """Cache function results"""
    def decorator(func: Callable) -> Callable:
        cache = {}
        
        @functools.wraps(func)
        def wrapper(*args, **kwargs) -> Any:
            # Create cache key
            key = str(args) + str(kwargs)
            
            if key in cache:
                return cache[key]
            
            result = func(*args, **kwargs)
            
            # Maintain cache size
            if len(cache) >= maxsize:
                cache.pop(next(iter(cache)))
            
            cache[key] = result
            return result
        
        return wrapper
    return decorator


def validate_input(validator: Callable):
    """Validate function input"""
    def decorator(func: Callable) -> Callable:
        @functools.wraps(func)
        def wrapper(*args, **kwargs) -> Any:
            validator(*args, **kwargs)
            return func(*args, **kwargs)
        return wrapper
    return decorator
