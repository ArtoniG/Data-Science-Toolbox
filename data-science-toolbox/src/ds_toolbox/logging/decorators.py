"""
src/credit_toolbox/logging/decorators.py

Decorators to non-invasively inject observability and audit trails into client code.
By wrapping functions with these decorators, we capture execution times, success/failure 
states, and handle exceptions without cluttering the core business logic.
"""

import functools
import time
from typing import Any, Callable, TypeVar, cast

from credit_toolbox.core.exceptions import CreditToolboxError
from credit_toolbox.logging.logger import get_logger

# Initialize the module-level logger
logger = get_logger(__name__)

# TypeVar to preserve the original function's signature and docstrings in IDEs (Type Hinting)
F = TypeVar("F", bound=Callable[..., Any])


def log_execution_time(func: F) -> F:
    """
    Decorator to measure and log the execution time of a function.
    Perfect for tracking latency in API endpoints or feature engineering steps.
    """
    @functools.wraps(func)
    def wrapper(*args: Any, **kwargs: Any) -> Any:
        start_time = time.perf_counter()
        
        # Execute the actual function
        result = func(*args, **kwargs)
        
        end_time = time.perf_counter()
        duration_ms = round((end_time - start_time) * 1000, 2)

        logger.info(
            f"Execution completed: {func.__name__}",
            extra={
                "duration_ms": duration_ms,
                "function_name": func.__name__,
                "module_name": func.__module__,
            }
        )
        return result

    return cast(F, wrapper)


def audit_trail(event_name: str | None = None) -> Callable[[F], F]:
    """
    Decorator for critical operations (e.g., model training, data ingestion).
    Automatically logs the start, success, or failure of the operation.
    Catches and categorizes exceptions as either controlled business errors 
    or critical system failures.
    
    Args:
        event_name: An optional custom name for the audit event. Defaults to the function name.
    """
    def decorator(func: F) -> F:
        @functools.wraps(func)
        def wrapper(*args: Any, **kwargs: Any) -> Any:
            actual_event = event_name or func.__name__
            
            logger.info(
                f"Audit event started: {actual_event}", 
                extra={"event_name": actual_event, "status": "STARTED"}
            )
            
            start_time = time.perf_counter()
            
            try:
                result = func(*args, **kwargs)
                duration_ms = round((time.perf_counter() - start_time) * 1000, 2)
                
                logger.info(
                    f"Audit event succeeded: {actual_event}", 
                    extra={
                        "event_name": actual_event, 
                        "status": "SUCCESS", 
                        "duration_ms": duration_ms
                    }
                )
                return result
                
            except CreditToolboxError as e:
                # Controlled SDK Error (e.g., SchemaMismatchError, ModelDriftError)
                duration_ms = round((time.perf_counter() - start_time) * 1000, 2)
                logger.error(
                    f"Audit event failed (Business Error): {actual_event}", 
                    extra={
                        "event_name": actual_event, 
                        "status": "FAILED", 
                        "error_type": type(e).__name__,
                        "duration_ms": duration_ms
                    },
                    exc_info=True # Captures the traceback cleanly into the JSON output
                )
                raise
                
            except Exception as e:
                # Unhandled System Error (e.g., MemoryError, KeyError)
                duration_ms = round((time.perf_counter() - start_time) * 1000, 2)
                logger.critical(
                    f"Audit event failed (System Error): {actual_event}", 
                    extra={
                        "event_name": actual_event, 
                        "status": "CRITICAL_FAILURE", 
                        "error_type": type(e).__name__,
                        "duration_ms": duration_ms
                    },
                    exc_info=True
                )
                raise
                
        return cast(F, wrapper)
    return decorator