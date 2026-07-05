"""
Structured logging module for the Financial Document Intelligence System.
Provides file + console logging with request tracing.
"""
import logging
import sys
from pathlib import Path
from datetime import datetime
from config.settings import settings


def setup_logger(name: str = "findoc") -> logging.Logger:
    """
    Set up a structured logger with file and console handlers.
    
    Args:
        name: Logger name identifier
        
    Returns:
        Configured logger instance
    """
    logger = logging.getLogger(name)
    
    if logger.handlers:
        return logger
    
    logger.setLevel(getattr(logging, settings.log_level.upper(), logging.INFO))
    
    # Console handler with colored output
    console_handler = logging.StreamHandler(sys.stdout)
    console_handler.setLevel(logging.INFO)
    console_format = logging.Formatter(
        "%(asctime)s │ %(levelname)-8s │ %(name)-20s │ %(message)s",
        datefmt="%H:%M:%S"
    )
    console_handler.setFormatter(console_format)
    logger.addHandler(console_handler)
    
    # File handler
    log_dir = Path(settings.log_dir)
    log_dir.mkdir(parents=True, exist_ok=True)
    
    log_file = log_dir / f"findoc_{datetime.now().strftime('%Y%m%d')}.log"
    file_handler = logging.FileHandler(log_file, encoding="utf-8")
    file_handler.setLevel(logging.DEBUG)
    file_format = logging.Formatter(
        "%(asctime)s | %(levelname)-8s | %(name)-25s | %(funcName)-20s | %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S"
    )
    file_handler.setFormatter(file_format)
    logger.addHandler(file_handler)
    
    return logger


# Pre-configured loggers for each module
def get_logger(module_name: str) -> logging.Logger:
    """Get a logger for a specific module."""
    return setup_logger(f"findoc.{module_name}")


class PipelineTracer:
    """Traces pipeline execution for debugging and monitoring."""
    
    def __init__(self):
        self.logger = get_logger("tracer")
        self.traces = []
    
    def trace(self, step: str, details: dict = None):
        """Record a pipeline step."""
        entry = {
            "timestamp": datetime.now().isoformat(),
            "step": step,
            "details": details or {}
        }
        self.traces.append(entry)
        self.logger.info(f"[TRACE] {step} | {details}")
    
    def get_trace_summary(self) -> list:
        """Get all recorded traces."""
        return self.traces
    
    def clear(self):
        """Clear all traces."""
        self.traces = []


# Global tracer instance
tracer = PipelineTracer()
