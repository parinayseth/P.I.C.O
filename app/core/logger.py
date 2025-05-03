import logging
import sys
from logging.handlers import RotatingFileHandler
import os
from pathlib import Path
from app.core.config import LogConfig

def setup_logging(log_level=logging.INFO, log_file=None):
    """Configure global logging for the application
    
    Args:
        log_level: The minimum log level to capture
        log_file: Optional path to log file. If None, logs will only go to console
    """
    # Create logs directory if logging to file and directory doesn't exist
    if log_file:
        log_dir = os.path.dirname(log_file)
        if log_dir:
            Path(log_dir).mkdir(parents=True, exist_ok=True)
    
    # Get the root logger
    root_logger = logging.getLogger()
    root_logger.setLevel(log_level)
    
    # Clear existing handlers to avoid duplicate logs
    if root_logger.handlers:
        root_logger.handlers.clear()
    
    # Create formatter
    formatter = logging.Formatter(
        '%(asctime)s - %(name)s - %(levelname)s - %(message)s',
        datefmt='%Y-%m-%d %H:%M:%S'
    )
    
    # Console handler
    console_handler = logging.StreamHandler(sys.stdout)
    console_handler.setFormatter(formatter)
    root_logger.addHandler(console_handler)
    
    # File handler (optional)
    if log_file:
        file_handler = RotatingFileHandler(
            log_file,
            maxBytes=10485760,  # 10MB
            backupCount=10
        )
        file_handler.setFormatter(formatter)
        root_logger.addHandler(file_handler)
    
    # Set uvicorn access log level
    logging.getLogger("uvicorn.access").setLevel(logging.WARNING)
    
    # Return logger for immediate use
    return root_logger

# For direct import and use
logger = logging.getLogger(__name__)

# Setup logging during module import if run directly
if __name__ == "__main__":
    setup_logging(
        log_level=LogConfig.log_level_enum, 
        log_file=LogConfig.LOG_FILE if LogConfig.LOG_FILE else None
    )