"""
Logging configuration for Company Knowledge Worker
"""

import logging
import logging.handlers
import os
from pathlib import Path

def setup_logging(log_level: str = "INFO", log_to_file: bool = True, log_dir: str = None):
    """Setup logging configuration for the application"""
    
    # Convert log level string to logging constant
    numeric_level = getattr(logging, log_level.upper(), logging.INFO)
    
    # Create formatter
    formatter = logging.Formatter(
        '%(asctime)s - %(name)s - %(levelname)s - %(message)s',
        datefmt='%Y-%m-%d %H:%M:%S'
    )
    
    # Setup root logger
    root_logger = logging.getLogger()
    root_logger.setLevel(numeric_level)
    
    # Clear any existing handlers
    root_logger.handlers.clear()
    
    # Console handler
    console_handler = logging.StreamHandler()
    console_handler.setLevel(numeric_level)
    console_handler.setFormatter(formatter)
    root_logger.addHandler(console_handler)
    
    # File handler (if enabled)
    if log_to_file:
        # Determine log directory
        if log_dir is None:
            log_dir = Path(__file__).parent.parent / "logs"
        else:
            log_dir = Path(log_dir)
        
        # Create log directory if it doesn't exist
        log_dir.mkdir(exist_ok=True)
        
        # Setup rotating file handler
        log_file = log_dir / "company_knowledge_worker.log"
        file_handler = logging.handlers.RotatingFileHandler(
            log_file,
            maxBytes=10*1024*1024,  # 10MB
            backupCount=5
        )
        file_handler.setLevel(numeric_level)
        file_handler.setFormatter(formatter)
        root_logger.addHandler(file_handler)
        
        print(f"Logging to file: {log_file}")
    
    # Reduce verbosity of some third-party loggers
    logging.getLogger("urllib3").setLevel(logging.WARNING)
    logging.getLogger("requests").setLevel(logging.WARNING)
    logging.getLogger("chromadb").setLevel(logging.WARNING)
    logging.getLogger("openai").setLevel(logging.WARNING)
    logging.getLogger("httpx").setLevel(logging.WARNING)
    
    print(f"Logging configured with level: {log_level}")
    return root_logger

def get_logger(name: str) -> logging.Logger:
    """Get a logger with the specified name"""
    return logging.getLogger(name)