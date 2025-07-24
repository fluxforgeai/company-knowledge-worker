"""
Configuration management for Company Knowledge Worker
"""

import os
from dotenv import load_dotenv
from pathlib import Path

class Config:
    """Application configuration"""
    
    def __init__(self):
        # Load environment variables
        load_dotenv(override=True)
        
        # Model configuration
        self.MODEL = os.getenv('OPENAI_MODEL', 'gpt-4-turbo-preview')
        self.OPENAI_API_KEY = os.getenv('OPENAI_API_KEY')
        
        # Database configuration
        self.DB_NAME = os.getenv('DB_NAME', 'vector_db')
        
        # Document processing configuration
        self.BASE_PATH = os.getenv('BASE_PATH', '/Users/johanjgenis/Documents/Artiligence')
        self.MAX_FILE_SIZE = int(os.getenv('MAX_FILE_SIZE', '100000'))  # 100KB
        
        # Chunking configuration
        self.CHUNK_SIZE = int(os.getenv('CHUNK_SIZE', '1200'))
        self.CHUNK_OVERLAP = int(os.getenv('CHUNK_OVERLAP', '150'))
        
        # Retrieval configuration  
        self.RETRIEVAL_K = int(os.getenv('RETRIEVAL_K', '25'))  # Increased for better coverage
        
        # Gradio configuration
        self.GRADIO_PORT = int(os.getenv('GRADIO_PORT', '7860'))
        self.GRADIO_SHARE = os.getenv('GRADIO_SHARE', 'False').lower() == 'true'
        
        # Supported file extensions
        self.SUPPORTED_EXTENSIONS = {
            '.md', '.txt', '.py', '.js', '.html', '.css', '.json', '.yml', '.yaml', 
            '.xml', '.csv', '.rst', '.tex', '.pdf', '.docx', '.doc', '.xlsx', '.xls'
        }
        
        # Validate required settings
        self._validate_config()
    
    def _validate_config(self):
        """Validate required configuration"""
        if not self.OPENAI_API_KEY:
            raise ValueError("OPENAI_API_KEY environment variable is required")
        
        if not os.path.exists(self.BASE_PATH):
            raise ValueError(f"BASE_PATH '{self.BASE_PATH}' does not exist")
    
    def get_project_root(self) -> Path:
        """Get the project root directory"""
        return Path(__file__).parent.parent
    
    def get_data_dir(self) -> Path:
        """Get the data directory"""
        return self.get_project_root() / 'data'
    
    def get_vector_db_path(self) -> str:
        """Get the vector database path"""
        return str(self.get_project_root() / self.DB_NAME)

# Global configuration instance
config = Config()