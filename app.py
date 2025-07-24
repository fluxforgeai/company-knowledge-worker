#!/usr/bin/env python3
"""
Company Knowledge Worker - Main Application Entry Point

A comprehensive RAG-powered Q&A assistant for company documents
Created with Claude Code - AI-powered development assistant
"""

import os
import sys
import argparse
from pathlib import Path

# Add src directory to Python path
sys.path.insert(0, str(Path(__file__).parent / "src"))

from src.logging_config import setup_logging, get_logger
from src.config import config
from src.document_loader import DocumentProcessor
from src.vector_store import VectorStoreManager
from src.rag_pipeline import RAGPipeline
from src.chat_interface import ChatInterface, SimpleChatInterface

def parse_arguments():
    """Parse command line arguments"""
    parser = argparse.ArgumentParser(
        description="Company Knowledge Worker - RAG-powered Q&A for company documents"
    )
    
    parser.add_argument(
        "--mode", 
        choices=["web", "cli", "build"], 
        default="web",
        help="Application mode: web (Gradio), cli (command line), or build (rebuild vector store)"
    )
    
    parser.add_argument(
        "--log-level", 
        choices=["DEBUG", "INFO", "WARNING", "ERROR"], 
        default="INFO",
        help="Logging level"
    )
    
    parser.add_argument(
        "--rebuild-db", 
        action="store_true",
        help="Force rebuild of vector database"
    )
    
    parser.add_argument(
        "--port", 
        type=int, 
        default=None,
        help="Port for web interface (overrides config)"
    )
    
    parser.add_argument(
        "--share", 
        action="store_true",
        help="Create public Gradio share link"
    )
    
    parser.add_argument(
        "--no-browser", 
        action="store_true",
        help="Don't open browser automatically"
    )
    
    return parser.parse_args()

def initialize_system(rebuild_db: bool = False):
    """Initialize the complete system"""
    logger = get_logger(__name__)
    
    try:
        logger.info("🔄 Initializing Company Knowledge Worker system...")
        
        # Initialize document processor
        logger.info("📚 Initializing document processor...")
        doc_processor = DocumentProcessor()
        
        # Initialize vector store manager
        logger.info("🗂️ Initializing vector store manager...")
        vector_store_manager = VectorStoreManager()
        
        # Load or create vector store
        if rebuild_db:
            logger.info("🔄 Rebuilding vector database...")
            documents = doc_processor.load_all_documents()
            vector_store_manager.create_vectorstore(documents, force_recreate=True)
        else:
            # Try to load existing vector store first
            logger.info("🔍 Checking for existing vector store...")
            vectorstore = vector_store_manager.load_existing_vectorstore()
            
            if vectorstore is None:
                logger.info("🔄 No existing vector store found. Creating new one...")
                documents = doc_processor.load_all_documents()
                vector_store_manager.create_vectorstore(documents)
        
        # Initialize RAG pipeline
        logger.info("🤖 Initializing RAG pipeline...")
        rag_pipeline = RAGPipeline(vector_store_manager)
        
        # Test the pipeline
        logger.info("🧪 Testing RAG pipeline...")
        test_result = rag_pipeline.test_pipeline()
        
        if test_result["success"]:
            logger.info("✅ System initialization completed successfully!")
            return rag_pipeline
        else:
            logger.error("❌ System test failed!")
            raise RuntimeError(f"System test failed: {test_result.get('error', 'Unknown error')}")
            
    except Exception as e:
        logger.error(f"❌ System initialization failed: {e}")
        raise

def run_web_interface(rag_pipeline: RAGPipeline, port: int = None, share: bool = False, open_browser: bool = True):
    """Run the web interface"""
    logger = get_logger(__name__)
    
    try:
        logger.info("🌐 Starting web interface...")
        
        # Create chat interface
        chat_interface = ChatInterface(rag_pipeline)
        interface = chat_interface.create_interface()
        
        # Launch interface
        chat_interface.launch(
            server_port=port,
            share=share,
            inbrowser=open_browser
        )
        
    except Exception as e:
        logger.error(f"❌ Web interface failed: {e}")
        logger.info("💡 Falling back to command line interface...")
        run_cli_interface(rag_pipeline)

def run_cli_interface(rag_pipeline: RAGPipeline):
    """Run the command line interface"""
    logger = get_logger(__name__)
    
    try:
        logger.info("💻 Starting command line interface...")
        
        # Create simple chat interface
        simple_chat = SimpleChatInterface(rag_pipeline)
        simple_chat.run()
        
    except Exception as e:
        logger.error(f"❌ Command line interface failed: {e}")
        raise

def build_vector_store():
    """Build/rebuild vector store only"""
    logger = get_logger(__name__)
    
    try:
        logger.info("🔨 Building vector store...")
        
        # Initialize document processor
        doc_processor = DocumentProcessor()
        
        # Load all documents
        documents = doc_processor.load_all_documents()
        
        # Initialize vector store manager and create store
        vector_store_manager = VectorStoreManager()
        vector_store_manager.create_vectorstore(documents, force_recreate=True)
        
        # Get stats
        stats = vector_store_manager.get_stats()
        logger.info(f"✅ Vector store built successfully!")
        logger.info(f"📊 Stats: {stats}")
        
    except Exception as e:
        logger.error(f"❌ Vector store build failed: {e}")
        raise

def create_env_file():
    """Create a sample .env file if it doesn't exist"""
    env_file = Path(__file__).parent / ".env"
    
    if not env_file.exists():
        sample_content = """# Company Knowledge Worker Configuration

# Required: OpenAI API Key
OPENAI_API_KEY=your_openai_api_key_here

# Optional: Model configuration
OPENAI_MODEL=gpt-4-turbo-preview

# Optional: Document path (default: /path/to/company/documents)
BASE_PATH=/path/to/company/documents

# Optional: Database configuration
DB_NAME=vector_db

# Optional: Processing configuration
MAX_FILE_SIZE=100000
CHUNK_SIZE=1200
CHUNK_OVERLAP=150

# Optional: Retrieval configuration
RETRIEVAL_K=10

# Optional: Gradio configuration
GRADIO_PORT=7860
GRADIO_SHARE=False
"""
        
        with open(env_file, 'w') as f:
            f.write(sample_content)
        
        print(f"📝 Created sample .env file: {env_file}")
        print("🔑 Please edit the .env file and add your OpenAI API key!")
        return False
    
    return True

def main():
    """Main application entry point"""
    # Parse arguments
    args = parse_arguments()
    
    # Setup logging
    setup_logging(log_level=args.log_level)
    logger = get_logger(__name__)
    
    # Show startup banner
    logger.info("="*60)
    logger.info("🏢 COMPANY KNOWLEDGE WORKER")
    logger.info("   RAG-powered Q&A for Company Documents")
    logger.info("   Created by Claude Code")
    logger.info("="*60)
    
    try:
        # Check for .env file
        if not create_env_file():
            return 1
        
        # Validate configuration
        try:
            config._validate_config()
        except ValueError as e:
            logger.error(f"❌ Configuration error: {e}")
            logger.error("💡 Please check your .env file and ensure OPENAI_API_KEY is set")
            return 1
        
        # Handle different modes
        if args.mode == "build":
            build_vector_store()
            logger.info("✅ Vector store build completed!")
            return 0
        
        # Initialize system
        rag_pipeline = initialize_system(rebuild_db=args.rebuild_db)
        
        # Run appropriate interface
        if args.mode == "web":
            run_web_interface(
                rag_pipeline, 
                port=args.port, 
                share=args.share, 
                open_browser=not args.no_browser
            )
        elif args.mode == "cli":
            run_cli_interface(rag_pipeline)
        
        return 0
        
    except KeyboardInterrupt:
        logger.info("👋 Application interrupted by user")
        return 0
    except Exception as e:
        logger.error(f"❌ Application failed: {e}")
        return 1

if __name__ == "__main__":
    sys.exit(main())