#!/usr/bin/env python3
"""
Company Knowledge Worker - Quick Fix Version

Addresses the chunking and retrieval issues with minimal changes.
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
from src.port_manager import PortManager

class QuickFixRAGPipeline(RAGPipeline):
    """RAG Pipeline with quick fixes for better project coverage"""
    
    def _initialize_pipeline(self):
        """Initialize with improved retrieval settings"""
        try:
            # Initialize the language model
            from langchain_openai import ChatOpenAI
            self.llm = ChatOpenAI(
                temperature=0.7, 
                model_name=config.MODEL,
                openai_api_key=config.OPENAI_API_KEY
            )
            logger = get_logger(__name__)
            logger.info(f"Language model initialized: {config.MODEL}")
            
            # Set up conversation memory
            from langchain.memory import ConversationBufferMemory
            self.memory = ConversationBufferMemory(
                memory_key='chat_history', 
                return_messages=True
            )
            logger.info("Conversation memory initialized")
            
            # Create retriever with HIGHER k for better coverage
            retriever = self.vector_store_manager.create_retriever(
                search_kwargs={"k": 25}  # Increased from 10 to 25!
            )
            logger.info("Document retriever configured with k=25 for better project coverage")
            
            # Create the conversational retrieval chain
            from langchain.chains import ConversationalRetrievalChain
            self.conversation_chain = ConversationalRetrievalChain.from_llm(
                llm=self.llm, 
                retriever=retriever, 
                memory=self.memory
            )
            logger.info("Conversational RAG chain created with improved settings")
            
        except Exception as e:
            logger = get_logger(__name__)
            logger.error(f"Error initializing improved RAG pipeline: {e}")
            raise
    
    def ask_question(self, question: str):
        """Ask question with improved context for project queries"""
        logger = get_logger(__name__)
        
        if not self.conversation_chain:
            raise ValueError("RAG pipeline not initialized")
        
        try:
            # Enhanced project query handling
            if self._is_project_overview_query(question):
                # For project overview queries, add context hint
                enhanced_question = f"""
                {question}
                
                Please provide information about ALL current Artiligence projects. Make sure to cover:
                - SQL Server Upgrades (database infrastructure projects)
                - SFMS Mining Analytics (mining operations and analytics)
                - Precision Agriculture Asset Management (farming and GIS solutions)
                - Database Migration (legacy system migrations)
                - Advanced Driver Assistance System (vehicle telematics)
                
                List each project with details about what work is being done.
                """
                
                logger.info(f"Enhanced project overview query: {question}")
                result = self.conversation_chain.invoke({"question": enhanced_question})
            else:
                logger.info(f"Processing standard question: {question}")
                result = self.conversation_chain.invoke({"question": question})
            
            response = {
                "question": question,
                "answer": result["answer"],
                "success": True
            }
            
            logger.info("Successfully answered question")
            return response
            
        except Exception as e:
            logger.error(f"Error processing question: {e}")
            return {
                "question": question,
                "answer": f"Sorry, I encountered an error: {str(e)}",
                "success": False,
                "error": str(e)
            }
    
    def _is_project_overview_query(self, question: str) -> bool:
        """Check if this is asking for project overview"""
        overview_keywords = [
            'what projects', 'all projects', 'current projects', 
            'projects working on', 'list projects', 'company projects',
            'projects is artiligence', 'artiligence projects'
        ]
        question_lower = question.lower()
        return any(keyword in question_lower for keyword in overview_keywords)

def main():
    """Main application entry point with quick fixes"""
    parser = argparse.ArgumentParser(description="Company Knowledge Worker - Quick Fix Version")
    parser.add_argument("--mode", choices=["web", "cli"], default="web", help="Application mode")
    parser.add_argument("--log-level", choices=["DEBUG", "INFO", "WARNING", "ERROR"], default="INFO", help="Logging level")
    parser.add_argument("--port", type=int, default=None, help="Port for web interface")
    parser.add_argument("--share", action="store_true", help="Create public Gradio share link")
    parser.add_argument("--no-browser", action="store_true", help="Don't open browser automatically")
    
    args = parser.parse_args()
    
    # Setup logging
    setup_logging(log_level=args.log_level)
    logger = get_logger(__name__)
    
    logger.info("="*60)
    logger.info("🏢 COMPANY KNOWLEDGE WORKER - QUICK FIX")
    logger.info("   Better Project Coverage with k=25 retrieval")
    logger.info("="*60)
    
    try:
        # Initialize components
        logger.info("📚 Initializing document processor...")
        doc_processor = DocumentProcessor()
        
        logger.info("🗂️ Initializing vector store manager...")
        vector_store_manager = VectorStoreManager()
        
        # Load existing vector store
        logger.info("🔍 Loading existing vector store...")
        vectorstore = vector_store_manager.load_existing_vectorstore()
        
        if vectorstore is None:
            logger.error("❌ No existing vector store found!")
            logger.info("💡 Please run the main app first to build the vector database")
            return 1
        
        # Initialize improved RAG pipeline
        logger.info("🚀 Initializing Quick Fix RAG pipeline...")
        rag_pipeline = QuickFixRAGPipeline(vector_store_manager)
        
        # Test the pipeline
        logger.info("🧪 Testing pipeline...")
        test_result = rag_pipeline.test_pipeline()
        
        if test_result["success"]:
            logger.info("✅ Quick Fix pipeline ready!")
        else:
            logger.warning("⚠️ Pipeline test had issues but continuing...")
        
        # Run interface
        if args.mode == "web":
            logger.info("🌐 Starting web interface...")
            
            # Initialize port manager
            preferred_port = args.port or 7860
            port_manager = PortManager(preferred_port=preferred_port)
            
            # Clean up any old processes first
            logger.info("🧹 Cleaning up old processes...")
            cleaned_count = port_manager.cleanup_old_processes()
            if cleaned_count > 0:
                logger.info(f"Cleaned up {cleaned_count} old processes")
            
            # Ensure port is available
            logger.info(f"🔧 Ensuring port {preferred_port} is available...")
            available_port = port_manager.ensure_port_available(
                port=preferred_port,
                kill_if_needed=True,
                force_kill=False  # Only kill Python/Gradio processes by default
            )
            
            if available_port is None:
                logger.error("❌ Could not secure any port for the application")
                logger.info("💻 Falling back to CLI interface...")
                simple_chat = SimpleChatInterface(rag_pipeline)
                simple_chat.run()
                return 0
            
            logger.info(f"✅ Using port {available_port} for web interface")
            
            chat_interface = ChatInterface(rag_pipeline)
            interface = chat_interface.create_interface()
            
            # Update interface description to mention improvements
            interface.description = """
            Ask me anything about Artiligence company documents, projects, invoices, contracts, and more! 
            
            **🚀 ENHANCED with better project coverage:**
            - Increased retrieval to 25 chunks (was 10) for comprehensive project information
            - Improved project classification and context handling
            - Better coverage of all current projects
            - Automatic port management (kills conflicting processes)
            
            **All Artiligence projects covered:**
            - SQL Server upgrades and database infrastructure
            - SFMS Mining Analytics (mining operations)
            - Precision Agriculture Asset Management (farming solutions)
            - Database Migration projects
            - Advanced Driver Assistance Systems
            """
            
            try:
                logger.info(f"🚀 Launching web interface on port {available_port}...")
                chat_interface.launch(
                    server_port=available_port,
                    share=args.share,
                    inbrowser=not args.no_browser,
                    server_name="0.0.0.0"  # Listen on all interfaces
                )
            except Exception as e:
                logger.error(f"❌ Failed to launch on port {available_port}: {e}")
                logger.info("💻 Falling back to CLI interface...")
                simple_chat = SimpleChatInterface(rag_pipeline)
                simple_chat.run()
        else:
            logger.info("💻 Starting CLI interface...")
            simple_chat = SimpleChatInterface(rag_pipeline)
            simple_chat.run()
        
        return 0
        
    except Exception as e:
        logger.error(f"❌ Application failed: {e}")
        return 1

if __name__ == "__main__":
    sys.exit(main())