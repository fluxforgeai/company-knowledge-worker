"""
RAG (Retrieval Augmented Generation) pipeline for Company Knowledge Worker
"""

import logging
from typing import Dict, Any, Optional

from langchain_openai import ChatOpenAI
from langchain.memory import ConversationBufferMemory
from langchain.chains import ConversationalRetrievalChain

from .config import config
from .vector_store import VectorStoreManager

logger = logging.getLogger(__name__)

class RAGPipeline:
    """Manages the RAG pipeline for question answering"""
    
    def __init__(self, vector_store_manager: VectorStoreManager):
        self.vector_store_manager = vector_store_manager
        self.llm = None
        self.memory = None
        self.conversation_chain = None
        self._initialize_pipeline()
    
    def _initialize_pipeline(self):
        """Initialize the RAG pipeline components"""
        try:
            # Initialize the language model
            self.llm = ChatOpenAI(
                temperature=0.7, 
                model_name=config.MODEL,
                openai_api_key=config.OPENAI_API_KEY
            )
            logger.info(f"Language model initialized: {config.MODEL}")
            
            # Set up conversation memory
            self.memory = ConversationBufferMemory(
                memory_key='chat_history', 
                return_messages=True
            )
            logger.info("Conversation memory initialized")
            
            # Create retriever from vector store
            retriever = self.vector_store_manager.create_retriever()
            logger.info("Document retriever configured")
            
            # Create the conversational retrieval chain
            self.conversation_chain = ConversationalRetrievalChain.from_llm(
                llm=self.llm, 
                retriever=retriever, 
                memory=self.memory
            )
            logger.info("Conversational RAG chain created")
            
        except Exception as e:
            logger.error(f"Error initializing RAG pipeline: {e}")
            raise
    
    def ask_question(self, question: str) -> Dict[str, Any]:
        """Ask a question and get an answer from the RAG pipeline"""
        if not self.conversation_chain:
            raise ValueError("RAG pipeline not initialized")
        
        try:
            logger.info(f"Processing question: {question}")
            result = self.conversation_chain.invoke({"question": question})
            
            response = {
                "question": question,
                "answer": result["answer"],
                "success": True
            }
            
            logger.info(f"Successfully answered question")
            return response
            
        except Exception as e:
            logger.error(f"Error processing question: {e}")
            return {
                "question": question,
                "answer": f"Sorry, I encountered an error: {str(e)}",
                "success": False,
                "error": str(e)
            }
    
    def get_conversation_history(self) -> list:
        """Get the conversation history"""
        if not self.memory:
            return []
        
        try:
            return self.memory.chat_memory.messages
        except Exception as e:
            logger.error(f"Error getting conversation history: {e}")
            return []
    
    def clear_conversation_history(self):
        """Clear the conversation history"""
        if self.memory:
            try:
                self.memory.clear()
                logger.info("Conversation history cleared")
            except Exception as e:
                logger.error(f"Error clearing conversation history: {e}")
    
    def test_pipeline(self) -> Dict[str, Any]:
        """Test the RAG pipeline with a sample question"""
        test_question = "What information do you have about Artiligence? Give me a brief overview."
        
        logger.info("Testing RAG pipeline...")
        result = self.ask_question(test_question)
        
        if result["success"]:
            logger.info("RAG pipeline test successful")
        else:
            logger.error("RAG pipeline test failed")
        
        return result
    
    def get_pipeline_status(self) -> Dict[str, Any]:
        """Get the status of the RAG pipeline components"""
        status = {
            "llm_initialized": self.llm is not None,
            "memory_initialized": self.memory is not None,
            "conversation_chain_initialized": self.conversation_chain is not None,
            "vector_store_available": self.vector_store_manager.get_vectorstore() is not None
        }
        
        status["overall_ready"] = all(status.values())
        
        # Add vector store stats if available
        if status["vector_store_available"]:
            status["vector_store_stats"] = self.vector_store_manager.get_stats()
        
        return status
    
    def create_debug_chain(self):
        """Create a debug version of the chain with verbose output"""
        try:
            from langchain_core.callbacks import StdOutCallbackHandler
            
            debug_llm = ChatOpenAI(
                temperature=0.7, 
                model_name=config.MODEL,
                openai_api_key=config.OPENAI_API_KEY
            )
            debug_memory = ConversationBufferMemory(
                memory_key='chat_history', 
                return_messages=True
            )
            debug_retriever = self.vector_store_manager.create_retriever(
                search_kwargs={"k": 15}
            )
            
            debug_chain = ConversationalRetrievalChain.from_llm(
                llm=debug_llm, 
                retriever=debug_retriever, 
                memory=debug_memory, 
                callbacks=[StdOutCallbackHandler()]
            )
            
            logger.info("Debug chain created with verbose output")
            return debug_chain
            
        except Exception as e:
            logger.error(f"Error creating debug chain: {e}")
            return None