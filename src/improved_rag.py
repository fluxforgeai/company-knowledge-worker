"""
Improved RAG Pipeline with better project coverage
"""

import logging
from typing import Dict, Any, Optional, List
from collections import defaultdict

from langchain_openai import ChatOpenAI
from langchain.memory import ConversationBufferMemory
from langchain.chains import ConversationalRetrievalChain
from langchain_core.documents import Document
from langchain_core.retrievers import BaseRetriever
from langchain_core.callbacks import CallbackManagerForRetrieverRun

from .config import config
from .vector_store import VectorStoreManager

logger = logging.getLogger(__name__)

class ImprovedRAGPipeline:
    """Enhanced RAG pipeline with better project coverage"""
    
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
            
            # Create enhanced retriever
            retriever = self._create_enhanced_retriever()
            logger.info("Enhanced document retriever configured")
            
            # Create the conversational retrieval chain
            self.conversation_chain = ConversationalRetrievalChain.from_llm(
                llm=self.llm, 
                retriever=retriever, 
                memory=self.memory
            )
            logger.info("Enhanced conversational RAG chain created")
            
        except Exception as e:
            logger.error(f"Error initializing enhanced RAG pipeline: {e}")
            raise
    
    def _create_enhanced_retriever(self):
        """Create an enhanced retriever with better project coverage"""
        
        class EnhancedRetriever(BaseRetriever):
            def __init__(self, vector_store_manager, base_k=30):
                super().__init__()
                self.vector_store_manager = vector_store_manager
                self.base_k = base_k
                self.vectorstore = vector_store_manager.get_vectorstore()
            
            def _get_relevant_documents(
                self, query: str, *, run_manager: CallbackManagerForRetrieverRun
            ) -> List[Document]:
                """Get relevant documents with improved project diversity"""
                
                # Check if this is a broad project query
                project_keywords = ['projects', 'working on', 'current projects', 'all projects']
                is_project_query = any(keyword in query.lower() for keyword in project_keywords)
                
                if is_project_query:
                    return self._get_diverse_project_documents(query)
                else:
                    # For specific queries, use higher k but standard retrieval
                    return self.vectorstore.similarity_search(query, k=min(self.base_k, 25))
            
            def _get_diverse_project_documents(self, query: str) -> List[Document]:
                """Get diverse documents covering multiple projects"""
                
                # Get more initial candidates
                candidates = self.vectorstore.similarity_search(query, k=self.base_k)
                
                # Group by project to ensure diversity
                project_groups = defaultdict(list)
                other_docs = []
                
                for doc in candidates:
                    project = doc.metadata.get('project')
                    if project and project != 'Unknown':
                        project_groups[project].append(doc)
                    else:
                        other_docs.append(doc)
                
                # Select diverse documents
                selected_docs = []
                
                # Take top documents from each project (up to 4 per project)
                for project, docs in project_groups.items():
                    selected_docs.extend(docs[:4])
                
                # Add some general documents
                selected_docs.extend(other_docs[:5])
                
                # Limit total to reasonable number
                return selected_docs[:25]
        
        return EnhancedRetriever(self.vector_store_manager)
    
    def ask_question(self, question: str) -> Dict[str, Any]:
        """Ask a question with enhanced project coverage"""
        if not self.conversation_chain:
            raise ValueError("Enhanced RAG pipeline not initialized")
        
        try:
            logger.info(f"Processing enhanced question: {question}")
            
            # Check if this needs special project handling
            if self._is_comprehensive_project_query(question):
                return self._handle_comprehensive_project_query(question)
            else:
                # Use standard enhanced pipeline
                result = self.conversation_chain.invoke({"question": question})
                
                response = {
                    "question": question,
                    "answer": result["answer"],
                    "success": True,
                    "enhanced": True
                }
                
                logger.info(f"Successfully answered enhanced question")
                return response
                
        except Exception as e:
            logger.error(f"Error processing enhanced question: {e}")
            return {
                "question": question,
                "answer": f"Sorry, I encountered an error: {str(e)}",
                "success": False,
                "error": str(e),
                "enhanced": True
            }
    
    def _is_comprehensive_project_query(self, question: str) -> bool:
        """Check if this is a query that needs comprehensive project coverage"""
        comprehensive_keywords = [
            'what projects', 'all projects', 'current projects', 
            'list projects', 'projects working on', 'company projects'
        ]
        return any(keyword in question.lower() for keyword in comprehensive_keywords)
    
    def _handle_comprehensive_project_query(self, question: str) -> Dict[str, Any]:
        """Handle queries that need comprehensive project information"""
        
        try:
            # Get project-specific information
            project_info = self._get_all_project_information()
            
            # Create enhanced context
            enhanced_context = f"""
            Based on comprehensive analysis of Artiligence documents, here are ALL the current projects:

            {project_info}
            
            Original question: {question}
            """
            
            # Get LLM response with enhanced context
            response = self.llm.invoke(enhanced_context)
            
            return {
                "question": question,
                "answer": response.content,
                "success": True,
                "enhanced": True,
                "comprehensive": True
            }
            
        except Exception as e:
            logger.error(f"Error in comprehensive project query: {e}")
            # Fallback to standard method
            result = self.conversation_chain.invoke({"question": question})
            return {
                "question": question,
                "answer": result["answer"],
                "success": True,
                "enhanced": True,
                "fallback": True
            }
    
    def _get_all_project_information(self) -> str:
        """Get comprehensive information about all projects"""
        
        try:
            vectorstore = self.vector_store_manager.get_vectorstore()
            
            # Define all known projects
            projects = [
                "SQL Server Upgrades",
                "SFMS Mining Analytics", 
                "Precision Agriculture Asset Management",
                "Database Migration",
                "Advanced Driver Assistance System"
            ]
            
            project_summaries = []
            
            for project in projects:
                # Get project-specific documents
                project_query = f"ARTILIGENCE PROJECT: {project}"
                docs = vectorstore.similarity_search(project_query, k=8)
                
                # Filter for actual project docs
                project_docs = [doc for doc in docs if doc.metadata.get('project') == project]
                
                if project_docs:
                    # Get representative content
                    content_samples = []
                    sources = set()
                    
                    for doc in project_docs[:4]:
                        content_samples.append(doc.page_content[:300])
                        source = doc.metadata.get('source', 'unknown')
                        if '/' in source:
                            sources.add(source.split('/')[-1])
                        else:
                            sources.add(source)
                    
                    project_summary = f"""
                    **{project}:**
                    - Document sources: {', '.join(list(sources)[:5])}
                    - Content sample: {content_samples[0] if content_samples else 'No content available'}
                    """
                    project_summaries.append(project_summary)
            
            return "\n".join(project_summaries)
            
        except Exception as e:
            logger.error(f"Error getting all project information: {e}")
            return "Unable to retrieve comprehensive project information."
    
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
        """Test the enhanced RAG pipeline"""
        test_question = "What projects is Artiligence working on? List all current projects with details."
        
        logger.info("Testing enhanced RAG pipeline...")
        result = self.ask_question(test_question)
        
        if result["success"]:
            logger.info("Enhanced RAG pipeline test successful")
        else:
            logger.error("Enhanced RAG pipeline test failed")
        
        return result