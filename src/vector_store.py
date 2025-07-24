"""
Vector store management for Company Knowledge Worker
"""

import os
import logging
from typing import List, Optional
import numpy as np

from langchain.schema import Document
from langchain_openai import OpenAIEmbeddings
from langchain_chroma import Chroma

from .config import config

logger = logging.getLogger(__name__)

class VectorStoreManager:
    """Manages ChromaDB vector store operations"""
    
    def __init__(self):
        self.embeddings = OpenAIEmbeddings()
        self.db_path = config.get_vector_db_path()
        self.vectorstore: Optional[Chroma] = None
        
    def create_vectorstore(self, documents: List[Document], force_recreate: bool = False) -> Chroma:
        """Create or load vector store from documents"""
        
        # Delete existing database if it exists and force_recreate is True
        if force_recreate and os.path.exists(self.db_path):
            try:
                existing_store = Chroma(persist_directory=self.db_path, embedding_function=self.embeddings)
                existing_store.delete_collection()
                logger.info("Deleted existing vector database")
            except Exception as e:
                logger.warning(f"Could not delete existing database: {e}")
        
        # Create vectorstore from documents
        logger.info(f"Creating embeddings and storing in Chroma database at {self.db_path}...")
        
        try:
            self.vectorstore = Chroma.from_documents(
                documents=documents, 
                embedding=self.embeddings, 
                persist_directory=self.db_path
            )
            
            # Get collection info
            collection = self.vectorstore._collection
            count = collection.count()
            
            logger.info(f"Vectorstore created with {count} document chunks")
            
            # Get embedding dimensions for info
            if count > 0:
                sample_embedding = collection.get(limit=1, include=["embeddings"])["embeddings"][0]
                dimensions = len(sample_embedding)
                logger.info(f"Vector store contains {count:,} vectors with {dimensions:,} dimensions")
            
            return self.vectorstore
            
        except Exception as e:
            logger.error(f"Error creating vector store: {e}")
            raise
    
    def load_existing_vectorstore(self) -> Optional[Chroma]:
        """Load existing vector store if it exists"""
        if os.path.exists(self.db_path):
            try:
                self.vectorstore = Chroma(
                    persist_directory=self.db_path, 
                    embedding_function=self.embeddings
                )
                
                # Verify it has content
                collection = self.vectorstore._collection
                count = collection.count()
                
                if count > 0:
                    logger.info(f"Loaded existing vector store with {count} document chunks")
                    return self.vectorstore
                else:
                    logger.warning("Existing vector store is empty")
                    return None
                    
            except Exception as e:
                logger.error(f"Error loading existing vector store: {e}")
                return None
        else:
            logger.info(f"No existing vector store found at {self.db_path}")
            return None
    
    def get_vectorstore(self) -> Optional[Chroma]:
        """Get the current vector store instance"""
        return self.vectorstore
    
    def create_retriever(self, search_kwargs: dict = None):
        """Create a retriever from the vector store"""
        if not self.vectorstore:
            raise ValueError("Vector store not initialized. Call create_vectorstore or load_existing_vectorstore first.")
        
        if search_kwargs is None:
            search_kwargs = {"k": config.RETRIEVAL_K}
        
        retriever = self.vectorstore.as_retriever(search_kwargs=search_kwargs)
        logger.info(f"Created retriever with search kwargs: {search_kwargs}")
        
        return retriever
    
    def get_visualization_data(self):
        """Get data for vector store visualization"""
        if not self.vectorstore:
            raise ValueError("Vector store not initialized")
        
        try:
            collection = self.vectorstore._collection
            result = collection.get(include=['embeddings', 'documents', 'metadatas'])
            
            vectors = np.array(result['embeddings'])
            documents = result['documents']
            metadatas = result['metadatas']
            doc_types = [metadata.get('doc_type', 'unknown') for metadata in metadatas]
            
            logger.info(f"Retrieved {len(vectors)} vectors for visualization")
            
            return {
                'vectors': vectors,
                'documents': documents,
                'metadatas': metadatas,
                'doc_types': doc_types
            }
            
        except Exception as e:
            logger.error(f"Error getting visualization data: {e}")
            raise
    
    def search_similar_documents(self, query: str, k: int = 5):
        """Search for similar documents"""
        if not self.vectorstore:
            raise ValueError("Vector store not initialized")
        
        try:
            results = self.vectorstore.similarity_search(query, k=k)
            logger.info(f"Found {len(results)} similar documents for query: '{query[:50]}...'")
            return results
            
        except Exception as e:
            logger.error(f"Error searching similar documents: {e}")
            raise
    
    def get_stats(self) -> dict:
        """Get vector store statistics"""
        if not self.vectorstore:
            return {"error": "Vector store not initialized"}
        
        try:
            collection = self.vectorstore._collection
            count = collection.count()
            
            # Get sample data to determine dimensions
            sample_data = collection.get(limit=1, include=["embeddings", "metadatas"])
            
            stats = {
                "total_documents": count,
                "embedding_dimensions": len(sample_data["embeddings"][0]) if sample_data["embeddings"] else 0,
                "database_path": self.db_path
            }
            
            # Get document type breakdown
            all_metadata = collection.get(include=["metadatas"])["metadatas"]
            doc_types = {}
            for metadata in all_metadata:
                doc_type = metadata.get('doc_type', 'unknown')
                doc_types[doc_type] = doc_types.get(doc_type, 0) + 1
            
            stats["doc_type_breakdown"] = doc_types
            
            return stats
            
        except Exception as e:
            logger.error(f"Error getting vector store stats: {e}")
            return {"error": str(e)}