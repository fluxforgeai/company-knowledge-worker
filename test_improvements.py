#!/usr/bin/env python3
"""
Test script to validate the RAG improvements
"""

import sys
sys.path.insert(0, 'src')

from src.config import config
from src.vector_store import VectorStoreManager
from src.improved_rag import ImprovedRAGPipeline

def test_improved_rag():
    """Test the improved RAG pipeline"""
    
    print("🧪 TESTING IMPROVED RAG PIPELINE")
    print("="*50)
    
    try:
        # Initialize components
        print("🔧 Initializing vector store manager...")
        vector_store_manager = VectorStoreManager()
        
        # Load existing vector store
        vectorstore = vector_store_manager.load_existing_vectorstore()
        if not vectorstore:
            print("❌ No existing vector store found! Please run the app first to build it.")
            return
        
        print("✅ Vector store loaded")
        
        # Initialize improved RAG pipeline
        print("🔧 Initializing improved RAG pipeline...")
        improved_rag = ImprovedRAGPipeline(vector_store_manager)
        print("✅ Improved RAG pipeline ready")
        
        # Test comprehensive project query
        print("\n🔍 Testing comprehensive project query...")
        test_question = "What projects is Artiligence currently working on? List all projects with details."
        
        result = improved_rag.ask_question(test_question)
        
        print(f"\n📋 Question: {test_question}")
        print(f"✅ Success: {result['success']}")
        print(f"🔧 Enhanced: {result.get('enhanced', False)}")
        print(f"📊 Comprehensive: {result.get('comprehensive', False)}")
        print(f"\n💬 Answer:\n{result['answer']}")
        
        # Test specific project query
        print("\n" + "="*50)
        print("🔍 Testing specific project query...")
        specific_question = "Tell me about the SQL Server upgrade project"
        
        result2 = improved_rag.ask_question(specific_question)
        
        print(f"\n📋 Question: {specific_question}")
        print(f"✅ Success: {result2['success']}")
        print(f"🔧 Enhanced: {result2.get('enhanced', False)}")
        print(f"\n💬 Answer:\n{result2['answer'][:500]}...")
        
        # Test regular query
        print("\n" + "="*50)
        print("🔍 Testing regular query...")
        regular_question = "What contracts does Artiligence have?"
        
        result3 = improved_rag.ask_question(regular_question)
        
        print(f"\n📋 Question: {regular_question}")
        print(f"✅ Success: {result3['success']}")
        print(f"🔧 Enhanced: {result3.get('enhanced', False)}")
        print(f"\n💬 Answer:\n{result3['answer'][:500]}...")
        
        print("\n" + "="*50)
        print("✅ All tests completed!")
        
    except Exception as e:
        print(f"❌ Test failed: {e}")
        import traceback
        traceback.print_exc()

def compare_retrieval_methods():
    """Compare standard vs improved retrieval"""
    
    print("\n🔄 COMPARING RETRIEVAL METHODS")
    print("="*50)
    
    try:
        vector_store_manager = VectorStoreManager()
        vectorstore = vector_store_manager.load_existing_vectorstore()
        
        if not vectorstore:
            print("❌ No vector store available")
            return
        
        test_query = "What projects is Artiligence working on?"
        
        # Standard retrieval (k=10)
        print(f"🔍 Standard retrieval (k=10):")
        standard_results = vector_store_manager.search_similar_documents(test_query, k=10)
        
        projects_standard = {}
        for doc in standard_results:
            project = doc.metadata.get('project', 'Unknown')
            projects_standard[project] = projects_standard.get(project, 0) + 1
        
        print(f"  Projects found: {projects_standard}")
        
        # Enhanced retrieval (k=25) 
        print(f"\n🔍 Enhanced retrieval (k=25):")
        enhanced_results = vector_store_manager.search_similar_documents(test_query, k=25)
        
        projects_enhanced = {}
        for doc in enhanced_results:
            project = doc.metadata.get('project', 'Unknown')
            projects_enhanced[project] = projects_enhanced.get(project, 0) + 1
        
        print(f"  Projects found: {projects_enhanced}")
        
        print(f"\n📊 Improvement:")
        print(f"  Standard: {len(projects_standard)} unique projects")
        print(f"  Enhanced: {len(projects_enhanced)} unique projects")
        print(f"  Gain: +{len(projects_enhanced) - len(projects_standard)} projects")
        
    except Exception as e:
        print(f"❌ Comparison failed: {e}")

if __name__ == "__main__":
    test_improved_rag()
    compare_retrieval_methods()