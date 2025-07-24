#!/usr/bin/env python3
"""
Debug script to investigate retrieval issues with project information
"""

import sys
sys.path.insert(0, 'src')

from src.config import config
from src.vector_store import VectorStoreManager
from src.document_loader import DocumentProcessor

def debug_project_retrieval():
    """Debug what gets retrieved when asking about projects"""
    
    print("🔍 DEBUGGING PROJECT RETRIEVAL")
    print("="*50)
    
    # Initialize vector store manager
    vector_store_manager = VectorStoreManager()
    
    # Try to load existing vector store
    vectorstore = vector_store_manager.load_existing_vectorstore()
    
    if not vectorstore:
        print("❌ No existing vector store found!")
        return
    
    # Get stats
    stats = vector_store_manager.get_stats()
    print(f"📊 Vector Store Stats:")
    print(f"  - Total documents: {stats.get('total_documents', 'Unknown')}")
    print(f"  - Document types: {stats.get('doc_type_breakdown', {})}")
    
    # Test project-related queries
    test_queries = [
        "What projects is the company working on?",
        "List all current projects",
        "Tell me about company projects", 
        "SQL Server upgrade project",
        "Precision Agriculture project"
    ]
    
    for query in test_queries:
        print(f"\n🔍 Query: '{query}'")
        print("-" * 40)
        
        # Get similar documents with different k values
        for k in [5, 10, 20]:
            print(f"\n📋 Retrieving top {k} chunks:")
            try:
                results = vector_store_manager.search_similar_documents(query, k=k)
                
                # Group by document type and project
                doc_types = {}
                projects = {}
                
                for i, doc in enumerate(results):
                    doc_type = doc.metadata.get('doc_type', 'unknown')
                    project = doc.metadata.get('project', 'N/A')
                    source = doc.metadata.get('source', 'unknown')
                    
                    # Count by doc type
                    doc_types[doc_type] = doc_types.get(doc_type, 0) + 1
                    
                    # Count by project  
                    if project != 'N/A':
                        projects[project] = projects.get(project, 0) + 1
                    
                    # Show first few results in detail
                    if i < 3:
                        print(f"  {i+1}. Type: {doc_type}, Project: {project}")
                        print(f"     Source: {source.split('/')[-1] if '/' in source else source}")
                        print(f"     Content preview: {doc.page_content[:150]}...")
                        print()
                
                print(f"  📊 Summary for k={k}:")
                print(f"     Doc types: {doc_types}")
                print(f"     Projects found: {projects}")
                
            except Exception as e:
                print(f"  ❌ Error: {e}")
    
    print("\n" + "="*50)

def analyze_project_chunks():
    """Analyze how project information is distributed across chunks"""
    
    print("\n🔍 ANALYZING PROJECT CHUNKS")
    print("="*50)
    
    vector_store_manager = VectorStoreManager()
    vectorstore = vector_store_manager.load_existing_vectorstore()
    
    if not vectorstore:
        print("❌ No existing vector store found!")
        return
    
    # Get all chunks
    try:
        viz_data = vector_store_manager.get_visualization_data()
        metadatas = viz_data['metadatas']
        documents = viz_data['documents']
        
        # Analyze project chunks
        project_chunks = {}
        project_content_lengths = {}
        
        for i, metadata in enumerate(metadatas):
            if metadata.get('doc_type') == 'Projects':
                project = metadata.get('project', 'Unknown')
                content = documents[i]
                
                if project not in project_chunks:
                    project_chunks[project] = []
                    project_content_lengths[project] = []
                
                project_chunks[project].append({
                    'source': metadata.get('source', 'unknown'),
                    'content_length': len(content),
                    'content_preview': content[:200] + '...' if len(content) > 200 else content
                })
                project_content_lengths[project].append(len(content))
        
        # Report findings
        print(f"📊 Project Analysis:")
        for project, chunks in project_chunks.items():
            avg_length = sum(project_content_lengths[project]) / len(project_content_lengths[project])
            print(f"\n🎯 {project}:")
            print(f"   - Number of chunks: {len(chunks)}")
            print(f"   - Average chunk length: {avg_length:.0f} characters")
            print(f"   - Sources: {set(chunk['source'].split('/')[-1] for chunk in chunks)}")
            
            # Show a sample chunk
            if chunks:
                print(f"   - Sample content: {chunks[0]['content_preview']}")
    
    except Exception as e:
        print(f"❌ Error analyzing chunks: {e}")

if __name__ == "__main__":
    debug_project_retrieval()
    analyze_project_chunks()