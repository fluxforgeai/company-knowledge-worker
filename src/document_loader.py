"""
Document loading and processing module for Company Knowledge Worker
"""

import os
import glob
from pathlib import Path
from typing import List, Dict, Any
import logging

import pandas as pd
import docx2txt
from langchain_community.document_loaders import DirectoryLoader, TextLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_core.documents import Document
from langchain_community.document_loaders import PyPDFLoader, Docx2txtLoader

from .config import config

logger = logging.getLogger(__name__)

class DocumentProcessor:
    """Handles document loading and processing"""
    
    def __init__(self):
        self.supported_extensions = config.SUPPORTED_EXTENSIONS
        self.max_file_size = config.MAX_FILE_SIZE
        self.base_path = config.BASE_PATH
        
    def find_all_directories(self, base_path: str) -> List[str]:
        """Recursively find all directories under base_path"""
        directories = []
        try:
            for root, dirs, files in os.walk(base_path):
                # Skip hidden directories and common non-document directories
                dirs[:] = [d for d in dirs if not d.startswith('.') and d.lower() not in ['__pycache__', 'node_modules']]
                if root != base_path:  # Skip the base directory itself
                    directories.append(root)
        except Exception as e:
            logger.error(f"Error walking directory {base_path}: {e}")
        return directories

    def find_all_files_recursive(self, base_path: str) -> List[str]:
        """Recursively find all supported files"""
        files = []
        try:
            for root, dirs, filenames in os.walk(base_path):
                # Skip hidden directories
                dirs[:] = [d for d in dirs if not d.startswith('.')]
                for filename in filenames:
                    if not filename.startswith('.'):  # Skip hidden files
                        file_ext = os.path.splitext(filename)[1].lower()
                        if file_ext in self.supported_extensions:
                            files.append(os.path.join(root, filename))
        except Exception as e:
            logger.error(f"Error walking directory {base_path}: {e}")
        return files

    def get_document_type(self, file_path: str, base_path: str) -> str:
        """Determine document type based on directory structure"""
        rel_path = os.path.relpath(file_path, base_path)
        path_parts = rel_path.split(os.sep)
        
        if len(path_parts) == 1:
            return "root_files"
        else:
            # Use the top-level directory as the document type
            return path_parts[0]

    def add_metadata(self, doc: Document, doc_type: str, file_type: str = None, subdirectory: str = None) -> Document:
        """Add comprehensive metadata to documents"""
        doc.metadata["doc_type"] = doc_type
        if file_type:
            doc.metadata["file_type"] = file_type
        if subdirectory:
            doc.metadata["subdirectory"] = subdirectory
        return doc

    def load_documents_recursive(self) -> List[Document]:
        """Recursively load all supported documents from nested directory structure"""
        documents = []
        
        # Find all supported files recursively
        all_files = self.find_all_files_recursive(self.base_path)
        
        logger.info(f"Processing {len(all_files)} files...")
        
        loaded_count = 0
        skipped_large = 0
        skipped_errors = 0
        
        for file_path in all_files:
            try:
                # Check file size first
                file_size = os.path.getsize(file_path)
                if file_size > self.max_file_size:
                    skipped_large += 1
                    continue
                    
                file_ext = os.path.splitext(file_path)[1].lower()
                filename = os.path.basename(file_path)
                doc_type = self.get_document_type(file_path, self.base_path)
                
                # Get subdirectory for metadata
                rel_path = os.path.relpath(file_path, self.base_path)
                subdirectory = os.path.dirname(rel_path) if os.path.dirname(rel_path) != '.' else None
                
                # Load based on file type
                if file_ext in ['.md', '.txt', '.py', '.js', '.html', '.css', '.json', '.yml', '.yaml', '.xml', '.csv', '.rst', '.tex']:
                    try:
                        loader = TextLoader(file_path, encoding='utf-8')
                        doc = loader.load()[0]
                        documents.append(self.add_metadata(doc, doc_type, file_ext[1:], subdirectory))
                        loaded_count += 1
                    except UnicodeDecodeError:
                        # Try with different encoding
                        try:
                            loader = TextLoader(file_path, encoding='latin-1')
                            doc = loader.load()[0]
                            documents.append(self.add_metadata(doc, doc_type, file_ext[1:], subdirectory))
                            loaded_count += 1
                        except Exception as e:
                            logger.warning(f"Could not load {filename}: {e}")
                            skipped_errors += 1
                            
                elif file_ext == '.pdf':
                    try:
                        loader = PyPDFLoader(file_path)
                        pdf_docs = loader.load()
                        for pdf_doc in pdf_docs:
                            documents.append(self.add_metadata(pdf_doc, doc_type, "pdf", subdirectory))
                        loaded_count += len(pdf_docs)
                    except Exception as e:
                        logger.warning(f"Could not load PDF {filename}: {e}")
                        skipped_errors += 1
                        
                elif file_ext in ['.docx', '.doc']:
                    try:
                        loader = Docx2txtLoader(file_path)
                        doc = loader.load()[0]
                        documents.append(self.add_metadata(doc, doc_type, "docx", subdirectory))
                        loaded_count += 1
                    except Exception as e:
                        logger.warning(f"Could not load Word file {filename}: {e}")
                        skipped_errors += 1
                        
                elif file_ext in ['.xlsx', '.xls']:
                    try:
                        excel_file = pd.ExcelFile(file_path)
                        content_parts = []
                        
                        for sheet_name in excel_file.sheet_names:
                            df = pd.read_excel(file_path, sheet_name=sheet_name)
                            sheet_content = f"Sheet: {sheet_name}\\n"
                            sheet_content += f"Shape: {df.shape[0]} rows, {df.shape[1]} columns\\n"
                            sheet_content += f"Columns: {', '.join(df.columns.astype(str))}\\n\\n"
                            df_sample = df.head(100)
                            sheet_content += df_sample.to_string(index=False)
                            content_parts.append(sheet_content)
                        
                        combined_content = f"Excel File: {filename}\\n\\n" + "\\n\\n---\\n\\n".join(content_parts)
                        doc = Document(page_content=combined_content, metadata={"source": file_path})
                        documents.append(self.add_metadata(doc, doc_type, "excel", subdirectory))
                        loaded_count += 1
                    except Exception as e:
                        logger.warning(f"Could not load Excel file {filename}: {e}")
                        skipped_errors += 1
                        
            except Exception as e:
                logger.error(f"Error processing {file_path}: {e}")
                skipped_errors += 1
        
        logger.info(f"Loading Summary: {loaded_count} documents loaded, {skipped_large} skipped (too large), {skipped_errors} skipped (errors)")
        
        return documents

    def determine_project_name(self, file_path: str) -> str:
        """Determine project name from file path with improved classification"""
        file_path_lower = file_path.lower()
        filename = os.path.basename(file_path).lower()
        
        # Enhanced pattern matching for better classification
        if any(keyword in file_path_lower for keyword in ['aa_sql', 'sql_server', 'sql server', 'version_upgrade', 'cumulative', 'aiskoldb', 'aiscengdb']):
            return "SQL Server Upgrades"
        elif any(keyword in file_path_lower for keyword in ['itgisworx', 'precision', 'agriculture', 'sensaas', 'asset_management', 'assets_datastructure']):
            return "Precision Agriculture Asset Management"
        elif any(keyword in file_path_lower for keyword in ['aa_sfms', 'sfms', 'mining', 'quellaveco', 'amps', 'operator', 'hexagon', 'realtime', 'truck']):
            return "SFMS Mining Analytics"
        elif any(keyword in file_path_lower for keyword in ['eben_db_migration', 'migration', 'consolidated', 'modem', 'meters', 'suppliers', 'accounts', 'tariffs']):
            return "Database Migration"
        elif any(keyword in file_path_lower for keyword in ['aa adas', 'adas', 'mix integrate', 'driver assistance', 'events', 'dictionary']):
            return "Advanced Driver Assistance System"
        else:
            # Extract folder name as project for other directories
            path_parts = file_path.split('/')
            for part in reversed(path_parts):
                if part != os.path.basename(file_path) and part != 'Projects':
                    # Apply same enhanced matching to folder names
                    part_lower = part.lower()
                    if any(keyword in part_lower for keyword in ['sql', 'server', 'upgrade']):
                        return "SQL Server Upgrades"
                    elif any(keyword in part_lower for keyword in ['precision', 'agriculture', 'gis']):
                        return "Precision Agriculture Asset Management"
                    elif any(keyword in part_lower for keyword in ['sfms', 'mining', 'analytics']):
                        return "SFMS Mining Analytics"
                    elif any(keyword in part_lower for keyword in ['migration', 'db']):
                        return "Database Migration"
                    elif any(keyword in part_lower for keyword in ['adas', 'driver']):
                        return "Advanced Driver Assistance System"
                    else:
                        return part.replace('_', ' ').title()
        return "General Project"

    def load_enhanced_project_documents(self) -> List[Document]:
        """Load project documents with enhanced context and processing"""
        logger.info("Loading company project documents with enhanced context...")
        
        # Get all project files specifically
        project_files = glob.glob(os.path.join(self.base_path, "Projects/**/*"), recursive=True)
        project_files = [f for f in project_files if os.path.isfile(f)]
        
        logger.info(f"Found {len(project_files)} project files")
        
        documents = []
        
        for file_path in project_files:
            try:
                filename = os.path.basename(file_path)
                file_ext = os.path.splitext(filename)[1].lower()
                project_name = self.determine_project_name(file_path)
                
                # Skip temporary files
                if filename.startswith('~$') or filename.startswith('.'):
                    continue
                    
                logger.debug(f"Processing: {filename} (Project: {project_name})")
                
                # Enhanced metadata function
                def add_enhanced_metadata(doc, doc_type, file_type, project_name):
                    doc.metadata["doc_type"] = doc_type
                    doc.metadata["file_type"] = file_type
                    doc.metadata["company"] = "Company"
                    doc.metadata["project"] = project_name
                    return doc
                
                # Load different file types with enhanced context
                if file_ext == '.pdf':
                    try:
                        loader = PyPDFLoader(file_path)
                        docs = loader.load()
                        for doc in docs:
                            # Add company context to the document content
                            enhanced_content = f"ARTILIGENCE PROJECT: {project_name}\\n\\nDocument: {filename}\\n\\n{doc.page_content}"
                            doc.page_content = enhanced_content
                            documents.append(add_enhanced_metadata(doc, "Projects", "pdf", project_name))
                    except Exception as e:
                        logger.warning(f"Could not read PDF {filename}: {e}")
                        
                elif file_ext in ['.docx', '.doc']:
                    try:
                        content = docx2txt.process(file_path)
                        if content.strip():
                            enhanced_content = f"ARTILIGENCE PROJECT: {project_name}\\n\\nDocument: {filename}\\n\\n{content}"
                            doc = Document(page_content=enhanced_content, metadata={"source": file_path})
                            documents.append(add_enhanced_metadata(doc, "Projects", "docx", project_name))
                    except Exception as e:
                        logger.warning(f"Could not read Word doc {filename}: {e}")
                        
                elif file_ext == '.txt':
                    try:
                        encodings = ['utf-8', 'latin-1', 'cp1252']
                        content = None
                        for encoding in encodings:
                            try:
                                with open(file_path, 'r', encoding=encoding) as f:
                                    content = f.read()
                                break
                            except UnicodeDecodeError:
                                continue
                        
                        if content:
                            enhanced_content = f"ARTILIGENCE PROJECT: {project_name}\\n\\nScript/Document: {filename}\\n\\n{content}"
                            doc = Document(page_content=enhanced_content, metadata={"source": file_path})
                            documents.append(add_enhanced_metadata(doc, "Projects", "text", project_name))
                    except Exception as e:
                        logger.warning(f"Could not read text file {filename}: {e}")
                        
                elif file_ext in ['.sql']:
                    try:
                        with open(file_path, 'r', encoding='utf-8') as f:
                            content = f.read()
                        enhanced_content = f"ARTILIGENCE PROJECT: {project_name}\\n\\nSQL Script: {filename}\\n\\n{content}"
                        doc = Document(page_content=enhanced_content, metadata={"source": file_path})
                        documents.append(add_enhanced_metadata(doc, "Projects", "sql", project_name))
                    except Exception as e:
                        logger.warning(f"Could not read SQL file {filename}: {e}")
                        
                elif file_ext in ['.xlsx', '.xls', '.csv']:
                    try:
                        if file_ext == '.csv':
                            df = pd.read_csv(file_path)
                        else:
                            df = pd.read_excel(file_path)
                        
                        # Create a summary of the spreadsheet
                        summary = f"Spreadsheet with {len(df)} rows and {len(df.columns)} columns\\n"
                        summary += f"Columns: {', '.join(df.columns.tolist())}\\n"
                        summary += f"Sample data:\\n{df.head().to_string()}"
                        
                        enhanced_content = f"ARTILIGENCE PROJECT: {project_name}\\n\\nSpreadsheet: {filename}\\n\\n{summary}"
                        doc = Document(page_content=enhanced_content, metadata={"source": file_path})
                        documents.append(add_enhanced_metadata(doc, "Projects", "spreadsheet", project_name))
                    except Exception as e:
                        logger.warning(f"Could not read spreadsheet {filename}: {e}")
                        
            except Exception as e:
                logger.warning(f"Could not process {filename}: {e}")
        
        logger.info(f"Loaded {len(documents)} enhanced project documents")
        
        # Show project breakdown
        projects = {}
        for doc in documents:
            project = doc.metadata.get('project', 'Unknown')
            projects[project] = projects.get(project, 0) + 1
        
        logger.info(f"Project breakdown: {projects}")
        
        return documents

    def chunk_documents(self, documents: List[Document]) -> List[Document]:
        """Chunk documents using RecursiveCharacterTextSplitter"""
        text_splitter = RecursiveCharacterTextSplitter(
            chunk_size=config.CHUNK_SIZE,
            chunk_overlap=config.CHUNK_OVERLAP,
            length_function=len,
            separators=["\\n\\n", "\\n", " ", ""]
        )
        
        chunks = text_splitter.split_documents(documents)
        logger.info(f"Created {len(chunks)} document chunks")
        
        return chunks

    def load_all_documents(self) -> List[Document]:
        """Load all documents (standard + enhanced projects) and chunk them"""
        logger.info("Starting comprehensive document loading...")
        
        # Load standard documents
        documents = self.load_documents_recursive()
        
        # Load enhanced project documents
        try:
            enhanced_project_docs = self.load_enhanced_project_documents()
            documents.extend(enhanced_project_docs)
            logger.info(f"Added {len(enhanced_project_docs)} enhanced project documents")
        except Exception as e:
            logger.error(f"Could not load enhanced project documents: {e}")
        
        # Filter large documents (except projects)
        filtered_docs = []
        for doc in documents:
            doc_size = len(doc.page_content)
            doc_type = doc.metadata.get('doc_type', 'unknown')
            
            # Keep all project documents regardless of size
            if doc_type == 'Projects' or doc_size <= self.max_file_size:
                filtered_docs.append(doc)
        
        logger.info(f"Using {len(filtered_docs)} documents after filtering")
        
        # Chunk the documents
        chunks = self.chunk_documents(filtered_docs)
        
        # Safety check - limit chunks if too many
        if len(chunks) > 5000:
            logger.warning(f"Too many chunks ({len(chunks)}). Sampling 5000 representative chunks.")
            import random
            random.seed(42)
            chunks = random.sample(chunks, 5000)
        
        logger.info(f"Document processing completed - Total: {len(chunks)} chunks")
        return chunks