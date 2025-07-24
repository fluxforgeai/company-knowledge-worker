# 🏢 Company Knowledge Worker

A powerful RAG-powered Q&A assistant for company documents, built with Python, LangChain, and Gradio.

## 🌟 Features

- **Enhanced RAG Pipeline**: Uses GPT-4.1 with k=25 retrieval for comprehensive project coverage
- **Persistent Quick Questions**: Always-visible example questions for easy access
- **Automatic Port Management**: Intelligent port conflict resolution
- **Multi-format Document Support**: PDF, Word, Excel, text files, and more
- **Project-Aware Search**: Enhanced retrieval for better project information coverage
- **Web & CLI Interfaces**: Choose between Gradio web UI or command-line interface
- **Comprehensive Project Knowledge**:
  - SQL Server upgrades and database infrastructure
  - Precision agriculture and asset management systems
  - Mining analytics and fleet management (SFMS)
  - Database migration projects
  - Advanced driver assistance systems (ADAS)

## Quick Start

### 1. Setup Environment

```bash
# Create and activate virtual environment
python3.13 -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate

# Install dependencies
pip install -r requirements.txt
```

### 2. Configuration

Create a `.env` file with your settings:

```bash
# Required: OpenAI API Key
OPENAI_API_KEY=your_openai_api_key_here

# Optional: Document path
BASE_PATH=/Users/johanjgenis/Documents/Artiligence

# Optional: Model configuration (now supports GPT-4.1)
OPENAI_MODEL=gpt-4.1
```

### 3. Run the Application

```bash
# Web interface with enhanced features (recommended)
python quick_fix_app.py --mode web

# Command line interface
python quick_fix_app.py --mode cli

# Build vector database only
python quick_fix_app.py --mode build
```

## Usage Options

### Web Interface with Enhanced Features
The web interface now includes:
- **Persistent Quick Questions**: Always-visible clickable example questions
- **Enhanced RAG**: k=25 retrieval for comprehensive project coverage
- **Automatic Port Management**: Resolves port conflicts automatically

```bash
python quick_fix_app.py --mode web --port 7860 --share
```

### Command Line Interface
```bash
python quick_fix_app.py --mode cli
```

### Build Vector Store
```bash
python quick_fix_app.py --mode build --rebuild-db
```

## Command Line Arguments

- `--mode`: Application mode (`web`, `cli`, `build`)
- `--log-level`: Logging level (`DEBUG`, `INFO`, `WARNING`, `ERROR`)
- `--rebuild-db`: Force rebuild of vector database
- `--port`: Port for web interface
- `--share`: Create public Gradio share link
- `--no-browser`: Don't open browser automatically

## Project Structure

```
company_knowledge_worker/
├── src/
│   ├── config.py              # Configuration management
│   ├── document_loader.py     # Document loading & processing
│   ├── vector_store.py        # ChromaDB vector store management
│   ├── rag_pipeline.py        # RAG chain setup
│   ├── chat_interface.py      # Gradio interface
│   └── logging_config.py      # Logging configuration
├── data/                      # Local data directory
├── logs/                      # Application logs
├── vector_db/                 # ChromaDB database
├── requirements.txt           # Python dependencies
├── .env                       # Environment variables
└── app.py                     # Main application entry point
```

## Configuration Options

All configuration can be set via environment variables in `.env`:

```bash
# Core settings
OPENAI_API_KEY=your_key
OPENAI_MODEL=gpt-4-turbo-preview
BASE_PATH=/path/to/documents

# Processing settings
MAX_FILE_SIZE=100000
CHUNK_SIZE=1200
CHUNK_OVERLAP=150

# Retrieval settings (enhanced)
RETRIEVAL_K=25

# Interface settings
GRADIO_PORT=7860
GRADIO_SHARE=False
```

## Supported Document Types

- **Text files**: `.md`, `.txt`, `.py`, `.js`, `.html`, `.css`, `.json`, `.yml`, `.yaml`, `.xml`, `.csv`, `.rst`, `.tex`
- **PDFs**: `.pdf`
- **Word documents**: `.docx`, `.doc`
- **Spreadsheets**: `.xlsx`, `.xls`

## Example Questions

- "What projects is Artiligence working on?"
- "Tell me about the SQL Server upgrade project"
- "What is the Precision Agriculture Asset Management project?"
- "Describe the SFMS Mining Analytics project"
- "What database migration work is being done?"
- "Tell me about the company's invoices and financial information"
- "What contracts does Artiligence have?"

## Troubleshooting

### Common Issues

1. **API Key Error**: Ensure `OPENAI_API_KEY` is set in `.env`
2. **Document Path Error**: Verify `BASE_PATH` points to correct directory
3. **Port Already In Use**: Use `--port` to specify different port
4. **Memory Issues**: Reduce `MAX_FILE_SIZE` or `CHUNK_SIZE` in `.env`

### Logs

Check logs in `logs/company_knowledge_worker.log` for detailed error information.

### Debug Mode

Run with debug logging for detailed information:
```bash
python app.py --log-level DEBUG
```

## Development

The application is built with:
- **LangChain**: Document processing and RAG pipeline
- **ChromaDB**: Vector database for embeddings
- **OpenAI**: Language model and embeddings
- **Gradio**: Web interface
- **Python 3.13**: Runtime environment

Created by Claude Code for Johan Genis, co-owner of Artiligence (Pty) Ltd.