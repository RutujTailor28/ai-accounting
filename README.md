# RAG System for Accounting

A production-ready FastAPI backend implementing Retrieval-Augmented Generation (RAG) for an accounting company. The system processes documents (PDF, Excel, CSV, Word), stores embeddings locally, and answers queries using locally-running Ollama models.

## Features

- **Document Processing**: Support for PDF, Excel (.xlsx, .xls), CSV, and Word (.docx) files
- **Local AI Models**: Uses Ollama for embeddings (`nomic-embed-text`) and LLM (`llama3`)
- **Vector Storage**: ChromaDB for persistent vector storage with company-wise isolation
- **Data Privacy**: All processing happens locally, no external API calls
- **Anti-Hallucination**: Strict prompting ensures answers come only from uploaded documents

## Prerequisites

1. **Python 3.10+**
2. **Ollama** installed and running locally
   - Install from: https://ollama.ai
   - Pull required models:
     ```bash
     ollama pull nomic-embed-text
     ollama pull llama3
     ```

## Installation

1. **Clone the repository** (or navigate to the project directory)

2. **Create a virtual environment**:
   ```bash
   python -m venv venv
   ```

3. **Activate the virtual environment**:
   - Windows:
     ```bash
     venv\Scripts\activate
     ```
   - Linux/Mac:
     ```bash
     source venv/bin/activate
     ```

4. **Install dependencies**:
   ```bash
   pip install -r requirements.txt
   ```

5. **Create environment file**:
   ```bash
   copy .env.example .env
   ```
   
   Edit `.env` if needed to customize configuration.

## Running the Application

1. **Ensure Ollama is running**:
   ```bash
   ollama list
   ```
   You should see `nomic-embed-text` and `llama3` in the list.

2. **Start the FastAPI server**:
   ```bash
   uvicorn main:app --reload
   ```

3. **Access the API**:
   - API: http://localhost:8000
   - Interactive docs: http://localhost:8000/docs
   - Alternative docs: http://localhost:8000/redoc

## API Endpoints

### 1. Upload Document
**POST** `/api/upload`

Upload and process a document for the RAG system.

**Request**:
- `file`: Document file (PDF, Excel, CSV, or Word)
- `company_id`: Unique company identifier (form field)

**Response**:
```json
{
  "message": "Document uploaded and processed successfully",
  "document_name": "financial_report.pdf",
  "chunks_created": 42,
  "company_id": "company_123"
}
```

**Example using curl**:
```bash
curl -X POST "http://localhost:8000/api/upload" \
  -F "file=@document.pdf" \
  -F "company_id=company_123"
```

### 2. Query Documents
**POST** `/api/query`

Ask a question and get an answer based on uploaded documents.

**Request**:
```json
{
  "question": "What was the total revenue in Q4?",
  "company_id": "company_123"
}
```

**Response**:
```json
{
  "answer": "The total revenue in Q4 was $2.5 million according to the financial report.",
  "sources": ["financial_report.pdf"]
}
```

**Example using curl**:
```bash
curl -X POST "http://localhost:8000/api/query" \
  -H "Content-Type: application/json" \
  -d '{"question": "What was the total revenue?", "company_id": "company_123"}'
```

### 3. Health Check
**GET** `/health`

Check if the API is running.

**Response**:
```json
{
  "status": "healthy",
  "message": "All systems operational"
}
```

## Project Structure

```
ACC-BACKEND/
├── main.py                      # FastAPI application entry point
├── config.py                    # Configuration management
├── requirements.txt             # Python dependencies
├── .env.example                 # Environment variables template
├── app/
│   ├── models/
│   │   └── schemas.py          # Pydantic models
│   ├── routes/
│   │   ├── upload.py           # Upload endpoint
│   │   └── query.py            # Query endpoint
│   ├── services/
│   │   ├── document_parser.py  # Document processing
│   │   ├── text_chunker.py     # Text chunking
│   │   ├── embedding_service.py # Embedding generation
│   │   ├── vector_store.py     # Vector database management
│   │   └── llm_service.py      # LLM integration
│   └── utils/
│       └── logger.py           # Logging configuration
├── chroma_db/                   # ChromaDB storage (created automatically)
└── logs/                        # Application logs (created automatically)
```

## Configuration

Edit `.env` to customize settings:

```env
# Ollama Configuration
OLLAMA_BASE_URL=http://localhost:11434
EMBEDDING_MODEL=nomic-embed-text
LLM_MODEL=llama3

# Chunking Configuration
CHUNK_SIZE=500
CHUNK_OVERLAP=100

# Vector Database
CHROMA_PERSIST_DIRECTORY=./chroma_db

# API Configuration
API_HOST=0.0.0.0
API_PORT=8000

# Logging
LOG_LEVEL=INFO
```

## Company Isolation

The system ensures data privacy through company-wise isolation:
- Each document is tagged with a `company_id` during upload
- Queries are filtered by `company_id` to prevent data leakage
- Companies can only access their own documents

## Anti-Hallucination

The LLM is configured with strict prompting to prevent hallucinations:
- Answers are generated ONLY from retrieved document context
- If information is not found, the system responds: "Information not available in uploaded records"
- Low temperature (0.1) for factual responses

## Troubleshooting

### Ollama Connection Error
- Ensure Ollama is running: `ollama list`
- Check the Ollama URL in `.env` matches your setup

### Model Not Found
- Pull the required models:
  ```bash
  ollama pull nomic-embed-text
  ollama pull llama3
  ```

### Document Parsing Error
- Ensure the file format is supported (PDF, Excel, CSV, Word)
- Check that the file is not corrupted

### No Results for Query
- Verify documents were uploaded for the correct `company_id`
- Check ChromaDB directory exists and has data

## Development

To run in development mode with auto-reload:
```bash
uvicorn main:app --reload --host 0.0.0.0 --port 8000
```

## Production Deployment

For production:
1. Set `LOG_LEVEL=WARNING` in `.env`
2. Configure CORS origins in `main.py`
3. Use a production ASGI server:
   ```bash
   uvicorn main:app --host 0.0.0.0 --port 8000 --workers 4
   ```

## License

Proprietary - Accounting Company Internal Use
