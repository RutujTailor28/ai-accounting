from typing import List, Optional
from pydantic import Field
from .base import CamelModel

class UploadRequest(CamelModel):
    """Request model for document upload."""
    company_id: str = Field(..., description="Unique identifier for the company")

class UploadResponse(CamelModel):
    """Response model for document upload."""
    message: str
    document_name: str
    chunks_created: Optional[int] = 0
    company_id: str
    s3_key: Optional[str] = Field(None, description="S3 object key (path) where file is stored")
    s3_url: Optional[str] = Field(None, description="Presigned URL to access the file (expires in 1 hour)")

class QueryRequest(CamelModel):
    """Request model for querying the RAG system."""
    question: str = Field(..., description="Question to ask the system")
    company_id: str = Field(..., description="Unique identifier for the company")
    file_types: Optional[List[str]] = Field(None, description="Filter by file extensions (e.g. ['pdf', 'docx'])")
    folder_ids: Optional[List[str]] = Field(None, description="Filter by folder IDs")
    uploaded_by: Optional[List[str]] = Field(None, description="Filter by user IDs")
    start_date: Optional[str] = Field(None, description="Start date for filtering documents (ISO format)")
    end_date: Optional[str] = Field(None, description="End date for filtering documents (ISO format)")
    tags: Optional[List[str]] = Field(None, description="Filter by tags")
    customer_id: Optional[str] = Field(None, description="Filter by customer ID")

class QueryResponse(CamelModel):
    """Response model for query results."""
    answer: str
    sources: List[str] = Field(default_factory=list, description="List of source document names")
