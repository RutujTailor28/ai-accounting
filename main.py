from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from app.api.v1.router import api_router
from app.schemas.health import HealthResponse
from app.core.config import settings
import uvicorn

# Create FastAPI application
app = FastAPI(
    title="Accounting RAG System",
    description="Retrieval-Augmented Generation system for accounting documents with Supabase Auth and User/Role CRUD",
    version="1.0.0"
)

# Configure CORS
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # Configure this based on your frontend URL in production
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Include routers
app.include_router(api_router, prefix="/api/v1")


@app.get("/", response_model=HealthResponse)
async def root():
    """Root endpoint."""
    return HealthResponse(
        status="success",
        message="RAG System API is running"
    )


@app.get("/health", response_model=HealthResponse)
async def health_check():
    """Health check endpoint."""
    print(f"[INFO] Health check requested")
    return HealthResponse(
        status="healthy",
        message="All systems operational"
    )


@app.on_event("startup")
async def startup_event():
    """Run on application startup."""
    print("=" * 50)
    print(f"[INFO] RAG System Starting Up (Hybrid Mode)")
    print(f"[INFO] Local Embeddings (Ollama): {settings.embedding_model}")
    print(f"[INFO] Remote LLM (OpenRouter): {settings.openrouter_model}")
    print(f"[INFO] Chunk Size: {settings.chunk_size}")
    print(f"[INFO] Chunk Overlap: {settings.chunk_overlap}")
    print(f"[INFO] ChromaDB Directory: {settings.chroma_persist_directory}")
    print("=" * 50)


@app.on_event("shutdown")
async def shutdown_event():
    """Run on application shutdown."""
    print(f"[INFO] RAG System Shutting Down")


if __name__ == "__main__":
    uvicorn.run(
        "main:app",
        host=settings.api_host,
        port=settings.api_port,
        reload=True
    )
