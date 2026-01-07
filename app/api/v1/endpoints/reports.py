from fastapi import APIRouter, HTTPException, Depends
from app.schemas.document import QueryRequest, QueryResponse
from app.services.embedding_service import EmbeddingService
from app.ai.rag.retriever import vector_store
from app.services.ai_service import LLMService
from app.api.deps import get_current_user

router = APIRouter()

# Initialize services
embedding_service = EmbeddingService()
llm_service = LLMService()


@router.post("/query", response_model=QueryResponse)
async def query_documents(request: QueryRequest, user=Depends(get_current_user)):
    """
    Query the RAG system and generate an answer.
    """
    try:
        print(f"[INFO] Received query request: question='{request.question}', company_id={request.company_id}")
        
        # Step 1: Generate Embedding
        print(f"[INFO] Step 1: Generating query embedding...")
        query_embedding = embedding_service.generate_embedding(request.question)
        
        # Step 2: Retrieve Relevant Chunks
        print(f"[INFO] Step 2: Retrieving relevant chunks from vector store...")
        # Get count first as a sanity check
        count = vector_store.get_collection_count(company_id=request.company_id)
        if count == 0:
            return QueryResponse(
                answer="No documents found for your company. Please upload some documents first.",
                sources=[]
            )
            
        # Retrieval logic based on question intent
        is_exhaustive = any(kw in request.question.lower() for kw in ["all", "every", "list", "total", "sum", "exhaustive"])
        
        if is_exhaustive:
            print(f"[INFO] Exhaustive extraction mode triggered due to keywords in question")
            results = vector_store.query(query_embedding, company_id=request.company_id, n_results=100)
        else:
            results = vector_store.query(query_embedding, company_id=request.company_id, n_results=10)
        
        print(f"[INFO] DONE: Retrieved {len(results.get('documents', [[]])[0])} chunks from vector store.")
        
        # Extract documents and metadata
        documents = results.get('documents', [[]])[0]
        metadatas = results.get('metadatas', [[]])[0]
        
        if not documents:
            print(f"[WARNING] No documents found for companyId={request.company_id}. Returning 'Information not available' response.")
            return QueryResponse(
                answer="Information not available in uploaded records",
                sources=[]
            )
        
        # Extract source document names
        source_documents = [meta.get('documentName', 'Unknown') for meta in metadatas]
        
        # Step 3: Generate answer using LLM
        # Check if user is asking for a list/all items which requires exhaustive extraction
        exhaustive_keywords = ["all", "list", "every", "full", "complete", "total"]
        is_exhaustive = any(word in request.question.lower() for word in exhaustive_keywords)
        
        if is_exhaustive:
            print(f"[INFO] Exhaustive extraction mode triggered due to keywords in question")
            result = await llm_service.generate_exhaustive_answer(
                question=request.question,
                context_chunks=documents,
                source_documents=source_documents
            )
        else:
            print(f"[INFO] Standard answer mode triggered")
            result = await llm_service.generate_answer(
                question=request.question,
                context_chunks=documents,
                source_documents=source_documents
            )
        
        print(f"[INFO] Successfully generated answer for companyId={request.company_id} with {len(result['sources'])} sources")
        
        return QueryResponse(
            answer=result['answer'],
            sources=result['sources']
        )
        
    except Exception as e:
        print(f"[ERROR] Error processing query: {str(e)}")
        raise HTTPException(status_code=500, detail=f"Internal server error: {str(e)}")
