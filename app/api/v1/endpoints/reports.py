from fastapi import APIRouter, HTTPException, Depends
from fastapi.responses import StreamingResponse
from app.schemas.document import QueryRequest, QueryResponse
from app.ai.rag.retriever import vector_store
from app.services.deps import embedding_service, llm_service
from app.api.deps import get_current_user
import json

router = APIRouter()

@router.post("/query/stream")
async def stream_query_documents(request: QueryRequest, user=Depends(get_current_user)):
    """
    Stream query results for real-time updates (NDJSON format).
    """
    try:
        print(f"[INFO] Received STREAM query request: question='{request.question}', company_id={request.company_id}")
        
        # Step 1: Generate Embedding
        query_embedding = embedding_service.generate_embedding(request.question)
        
        # Step 2: Retrieve Relevant Chunks
        count = vector_store.get_collection_count(company_id=request.company_id)
        
        if count == 0:
            async def empty_gen():
                yield json.dumps({"type": "error", "message": "No documents found"}) + "\n"
            return StreamingResponse(empty_gen(), media_type="application/x-ndjson")

        # Retrieval logic
        is_exhaustive = any(kw in request.question.lower() for kw in [
            "all", "every", "list", "total", "sum", "exhaustive", 
            "transaction", "records", "record", "history", "statement", 
            "report", "upi", "summary", "fetch", "provide"
        ])
        
        if is_exhaustive:
            total_chunks = count
            print(f"[INFO] Streaming exhaustive extraction for {total_chunks} chunks")
            results = vector_store.query(query_embedding, company_id=request.company_id, n_results=total_chunks)
        else:
            results = vector_store.query(query_embedding, company_id=request.company_id, n_results=10)
        
        documents = results.get('documents', [[]])[0]
        metadatas = results.get('metadatas', [[]])[0]
        
        if not documents:
            async def no_docs_gen():
                 yield json.dumps({"type": "summary", "total_transactions": 0, "sources": []}) + "\n"
            return StreamingResponse(no_docs_gen(), media_type="application/x-ndjson")

        source_documents = [meta.get('document_name', 'Unknown') for meta in metadatas]
        
        # Step 3: Stream generation
        async def response_generator():
            try:
                if is_exhaustive:
                    async for chunk in llm_service.stream_exhaustive_answer(
                        question=request.question,
                        context_chunks=documents,
                        source_documents=source_documents,
                        batch_size=40 # Reduced for higher precision and completeness
                    ):
                        yield chunk + "\n"
                else:
                    # Standard mode: just generate and yield once
                    print(f"[INFO] Standard answer mode triggered (Streaming wrapper)")
                    result = await llm_service.generate_answer(
                        question=request.question,
                        context_chunks=documents,
                        source_documents=source_documents
                    )
                    total_txs = 0
                    try:
                        content = json.loads(result['answer'])
                        if "transactions" in content:
                            txs = content['transactions']
                            total_txs = len(txs)
                            yield json.dumps({"type": "data", "transactions": txs}) + "\n"
                    except:
                        pass
                        
                    yield json.dumps({
                        "type": "summary", 
                        "total_transactions": total_txs,
                        "sources": result['sources'],
                        "full_answer": result['answer']
                    }) + "\n"
            except Exception as e:
                print(f"[ERROR] Stream generation error: {e}")
                yield json.dumps({"type": "error", "message": str(e)}) + "\n"

        return StreamingResponse(response_generator(), media_type="application/x-ndjson")

    except Exception as e:
        print(f"[ERROR] Error setup stream: {str(e)}")
        raise HTTPException(status_code=500, detail=str(e))


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
        # Added more keywords to ensure financial queries trigger exhaustive search
        is_exhaustive = any(kw in request.question.lower() for kw in [
            "all", "every", "list", "total", "sum", "exhaustive", 
            "transaction", "records", "record", "history", "statement", 
            "report", "upi", "summary", "fetch", "provide"
        ])
        
        if is_exhaustive:
            print(f"[INFO] Exhaustive extraction mode triggered due to keywords in question")
            # User has paid plan: Retrieve ALL chunks to ensure we don't miss transaction data
            total_chunks = count
            print(f"[INFO] Retrieving ALL {total_chunks} chunks for exhaustive extraction (Paid Plan Enabled)")
            results = vector_store.query(query_embedding, company_id=request.company_id, n_results=total_chunks)
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
        source_documents = [meta.get('document_name', 'Unknown') for meta in metadatas]
        
        # Step 3: Generate answer using LLM
        # Use the same 'is_exhaustive' flag determined earlier for retrieval
        
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
