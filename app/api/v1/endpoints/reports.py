from fastapi import APIRouter, HTTPException, Depends, Query
from fastapi.responses import StreamingResponse
from app.schemas.document import QueryRequest, QueryResponse
from app.ai.rag.retriever import vector_store
from app.services.deps import embedding_service, llm_service
from app.api.deps import get_current_user
import json

router = APIRouter()

@router.get("/debug/chunks")
async def debug_document_chunks(
    company_id: str = Query(..., description="Company id used in vector metadata"),
    document_name: str = Query(..., description="Exact document filename as stored in metadata (e.g. karm-bl.pdf)"),
    limit: int = Query(5, ge=1, le=50, description="How many chunks to return"),
    user=Depends(get_current_user)
):
    """
    Debug helper: return a few stored chunks for a given document.
    This lets you verify what text the model is actually seeing (OCR/PDF extraction quality).
    """
    try:
        # Fetch stored chunks directly from Chroma by metadata.
        # NOTE: We include a company_id filter to avoid cross-tenant leakage.
        res = vector_store.collection.get(
            where={
                "$and": [
                    {"company_id": {"$eq": company_id}},
                    {"document_name": {"$eq": document_name}},
                ]
            },
            include=["documents", "metadatas"],
            limit=limit,
        )

        docs = res.get("documents") or []
        metas = res.get("metadatas") or []

        previews = []
        for i, doc in enumerate(docs):
            meta = metas[i] if i < len(metas) else {}
            previews.append(
                {
                    "index": i,
                    "chunk_index": meta.get("chunk_index"),
                    "document_name": meta.get("document_name"),
                    "file_type": meta.get("file_type"),
                    "preview": (doc or "")[:1200],
                }
            )

        return {
            "company_id": company_id,
            "document_name": document_name,
            "returned": len(previews),
            "chunks": previews,
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@router.post("/query/stream")
async def stream_query_documents(request: QueryRequest, user=Depends(get_current_user)):
    """
    Stream query results for real-time updates (NDJSON format).
    """
    try:
        print(f"[INFO] Received STREAM query request: question='{request.question}', company_id={request.company_id}")
        print(f"[INFO] Filters - file_types: {request.file_types}, folder_ids: {request.folder_ids}, uploaded_by: {request.uploaded_by}, tags: {request.tags}")
        
        # Step 1: Generate Embedding
        query_embedding = embedding_service.generate_embedding(request.question)
        
        # Step 2: Retrieve Relevant Chunks
        count = vector_store.get_collection_count(company_id=request.company_id)
        
        if count == 0:
            async def empty_gen():
                yield json.dumps({"type": "error", "message": "No documents found"}) + "\n"
            return StreamingResponse(empty_gen(), media_type="application/x-ndjson")

        # Retrieval logic
        q_lower = request.question.lower()
        is_exhaustive = any(kw in q_lower for kw in [
            "all", "every", "sum", "total", "exhaustive", "upi", "cheque", "chq", "instrument"
        ])
        # Summary/report questions should also scan broadly; otherwise top-k may miss the table.
        is_summary = any(kw in q_lower for kw in ["balance sheet", "p&l", "profit", "loss", "report", "summary", "computation"])
        
        if is_exhaustive or is_summary:
            total_chunks = count
            print(f"[INFO] Streaming exhaustive extraction for {total_chunks} chunks (is_exhaustive={is_exhaustive}, is_summary={is_summary})")
            results = vector_store.query(
                query_embedding, 
                company_id=request.company_id, 
                n_results=total_chunks,
                file_types=request.file_types,
                folder_ids=request.folder_ids,
                uploaded_by=request.uploaded_by,
                tags=request.tags
            )
        else:
            n_results = 10
            print(f"[INFO] Standard answer mode triggered (n_results={n_results})")
            results = vector_store.query(
                query_embedding, 
                company_id=request.company_id, 
                n_results=n_results,
                file_types=request.file_types,
                folder_ids=request.folder_ids,
                uploaded_by=request.uploaded_by,
                tags=request.tags
            )
        
        # Access the query results correctly
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
                if is_summary:
                    # Filter down to likely report documents to avoid massive contexts
                    # Match by keywords in question and document name
                    ql = (request.question or "").lower()
                    
                    def _is_relevant_doc(doc_name: str) -> bool:
                        dn = (doc_name or "").lower()
                        if "balance sheet" in ql and any(k in dn for k in ["balance", "bl.", "bl_", "-bl", "bs."]):
                            return True
                        if ("p&l" in ql or "profit" in ql or "loss" in ql) and any(k in dn for k in ["p&l", "pl.", "pl_", "-pl", "profit"]):
                            return True
                        if "computation" in ql and "computation" in dn:
                            return True
                        return False

                    filtered = [(c, d) for c, d in zip(documents, source_documents) if _is_relevant_doc(d)]
                    
                    if filtered:
                        documents_local = [c for c, _ in filtered]
                        source_documents_local = [d for _, d in filtered]
                        print(f"[INFO] Summary focus: filtered to {len(documents_local)} chunks from relevant documents")
                    else:
                        # Fallback to all documents if no naming matches found
                        documents_local = documents
                        source_documents_local = source_documents
                        print(f"[INFO] Summary focus: No name matches found, using all {len(documents)} chunks")

                    # Summary reports: return a single deterministic summary when possible
                    result = await llm_service.generate_summary_report(
                        question=request.question,
                        context_chunks=documents_local,
                        source_documents=source_documents_local,
                    )
                    yield json.dumps({
                        "type": "summary",
                        "total_transactions": 0,
                        "total_credits": 0,
                        "total_debits": 0,
                        "sources": result.get("sources", []),
                        "full_answer": result.get("full_answer", result.get("answer", "")),
                    }) + "\n"
                    return
                elif is_exhaustive:
                    async for chunk in llm_service.stream_exhaustive_answer(
                        question=request.question,
                        context_chunks=documents,
                        source_documents=source_documents,
                        batch_size=20 # Reduced for higher precision and completeness
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
                        # Use the helper to extract JSON even if markdown is present
                        content = llm_service._extract_json(result['answer'])
                        if "transactions" in content:
                            txs = content['transactions']
                            total_txs = len(txs)
                            yield json.dumps({"type": "data", "transactions": txs}) + "\n"
                    except Exception as e:
                        print(f"[DEBUG] Failed to parse internal JSON from answer: {e}")
                        
                    yield json.dumps({
                        "type": "summary", 
                        "total_transactions": total_txs,
                        "sources": result['sources'],
                        "full_answer": result.get('full_answer', result['answer'])
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
        print(f"[INFO] Filters - file_types: {request.file_types}, folder_ids: {request.folder_ids}, uploaded_by: {request.uploaded_by}, tags: {request.tags}")
        
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
            "all", "every", "sum", "total", "exhaustive", "upi", "cheque", "chq", "instrument"
        ])
        
        if is_exhaustive:
            print(f"[INFO] Exhaustive extraction mode triggered due to keywords in question")
            # User has paid plan: Retrieve ALL chunks to ensure we don't miss transaction data
            total_chunks = count
            print(f"[INFO] Retrieving ALL {total_chunks} chunks for exhaustive extraction (Paid Plan Enabled)")
            results = vector_store.query(
                query_embedding, 
                company_id=request.company_id, 
                n_results=total_chunks,
                file_types=request.file_types,
                folder_ids=request.folder_ids,
                uploaded_by=request.uploaded_by,
                tags=request.tags
            )
        else:
            # For summary reports, increase chunk count to ensure we find the report table
            is_summary = any(kw in request.question.lower() for kw in ["balance sheet", "p&l", "profit", "loss", "report", "summary", "computation"])
            n_results = 100 if is_summary else 10
            print(f"[INFO] Standard answer mode triggered (n_results={n_results}, is_summary={is_summary})")
            results = vector_store.query(
                query_embedding, 
                company_id=request.company_id, 
                n_results=n_results,
                file_types=request.file_types,
                folder_ids=request.folder_ids,
                uploaded_by=request.uploaded_by,
                tags=request.tags
            )
        
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
            answer=result.get('full_answer', result['answer']),
            sources=result['sources']
        )
        
    except Exception as e:
        print(f"[ERROR] Error processing query: {str(e)}")
        raise HTTPException(status_code=500, detail=f"Internal server error: {str(e)}")
