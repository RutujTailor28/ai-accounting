from fastapi import APIRouter, HTTPException, Depends, Query, Request
from fastapi.responses import StreamingResponse
from app.schemas.document import QueryRequest, QueryResponse
from app.ai.rag.retriever import vector_store
from app.services.deps import embedding_service, llm_service
from app.api.deps import get_current_user
import json
from app.core.supabase import supabase

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
async def stream_query_documents(request_body: QueryRequest, request: Request, user=Depends(get_current_user)):
    """
    Stream query results for real-time updates (NDJSON format).
    """
    try:
        # Step 0: Get user's verified company_id from profile
        profile_res = supabase.table("profiles").select("company_id").eq("id", user.id).single().execute()
        company_id = profile_res.data["company_id"]
        
        # Override the request body's company_id to ensure consistency
        request_body.company_id = company_id

        print(f"[INFO] Received STREAM query request: question='{request_body.question}', company_id={request_body.company_id}")
        print(f"[INFO] Filters - file_types: {request_body.file_types}, folder_ids: {request_body.folder_ids}, uploaded_by: {request_body.uploaded_by}, tags: {request_body.tags}")
        
        # Step 1: Generate Embedding
        query_embedding = embedding_service.generate_embedding(request_body.question)
        

        # Resolve customer_id to folder_ids if provided
        # Resolve customer_id to folder_ids if provided
        folder_ids = request_body.folder_ids or []
        target_document_names = []
        
        # Date Filter Injection
        effective_question = request_body.question
        if request_body.start_date and request_body.end_date:
            effective_question += f" from {request_body.start_date} to {request_body.end_date}"
            print(f"[INFO] Injected date range into question: {effective_question}")
                
        if request_body.customer_id:
            customer_folders = supabase.table("folders") \
                .select("id") \
                .eq("customer_id", request_body.customer_id) \
                .is_("deleted_at", "null") \
                .execute()
            
            if customer_folders.data:
                customer_folder_ids = [f["id"] for f in customer_folders.data]
                
                if folder_ids:
                    # If folder_ids were already provided, intersect them
                    effective_folder_ids = list(set(folder_ids) & set(customer_folder_ids))
                else:
                    effective_folder_ids = customer_folder_ids

                # Fetch all files in these effective folders to get document names
                if effective_folder_ids:
                    customer_files = supabase.table("files") \
                        .select("name") \
                        .in_("folder_id", effective_folder_ids) \
                        .is_("deleted_at", "null") \
                        .execute()
                    
                    if customer_files.data:
                        target_document_names = [f["name"] for f in customer_files.data]
                        print(f"[INFO] Resolved {len(target_document_names)} documents for customer {request_body.customer_id} in {len(effective_folder_ids)} folders")
                
                folder_ids = effective_folder_ids
            
            if not folder_ids and not target_document_names:
                # If a customer was selected but has no folders/documents
                async def no_customer_docs_gen():
                    yield json.dumps({"type": "summary", "total_transactions": 0, "sources": [], "message": "No documents found for this customer."}) + "\n"
                return StreamingResponse(no_customer_docs_gen(), media_type="application/x-ndjson")

        # Step 2: Retrieve Relevant Chunks
        count = vector_store.get_collection_count(company_id=request_body.company_id)
        
        if count == 0:
            async def empty_gen():
                yield json.dumps({"type": "error", "message": "No documents found"}) + "\n"
            return StreamingResponse(empty_gen(), media_type="application/x-ndjson")

        # Retrieval logic
        q_lower = request_body.question.lower()
        
        # 1. Check for explicit limit (e.g. "10 records", "top 5 transactions")
        import re
        limit_match = re.search(r'\b(\d+)\s*(?:records|rows|transactions|items|results|entries)\b', q_lower)
        explicit_limit = int(limit_match.group(1)) if limit_match else None

        # 2. Check for keywords that imply "ALL"
        has_exhaustive_keywords = any(kw in q_lower for kw in [
            "all", "every", "sum", "total", "exhaustive", "upi", "cheque", "chq", "instrument",
            "cash", "atm", "self", "withdrawal"
        ])
        
        # 3. Check for summary/report keywords (Balance Sheet, P&L, etc.)
        has_summary_keywords = any(kw in q_lower for kw in [
            "balance sheet", "p&l", "p & l", "profit", "loss", "report", "summary", "computation"
        ])

        # 4. Check for active filters
        # If specific folders, date range, file types, or tags are provided, user expects ALL matching data
        has_active_filters = bool(
            (request_body.start_date and request_body.end_date) or 
            request_body.folder_ids or 
            request_body.file_types or 
            request_body.tags or
            request_body.uploaded_by
        )

        # Decision Logic
        if explicit_limit:
            # Case A: User asked for a specific number. Honor it strictly.
            is_exhaustive = False
            is_summary = False
            n_results = explicit_limit
            print(f"[INFO] Explicit limit detected: {n_results}. Mode: Standard (Limited)")
            
        else:
            # Case B: Default (Filters, Keywords, or General). Exhaustive Search.
            # User wants "all records" by default unless a specific number is requested.
            is_exhaustive = True
            is_summary = has_summary_keywords
            print(f"[INFO] Defaulting to Exhaustive Search (All Records). keywords={has_exhaustive_keywords}, filters={has_active_filters}")
        
        if is_exhaustive:
            print(f"[INFO] Streaming exhaustive extraction for ALL chunks (is_summary={is_summary})")
            
            # Manual Pagination Loop to fetch ALL chunks safely
            all_chunks = []
            all_metadatas = []
            
            offset = 0
            limit = 5000
            
            where_filter = vector_store._build_where_filter(
                 company_id=request_body.company_id,
                 document_names=target_document_names if target_document_names else None,
                 file_types=request_body.file_types,
                 # Strict folder filter: always apply if provided
                 folder_ids=folder_ids,
                 uploaded_by=request_body.uploaded_by,
                 tags=request_body.tags
            )
            
            print(f"[INFO] Fetching all chunks with filter: {where_filter}...")
            
            while True:
                batch = vector_store.collection.get(
                    where=where_filter,
                    include=['documents', 'metadatas'],
                    limit=limit,
                    offset=offset
                )
                
                b_docs = batch.get('documents', [])
                b_metas = batch.get('metadatas', [])
                
                if not b_docs:
                    break
                    
                all_chunks.extend(b_docs)
                all_metadatas.extend(b_metas)
                
                offset += len(b_docs)
                if len(b_docs) < limit:
                    break
            
            # Deterministic Sorting
            # Key: (Document Name, Chunk Index)
            combined = []
            for i in range(len(all_chunks)):
                meta = all_metadatas[i] or {}
                # Tie-breaker: content snippet
                sort_key = (
                    meta.get('document_name', ''),
                    int(meta.get('chunk_index', 0)),
                    all_chunks[i][:20]
                )
                combined.append((sort_key, all_chunks[i], meta))
            
            combined.sort(key=lambda x: x[0])
            
            sorted_chunks = [x[1] for x in combined]
            sorted_metas = [x[2] for x in combined]
            
            print(f"[INFO] Retrieved and sorted {len(sorted_chunks)} chunks for processing")
            
            # Mimic expected structure
            results = {
                'documents': [sorted_chunks],
                'metadatas': [sorted_metas]
            }
        else:
            # n_results is already set in the Decision Logic block
            print(f"[INFO] Standard answer mode triggered (n_results={n_results})")
            results = vector_store.query(
                query_embedding, 
                company_id=request_body.company_id, 
                n_results=n_results,
                file_types=request_body.file_types,
                folder_ids=folder_ids, 
                document_names=target_document_names if target_document_names else None,
                uploaded_by=request_body.uploaded_by,
                tags=request_body.tags
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
                # Check if client disconnected before starting
                if await request.is_disconnected():
                    print("[INFO] Client disconnected before streaming started, aborting")
                    return
                
                if is_summary:
                    # Filter down to likely report documents to avoid massive contexts
                    # Match by keywords in question and document name
                    ql = (request_body.question or "").lower()
                    
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
                        question=effective_question,
                        context_chunks=documents_local,
                        source_documents=source_documents_local,
                    )
                    
                    # Check disconnection before yielding
                    if await request.is_disconnected():
                        print("[INFO] Client disconnected during summary generation, stopping")
                        return
                    
                    summary_payload = {
                        "type": "summary",
                        "total_transactions": 0,
                        "total_credits": 0,
                        "total_debits": 0,
                        "sources": list(result.get("sources", [])), # Ensure it's a list
                        "full_answer": result.get("full_answer", result.get("answer", "")),
                        "data": json.loads(result.get("answer", "{}")) # Pass structured data for renderer
                    }
                    if "balance_sheet" in result.get("answer", ""):
                         print(f"[INFO] Summary report generated with {len(result.get('sources', []))} sources")
                    
                    yield json.dumps(summary_payload) + "\n"
                    return
                elif is_exhaustive:
                    async for chunk in llm_service.stream_exhaustive_answer(
                        question=effective_question,
                        context_chunks=documents,
                        source_documents=source_documents,
                        batch_size=5, # Reduced for faster initial response
                        company_id=request_body.company_id
                    ):
                        # Check if client disconnected before yielding each chunk
                        if await request.is_disconnected():
                            print("[INFO] ✅ Client disconnected during exhaustive streaming, stopping processing")
                            return
                        yield chunk + "\n"
                else:
                    # Standard mode: just generate and yield once
                    print(f"[INFO] Standard answer mode triggered (Streaming wrapper)")
                    result = await llm_service.generate_answer(
                        question=effective_question,
                        context_chunks=documents,
                        source_documents=source_documents,
                        company_id=request_body.company_id
                    )
                    
                    # Check disconnection before processing result
                    if await request.is_disconnected():
                        print("[INFO] Client disconnected during standard answer generation, stopping")
                        return
                    
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
        # Step 0: Get user's verified company_id from profile
        profile_res = supabase.table("profiles").select("company_id").eq("id", user.id).single().execute()
        company_id = profile_res.data["company_id"]
        
        # Override
        request.company_id = company_id
        
        print(f"[INFO] Received query request: question='{request.question}', company_id={request.company_id}")
        print(f"[INFO] Filters - file_types: {request.file_types}, folder_ids: {request.folder_ids}, uploaded_by: {request.uploaded_by}, tags: {request.tags}")
        
        # Step 1: Generate Embedding
        print(f"[INFO] Step 1: Generating query embedding...")
        query_embedding = embedding_service.generate_embedding(request.question)
        
        # Resolve customer_id to folder_ids if provided
        # Resolve customer_id to folder_ids if provided
        folder_ids = request.folder_ids or []
        target_document_names = []

        # Date Filter Injection
        effective_question = request.question
        if request.start_date and request.end_date:
            effective_question += f" from {request.start_date} to {request.end_date}"
            print(f"[INFO] Injected date range into question: {effective_question}")
        
        if request.customer_id:
            customer_folders = supabase.table("folders") \
                .select("id") \
                .eq("customer_id", request.customer_id) \
                .is_("deleted_at", "null") \
                .execute()
            
            if customer_folders.data:
                customer_folder_ids = [f["id"] for f in customer_folders.data]
                
                if folder_ids:
                    # If folder_ids were already provided, intersect them
                    effective_folder_ids = list(set(folder_ids) & set(customer_folder_ids))
                else:
                    effective_folder_ids = customer_folder_ids

                # Fetch all files in these effective folders
                if effective_folder_ids:
                    customer_files = supabase.table("files") \
                        .select("name") \
                        .in_("folder_id", effective_folder_ids) \
                        .is_("deleted_at", "null") \
                        .execute()
                    
                    if customer_files.data:
                        target_document_names = [f["name"] for f in customer_files.data]
                        print(f"[INFO] Resolved {len(target_document_names)} documents for customer {request.customer_id} in {len(effective_folder_ids)} folders")

                folder_ids = effective_folder_ids
            
            if not folder_ids and not target_document_names:
                # If a customer was selected but has no folders/documents
                return QueryResponse(
                    answer="No documents found for this customer.",
                    sources=[]
                )

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
        q_lower = request.question.lower()
        
        # 1. Check for explicit limit
        import re
        limit_match = re.search(r'\b(\d+)\s*(?:records|rows|transactions|items|results|entries)\b', q_lower)
        explicit_limit = int(limit_match.group(1)) if limit_match else None

        # 2. Check for keywords that imply "ALL"
        has_exhaustive_keywords = any(kw in q_lower for kw in [
            "all", "every", "sum", "total", "exhaustive", "upi", "cheque", "chq", "instrument",
            "cash", "atm", "self", "withdrawal"
        ])
        
        # 3. Check for summary/report keywords (Balance Sheet, P&L, etc.)
        has_summary_keywords = any(kw in q_lower for kw in [
            "balance sheet", "p&l", "p & l", "profit", "loss", "report", "summary", "computation"
        ])

        # 4. Check for active filters
        has_active_filters = bool(
            (request.start_date and request.end_date) or 
            request.folder_ids or 
            request.file_types or 
            request.tags or
            request.uploaded_by
        )

        # Decision Logic - Identical to streaming endpoint
        if explicit_limit:
            is_exhaustive = False
            n_results = explicit_limit
            print(f"[INFO] Explicit limit detected: {n_results}. Mode: Standard (Limited)")
            
        else:
            # Default to Exhaustive
            is_exhaustive = True
            print(f"[INFO] Defaulting to Exhaustive Search (All Records). keywords={has_exhaustive_keywords}, filters={has_active_filters}")
        
        if is_exhaustive:
            print(f"[INFO] Exhaustive extraction mode triggered")
            # User has paid plan: Retrieve ALL chunks to ensure we don't miss transaction data
            print(f"[INFO] Retrieving ALL chunks for exhaustive extraction (Paid Plan Enabled) - Pagination & Sort")
            
            # Manual Pagination Loop
            all_chunks = []
            all_metadatas = []
            offset = 0
            limit = 5000
            
            where_filter = vector_store._build_where_filter(
                 company_id=request.company_id,
                 document_names=target_document_names if target_document_names else None,
                 file_types=request.file_types,
                 # Strict folder filter: always apply if provided
                 folder_ids=folder_ids,
                 uploaded_by=request.uploaded_by,
                 tags=request.tags
            )
            
            while True:
                batch = vector_store.collection.get(
                    where=where_filter,
                    include=['documents', 'metadatas'],
                    limit=limit,
                    offset=offset
                )
                
                b_docs = batch.get('documents', [])
                b_metas = batch.get('metadatas', [])
                
                if not b_docs:
                    break
                    
                all_chunks.extend(b_docs)
                all_metadatas.extend(b_metas)
                
                offset += len(b_docs)
                if len(b_docs) < limit:
                    break
            
            # Deterministic Sorting
            combined = []
            for i in range(len(all_chunks)):
                meta = all_metadatas[i] or {}
                sort_key = (
                    meta.get('document_name', ''),
                    int(meta.get('chunk_index', 0)),
                    all_chunks[i][:20]
                )
                combined.append((sort_key, all_chunks[i], meta))
            
            combined.sort(key=lambda x: x[0])
            
            sorted_chunks = [x[1] for x in combined]
            sorted_metas = [x[2] for x in combined]
            
            print(f"[INFO] Retrieved and sorted {len(sorted_chunks)} chunks for processing")
            
            results = {
                'documents': [sorted_chunks],
                'metadatas': [sorted_metas]
            }
        else:
            # n_results is already set in the Decision Logic block
            print(f"[INFO] Standard answer mode triggered (n_results={n_results})")
            results = vector_store.query(
                query_embedding, 
                company_id=request.company_id, 
                n_results=n_results,
                file_types=request.file_types,
                folder_ids=folder_ids, 
                document_names=target_document_names if target_document_names else None,
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
                question=effective_question,
                context_chunks=documents,
                source_documents=source_documents,
                company_id=request.company_id
            )
        else:
            print(f"[INFO] Standard answer mode triggered")
            result = await llm_service.generate_answer(
                question=effective_question,
                context_chunks=documents,
                source_documents=source_documents,
                company_id=request.company_id
            )
        
        print(f"[INFO] Successfully generated answer for companyId={request.company_id} with {len(result['sources'])} sources")
        
        return QueryResponse(
            answer=result.get('full_answer', result['answer']),
            sources=result['sources']
        )
        
    except Exception as e:
        print(f"[ERROR] Error processing query: {str(e)}")
        raise HTTPException(status_code=500, detail=f"Internal server error: {str(e)}")
