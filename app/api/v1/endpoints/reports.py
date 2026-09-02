from fastapi import APIRouter, HTTPException, Depends, Query, Request
from fastapi.responses import StreamingResponse
from app.schemas.document import QueryRequest, QueryResponse
from app.ai.rag.retriever import vector_store
from app.services.deps import embedding_service, llm_service
from app.api.deps import get_current_user, require_role
import json
from app.core.supabase import supabase_admin

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
async def stream_query_documents(
    request_body: QueryRequest, 
    request: Request, 
    user=Depends(require_role(["user", "manager", "admin", "superadmin"], required_permission="create_reports"))
):
    """
    Stream query results for real-time updates (NDJSON format).
    """
    try:

        profile_res = supabase_admin.table("profiles").select("company_id").eq("id", user.id).single().execute()
        company_id = profile_res.data["company_id"]

        request_body.company_id = company_id

        print(f"[INFO] Received STREAM query request: question='{request_body.question}', company_id={request_body.company_id}")
        print(f"[INFO] Filters - file_types: {request_body.file_types}, folder_ids: {request_body.folder_ids}, uploaded_by: {request_body.uploaded_by}, tags: {request_body.tags}")

        query_embedding = embedding_service.generate_embedding(request_body.question)

        folder_ids = request_body.folder_ids or []
        target_document_names = []

        effective_question = request_body.question
        if request_body.start_date and request_body.end_date:
            effective_question += f" from {request_body.start_date} to {request_body.end_date}"
            print(f"[INFO] Injected date range into question: {effective_question}")

        if request_body.customer_id:
            customer_folders = supabase_admin.table("folders") \
                .select("id") \
                .eq("customer_id", request_body.customer_id) \
                .is_("deleted_at", "null") \
                .execute()

            if customer_folders.data:
                customer_folder_ids = [f["id"] for f in customer_folders.data]

                if folder_ids:

                    effective_folder_ids = list(set(folder_ids) & set(customer_folder_ids))
                else:
                    effective_folder_ids = customer_folder_ids

                if effective_folder_ids:
                    customer_files = supabase_admin.table("files") \
                        .select("name") \
                        .in_("folder_id", effective_folder_ids) \
                        .is_("deleted_at", "null") \
                        .execute()

                    if customer_files.data:
                        target_document_names = list(set([f["name"] for f in customer_files.data]))
                        print(f"[INFO] Resolved {len(target_document_names)} documents for customer {request_body.customer_id} in {len(effective_folder_ids)} folders")

                folder_ids = effective_folder_ids

            if not folder_ids and not target_document_names:

                async def no_customer_docs_gen():
                    yield json.dumps({"type": "summary", "total_transactions": 0, "sources": [], "message": "No documents found for this customer."}) + "\n"
                return StreamingResponse(no_customer_docs_gen(), media_type="application/x-ndjson")

        count = vector_store.get_collection_count(company_id=request_body.company_id)

        if count == 0:
            async def empty_gen():
                yield json.dumps({"type": "error", "message": "No documents found"}) + "\n"
            return StreamingResponse(empty_gen(), media_type="application/x-ndjson")

        intent_data = await llm_service.classify_query_intent(request_body.question)
        intent = intent_data.get("intent", "EXTRACTION")
        explicit_limit = intent_data.get("explicit_limit")

        is_summary = (intent == "SUMMARY")

        is_exhaustive = (intent == "SUMMARY" or intent == "EXTRACTION")

        if explicit_limit:
            n_results = explicit_limit
            is_exhaustive = False
            print(f"[INFO] Intent: EXTRACTION with Limit={n_results}")
        else:
            n_results = 50
            print(f"[INFO] Intent: {intent} (Exhaustive={is_exhaustive})")

        if is_exhaustive:
            print(f"[INFO] Streaming exhaustive extraction for ALL chunks (is_summary={is_summary})")

            all_chunks = []
            all_metadatas = []

            offset = 0
            limit = 5000

            where_filter = vector_store._build_where_filter(
                 company_id=request_body.company_id,
                 document_names=target_document_names if target_document_names else None,
                 file_types=request_body.file_types,

                 folder_ids=folder_ids if not target_document_names else None,
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

            print(f"[INFO] Standard answer mode triggered (n_results={n_results})")
            results = vector_store.query(
                query_embedding,
                company_id=request_body.company_id,
                n_results=n_results,
                file_types=request_body.file_types,
                folder_ids=folder_ids if not target_document_names else None,
                document_names=target_document_names if target_document_names else None,
                uploaded_by=request_body.uploaded_by,
                tags=request_body.tags
            )

        documents = results.get('documents', [[]])[0]
        metadatas = results.get('metadatas', [[]])[0]

        if not documents:
            async def no_docs_gen():
                 yield json.dumps({"type": "summary", "total_transactions": 0, "sources": []}) + "\n"
            return StreamingResponse(no_docs_gen(), media_type="application/x-ndjson")

        source_documents = [meta.get('document_name', 'Unknown') for meta in metadatas]

        async def response_generator():
            try:

                if await request.is_disconnected():
                    print("[INFO] Client disconnected before streaming started, aborting")
                    return

                if is_summary:

                    CHUNK_CAP = 200
                    documents_local = documents[:CHUNK_CAP]
                    source_documents_local = source_documents[:CHUNK_CAP]
                    print(f"[INFO] Summary focus: Capping at first {len(documents_local)} chunks (original: {len(documents)})")

                    result = await llm_service.generate_summary_report(
                        question=effective_question,
                        context_chunks=documents_local,
                        source_documents=source_documents_local,
                    )

                    if await request.is_disconnected():
                        print("[INFO] Client disconnected during summary generation, stopping")
                        return

                    summary_payload = {
                        "type": "summary",
                        "intent": "SUMMARY",
                        "total_transactions": 0,
                        "total_credits": 0,
                        "total_debits": 0,
                        "sources": list(result.get("sources", [])),
                        "full_answer": result.get("full_answer", result.get("answer", "")),
                        "data": json.loads(result.get("answer", "{}"))
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
                        batch_size=3,
                        company_id=request_body.company_id
                    ):

                        if await request.is_disconnected():
                            print("[INFO] Client disconnected during exhaustive streaming, stopping processing")
                            return
                        yield chunk + "\n"
                else:

                    print(f"[INFO] Standard answer mode triggered (Streaming wrapper)")
                    result = await llm_service.generate_answer(
                        question=effective_question,
                        context_chunks=documents,
                        source_documents=source_documents,
                        company_id=request_body.company_id
                    )

                    if await request.is_disconnected():
                        print("[INFO] Client disconnected during standard answer generation, stopping")
                        return

                    total_txs = 0
                    try:

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
async def query_documents(
    request: QueryRequest, 
    user=Depends(require_role(["user", "manager", "admin", "superadmin"], required_permission="create_reports"))
):
    """
    Query the RAG system and generate an answer.
    """
    try:

        profile_res = supabase_admin.table("profiles").select("company_id").eq("id", user.id).single().execute()
        company_id = profile_res.data["company_id"]

        request.company_id = company_id

        print(f"[INFO] Received query request: question='{request.question}', company_id={request.company_id}")
        print(f"[INFO] Filters - file_types: {request.file_types}, folder_ids: {request.folder_ids}, uploaded_by: {request.uploaded_by}, tags: {request.tags}")

        print(f"[INFO] Step 1: Generating query embedding...")
        query_embedding = embedding_service.generate_embedding(request.question)

        folder_ids = request.folder_ids or []
        target_document_names = []

        effective_question = request.question
        if request.start_date and request.end_date:
            effective_question += f" from {request.start_date} to {request.end_date}"
            print(f"[INFO] Injected date range into question: {effective_question}")

        if request.customer_id:
            customer_folders = supabase_admin.table("folders") \
                .select("id") \
                .eq("customer_id", request.customer_id) \
                .is_("deleted_at", "null") \
                .execute()

            if customer_folders.data:
                customer_folder_ids = [f["id"] for f in customer_folders.data]

                if folder_ids:

                    effective_folder_ids = list(set(folder_ids) & set(customer_folder_ids))
                else:
                    effective_folder_ids = customer_folder_ids

                if effective_folder_ids:
                    customer_files = supabase_admin.table("files") \
                        .select("name") \
                        .in_("folder_id", effective_folder_ids) \
                        .is_("deleted_at", "null") \
                        .execute()

                    if customer_files.data:
                        target_document_names = list(set([f["name"] for f in customer_files.data]))
                        print(f"[INFO] Resolved {len(target_document_names)} documents for customer {request.customer_id} in {len(effective_folder_ids)} folders")

                folder_ids = effective_folder_ids

            if not folder_ids and not target_document_names:

                return QueryResponse(
                    answer="No documents found for this customer.",
                    sources=[]
                )

        count = vector_store.get_collection_count(company_id=request.company_id)
        if count == 0:
            return QueryResponse(
                answer="No documents found for your company. Please upload some documents first.",
                sources=[]
            )

        intent_data = await llm_service.classify_query_intent(request.question)
        intent = intent_data.get("intent", "EXTRACTION")
        explicit_limit = intent_data.get("explicit_limit")

        is_summary = (intent == "SUMMARY")
        is_exhaustive = (intent == "SUMMARY" or intent == "EXTRACTION")

        if explicit_limit:
            n_results = explicit_limit
            is_exhaustive = False
            print(f"[INFO] Intent: EXTRACTION with Limit={n_results}")
        else:
            n_results = 50
            print(f"[INFO] Intent: {intent} (Exhaustive={is_exhaustive})")

        if is_exhaustive:
            print(f"[INFO] Exhaustive extraction mode triggered")

            print(f"[INFO] Retrieving ALL chunks for exhaustive extraction (Paid Plan Enabled) - Pagination & Sort")

            all_chunks = []
            all_metadatas = []
            offset = 0
            limit = 5000

            where_filter = vector_store._build_where_filter(
                 company_id=request.company_id,
                 document_names=target_document_names if target_document_names else None,
                 file_types=request.file_types,

                 folder_ids=folder_ids if not target_document_names else None,
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

            print(f"[INFO] Standard answer mode triggered (n_results={n_results})")
            results = vector_store.query(
                query_embedding,
                company_id=request.company_id,
                n_results=n_results,
                file_types=request.file_types,
                folder_ids=folder_ids if not target_document_names else None,
                document_names=target_document_names if target_document_names else None,
                uploaded_by=request.uploaded_by,
                tags=request.tags
            )

        print(f"[INFO] DONE: Retrieved {len(results.get('documents', [[]])[0])} chunks from vector store.")

        documents = results.get('documents', [[]])[0]
        metadatas = results.get('metadatas', [[]])[0]

        if not documents:
            print(f"[WARNING] No documents found for companyId={request.company_id}. Returning 'Information not available' response.")
            return QueryResponse(
                answer="Information not available in uploaded records",
                sources=[]
            )

        source_documents = [meta.get('document_name', 'Unknown') for meta in metadatas]

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
