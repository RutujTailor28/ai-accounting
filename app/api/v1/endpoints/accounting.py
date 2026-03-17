from fastapi import APIRouter, HTTPException, Depends, Request
from fastapi.responses import StreamingResponse
from typing import List, Dict, Any
from uuid import UUID
import json
import re
import uuid
import asyncio
from app.schemas.chat import ChatQueryRequest, ChatMessageResponse
from app.api.deps import get_current_user
from app.core.supabase import supabase, supabase_admin
from app.services.deps import embedding_service
from app.ai.rag.retriever import vector_store
from app.services.accounting_service import AccountingService

router = APIRouter()

# Initialize services
accounting_service = AccountingService()

@router.post("/query")
async def accounting_query(request_body: ChatQueryRequest, request: Request, user=Depends(get_current_user)):
    """
    Generate accounting reports (entries, balance sheets) from workspace documents with streaming.
    """
    try:
        # Step 0: Input Guard (Agent 0)
        # Prevent accidental or meaningless inputs (like "nw", ".", "hi") from triggering full processing.
        q_low = str(request_body.question).strip().lower()
        # Regex to catch gibberish or very short non-accounting tokens
        # Keywords that indicate a valid accounting intent
        valid_keywords = [
            "p&l", "p & l", "profit", "loss", "statement", "ledger", "journal", "audit", 
            "tally", "sheet", "categorize", "category", "account", "expense", "amount", 
            "gst", "interest", "analyze", "analysis", "find", "show", "get", "instead", 
            "change", "move", "shift", "put", "update", "edit", "modify", "balance", 
            "total", "summary", "report", "extract", "transaction", "analyze", "list"
        ]
        is_meaningless = len(q_low) < 3 or (not any(kw in q_low for kw in valid_keywords))
        
        # Exceptions for common shorthand
        if q_low in ["p&l", "bs", "tb", "p & l"]: is_meaningless = False
        
        if is_meaningless:
            async def meaningless_stream():
                msg = "I didn't quite catch that. Could you please ask a specific question about your documents? For example: 'Analyze my P&L' or 'Show my journal entries'."
                yield f"data: {json.dumps({'token': msg})}\n\n"
            return StreamingResponse(meaningless_stream(), media_type="text/event-stream")

        # Step 0.1: Get company_id
        profile_res = supabase.table("profiles").select("company_id").eq("id", user.id).single().execute()
        company_id = profile_res.data["company_id"]

        # Step 1: Pre-resolve session history for efficiency (Iteration Check)
        previous_context = None
        cached_all_tables = []
        cached_transactions = []
        aggregated_instructions = ""

        try:
            # Fetch full session history to aggregate refinements
            history_res = supabase.table("chat_messages") \
                .select("role, content, data") \
                .eq("session_id", str(request_body.session_id)) \
                .order("created_at", desc=True) \
                .execute()
            
            if history_res.data:
                # 1. Get the last assistant message for context (tables/txns)
                last_assistant = next((m for m in history_res.data if m["role"] == "assistant"), None)
                if last_assistant:
                    content = last_assistant.get("content", "")
                    data_obj = last_assistant.get("data", {}) or {}
                    data_tables = data_obj.get("tables", [])
                    cached_all_tables = data_obj.get("all_tables", data_tables)
                    cached_transactions = data_obj.get("transactions", [])
                    
                    # Reconstruct full context with ALL cached tables
                    full_previous_context = ""
                    for t in cached_all_tables:
                        if t.get("type") in ["profit_loss", "balance_sheet", "p_and_l"]:
                            title = t.get("title", "")
                            headers = t.get("headers", [])
                            rows = t.get("rows", [])
                            full_previous_context += f"### {title}\n"
                            full_previous_context += f"| {' | '.join(headers)} |\n"
                            full_previous_context += f"| {' | '.join(['---'] * len(headers))} |\n"
                            for row in rows:
                                clean_row = [str(c) if c is not None else "" for c in row]
                                full_previous_context += f"| {' | '.join(clean_row)} |\n"
                            full_previous_context += "\n"
                    
                    full_previous_context += content
                    if "Balance Sheet" in full_previous_context or "Profit & Loss" in full_previous_context or "|---|" in full_previous_context:
                        previous_context = full_previous_context

                # 2. Aggregate ALL user instructions for Cumulative Refinement
                user_msgs = [m["content"] for m in reversed(history_res.data) if m["role"] == "user"]
                if len(user_msgs) > 0:
                    instructions = user_msgs # Use all history for context
                    aggregated_instructions = " + ".join(instructions)
        except Exception as e:
            print(f"[WARNING] Failed to retrieve session history early: {e}")

        # Step 2: Resolve context and documents to query
        doc_names = []
        chunks = []
        metadatas = []
        
        effective_workspace_id = str(request_body.workspace_id) if request_body.workspace_id else None
        effective_customer_id = str(request_body.customer_id) if request_body.customer_id else None

        if effective_customer_id:
            # Get all folders for this customer
            folders_res = supabase.table("folders") \
                .select("id") \
                .eq("customer_id", effective_customer_id) \
                .is_("deleted_at", "null") \
                .execute()
            
            customer_folder_ids = [f["id"] for f in folders_res.data]

            if not customer_folder_ids:
                raise HTTPException(status_code=400, detail="No folders found for this customer.")

            query = supabase.table("files") \
                .select("name") \
                .in_("folder_id", customer_folder_ids) \
                .is_("deleted_at", "null")
            
            if request_body.file_ids or request_body.folder_ids:
                or_conditions = []
                if request_body.file_ids:
                    uuids = []
                    names = []
                    for fid in request_body.file_ids:
                        fid_str = str(fid)
                        try:
                            uuid.UUID(fid_str)
                            uuids.append(f'"{fid_str}"')
                        except ValueError:
                            names.append(f'"{fid_str}"')
                    
                    if uuids:
                        or_conditions.append(f"id.in.({','.join(uuids)})")
                    if names:
                        or_conditions.append(f"name.in.({','.join(names)})")
                        
                if request_body.folder_ids:
                    quoted_folder_ids = [f'"{str(foid)}"' for foid in request_body.folder_ids]
                    or_conditions.append(f"folder_id.in.({','.join(quoted_folder_ids)})")
                
                if or_conditions:
                    query = query.or_(",".join(or_conditions))
            
            files_res = query.execute()
            doc_names.extend([f["name"] for f in files_res.data])

            if not doc_names:
                raise HTTPException(status_code=400, detail="No documents found for the selected customer context.")
        else:
            if not effective_workspace_id:
                raise HTTPException(status_code=400, detail="Either workspace_id or customer_id must be provided")

            ws_res = supabase.table("workspaces").select("*").eq("id", effective_workspace_id).single().execute()
            if not ws_res.data:
                raise HTTPException(status_code=404, detail="Workspace not found")
                
            file_ids = ws_res.data.get("file_ids", [])
            folder_ids = ws_res.data.get("folder_ids", [])

            if file_ids:
                files_res = supabase.table("files").select("name").in_("id", file_ids).execute()
                doc_names.extend([f["name"] for f in files_res.data])
            if folder_ids:
                folder_files_res = supabase.table("files").select("name").in_("folder_id", folder_ids).execute()
                doc_names.extend([f["name"] for f in folder_files_res.data])
            
            doc_names = list(set(doc_names))

            if not doc_names:
                raise HTTPException(status_code=400, detail="Workspace has no documents.")

        # Step 3: Retrieve ALL Chunks
        query_embedding = embedding_service.generate_embedding(request_body.question)
        query_filters = {
            "file_types": request_body.file_types,
            "folder_ids": request_body.folder_ids,
            "uploaded_by": request_body.uploaded_by,
            "start_date": request_body.start_date,
            "end_date": request_body.end_date,
            "tags": request_body.tags
        }

        count_result = vector_store.collection.get(
            where=vector_store._build_where_filter(
                company_id=company_id,
                document_names=doc_names,
                **query_filters
            ),
            include=[],
            limit=10000
        )
        total_available = len(count_result['ids'])
        
        if total_available > 0:
            all_results = vector_store.collection.get(
                where=vector_store._build_where_filter(
                    company_id=company_id,
                    document_names=doc_names,
                    **query_filters
                ),
                include=['documents', 'metadatas'],
                limit=10000
            )
            raw_chunks = all_results.get('documents', [])
            raw_metadatas = all_results.get('metadatas', [])
            
            combined = []
            for i in range(len(raw_chunks)):
                meta = raw_metadatas[i] or {}
                sort_key = (meta.get('document_name', ''), int(meta.get('chunk_index', 0)), raw_chunks[i][:20])
                combined.append((sort_key, raw_chunks[i], meta))
            
            combined.sort(key=lambda x: x[0])
            chunks = [x[1] for x in combined]
            metadatas = [x[2] for x in combined]

            transaction_keywords = ['CASH', 'WDL', 'ATM', 'WITHDRAWAL', 'SELF', 'UPI', 'NEFT', 'IMPS', 'RTGS', 
                                'PAYMENT', 'RECEIVED', 'TRANSFER', 'DEBIT', 'CREDIT', 'CR', 'DR',
                                'DEPOSIT', 'DATE', '/', 'Rs.', 'AMOUNT', 'CHQ', 'CHEQUE']
            
            filtered_chunks = []
            filtered_metadatas = []
            for chunk, metadata in zip(chunks, metadatas):
                chunk_upper = chunk.upper()
                has_transaction_pattern = any(keyword in chunk_upper for keyword in transaction_keywords)
                has_date_pattern = bool(re.search(r'\d{1,2}[/-]\d{1,2}[/-]\d{2,4}', chunk))
                has_amount_pattern = bool(re.search(r'\d+[,\.]\d+', chunk))
                is_header_footer = any(keyword in chunk_upper for keyword in ['CLOSING BALANCE INCLUDES', 'GSTIN NUMBER', 'IFSC'])
                
                if has_transaction_pattern or (has_date_pattern and has_amount_pattern) or (not is_header_footer and len(chunk) > 50):
                    filtered_chunks.append(chunk)
                    filtered_metadatas.append(metadata)
            
            if filtered_chunks:
                chunks = filtered_chunks
                metadatas = filtered_metadatas
        else:
            # Fallback
            results = vector_store.workspace_query(
                query_embedding=query_embedding,
                company_id=company_id,
                document_names=doc_names,
                n_results=100,
                **query_filters
            )
            chunks = results.get('documents', [[]])[0]
            metadatas = results.get('metadatas', [[]])[0]
        
        if not chunks and not cached_transactions:
            raise HTTPException(
                status_code=400, 
                detail="No document content found and no cached data available."
            )

        # Step 4: Intent and Refinement Logic
        q_lower = request_body.question.lower()
        q_words = re.sub(r'[^a-z0-9]', ' ', q_lower).split()
        
        show_tables_requested = []
        if any(kw in q_lower for kw in ['balanc', 'asset', 'liabilit', 'equity']) or 'bs' in q_words:
            show_tables_requested.append("balance_sheet")
        if any(kw in q_lower for kw in ['profit', 'loss', 'p&l', 'pnl', 'income', 'expense', 'revenue']):
            show_tables_requested.append("profit_loss")
        if "journal" in q_lower or "entries" in q_lower or "ledger" in q_lower: 
            show_tables_requested.append("journal")

        # Step 3: Define Streaming Generator
        async def stream_generator():
            # Check if client disconnected before starting
            if await request.is_disconnected():
                print("[INFO] Client disconnected before accounting stream started, aborting")
                return
            
            full_response = ""
            final_structured_tables = []
            final_all_tables = []
            final_transactions = []
            
            # Check if we can fulfill this from cache instantly
            tables_to_show = [t for t in cached_all_tables if t.get("type") in show_tables_requested]
            if cached_all_tables and tables_to_show and not any(kw in q_lower for kw in ["change", "update", "modify", "edit", "fix", "instead", "move", "remove", "add", "to"]):
                print(f"[INFO] Instant Cache Hit! Serving {len(tables_to_show)} tables from session cache.")
                full_response = "Here is the requested report based on the already analyzed statement.\n\n"
                yield f"data: {json.dumps({'token': full_response})}\n\n"
                
                for t in tables_to_show:
                    title = t.get("title", "")
                    headers = t.get("headers", [])
                    rows = t.get("rows", [])
                    table_md = f"### {title}\n"
                    table_md += f"| {' | '.join(headers)} |\n"
                    table_md += f"| {' | '.join(['---'] * len(headers))} |\n"
                    for row in rows:
                        clean_row = [str(c) if c is not None else "" for c in row]
                        table_md += f"| {' | '.join(clean_row)} |\n"
                    table_md += "\n"
                    
                    full_response += table_md
                    yield f"data: {json.dumps({'token': table_md})}\n\n"
                    await asyncio.sleep(0.1)
                
                final_structured_tables = tables_to_show
                final_all_tables = cached_all_tables
            else:
                # Prepare streaming from AccountingService
                # Pass previous_context to enable refinement mode
                # Use aggregated_instructions if present to ensure cumulative refinement
                async for token in accounting_service.stream_accounting_synthesis(
                    request_body.question, 
                    chunks, 
                    metadatas, 
                    company_id=company_id,
                    customer_id=effective_customer_id,
                    previous_context=previous_context,
                    input_transactions=cached_transactions if cached_transactions else None,
                    feedback_history=aggregated_instructions if aggregated_instructions else None
                ):
                    # Check if client disconnected before yielding each token
                    if await request.is_disconnected():
                        print("[INFO] Client disconnected during accounting streaming, stopping processing")
                        return
                    
                    if isinstance(token, dict):
                         if "structured_tables" in token:
                             final_structured_tables = token["structured_tables"]
                             final_all_tables = token.get("all_tables", final_structured_tables)
                         elif "all_transactions" in token:
                             # Capture transactions for persistence
                             final_transactions = token["all_transactions"]
                         else:
                             # Status update (already a dict, just wrap in data)
                             yield f"data: {json.dumps(token)}\n\n"
                    else:
                        full_response += token
                        # SSE Format: data: <payload>\n\n
                        yield f"data: {json.dumps({'token': token})}\n\n"
            
            # Check disconnection before saving history
            if await request.is_disconnected():
                print("[INFO] Client disconnected before saving history, skipping save")
                return
            
            # Step 4: After stream finishes, save to history
            try:
                # Check if session is new to set the title
                history_check = supabase.table("chat_messages").select("id").eq("session_id", str(request_body.session_id)).limit(1).execute()
                session_title = None
                if not history_check.data:
                    session_title = request_body.question[:100]

                # Save User Question
                user_msg_data = {
                    "session_id": str(request_body.session_id),
                    "workspace_id": effective_workspace_id,
                    "customer_id": effective_customer_id,
                    "role": "user",
                    "content": request_body.question,
                    "company_id": company_id,
                    "created_by": user.id,
                    "file_names": doc_names
                }
                if session_title:
                    user_msg_data["session_title"] = session_title
                
                supabase_admin.table("chat_messages").insert(user_msg_data).execute()

                # Save AI Synthesis
                if not final_structured_tables:
                    final_structured_tables = await accounting_service._parse_markdown_tables(full_response)
                    
                    # Merge with cached_all_tables so we don't lose the un-edited tables
                    merged_all = list(cached_all_tables) if cached_all_tables else []
                    
                    for refined_t in final_structured_tables:
                        matched = False
                        for i, old_t in enumerate(merged_all):
                            if old_t.get('type') == refined_t.get('type'):
                                merged_all[i] = refined_t
                                matched = True
                                break
                        if not matched:
                            merged_all.append(refined_t)
                            
                    final_all_tables = merged_all
                
                structured_tables = final_structured_tables
                
                msg_data = {
                    "session_id": str(request_body.session_id),
                    "workspace_id": effective_workspace_id,
                    "customer_id": effective_customer_id,
                    "role": "assistant",
                    "content": full_response,
                    "company_id": company_id,
                    "created_by": user.id,
                    "data": {
                        "tables": structured_tables, 
                        "all_tables": final_all_tables,
                        "transactions": final_transactions or cached_transactions
                    },
                    "file_names": doc_names
                }
                if session_title:
                    msg_data["session_title"] = session_title
                
                saved_msg = supabase_admin.table("chat_messages").insert(msg_data).execute()
                final_msg_id = saved_msg.data[0]['id'] if saved_msg.data else None
                
                # Signal end of stream with actual message ID and structured data for frontend
                if final_msg_id:
                    yield f"data: {json.dumps({'message_id': final_msg_id, 'data': {'tables': structured_tables}})}\n\n"
                
                yield "data: [DONE]\n\n"
            except Exception as e:
                print(f"[ERROR] Failed to save synthesis history: {str(e)}")
                yield f"data: {json.dumps({'error': 'History save failed'})}\n\n"
                yield "data: [DONE]\n\n"

        return StreamingResponse(stream_generator(), media_type="text/event-stream")

    except Exception as e:
        print(f"[ERROR] Accounting Query Error: {str(e)}")
        raise HTTPException(status_code=500, detail=str(e))

@router.patch("/messages/{message_id}")
async def update_accounting_message(message_id: UUID, content: str = None, data: Dict[str, Any] = None, user=Depends(get_current_user)):
    """
    Update a chat message's content and structured data.
    Only the creator can edit their messages.
    """
    try:
        # Verify ownership
        msg_check = supabase.table("chat_messages").select("created_by").eq("id", str(message_id)).single().execute()
        if not msg_check.data or msg_check.data["created_by"] != user.id:
            raise HTTPException(status_code=403, detail="Forbidden")
            
        update_fields = {}
        if content is not None:
            update_fields["content"] = content
        if data is not None:
            update_fields["data"] = data
            
        if not update_fields:
            return {"message": "No fields to update"}
            
        res = supabase.table("chat_messages").update(update_fields).eq("id", str(message_id)).execute()
        return res.data[0]
    except Exception as e:
        print(f"[ERROR] Update Message Error: {str(e)}")
        raise HTTPException(status_code=500, detail=str(e))
