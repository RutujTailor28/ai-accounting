from fastapi import APIRouter, HTTPException, Depends
from fastapi.responses import StreamingResponse
from typing import List, Dict, Any
from uuid import UUID
import json
import re
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
async def accounting_query(request: ChatQueryRequest, user=Depends(get_current_user)):
    """
    Generate accounting reports (entries, balance sheets) from workspace documents with streaming.
    """
    try:
        # Step 0: Get company_id
        profile_res = supabase.table("profiles").select("company_id").eq("id", user.id).single().execute()
        company_id = profile_res.data["company_id"]

        # Step 1: Resolve Workspace and its documents (High Context)
        ws_res = supabase.table("workspaces").select("*").eq("id", str(request.workspace_id)).single().execute()
        if not ws_res.data:
            raise HTTPException(status_code=404, detail="Workspace not found")
            
        file_ids = ws_res.data.get("file_ids", [])
        folder_ids = ws_res.data.get("folder_ids", [])

        # Fetch all filenames
        doc_names = []
        if file_ids:
            files_res = supabase.table("files").select("name").in_("id", file_ids).execute()
            doc_names.extend([f["name"] for f in files_res.data])
        if folder_ids:
            folder_files_res = supabase.table("files").select("name").in_("folder_id", folder_ids).execute()
            doc_names.extend([f["name"] for f in folder_files_res.data])
        
        doc_names = list(set(doc_names))

        if not doc_names:
            raise HTTPException(status_code=400, detail="Workspace has no documents.")

        # Step 2: Retrieve ALL Chunks (High Context for Reports)
        # For accounting reports, we often need the full story, so we pull more results than usual
        query_embedding = embedding_service.generate_embedding(request.question)
        
        # Prepare filters
        query_filters = {
            "file_types": request.file_types,
            "folder_ids": request.folder_ids,
            "uploaded_by": request.uploaded_by,
            "start_date": request.start_date,
            "end_date": request.end_date,
            "tags": request.tags
        }

        # For accounting, we need to get ALL chunks from the documents, not just semantically similar ones
        # First get count to retrieve all available chunks
        
        count_result = vector_store.collection.get(
            where=vector_store._build_where_filter(
                company_id=company_id,
                document_names=doc_names,
                **query_filters
            ),
            include=[],
            limit=10000  # Explicitly set high limit to get actual count
        )
        total_available = len(count_result['ids'])
        print(f"[INFO] Accounting endpoint: Total available chunks for documents: {total_available}")
        
        # Retrieve ALL chunks for these documents (not just semantically similar ones)
        # This ensures we get transaction data, not just headers/footers
        if total_available > 0:
            # Get all chunks without semantic filtering - just filter by document
            all_results = vector_store.collection.get(
                where=vector_store._build_where_filter(
                    company_id=company_id,
                    document_names=doc_names,
                    **query_filters
                ),
                include=['documents', 'metadatas'],
                limit=10000  # Explicitly set high limit for accounting synthesis
            )
            chunks = all_results.get('documents', [])
            metadatas = all_results.get('metadatas', [])
            print(f"[INFO] Accounting endpoint: Retrieved {len(chunks)} chunks (ALL chunks from documents) for accounting synthesis")
            
            # Filter out header/footer chunks that don't contain transaction-like patterns
            transaction_keywords = ['UPI', 'NEFT', 'IMPS', 'RTGS', 'PAYMENT', 'RECEIVED', 'TRANSFER', 
                                  'DEBIT', 'CREDIT', 'WITHDRAWAL', 'DEPOSIT', 'DATE', '/', 'Rs.', 'AMOUNT',
                                  '22/', '23/', '24/', '25/', '01/', '02/', '03/', '04/', '05/',
                                  '06/', '07/', '08/', '09/', '10/', '11/', '12/']
            
            filtered_chunks = []
            filtered_metadatas = []
            
            for chunk, metadata in zip(chunks, metadatas):
                chunk_upper = chunk.upper()
                # Check if chunk contains transaction-like patterns
                has_transaction_pattern = any(keyword in chunk_upper for keyword in transaction_keywords)
                
                # Check if chunk has date-like patterns (DD/MM/YY or DD/MM/YYYY)
                has_date_pattern = bool(re.search(r'\d{1,2}[/-]\d{1,2}[/-]\d{2,4}', chunk))
                
                # Check if chunk has amount patterns (numbers with commas or decimals)
                has_amount_pattern = bool(re.search(r'\d+[,\.]\d+', chunk))
                
                # Exclude chunks that are clearly headers/footers
                is_header_footer = any(keyword in chunk_upper for keyword in [
                    'CLOSING BALANCE INCLUDES', 'STATEMENT WILL BE CONSIDERED CORRECT',
                    'ERROR IS REPORTED WITHIN', 'GSTIN NUMBER', 'GST NUMBER',
                    'REGISTERED OFFICE', 'BRANCH CODE', 'IFSC', 'MICR', 'ACCOUNT TYPE',
                    'STATEMENT OF ACCOUNT FROM', 'ACCOUNT BRANCH', 'PHONE NO', 'EMAIL',
                    'CUST ID', 'ACCOUNT NO', 'A/C OPEN DATE', 'ACCOUNT STATUS'
                ])
                
                # Include chunk if:
                # 1. It has transaction patterns, OR
                # 2. It has date AND amount patterns (likely a transaction), OR  
                # 3. It's not a header/footer (keep if we're not sure)
                if (has_transaction_pattern or 
                    (has_date_pattern and has_amount_pattern) or 
                    (not is_header_footer and len(chunk) > 50)):  # Exclude very short chunks too
                    filtered_chunks.append(chunk)
                    filtered_metadatas.append(metadata)
            
            # If we filtered too much, use original chunks (but log warning)
            if len(filtered_chunks) < 5 and len(chunks) > 10:
                print(f"[WARNING] Filtering removed too many chunks ({len(chunks)} -> {len(filtered_chunks)}). Using all chunks.")
                chunks = chunks
            elif filtered_chunks:
                original_count = len(chunks)
                chunks = filtered_chunks
                removed_count = original_count - len(chunks)
                print(f"[INFO] Accounting endpoint: After filtering, using {len(chunks)} chunks with transaction data (removed {removed_count} header/footer chunks)")
            else:
                print(f"[WARNING] All chunks were filtered out. Using original chunks.")
        else:
            # Fallback to semantic search if no chunks found
            results = vector_store.workspace_query(
                query_embedding=query_embedding,
                company_id=company_id,
                document_names=doc_names,
                n_results=100,  # Much higher for accounting
                **query_filters
            )
            chunks = results.get('documents', [[]])[0]
            print(f"[INFO] Accounting endpoint: Retrieved {len(chunks)} chunks via semantic search (fallback)")
        
        if not chunks:
            raise HTTPException(
                status_code=400, 
                detail="No document content found. Please ensure documents are uploaded and processed correctly."
            )
        
        # Log chunk preview for debugging
        if chunks:
            total_length = sum(len(chunk) for chunk in chunks)
            print(f"[INFO] Accounting endpoint: Total context length: {total_length} characters")
            # Show first chunk that looks like it has transactions
            for i, chunk in enumerate(chunks[:5]):
                if any(kw in chunk.upper() for kw in ['UPI', 'NEFT', 'DEBIT', 'CREDIT', 'DATE']):
                    print(f"[DEBUG] Accounting endpoint: Chunk {i} preview (first 300 chars): {chunk[:300]}...")
                    break
            else:
                print(f"[DEBUG] Accounting endpoint: First chunk preview (200 chars): {chunks[0][:200]}...")

        # Step 3: Define Streaming Generator
        async def stream_generator():
            full_response = ""
            # Prepare streaming from AccountingService
            async for token in accounting_service.stream_accounting_synthesis(request.question, chunks, metadatas):
                if isinstance(token, dict):
                     # Status update (already a dict, just wrap in data)
                     yield f"data: {json.dumps(token)}\n\n"
                else:
                    full_response += token
                    # SSE Format: data: <payload>\n\n
                    yield f"data: {json.dumps({'token': token})}\n\n"
            
            # Step 4: After stream finishes, save to history
            try:
                # Check if session is new to set the title
                history_check = supabase.table("chat_messages").select("id").eq("session_id", str(request.session_id)).limit(1).execute()
                session_title = None
                if not history_check.data:
                    session_title = request.question[:100]

                # Save User Question
                user_msg_data = {
                    "session_id": str(request.session_id),
                    "workspace_id": str(request.workspace_id),
                    "role": "user",
                    "content": request.question,
                    "company_id": company_id,
                    "created_by": user.id
                }
                if session_title:
                    user_msg_data["session_title"] = session_title
                
                supabase_admin.table("chat_messages").insert(user_msg_data).execute()

                # Save AI Synthesis
                # Phase 2: Extract ALL structured tables from combined generator output
                structured_tables = await accounting_service.get_all_structured_tables(full_response)
                
                msg_data = {
                    "session_id": str(request.session_id),
                    "workspace_id": str(request.workspace_id),
                    "role": "assistant",
                    "content": full_response,
                    "company_id": company_id,
                    "created_by": user.id,
                    "data": {"tables": structured_tables}
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
async def update_accounting_message(message_id: UUID, content: str, data: Dict[str, Any] = None, user=Depends(get_current_user)):
    """
    Update a chat message's content and structured data.
    Only the creator can edit their messages.
    """
    try:
        # Verify ownership
        msg_check = supabase.table("chat_messages").select("created_by").eq("id", str(message_id)).single().execute()
        if not msg_check.data or msg_check.data["created_by"] != user.id:
            raise HTTPException(status_code=403, detail="Forbidden")
            
        update_fields = {"content": content}
        if data is not None:
            update_fields["data"] = data
            
        res = supabase.table("chat_messages").update(update_fields).eq("id", str(message_id)).execute()
        return res.data[0]
    except Exception as e:
        print(f"[ERROR] Update Message Error: {str(e)}")
        raise HTTPException(status_code=500, detail=str(e))
