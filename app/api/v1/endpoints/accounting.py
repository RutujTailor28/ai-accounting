from fastapi import APIRouter, HTTPException, Depends
from fastapi.responses import StreamingResponse
from typing import List
from uuid import UUID
import json
from app.schemas.chat import ChatQueryRequest, ChatMessageResponse
from app.api.deps import get_current_user
from app.core.supabase import supabase
from app.services.embedding_service import EmbeddingService
from app.ai.rag.retriever import vector_store
from app.services.accounting_service import AccountingService

router = APIRouter()

# Initialize services
embedding_service = EmbeddingService()
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
        results = vector_store.workspace_query(
            query_embedding=query_embedding,
            company_id=company_id,
            document_names=doc_names,
            n_results=30 # Higher context for synthesis
        )
        
        chunks = results.get('documents', [[]])[0]

        # Step 3: Define Streaming Generator
        async def stream_generator():
            full_response = ""
            # Prepare streaming from AccountingService
            async for token in accounting_service.stream_accounting_synthesis(request.question, chunks):
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
                
                supabase.table("chat_messages").insert(user_msg_data).execute()

                # Save AI Synthesis
                msg_data = {
                    "session_id": str(request.session_id),
                    "workspace_id": str(request.workspace_id),
                    "role": "assistant",
                    "content": full_response,
                    "company_id": company_id,
                    "created_by": user.id
                }
                if session_title:
                    msg_data["session_title"] = session_title
                
                supabase.table("chat_messages").insert(msg_data).execute()
                
                # Signal end of stream
                yield "data: [DONE]\n\n"
            except Exception as e:
                print(f"[ERROR] Failed to save synthesis history: {str(e)}")
                yield f"data: {json.dumps({'error': 'History save failed'})}\n\n"
                yield "data: [DONE]\n\n"

        return StreamingResponse(stream_generator(), media_type="text/event-stream")

    except Exception as e:
        print(f"[ERROR] Accounting Query Error: {str(e)}")
        raise HTTPException(status_code=500, detail=str(e))
