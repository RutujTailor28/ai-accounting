from fastapi import APIRouter, HTTPException, Depends
from typing import List, Dict, Any
from uuid import UUID
import json
from app.schemas.chat import ChatQueryRequest, ChatMessageResponse, ChatHistoryItem
from app.api.deps import get_current_user
from app.core.supabase import supabase
from app.services.embedding_service import EmbeddingService
from app.ai.rag.retriever import vector_store
from app.services.ai_service import LLMService

router = APIRouter()

# Initialize services
embedding_service = EmbeddingService()
llm_service = LLMService()


@router.post("/query", response_model=ChatMessageResponse)
async def chat_query(request: ChatQueryRequest, user=Depends(get_current_user)):
    """
    Chat with the AI using documents from a specific workspace.
    """
    try:
        # Step 0: Get user's company_id
        profile_res = supabase.table("profiles").select("company_id").eq("id", user.id).single().execute()
        company_id = profile_res.data["company_id"]

        # Step 1: Resolve Workspace and its documents
        ws_res = supabase.table("workspaces").select("*").eq("id", str(request.workspace_id)).is_("deleted_at", "null").single().execute()
        if not ws_res.data:
            raise HTTPException(status_code=404, detail="Workspace not found")
        
        ws_data = ws_res.data
        file_ids = ws_data.get("file_ids", [])
        folder_ids = ws_data.get("folder_ids", [])

        # Fetch all file names (ChromaDB filters by document_name)
        # 1. Direct files
        doc_names = []
        if file_ids:
            files_res = supabase.table("files") \
                .select("name") \
                .in_("id", file_ids) \
                .is_("deleted_at", "null") \
                .execute()
            doc_names.extend([f["name"] for f in files_res.data])
        
        # 2. Files inside folders
        if folder_ids:
            folder_files_res = supabase.table("files") \
                .select("name") \
                .in_("folder_id", folder_ids) \
                .is_("deleted_at", "null") \
                .execute()
            doc_names.extend([f["name"] for f in folder_files_res.data])
        
        doc_names = list(set(doc_names)) # Unique names

        if not doc_names:
            raise HTTPException(status_code=400, detail="This workspace has no documents associated with it.")

        # Step 2: RAG Flow
        # a. Embedding
        query_embedding = embedding_service.generate_embedding(request.question)
        
        # b. Workspace Retrieval
        results = vector_store.workspace_query(
            query_embedding=query_embedding,
            company_id=company_id,
            document_names=doc_names,
            n_results=15
        )
        
        documents = results.get('documents', [[]])[0]
        metadatas = results.get('metadatas', [[]])[0]
        source_documents = [meta.get('document_name', 'Unknown') for meta in metadatas]

        # c. LLM Generation
        # Different prompt/logic can be added here if needed, 
        # for now using the standard generate_answer which returns stringified JSON
        llm_result = await llm_service.generate_answer(
            question=request.question,
            context_chunks=documents,
            source_documents=source_documents
        )
        
        parsed_answer = json.loads(llm_result["answer"])

        # Step 3: Save History
        # Check if session is new to set the title
        history_check = supabase.table("chat_messages").select("id").eq("session_id", str(request.session_id)).limit(1).execute()
        session_title = None
        if not history_check.data:
            session_title = request.question[:100]

        # a. Save User Message
        user_msg = {
            "session_id": str(request.session_id),
            "workspace_id": str(request.workspace_id),
            "role": "user",
            "content": request.question,
            "company_id": company_id,
            "created_by": user.id
        }
        if session_title:
            user_msg["session_title"] = session_title
        
        supabase.table("chat_messages").insert(user_msg).execute()

        # b. Save Assistant Message
        assistant_msg = {
            "session_id": str(request.session_id),
            "workspace_id": str(request.workspace_id),
            "role": "assistant",
            "content": parsed_answer.get("message", "I couldn't generate a response."),
            "data": parsed_answer.get("data", []),
            "sources": llm_result["sources"],
            "company_id": company_id,
            "created_by": user.id
        }
        if session_title:
            assistant_msg["session_title"] = session_title

        save_res = supabase.table("chat_messages").insert(assistant_msg).execute()
        
        return save_res.data[0]

    except Exception as e:
        print(f"[ERROR] Chat Query Error: {str(e)}")
        raise HTTPException(status_code=500, detail=str(e))


@router.get("/history", response_model=List[ChatHistoryItem])
async def list_chat_history(workspace_id: UUID, user=Depends(get_current_user)):
    """List all unique chat sessions for a workspace."""
    try:
        # Supabase doesn't have a clean DISTINCT ON for multiple columns easily via client
        # We'll fetch and group manually or use a trick
        result = supabase.table("chat_messages") \
            .select("session_id, session_title, created_at, workspace_id") \
            .eq("workspace_id", str(workspace_id)) \
            .is_("deleted_at", "null") \
            .order("created_at", desc=True) \
            .execute()
        
        session_map = {}
        for msg in result.data:
            sid = msg["session_id"]
            if sid not in session_map:
                session_map[sid] = {
                    "session_id": sid,
                    "session_title": msg.get("session_title"),
                    "last_message_at": msg["created_at"],
                    "workspace_id": msg["workspace_id"]
                }
            # If we didn't have a title for this session yet, but this message has one, use it
            elif not session_map[sid]["session_title"] and msg.get("session_title"):
                session_map[sid]["session_title"] = msg.get("session_title")
        
        # Final formatting and fallback
        history = []
        for sid in session_map:
            item = session_map[sid]
            if not item["session_title"]:
                item["session_title"] = "Untitled Chat"
            history.append(item)
        
        return history
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@router.get("/history/{session_id}", response_model=List[ChatMessageResponse])
async def get_chat_thread(session_id: UUID, user=Depends(get_current_user)):
    """Fetch the full thread of messages for a session."""
    try:
        result = supabase.table("chat_messages") \
            .select("*") \
            .eq("session_id", str(session_id)) \
            .is_("deleted_at", "null") \
            .order("created_at", desc=False) \
            .execute()
        
        return result.data
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@router.delete("/history/{session_id}")
async def delete_chat_session(session_id: UUID, user=Depends(get_current_user)):
    """Soft delete an entire chat session."""
    try:
        from datetime import datetime, timezone
        now = datetime.now(timezone.utc).isoformat()
        
        # Soft delete all messages in the session
        result = supabase.table("chat_messages") \
            .update({"deleted_at": now}) \
            .eq("session_id", str(session_id)) \
            .execute()
        
        return {"message": "Chat session soft-deleted successfully"}
    except Exception as e:
        print(f"[ERROR] Error deleting chat session: {str(e)}")
        raise HTTPException(status_code=500, detail=str(e))
