from fastapi import APIRouter, HTTPException, Depends
from typing import List, Dict, Any, Optional
from uuid import UUID
import json
import uuid
from app.schemas.chat import ChatQueryRequest, ChatMessageResponse, ChatHistoryItem, ChatFeedbackCreate, ChatSaveRequest, ChatUpdateTitleRequest, ChatMessageUpdateRequest
from app.api.deps import get_current_user
from app.core.supabase import supabase, supabase_admin
from app.services.embedding_service import EmbeddingService
from app.ai.rag.retriever import vector_store
from app.services.ai_service import LLMService

router = APIRouter()

embedding_service = EmbeddingService()
llm_service = LLMService()

@router.post("/query", response_model=ChatMessageResponse)
async def chat_query(request: ChatQueryRequest, user=Depends(get_current_user)):
    """
    Chat with the AI using documents from a specific workspace.
    """
    try:

        profile_res = supabase.table("profiles").select("company_id").eq("id", user.id).single().execute()
        company_id = profile_res.data["company_id"]

        doc_names = []
        if request.customer_id:

            folders_res = supabase.table("folders") \
                .select("id") \
                .eq("customer_id", str(request.customer_id)) \
                .is_("deleted_at", "null") \
                .execute()

            customer_folder_ids = [f["id"] for f in folders_res.data]

            if not customer_folder_ids:
                raise HTTPException(status_code=400, detail="No folders found for this customer.")

            query = supabase.table("files") \
                .select("name") \
                .in_("folder_id", customer_folder_ids) \
                .is_("deleted_at", "null")

            if request.file_ids or request.folder_ids:
                or_conditions = []
                if request.file_ids:
                    uuids = []
                    names = []
                    for fid in request.file_ids:
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

                if request.folder_ids:
                    quoted_folder_ids = [f'"{str(foid)}"' for foid in request.folder_ids]
                    or_conditions.append(f"folder_id.in.({','.join(quoted_folder_ids)})")

                if or_conditions:
                    query = query.or_(",".join(or_conditions))

            files_res = query.execute()

            doc_names.extend([f["name"] for f in files_res.data])
            if not doc_names:
                raise HTTPException(status_code=400, detail="No documents found for the selected context.")

            effective_workspace_id = None
        else:

            if not request.workspace_id:
                raise HTTPException(status_code=400, detail="Either workspace_id or customer_id must be provided")

            ws_res = supabase.table("workspaces").select("*").eq("id", str(request.workspace_id)).is_("deleted_at", "null").single().execute()
            if not ws_res.data:
                raise HTTPException(status_code=404, detail="Workspace not found")

            ws_data = ws_res.data
            file_ids = ws_data.get("file_ids", [])
            folder_ids = ws_data.get("folder_ids", [])

            if file_ids:
                files_res = supabase.table("files") \
                    .select("name") \
                    .in_("id", file_ids) \
                    .is_("deleted_at", "null") \
                    .execute()
                doc_names.extend([f["name"] for f in files_res.data])

            if folder_ids:
                folder_files_res = supabase.table("files") \
                    .select("name") \
                    .in_("folder_id", folder_ids) \
                    .is_("deleted_at", "null") \
                    .execute()
                doc_names.extend([f["name"] for f in folder_files_res.data])

            if not doc_names:
                raise HTTPException(status_code=400, detail="This workspace has no documents associated with it.")

            effective_workspace_id = str(request.workspace_id)

        doc_names = list(set(doc_names))

        query_embedding = embedding_service.generate_embedding(request.question)

        results = vector_store.workspace_query(
            query_embedding=query_embedding,
            company_id=company_id,
            document_names=doc_names,
            n_results=15
        )

        documents = results.get('documents', [[]])[0]
        metadatas = results.get('metadatas', [[]])[0]
        source_documents = [meta.get('document_name', 'Unknown') for meta in metadatas]

        llm_result = await llm_service.generate_answer(
            question=request.question,
            context_chunks=documents,
            source_documents=source_documents,
            company_id=company_id
        )

        parsed_answer = json.loads(llm_result["answer"])

        history_check = supabase.table("chat_messages").select("id").eq("session_id", str(request.session_id)).limit(1).execute()
        session_title = None
        if not history_check.data:
            session_title = request.question[:100]

        user_msg = {
            "session_id": str(request.session_id),
            "role": "user",
            "content": request.question,
            "company_id": company_id,
            "created_by": user.id,
            "file_names": doc_names
        }
        if effective_workspace_id:
            user_msg["workspace_id"] = effective_workspace_id
        if request.customer_id:
            user_msg["customer_id"] = str(request.customer_id)

        if session_title:
            user_msg["session_title"] = session_title

        supabase.table("chat_messages").insert(user_msg).execute()

        assistant_msg = {
            "session_id": str(request.session_id),
            "role": "assistant",
            "content": parsed_answer.get("message", "I couldn't generate a response."),
            "data": parsed_answer.get("data", []),
            "sources": llm_result["sources"],
            "company_id": company_id,
            "created_by": user.id,
            "file_names": doc_names
        }
        if effective_workspace_id:
            assistant_msg["workspace_id"] = effective_workspace_id
        if request.customer_id:
            assistant_msg["customer_id"] = str(request.customer_id)

        if session_title:
            assistant_msg["session_title"] = session_title

        save_res = supabase.table("chat_messages").insert(assistant_msg).execute()

        return save_res.data[0]

    except Exception as e:
        print(f"[ERROR] Chat Query Error: {str(e)}")
        raise HTTPException(status_code=500, detail=str(e))

@router.post("/save")
async def save_chat_session(request: ChatSaveRequest, user=Depends(get_current_user)):
    """Mark a chat session as saved and store assigned files."""
    try:
        res = supabase.table("chat_messages") \
            .update({
                "is_saved": True,
                "file_names": request.file_names
            }) \
            .eq("session_id", str(request.session_id)) \
            .execute()
        return {"status": "success"}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@router.get("/history", response_model=List[ChatHistoryItem])
async def list_chat_history(workspace_id: Optional[UUID] = None, customer_id: Optional[UUID] = None, user=Depends(get_current_user)):
    """List all unique chat sessions for a workspace or customer."""
    try:
        if not workspace_id and not customer_id:
            raise HTTPException(status_code=400, detail="Must provide either workspace_id or customer_id")

        query = supabase.table("chat_messages") \
            .select("session_id, session_title, created_at, workspace_id, customer_id, file_names") \
            .is_("deleted_at", "null") \
            .eq("is_saved", True) \
            .order("created_at", desc=True)

        if customer_id:
            query = query.eq("customer_id", str(customer_id))
        elif workspace_id:
            query = query.eq("workspace_id", str(workspace_id))

        result = query.execute()

        session_map = {}
        for msg in result.data:
            sid = msg["session_id"]
            if sid not in session_map:
                session_map[sid] = {
                    "session_id": sid,
                    "session_title": msg.get("session_title"),
                    "last_message_at": msg["created_at"],
                    "workspace_id": msg["workspace_id"] if msg.get("workspace_id") else None,
                    "customer_id": msg["customer_id"] if msg.get("customer_id") else None,
                    "file_names": msg.get("file_names") or []
                }
            else:

                if not session_map[sid]["session_title"] and msg.get("session_title"):
                    session_map[sid]["session_title"] = msg.get("session_title")

                if msg.get("file_names") and len(msg.get("file_names")) > 0:
                    session_map[sid]["file_names"] = list(set(session_map[sid]["file_names"] + msg.get("file_names")))

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
            .select("*, message_feedback(rating)") \
            .eq("session_id", str(session_id)) \
            .is_("deleted_at", "null") \
            .order("created_at", desc=False) \
            .execute()

        messages = []
        for msg in result.data:
            feedback = msg.get("message_feedback", [])
            msg["rating"] = feedback[0]["rating"] if feedback else None
            messages.append(msg)

        return messages
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@router.delete("/history/{session_id}")
async def delete_chat_session(session_id: UUID, user=Depends(get_current_user)):
    """Soft delete an entire chat session."""
    try:
        from datetime import datetime, timezone
        now = datetime.now(timezone.utc).isoformat()

        result = supabase.table("chat_messages") \
            .update({"deleted_at": now}) \
            .eq("session_id", str(session_id)) \
            .execute()

        return {"message": "Chat session soft-deleted successfully"}
    except Exception as e:
        print(f"[ERROR] Error deleting chat session: {str(e)}")
        raise HTTPException(status_code=500, detail=str(e))

@router.post("/feedback")
async def submit_feedback(request: ChatFeedbackCreate, user=Depends(get_current_user)):
    """
    Store user feedback/rating for a specific message.
    """
    try:
        feedback_data = {
            "message_id": str(request.message_id),
            "rating": request.rating,
            "feedback_text": request.feedback_text,
            "created_by": user.id
        }

        supabase_admin.table("message_feedback").insert(feedback_data).execute()
        return {"message": "Feedback submitted successfully"}
    except Exception as e:
        print(f"[ERROR] Feedback Submission Error: {str(e)}")
        raise HTTPException(status_code=500, detail=str(e))
@router.patch("/history/{session_id}/title")
async def update_chat_session_title(session_id: UUID, request: ChatUpdateTitleRequest, user=Depends(get_current_user)):
    """Update the title of a chat session."""
    try:

        result = supabase.table("chat_messages") \
            .update({"session_title": request.session_title}) \
            .eq("session_id", str(session_id)) \
            .execute()

        if not result.data:
            raise HTTPException(status_code=404, detail="Chat session not found")

        return {"status": "success", "title": request.session_title}
    except Exception as e:
        print(f"[ERROR] Error updating chat title: {str(e)}")
        raise HTTPException(status_code=500, detail=str(e))

@router.patch("/messages/{message_id}")
async def update_chat_message(message_id: UUID, request: ChatMessageUpdateRequest, user=Depends(get_current_user)):
    """Update a specific chat message's content or data."""
    try:
        update_data = {}
        if request.content is not None:
            update_data["content"] = request.content
        if request.data is not None:
            update_data["data"] = request.data

        if not update_data:
            return {"status": "success", "message": "No fields to update"}

        result = supabase.table("chat_messages") \
            .update(update_data) \
            .eq("id", str(message_id)) \
            .execute()

        if not result.data:
            raise HTTPException(status_code=404, detail="Chat message not found")

        return {"status": "success"}
    except Exception as e:
        print(f"[ERROR] Error updating chat message: {str(e)}")
        raise HTTPException(status_code=500, detail=str(e))
