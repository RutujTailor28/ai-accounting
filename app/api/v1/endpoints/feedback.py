from fastapi import APIRouter, HTTPException, Depends
from app.api.deps import get_current_user
from app.schemas.feedback import FeedbackCreate, FeedbackResponse
from app.ai.rag.retriever import vector_store
from app.services.deps import embedding_service
from datetime import datetime
import uuid

router = APIRouter()

from app.core.supabase import supabase

@router.post("/", response_model=FeedbackResponse)
async def submit_feedback(
    feedback: FeedbackCreate,
    user=Depends(get_current_user)
):
    """
    Submit user feedback/correction to the AI agent.
    This stores the correction in the vector database so the agent can learn from it next time.
    """
    try:
        # Step 0: Get user's verified company_id from profile
        profile_res = supabase.table("profiles").select("company_id").eq("id", user.id).single().execute()
        company_id = profile_res.data["company_id"]
        
        print(f"[INFO] Receiving feedback from user {user.id} for company {company_id} (requested: {feedback.company_id})")
        
        # 1. Create a "Rule" string that the AI can understand
        # Format: "When queried about '{original_query}', the user corrected the response '{ai_response}' with '{user_correction}'. Explanation: {explanation}"
        rule_text = f"USER CORRECTION: When asked '{feedback.original_query}', the correct validation/category is '{feedback.user_correction}'."
        if feedback.explanation:
            rule_text += f" Reason: {feedback.explanation}"
            
        # 2. Generate embedding for the ORIGINAL QUERY
        # We want to retrieve this rule when a SIMILAR query is asked in the future.
        embedding = embedding_service.generate_embedding(feedback.original_query)
        
        # 3. Store in Vector Store
        metadata = {
            "company_id": company_id,
            "user_id": str(user.id),
            "created_at": datetime.now().isoformat(),
            "type": "correction",
            "original_query": feedback.original_query
        }
        
        vector_store.add_feedback(
            text=rule_text,
            embedding=embedding,
            metadata=metadata
        )
        
        return FeedbackResponse(
            id=uuid.uuid4(),
            original_query=feedback.original_query,
            ai_response=feedback.ai_response,
            user_correction=feedback.user_correction,
            explanation=feedback.explanation,
            company_id=company_id,
            created_at=datetime.now()
        )

    except Exception as e:
        print(f"[ERROR] Failed to submit feedback: {str(e)}")
        raise HTTPException(status_code=500, detail=str(e))

@router.delete("/")
async def clear_feedback(user=Depends(get_current_user)):
    """
    Clear all feedback/corrections for the user's company.
    This effectively resets the AI's "learning" for this company.
    """
    try:
        # Step 0: Get user's verified company_id
        profile_res = supabase.table("profiles").select("company_id").eq("id", user.id).single().execute()
        company_id = profile_res.data["company_id"]
        
        success = vector_store.clear_company_feedback(company_id)
        
        if success:
            return {"message": "All feedback cleared successfully. AI memory reset for this company."}
        else:
            raise HTTPException(status_code=500, detail="Failed to clear feedback")
            
    except Exception as e:
        print(f"[ERROR] Failed to clear feedback: {str(e)}")
        raise HTTPException(status_code=500, detail=str(e))
