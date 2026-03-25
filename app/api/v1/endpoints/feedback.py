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

        profile_res = supabase.table("profiles").select("company_id").eq("id", user.id).single().execute()
        company_id = profile_res.data["company_id"]

        print(f"[INFO] Receiving feedback from user {user.id} for company {company_id} (requested: {feedback.company_id})")

        rule_text = f"USER CORRECTION: When asked '{feedback.original_query}', the correct validation/category is '{feedback.user_correction}'."
        if feedback.explanation:
            rule_text += f" Reason: {feedback.explanation}"

        embedding = embedding_service.generate_embedding(feedback.original_query)

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

        if feedback.customer_id:
            try:

                cust_res = supabase.table("customers").select("business_type").eq("id", feedback.customer_id).single().execute()
                business_type = cust_res.data.get("business_type") if cust_res.data else None

                rule_record = {
                    "customer_id": feedback.customer_id,
                    "business_type": business_type,
                    "rule_description": rule_text
                }

                try:
                    uuid.UUID(str(company_id))
                    rule_record["company_id"] = company_id
                except ValueError:

                    pass

                supabase.table("accounting_rules").insert(rule_record).execute()
                print(f"[INFO] Saved custom rule to database for customer {feedback.customer_id}")
            except Exception as e:
                print(f"[WARNING] Failed to save rule to accounting_rules table: {e}")

        return FeedbackResponse(
            id=uuid.uuid4(),
            original_query=feedback.original_query,
            ai_response=feedback.ai_response,
            user_correction=feedback.user_correction,
            explanation=feedback.explanation,
            company_id=company_id,
            customer_id=feedback.customer_id,
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
