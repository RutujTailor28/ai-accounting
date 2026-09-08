from fastapi import APIRouter
from app.api.v1.endpoints import auth, users, roles, documents, reports, folders, workspaces, chat, accounting, customers, feedback, demo, trial

api_router = APIRouter()

api_router.include_router(auth.router, prefix="/auth", tags=["Authentication"])
api_router.include_router(users.router, prefix="/users", tags=["User Management"])
api_router.include_router(roles.router, prefix="/roles", tags=["Role Management"])
api_router.include_router(documents.router, prefix="/documents", tags=["Documents"])
api_router.include_router(reports.router, prefix="/reports", tags=["Reports"])
api_router.include_router(folders.router, prefix="/folders", tags=["Folders"])
api_router.include_router(workspaces.router, prefix="/workspaces", tags=["Workspaces"])
api_router.include_router(chat.router, prefix="/chat", tags=["Chat"])
api_router.include_router(accounting.router, prefix="/accounting", tags=["Accounting"])
api_router.include_router(customers.router, prefix="/customers", tags=["Customers"])
api_router.include_router(feedback.router, prefix="/feedback", tags=["Feedback"])
api_router.include_router(demo.router, prefix="/demo", tags=["Demo Booking"])
# Public, unauthenticated. Phase 1 deterministic pipeline only - no LLM.
api_router.include_router(trial.router, prefix="/trial", tags=["Public Trial"])
