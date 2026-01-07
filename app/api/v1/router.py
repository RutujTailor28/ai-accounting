from fastapi import APIRouter
from app.api.v1.endpoints import auth, users, roles, documents, reports, folders

api_router = APIRouter()

api_router.include_router(auth.router, prefix="/auth", tags=["Authentication"])
api_router.include_router(users.router, prefix="/users", tags=["User Management"])
api_router.include_router(roles.router, prefix="/roles", tags=["Role Management"])
api_router.include_router(documents.router, prefix="/documents", tags=["Documents"])
api_router.include_router(reports.router, prefix="/reports", tags=["Reports"])
api_router.include_router(folders.router, prefix="/folders", tags=["Folders"])
