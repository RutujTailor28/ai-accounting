from fastapi import APIRouter, HTTPException, Depends
from typing import List
from uuid import UUID
from app.schemas.folder import FolderCreate, FolderResponse, FileResponse
from app.api.deps import get_current_user
from app.core.supabase import supabase
from app.core.supabase import supabase

router = APIRouter()


@router.post("/", response_model=FolderResponse)
async def create_folder(folder_data: FolderCreate, user=Depends(get_current_user)):
    """Create a new folder in the database."""
    try:
        # Step 0: Get user's company_id from profile (Security enforcement)
        profile_res = supabase.table("profiles").select("company_id").eq("id", user.id).single().execute()
        if not profile_res.data or not profile_res.data.get("company_id"):
            raise HTTPException(status_code=400, detail="User profile or company assignment missing")
        
        company_id = profile_res.data["company_id"]

        # Prepare data for Supabase
        data = {
            "name": folder_data.name,
            "company_id": company_id,
            "parent_id": str(folder_data.parent_id) if folder_data.parent_id else None,
            "created_by": user.id
        }

        result = supabase.table("folders").insert(data).execute()
        
        if not result.data:
            raise HTTPException(status_code=400, detail="Failed to create folder")

        print(f"[INFO] Folder created: {folder_data.name} for company {company_id}")
        return result.data[0]

    except Exception as e:
        print(f"[ERROR] Error creating folder: {str(e)}")
        if "duplicate key" in str(e).lower():
            raise HTTPException(status_code=400, detail="Folder with this name already exists in this company")
        raise HTTPException(status_code=500, detail=str(e))


@router.get("/", response_model=List[FolderResponse])
async def list_folders(user=Depends(get_current_user)):
    """List all folders for the authenticated user's company."""
    try:
        # Get company_id from profile
        profile_res = supabase.table("profiles").select("company_id").eq("id", user.id).single().execute()
        if not profile_res.data or not profile_res.data.get("company_id"):
            raise HTTPException(status_code=400, detail="User profile or company assignment missing")
        
        company_id = profile_res.data["company_id"]

        result = supabase.table("folders") \
            .select("*") \
            .eq("company_id", company_id) \
            .order("name") \
            .execute()
        
        return result.data or []
    except Exception as e:
        print(f"[ERROR] Error listing folders: {str(e)}")
        raise HTTPException(status_code=500, detail=str(e))


@router.get("/{folder_id}/files", response_model=List[FileResponse])
async def list_folder_files(folder_id: UUID, user=Depends(get_current_user)):
    """List all files within a specific folder."""
    try:
        result = supabase.table("files") \
            .select("*") \
            .eq("folder_id", str(folder_id)) \
            .order("created_at", desc=True) \
            .execute()
        
        return result.data or []
    except Exception as e:
        print(f"[ERROR] Error listing folder files: {str(e)}")
        raise HTTPException(status_code=500, detail=str(e))
