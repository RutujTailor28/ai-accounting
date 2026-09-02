from fastapi import APIRouter, HTTPException, Depends
from typing import List
from uuid import UUID
from app.schemas.workspace import WorkspaceCreate, WorkspaceResponse
from app.api.deps import get_current_user, require_role
from app.core.supabase import supabase_admin

router = APIRouter()

@router.post("/", response_model=WorkspaceResponse)
async def create_workspace(
    workspace_data: WorkspaceCreate, 
    user=Depends(require_role(["user", "manager", "admin", "superadmin"]))
):
    """Create a new workspace by selecting existing file and folder IDs."""
    try:

        profile_res = supabase_admin.table("profiles").select("company_id").eq("id", user.id).single().execute()
        if not profile_res.data or not profile_res.data.get("company_id"):
            raise HTTPException(status_code=400, detail="User profile or company assignment missing")

        company_id = profile_res.data["company_id"]

        data = {
            "name": workspace_data.name,
            "description": workspace_data.description,
            "company_id": company_id,
            "file_ids": [str(fid) for fid in workspace_data.file_ids],
            "folder_ids": [str(fid) for fid in workspace_data.folder_ids],
            "created_by": user.id
        }

        result = supabase_admin.table("workspaces").insert(data).execute()

        if not result.data:
            raise HTTPException(status_code=400, detail="Failed to create workspace")

        print(f"[INFO] Workspace created: {workspace_data.name} for company {company_id}")
        return result.data[0]

    except Exception as e:
        print(f"[ERROR] Error creating workspace: {str(e)}")
        raise HTTPException(status_code=500, detail=str(e))

@router.get("/", response_model=List[WorkspaceResponse])
async def list_workspaces(user=Depends(get_current_user)):
    """List all workspaces for the authenticated user's company."""
    try:

        profile_res = supabase_admin.table("profiles").select("company_id").eq("id", user.id).single().execute()
        if not profile_res.data or not profile_res.data.get("company_id"):
            raise HTTPException(status_code=400, detail="User profile or company assignment missing")

        company_id = profile_res.data["company_id"]

        result = supabase_admin.table("workspaces") \
            .select("*") \
            .eq("company_id", company_id) \
            .is_("deleted_at", "null") \
            .order("created_at", desc=True) \
            .execute()

        return result.data or []
    except Exception as e:
        print(f"[ERROR] Error listing workspaces: {str(e)}")
        raise HTTPException(status_code=500, detail=str(e))

@router.get("/{workspace_id}", response_model=WorkspaceResponse)
async def get_workspace(workspace_id: UUID, user=Depends(get_current_user)):
    """Get details of a specific workspace."""
    try:
        result = supabase_admin.table("workspaces") \
            .select("*") \
            .eq("id", str(workspace_id)) \
            .is_("deleted_at", "null") \
            .single() \
            .execute()

        if not result.data:
            raise HTTPException(status_code=404, detail="Workspace not found")

        return result.data
    except Exception as e:
        print(f"[ERROR] Error getting workspace: {str(e)}")
        raise HTTPException(status_code=500, detail=str(e))

@router.delete("/{workspace_id}")
async def delete_workspace(
    workspace_id: UUID, 
    user=Depends(require_role(["user", "manager", "admin", "superadmin"]))
):
    """Delete a workspace."""
    try:
        from datetime import datetime, timezone
        now = datetime.now(timezone.utc).isoformat()

        result = supabase_admin.table("workspaces") \
            .update({"deleted_at": now}) \
            .eq("id", str(workspace_id)) \
            .execute()

        return {"message": "Workspace soft-deleted successfully"}
    except Exception as e:
        print(f"[ERROR] Error deleting workspace: {str(e)}")
        raise HTTPException(status_code=500, detail=str(e))
