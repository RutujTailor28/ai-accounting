from fastapi import APIRouter, HTTPException, Depends, status
from app.schemas.role import RoleCreate, RoleUpdate, RoleResponse
from app.core.supabase import supabase_admin
from app.api.deps import require_role
from typing import List

admin_dep = Depends(require_role(["admin", "superadmin"], required_permission="manage_team"))
router = APIRouter(dependencies=[admin_dep])
from app.api.deps import get_user_context

@router.get("", response_model=List[RoleResponse])
async def list_roles(ctx: dict = Depends(get_user_context)):
    """List all available roles. Filters by company for admins."""
    try:
        query = supabase_admin.table("roles").select("*")

        if not ctx["is_superadmin"]:

            pass

        response = query.execute()
        return response.data
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@router.post("", response_model=RoleResponse, status_code=status.HTTP_201_CREATED)
async def create_role(data: RoleCreate, ctx: dict = Depends(get_user_context)):
    """Create a new role definition for the company."""
    try:
        target_company_id = data.company_id
        if not ctx["is_superadmin"]:
            target_company_id = ctx["company_id"]

        response = supabase_admin.table("roles").insert({
            "name": data.name,
            "description": data.description,
            "company_id": target_company_id,
            "permissions": data.permissions
        }).execute()

        if not response.data:
            raise HTTPException(status_code=400, detail="Failed to create role")

        return response.data[0]
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@router.patch("/{role_id}", response_model=RoleResponse)
async def update_role(role_id: int, data: RoleUpdate, ctx: dict = Depends(get_user_context)):
    """Update a role's name or description."""
    try:

        target_res = supabase_admin.table("roles").select("company_id").eq("id", role_id).single().execute()
        if not target_res.data:
            raise HTTPException(status_code=404, detail="Role not found")

        if not ctx["is_superadmin"] and target_res.data.get("company_id") != ctx["company_id"]:
            raise HTTPException(status_code=403, detail="Access denied")

        update_data = {}
        if data.name is not None: update_data["name"] = data.name
        if data.description is not None: update_data["description"] = data.description
        if data.permissions is not None: update_data["permissions"] = data.permissions

        response = supabase_admin.table("roles").update(update_data).eq("id", role_id).execute()

        if not response.data:
            raise HTTPException(status_code=404, detail="Role not found")

        return response.data[0]
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@router.delete("/{role_id}", status_code=status.HTTP_204_NO_CONTENT)
async def delete_role(role_id: int, ctx: dict = Depends(get_user_context)):
    """Delete a role definition."""
    try:

        target_res = supabase_admin.table("roles").select("company_id").eq("id", role_id).single().execute()
        if not target_res.data:
             return None

        if not ctx["is_superadmin"] and target_res.data.get("company_id") != ctx["company_id"]:
             raise HTTPException(status_code=403, detail="Access denied")

        if target_res.data.get("company_id") is None:
             raise HTTPException(status_code=400, detail="Cannot delete system roles")

        supabase_admin.table("roles").delete().eq("id", role_id).execute()
        return None
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))
