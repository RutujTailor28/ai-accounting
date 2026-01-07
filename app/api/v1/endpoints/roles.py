from fastapi import APIRouter, HTTPException, Depends, status
from app.schemas.role import RoleCreate, RoleUpdate, RoleResponse
from app.core.supabase import supabase_admin
from app.api.deps import require_role
from typing import List

router = APIRouter()

# All routes here require 'superadmin' role
superadmin_dep = Depends(require_role(["superadmin"]))

@router.get("", response_model=List[RoleResponse])
async def list_roles(_=superadmin_dep):
    """List all available roles."""
    try:
        response = supabase_admin.table("roles").select("*").execute()
        return response.data
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@router.post("", response_model=RoleResponse, status_code=status.HTTP_201_CREATED)
async def create_role(data: RoleCreate, _=superadmin_dep):
    """Create a new role definition."""
    try:
        response = supabase_admin.table("roles").insert({
            "name": data.name,
            "description": data.description,
            "company_id": data.company_id
        }).execute()
        
        if not response.data:
            raise HTTPException(status_code=400, detail="Failed to create role")
            
        return response.data[0]
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@router.patch("/{role_id}", response_model=RoleResponse)
async def update_role(role_id: int, data: RoleUpdate, _=superadmin_dep):
    """Update a role's name or description."""
    try:
        update_data = {}
        if data.name is not None: update_data["name"] = data.name
        if data.description is not None: update_data["description"] = data.description
        if data.company_id is not None: update_data["company_id"] = data.company_id
        
        response = supabase_admin.table("roles").update(update_data).eq("id", role_id).execute()
        
        if not response.data:
            raise HTTPException(status_code=404, detail="Role not found")
            
        return response.data[0]
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@router.delete("/{role_id}", status_code=status.HTTP_204_NO_CONTENT)
async def delete_role(role_id: int, _=superadmin_dep):
    """Delete a role definition."""
    try:
        supabase_admin.table("roles").delete().eq("id", role_id).execute()
        return None
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))
