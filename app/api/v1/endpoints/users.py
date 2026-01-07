from fastapi import APIRouter, HTTPException, Depends, status
from app.schemas.user import UserCreate, UserUpdate, UserResponse
from app.core.supabase import supabase_admin
from app.api.deps import require_role
from typing import List
import uuid

router = APIRouter()

# All routes here require 'admin' or 'superadmin' role
admin_dep = Depends(require_role(["admin", "superadmin"]))


@router.get("", response_model=List[UserResponse])
async def list_users(_=admin_dep):
    """List all users with their roles."""
    try:
        # Fetch profiles and join roles
        response = supabase_admin.table("profiles") \
            .select("*, user_roles(roles(name))") \
            .execute()

        users = []
        for item in response.data:
            roles = [ur['roles']['name'] for ur in item.get('user_roles', []) if ur.get('roles')]
            users.append(UserResponse(**item, roles=roles))

        return users
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@router.get("/{user_id}", response_model=UserResponse)
async def get_user(user_id: str, _=admin_dep):
    """Get detailed information for a specific user."""
    try:
        response = supabase_admin.table("profiles") \
            .select("*, user_roles(roles(name))") \
            .eq("id", user_id) \
            .single() \
            .execute()

        item = response.data
        if not item:
            raise HTTPException(status_code=404, detail="User not found")

        roles = [ur['roles']['name'] for ur in item.get('user_roles', []) if ur.get('roles')]
        return UserResponse(**item, roles=roles)
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@router.post("", response_model=UserResponse, status_code=status.HTTP_201_CREATED)
async def create_user(data: UserCreate, _=admin_dep):
    """Create a new user using Supabase Auth Admin API and assign roles."""
    try:
        # 1. Create user in Supabase Auth
        auth_response = supabase_admin.auth.admin.create_user({
            "email": data.email,
            "password": data.password,
            "email_confirm": True,
            "user_metadata": {
                "first_name": data.first_name,
                "last_name": data.last_name,
                "company_id": data.company_id
            }
        })

        new_user = auth_response.user

        # 2. Assign roles
        for role_name in data.roles:
            # Get role id
            role_res = supabase_admin.table("roles").select("id").eq("name", role_name).single().execute()
            if role_res.data:
                supabase_admin.table("user_roles").insert({
                    "user_id": new_user.id,
                    "role_id": role_res.data['id']
                }).execute()

        # Profile is usually created by a DB trigger on auth.users insert
        # We'll wait a bit or fetch it.
        profile_res = supabase_admin.table("profiles") \
            .select("*") \
            .eq("id", new_user.id) \
            .single() \
            .execute()

        return UserResponse(**profile_res.data, roles=data.roles)
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@router.patch("/{user_id}", response_model=UserResponse)
async def update_user(user_id: str, data: UserUpdate, _=admin_dep):
    """Update user details and roles."""
    try:
        # Update profile
        update_data = {}
        if data.first_name is not None:
            update_data["first_name"] = data.first_name
        if data.last_name is not None:
            update_data["last_name"] = data.last_name

        if data.company_id is not None:
            update_data["company_id"] = data.company_id

        if update_data:
            supabase_admin.table("profiles").update(update_data).eq("id", user_id).execute()

        # Update roles if provided
        if data.roles is not None:
            # Delete existing roles
            supabase_admin.table("user_roles").delete().eq("user_id", user_id).execute()
            # Assign new ones
            for role_name in data.roles:
                role_res = supabase_admin.table("roles").select("id").eq("name", role_name).single().execute()
                if role_res.data:
                    supabase_admin.table("user_roles").insert({
                        "user_id": user_id,
                        "role_id": role_res.data['id']
                    }).execute()

        return await get_user(user_id)
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@router.delete("/{user_id}", status_code=status.HTTP_204_NO_CONTENT)
async def delete_user(user_id: str, _=admin_dep):
    """Delete a user from Supabase Auth and database."""
    try:
        supabase_admin.auth.admin.delete_user(user_id)
        # Cascade delete should handle profiles and user_roles if FKs are set correctly
        return None
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))
