from typing import List
from fastapi import APIRouter, HTTPException, Depends, status
from app.schemas.user import UserCreate, UserUpdate, UserResponse
from app.core.supabase import supabase_admin
from app.api.deps import require_role, get_user_context

# All routes here require 'admin' or 'superadmin' role
admin_dep = Depends(require_role(["admin", "superadmin"]))

router = APIRouter(dependencies=[admin_dep])


@router.get("", response_model=List[UserResponse])
async def list_users(ctx: dict = Depends(get_user_context)):
    """List all users with their roles. Filters by company for admins."""
    try:
        query = supabase_admin.table("profiles").select("*, user_roles(roles(name))")
        
        # If not superadmin, restrict to their own company
        if not ctx["is_superadmin"]:
            if not ctx["company_id"]:
                return [] # Should not happen, but safety first
            query = query.eq("company_id", ctx["company_id"])
            
        response = query.execute()

        users = []
        for item in response.data:
            roles = [ur['roles']['name'] for ur in item.get('user_roles', []) if ur.get('roles')]
            users.append(UserResponse(**item, roles=roles))

        return users
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@router.get("/{user_id}", response_model=UserResponse)
async def get_user(user_id: str, ctx: dict = Depends(get_user_context)):
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

        # Check company boundary for non-superadmins
        if not ctx["is_superadmin"] and item.get("company_id") != ctx["company_id"]:
            raise HTTPException(status_code=403, detail="Access denied to this user's information")

        roles = [ur['roles']['name'] for ur in item.get('user_roles', []) if ur.get('roles')]
        return UserResponse(**item, roles=roles)
    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@router.post("", response_model=UserResponse, status_code=status.HTTP_201_CREATED)
async def create_user(data: UserCreate, ctx: dict = Depends(get_user_context)):
    """Create a new user and assign roles. Restricted to current company for admins."""
    try:
        # Business rules:
        # 1. Non-superadmins can only create users for their own company.
        # 2. Non-superadmins can only assign 'user' role (or subset of their roles, but here just 'user').
        
        target_company_id = data.company_id
        target_roles = data.roles
        
        # Enforce company isolation: admins can only create users for their own company
        if not ctx["is_superadmin"]:
            target_company_id = ctx["company_id"]
            # Filter out 'superadmin' role if an admin tries to assign it
            target_roles = [r for r in target_roles if r != "superadmin"]
            if not target_roles:
                target_roles = ["user"]
            print(f"[AUTH] Enforcing company {target_company_id} and roles {target_roles} for new user {data.email}")
        elif not target_company_id:
            # Superadmin must specify a company_id if they want one assigned
            target_company_id = data.company_id
        # 1. Create user in Supabase Auth
        auth_response = supabase_admin.auth.admin.create_user({
            "email": data.email,
            "password": data.password,
            "email_confirm": True,
            "user_metadata": {
                "first_name": data.first_name,
                "last_name": data.last_name,
                "company_id": target_company_id
            }
        })

        new_user = auth_response.user

        # 2. Assign roles
        for role_name in target_roles:
            # Get role id
            role_res = supabase_admin.table("roles").select("id").eq("name", role_name).single().execute()
            if role_res.data:
                supabase_admin.table("user_roles").insert({
                    "user_id": new_user.id,
                    "role_id": role_res.data['id']
                }).execute()

        # 3. Explicitly create/update profile to ensure company_id is set
        # This acts as a backup/override for any DB triggers
        profile_data = {
            "id": new_user.id,
            "email": data.email,
            "company_id": target_company_id,
            "first_name": data.first_name,
            "last_name": data.last_name
        }
        supabase_admin.table("profiles").upsert(profile_data).execute()

        # Wait for profile to be created (fetch the final state)
        profile_res = supabase_admin.table("profiles") \
            .select("*") \
            .eq("id", new_user.id) \
            .single() \
            .execute()

        return UserResponse(**profile_res.data, roles=target_roles)
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@router.patch("/{user_id}", response_model=UserResponse)
async def update_user(user_id: str, data: UserUpdate, ctx: dict = Depends(get_user_context)):
    """Update user details and roles. Restricted to current company for admins."""
    try:
        # Fetch target user first to check company
        target_res = supabase_admin.table("profiles").select("company_id").eq("id", user_id).single().execute()
        if not target_res.data:
             raise HTTPException(status_code=404, detail="User not found")
             
        if not ctx["is_superadmin"] and target_res.data.get("company_id") != ctx["company_id"]:
            raise HTTPException(status_code=403, detail="Access denied")

        # Update profile
        update_data = {}
        if data.first_name is not None:
            update_data["first_name"] = data.first_name
        if data.last_name is not None:
            update_data["last_name"] = data.last_name

        # Managers/Admins cannot change company_id
        if ctx["is_superadmin"] and data.company_id is not None:
            update_data["company_id"] = data.company_id

        if update_data:
            supabase_admin.table("profiles").update(update_data).eq("id", user_id).execute()

        # Update roles if provided
        if data.roles is not None:
            # Filter roles for non-superadmins
            if not ctx["is_superadmin"]:
                new_roles = [r for r in data.roles if r != "superadmin"]
                if not new_roles:
                    new_roles = ["user"]
            else:
                new_roles = data.roles
            
            # Delete existing roles
            supabase_admin.table("user_roles").delete().eq("user_id", user_id).execute()
            # Assign new ones
            for role_name in new_roles:
                role_res = supabase_admin.table("roles").select("id").eq("name", role_name).single().execute()
                if role_res.data:
                    supabase_admin.table("user_roles").insert({
                        "user_id": user_id,
                        "role_id": role_res.data['id']
                    }).execute()

        return await get_user(user_id, ctx=ctx)
    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@router.delete("/{user_id}", status_code=status.HTTP_204_NO_CONTENT)
async def delete_user(user_id: str, ctx: dict = Depends(get_user_context)):
    """Delete a user from Supabase Auth and database. Restricted to current company for admins."""
    try:
        # Fetch target user first to check company
        target_res = supabase_admin.table("profiles").select("company_id").eq("id", user_id).single().execute()
        if not target_res.data:
             raise HTTPException(status_code=404, detail="User not found")
             
        if not ctx["is_superadmin"] and target_res.data.get("company_id") != ctx["company_id"]:
            raise HTTPException(status_code=403, detail="Access denied")

        # 1. Delete associated roles
        supabase_admin.table("user_roles").delete().eq("user_id", user_id).execute()
        
        # 2. Delete profile
        supabase_admin.table("profiles").delete().eq("id", user_id).execute()
        
        # 3. Delete from Supabase Auth
        supabase_admin.auth.admin.delete_user(user_id)
        
        print(f"[AUTH] Successfully deleted user {user_id} and all associated data.")
        return None
    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))
