from fastapi import Depends, HTTPException, status
from fastapi.security import HTTPBearer, HTTPAuthorizationCredentials
from app.core.supabase import supabase, supabase_admin
from typing import List, Optional

security = HTTPBearer()

async def get_current_user(credentials: HTTPAuthorizationCredentials = Depends(security)):
    """
    Verifies the Supabase JWT and returns the user object.
    """
    token = credentials.credentials
    try:

        user_response = supabase.auth.get_user(token)

        if not user_response.user:
            print(f"[AUTH] 401: Token provided but no user found in response.")
            raise HTTPException(
                status_code=status.HTTP_401_UNAUTHORIZED,
                detail="Invalid or expired token",
                headers={"WWW-Authenticate": "Bearer"},
            )

        return user_response.user
    except Exception as e:
        error_str = str(e).lower()
        if "expired" in error_str:
            print(f"[AUTH] 401: Token expired. Detail: {str(e)}")
        elif "invalid" in error_str:
            print(f"[AUTH] 401: Token invalid. Detail: {str(e)}")
        else:
            print(f"[AUTH] 401: Auth error. Detail: {str(e)}")

        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail=f"Auth error: {str(e)}",
            headers={"WWW-Authenticate": "Bearer"},
        )

async def get_user_context(user=Depends(get_current_user)):
    """
    Returns a context object with user, roles, and company_id.
    """
    try:

        profile_res = supabase_admin.table("profiles") \
            .select("*, user_roles(roles(name, permissions))") \
            .eq("id", user.id) \
            .single() \
            .execute()

        profile = profile_res.data
        if not profile:
            raise HTTPException(status_code=404, detail="User profile not found")

        roles = []
        permissions = set()
        for ur in profile.get('user_roles', []):
            role_data = ur.get('roles')
            if role_data:
                roles.append(role_data['name'])
                if role_data.get('permissions'):
                    permissions.update(role_data['permissions'])

        return {
            "user": user,
            "profile": profile,
            "roles": roles,
            "permissions": list(permissions),
            "company_id": profile.get("company_id"),
            "is_superadmin": "superadmin" in roles
        }
    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Failed to fetch user context: {str(e)}")

def require_role(allowed_roles: List[str], required_permission: Optional[str] = None):
    """
    Dependency factory to check if a user has one of the allowed roles OR a specific permission.
    """
    async def checker(user=Depends(get_current_user)):
        try:
            # Fetch user roles and their permissions
            response = supabase_admin.table("user_roles") \
                .select("roles(name, permissions)") \
                .eq("user_id", user.id) \
                .execute()

            user_roles_data = response.data
            user_role_names = []
            user_permissions = set()

            for item in user_roles_data:
                role_data = item.get('roles')
                if role_data:
                    user_role_names.append(role_data['name'])
                    if role_data.get('permissions'):
                        user_permissions.update(role_data['permissions'])

            # Superadmin bypass
            if "superadmin" in user_role_names:
                return user

            # Check if role-based access is allowed
            if any(role in allowed_roles for role in user_role_names):
                return user

            # Check if permission-based access is allowed
            if required_permission and required_permission in user_permissions:
                return user

            raise HTTPException(
                status_code=status.HTTP_403_FORBIDDEN,
                detail="Operation not permitted for your role or permissions"
            )
        except HTTPException:
            raise
        except Exception as e:
            raise HTTPException(
                status_code=status.HTTP_403_FORBIDDEN,
                detail=f"Access check failed: {str(e)}"
            )

    return checker
