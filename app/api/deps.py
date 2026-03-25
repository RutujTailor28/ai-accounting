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
            .select("*, user_roles(roles(name))") \
            .eq("id", user.id) \
            .single() \
            .execute()

        profile = profile_res.data
        if not profile:
            raise HTTPException(status_code=404, detail="User profile not found")

        roles = [ur['roles']['name'] for ur in profile.get('user_roles', []) if ur.get('roles')]

        return {
            "user": user,
            "profile": profile,
            "roles": roles,
            "company_id": profile.get("company_id"),
            "is_superadmin": "superadmin" in roles
        }
    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Failed to fetch user context: {str(e)}")

def require_role(allowed_roles: List[str]):
    """
    Dependency factory to check if a user has one of the allowed roles.
    """
    async def role_checker(user=Depends(get_current_user)):

        try:

            response = supabase_admin.table("user_roles") \
                .select("roles(name)") \
                .eq("user_id", user.id) \
                .execute()

            user_roles_data = response.data
            user_role_names = [item['roles']['name'] for item in user_roles_data if item.get('roles')]

            if not any(role in allowed_roles for role in user_role_names):

                if "superadmin" in user_role_names:
                    return user

                raise HTTPException(
                    status_code=status.HTTP_403_FORBIDDEN,
                    detail="Operation not permitted for your role"
                )

            return user
        except HTTPException:
            raise
        except Exception as e:
            raise HTTPException(
                status_code=status.HTTP_403_FORBIDDEN,
                detail=f"Role check failed: {str(e)}"
            )

    return role_checker
