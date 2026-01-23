from fastapi import Depends, HTTPException, status
from fastapi.security import HTTPBearer, HTTPAuthorizationCredentials
from app.core.supabase import supabase
from typing import List, Optional

security = HTTPBearer()

async def get_current_user(credentials: HTTPAuthorizationCredentials = Depends(security)):
    """
    Verifies the Supabase JWT and returns the user object.
    """
    token = credentials.credentials
    try:
        # Verify the token with Supabase
        # Note: supabase.auth.get_user(token) implicitly validates the JWT
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

def require_role(allowed_roles: List[str]):
    """
    Dependency factory to check if a user has one of the allowed roles.
    """
    async def role_checker(user=Depends(get_current_user)):
        # Fetch roles for the user from our custom user_roles table
        try:
            # We use supabase_admin if needed, but here user's token might be enough 
            # depending on RLS. For simplicity, we'll check via the client.
            # Assuming 'roles' table has 'name' and 'user_roles' links user_id to role_id
            response = supabase.table("user_roles") \
                .select("roles(name)") \
                .eq("user_id", user.id) \
                .execute()
            
            user_roles_data = response.data
            user_role_names = [item['roles']['name'] for item in user_roles_data if 'roles' in item and 'name' in item['roles']]
            
            # Check if any of the user's roles are in the allowed list
            if not any(role in allowed_roles for role in user_role_names):
                raise HTTPException(
                    status_code=status.HTTP_403_FORBIDDEN,
                    detail="Operation not permitted for your role"
                )
            
            return user
        except Exception as e:
            raise HTTPException(
                status_code=status.HTTP_403_FORBIDDEN,
                detail=f"Role check failed: {str(e)}"
            )
            
    return role_checker
