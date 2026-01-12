from fastapi import APIRouter, HTTPException, Depends, status
from app.schemas.auth import UserLogin, LoginResponse, ForgotPasswordRequest, ResetPasswordRequest
from app.core.supabase import supabase
from app.api.deps import get_current_user

router = APIRouter()

@router.post("/login", response_model=LoginResponse)
async def login(credentials: UserLogin):
    """
    Authenticate user via Supabase and return JWT.
    """
    try:
        response = supabase.auth.sign_in_with_password({
            "email": credentials.email,
            "password": credentials.password
        })
        
        if not response.session:
            # Supabase sometimes returns successful sign_in but no session if email confirmation is required
            raise HTTPException(
                status_code=status.HTTP_401_UNAUTHORIZED,
                detail="Login succeeded but no session was created. Please check if your email is confirmed."
            )
            
        print(f"[DEBUG] Login successful for {response.user.email}")

        # Try to get company_id from user_metadata first (preferred method)
        company_id = None
        if hasattr(response.user, 'user_metadata') and response.user.user_metadata:
            company_id = response.user.user_metadata.get("company_id")
            if company_id:
                print(f"[INFO] Found company_id in user_metadata: {company_id}")
        
        # Fallback: Check profiles table if not in user_metadata
        if not company_id:
            try:
                profile_res = supabase.table("profiles") \
                    .select("company_id") \
                    .eq("id", response.user.id) \
                    .execute()
                
                if profile_res.data and len(profile_res.data) > 0:
                    company_id = profile_res.data[0].get("company_id")
                    print(f"[INFO] Found company_id in profiles table: {company_id}")
                else:
                    # Neither source has company_id - create profile with default
                    print(f"[INFO] No company_id found. Creating default profile...")
                    default_company_id = f"company-{response.user.id[:8]}"
                    
                    # Import admin client for bypassing RLS
                    from app.core.supabase import supabase_admin
                    
                    # Create profile using admin client to bypass RLS
                    new_profile = supabase_admin.table("profiles").insert({
                        "id": response.user.id,
                        "email": response.user.email,
                        "company_id": default_company_id,
                        "first_name": response.user.email.split("@")[0],
                        "last_name": ""
                    }).execute()
                    
                    if new_profile.data:
                        company_id = default_company_id
                        print(f"[INFO] Created profile with company_id: {company_id}")
                    else:
                        print(f"[ERROR] Profile creation failed: {new_profile}")
                    
            except Exception as profile_e:
                print(f"[WARNING] Profile fetch/create failed: {str(profile_e)}")
                import traceback
                traceback.print_exc()

        if not company_id:
            raise HTTPException(
                status_code=status.HTTP_400_BAD_REQUEST,
                detail="User profile could not be created. Please contact your administrator."
            )

        # Ensure we return valid strings, not None for required fields
        return LoginResponse(
            access_token=str(response.session.access_token),
            user_id=str(response.user.id),
            email=str(response.user.email),
            company_id=str(company_id)
        )
    except HTTPException:
        raise
    except Exception as e:
        print(f"[ERROR] unexpected login failure: {str(e)}")
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail=f"Login failed: {str(e)}"
        )

@router.get("/me")
async def get_me(user=Depends(get_current_user)):
    """
    Get current logged in user info.
    """
    # Fetch profile for company_id
    profile_res = supabase.table("profiles") \
        .select("company_id") \
        .eq("id", user.id) \
        .single() \
        .execute()
    
    company_id = profile_res.data.get("company_id") if profile_res.data else None

    return {
        "userId": user.id,
        "email": user.email,
        "companyId": company_id,
        "lastSignIn": user.last_sign_in_at
    }

@router.post("/forgot-password")
async def forgot_password(data: ForgotPasswordRequest):
    """
    Step 1: Request a password reset email.
    Supabase will send an email to the user with a link to reset their password.
    """
    try:
        supabase.auth.reset_password_for_email(data.email)
        return {"message": "Password reset email sent"}
    except Exception as e:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail=f"Failed to send reset email: {str(e)}"
        )

@router.post("/reset-password")
async def reset_password(data: ResetPasswordRequest, user=Depends(get_current_user)):
    """
    Step 2: Update the password.
    This endpoint should be called AFTER the user clicks the link in their email 
    and is redirected back to your app with a session.
    """
    try:
        supabase.auth.update_user({"password": data.new_password})
        return {"message": "Password updated successfully"}
    except Exception as e:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail=f"Failed to reset password: {str(e)}"
        )
