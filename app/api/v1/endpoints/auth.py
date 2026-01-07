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

        # Fetch profile for company_id
        company_id = None
        try:
            profile_res = supabase.table("profiles") \
                .select("company_id") \
                .eq("id", response.user.id) \
                .execute()
            
            if profile_res.data and len(profile_res.data) > 0:
                company_id = profile_res.data[0].get("company_id")
        except Exception as profile_e:
            print(f"[WARNING] Profile fetch failed: {str(profile_e)}")

        if not company_id:
            raise HTTPException(
                status_code=status.HTTP_400_BAD_REQUEST,
                detail="User profile not found or company_id not assigned. Please contact your administrator."
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
