from fastapi import APIRouter, HTTPException, Depends, status
from app.schemas.auth import UserLogin, LoginResponse, ForgotPasswordRequest, ResetPasswordRequest, TokenRefreshRequest, UserRegister
from app.core.supabase import supabase, supabase_admin
import uuid
from app.api.deps import get_current_user
from app.schemas.user import ProfileResponse, ProfileUpdate
from app.schemas.company import CompanyResponse, CompanyUpdate

router = APIRouter()

@router.post("/signup", response_model=dict)
async def signup(data: UserRegister):
    """
    Register a new user and initialize their company.
    """
    try:
        # Sanitize email
        email = data.email.strip().lower()
        print(f"[DEBUG] Attempting signup for email: '{email}'")

        # Step 2: Generate a unique company_id
        company_id = f"comp-{str(uuid.uuid4())[:8]}"
        print(f"[DEBUG] Generated company_id: {company_id}")

        # Step 3: Register user in Supabase Auth using Admin Client
        # Using Admin Client is more reliable for testing as it confirms the email automatically
        try:
            auth_res = supabase_admin.auth.admin.create_user({
                "email": email,
                "password": data.password,
                "email_confirm": True,
                "user_metadata": {
                    "full_name": f"{data.first_name} {data.last_name}",
                    "first_name": data.first_name,
                    "last_name": data.last_name,
                    "company_name": data.company_name,
                    "company_id": company_id
                }
            })
        except Exception as auth_e:
            err_msg = str(auth_e).lower()
            print(f"[ERROR] Supabase admin.create_user failed: {str(auth_e)}")
            
            if "already been registered" in err_msg or "already exists" in err_msg:
                raise HTTPException(
                    status_code=status.HTTP_400_BAD_REQUEST,
                    detail="User is already registered in Supabase Auth. Please delete the user from the Supabase Console (Authentication > Users) or use a different email."
                )
            
            raise HTTPException(
                status_code=status.HTTP_400_BAD_REQUEST,
                detail=f"Authentication system error: {str(auth_e)}"
            )

        if not auth_res.user:
            print(f"[ERROR] auth_res.user is None. Full response: {auth_res}")
            raise HTTPException(
                status_code=status.HTTP_400_BAD_REQUEST,
                detail="Failed to create user account"
            )

        user_id = auth_res.user.id
        print(f"[DEBUG] User created with ID: {user_id}")

        # Step 4: Create entry in 'company' table
        company_data = {
            "id": company_id,
            "name": data.company_name
        }
        supabase_admin.table("company").upsert(company_data).execute()

        # Step 5: Create profile in 'profiles' table
        profile_data = {
            "id": user_id,
            "email": email,
            "company_id": company_id,
            "first_name": data.first_name,
            "last_name": data.last_name
        }
        
        # Using upsert to handle case where profile might already exist (e.g. from previous failed attempt)
        profile_res = supabase_admin.table("profiles").upsert(profile_data).execute()

        # Step 6: Assign 'admin' role to the first user
        try:
            role_res = supabase_admin.table("roles").select("id").eq("name", "admin").single().execute()
            if role_res.data:
                supabase_admin.table("user_roles").insert({
                    "user_id": user_id,
                    "role_id": role_res.data['id']
                }).execute()
        except Exception as role_e:
            print(f"[WARNING] Failed to assign admin role: {str(role_e)}")
        
        if not profile_res.data:
            print(f"[ERROR] Failed to create profile for user {auth_res.user.id}")
            # We don't necessarily want to fail the whole signup if Auth succeeded, 
            # as login can recreate the profile as a fallback, but it's better to log it.

        return {
            "message": "User registered successfully",
            "user_id": auth_res.user.id,
            "company_id": company_id
        }

    except Exception as e:
        print(f"[ERROR] Signup failed: {str(e)}")
        if "already registered" in str(e).lower():
             raise HTTPException(status_code=400, detail="User already registered")
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail=str(e)
        )

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

        # Try to get company_id and company_name from user_metadata first
        company_id = None
        company_name = None
        if hasattr(response.user, 'user_metadata') and response.user.user_metadata:
            company_id = response.user.user_metadata.get("company_id")
        
        # Fallback/Refresh: Check company table
        if company_id:
            try:
                comp_res = supabase_admin.table("company") \
                    .select("name") \
                    .eq("id", company_id) \
                    .single() \
                    .execute()
                if comp_res.data:
                    company_name = comp_res.data.get("name")
            except Exception as e:
                print(f"[WARNING] Could not fetch company name: {str(e)}")

        # If still missing info, check profiles (legacy fallback)
        if not company_id or not company_name:
            try:
                profile_res = supabase_admin.table("profiles") \
                    .select("company_id") \
                    .eq("id", response.user.id) \
                    .execute()
                
                if profile_res.data and len(profile_res.data) > 0:
                    profile = profile_res.data[0]
                    company_id = company_id or profile.get("company_id")
                    # company_name will be fetched from company table below or is already None
                else:
                    # Create default company and profile if nothing exists
                    company_id = company_id or f"company-{response.user.id[:8]}"
                    company_name = company_name or "My Company"
                    
                    # 1. UPSERT Company
                    supabase_admin.table("company").upsert({
                        "id": company_id,
                        "name": company_name
                    }).execute()

                    # 2. UPSERT Profile
                    supabase_admin.table("profiles").upsert({
                        "id": response.user.id,
                        "email": response.user.email,
                        "company_id": company_id,
                        "first_name": response.user.email.split("@")[0],
                        "last_name": "",
                        "role": "admin"
                    }).execute()
            except Exception as profile_e:
                print(f"[WARNING] Profile/Company sync failed: {str(profile_e)}")
            
            # Final check: If we have company_id but still no name, try fetching from company table
            if company_id and not company_name:
                try:
                    c_res = supabase_admin.table("company").select("name").eq("id", company_id).single().execute()
                    if c_res.data:
                        company_name = c_res.data.get("name")
                except:
                    pass

        if not company_id:
            raise HTTPException(
                status_code=status.HTTP_400_BAD_REQUEST,
                detail="User profile could not be created. Please contact your administrator."
            )

        # Fetch role and permissions
        role = "user"
        permissions = []
        try:
            role_res = supabase_admin.table("user_roles").select("roles(name, permissions)").eq("user_id", response.user.id).execute()
            print(f"[DEBUG] Role fetch result for {response.user.email}: {role_res.data}")
            if role_res.data:
                # Handle potential structure variations
                item = role_res.data[0]
                role_info = item.get("roles", {})
                if isinstance(role_info, dict):
                    role = role_info.get("name", "user")
                    permissions = role_info.get("permissions", [])
                elif isinstance(role_info, list) and len(role_info) > 0:
                     role = role_info[0].get("name", "user")
                     permissions = role_info[0].get("permissions", [])
            print(f"[DEBUG] Determined role: {role}, permissions: {permissions}")
        except Exception as e:
            print(f"[WARNING] Could not fetch user role: {str(e)}")

        # Ensure we return valid strings, not None for required fields
        return LoginResponse(
            access_token=str(response.session.access_token),
            refresh_token=str(response.session.refresh_token),
            user_id=str(response.user.id),
            email=str(response.user.email),
            company_id=str(company_id),
            company_name=str(company_name) if company_name else None,
            role=role,
            permissions=permissions
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
    # Fetch profile and company info
    # Joining with company table for the name
    profile_res = supabase_admin.table("profiles") \
        .select("company_id, company(name)") \
        .eq("id", user.id) \
        .single() \
        .execute()
    
    company_id = None
    company_name = None

    if profile_res.data:
        company_id = profile_res.data.get("company_id")
        # Prefer name from company table
        company_name = profile_res.data.get("company", {}).get("name") if profile_res.data.get("company") else None

    # Fetch role and permissions
    role = "user"
    permissions = []
    try:
        role_res = supabase_admin.table("user_roles").select("roles(name, permissions)").eq("user_id", user.id).execute()
        if role_res.data:
            role_info = role_res.data[0].get("roles", {})
            role = role_info.get("name", "user")
            permissions = role_info.get("permissions", [])
    except Exception as e:
        print(f"[WARNING] Could not fetch user role in get_me: {str(e)}")

    return {
        "userId": user.id,
        "email": user.email,
        "companyId": company_id,
        "companyName": company_name,
        "lastSignIn": user.last_sign_in_at,
        "role": role,
        "permissions": permissions
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

@router.post("/refresh", response_model=LoginResponse)
async def refresh_token(data: TokenRefreshRequest):
    """
    Refresh JWT token using a refresh token.
    """
    try:
        response = supabase.auth.refresh_session(data.refresh_token)
        
        if not response.session:
            raise HTTPException(
                status_code=status.HTTP_401_UNAUTHORIZED,
                detail="Invalid or expired refresh token"
            )

        # Get company info from metadata or tables
        company_id = None
        company_name = None
        if hasattr(response.user, 'user_metadata') and response.user.user_metadata:
            company_id = response.user.user_metadata.get("company_id")
        
        if company_id:
            try:
                comp_res = supabase_admin.table("company").select("name").eq("id", company_id).single().execute()
                if comp_res.data:
                    company_name = comp_res.data.get("name")
            except:
                pass

        if not company_id or not company_name:
            try:
                profile_res = supabase_admin.table("profiles") \
                    .select("company_id, company(name)") \
                    .eq("id", response.user.id) \
                    .execute()
                if profile_res.data:
                    profile = profile_res.data[0]
                    company_id = company_id or profile.get("company_id")
                    company_name = company_name or (profile.get("company", {}).get("name") if profile.get("company") else None)
            except Exception as e:
                print(f"[AUTH] Failed to fetch company_info in refresh: {str(e)}")
 
        # Fetch role
        role = "user"
        try:
            role_res = supabase_admin.table("user_roles").select("roles(name)").eq("user_id", response.user.id).execute()
            if role_res.data:
                role = role_res.data[0].get("roles", {}).get("name", "user")
        except Exception as e:
            print(f"[WARNING] Could not fetch user role in refresh: {str(e)}")

        # Ensure we return valid strings, not None for required fields
        return LoginResponse(
            access_token=str(response.session.access_token),
            refresh_token=str(response.session.refresh_token),
            user_id=str(response.user.id),
            email=str(response.user.email),
            company_id=str(company_id) if company_id else "",
            company_name=str(company_name) if company_name else None,
            role=role
        )
    except Exception as e:
        print(f"[ERROR] token refresh failed: {str(e)}")
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail=f"Token refresh failed: {str(e)}"
        )

@router.get("/company/{company_id}")
async def get_company_details(company_id: str, _user=Depends(get_current_user)):
    """
    Get company details by ID.
    Uses supabase_admin to bypass RLS for service-to-service communication.
    """
    try:
        # Using supabase_admin to ensure we can fetch company even with RLS enabled
        res = supabase_admin.table("company").select("*").eq("id", company_id).single().execute()
        if not res.data:
            print(f"[DEBUG] Company {company_id} not found in database, returning placeholder")
            return {"id": company_id, "name": "My Company"}
        return res.data
    except Exception as e:
        # PGRST116 often means 0 rows found when .single() is used
        if "PGRST116" in str(e):
            print(f"[DEBUG] Company {company_id} not found (.single()), returning placeholder")
            return {"id": company_id, "name": "My Company"}
            
        print(f"[ERROR] Failed to fetch company {company_id}: {str(e)}")
        # Return a sensible fallback instead of 404/500 to keep UI happy
        return {"id": company_id, "name": "My Company"}


@router.get("/profile", response_model=ProfileResponse)
async def get_profile(user=Depends(get_current_user)):
    """Get current user's profile info."""
    try:
        res = supabase_admin.table("profiles").select("*").eq("id", user.id).single().execute()
        if not res.data:
            raise HTTPException(status_code=404, detail="Profile not found")
        return res.data
    except Exception as e:
        if isinstance(e, HTTPException): raise e
        raise HTTPException(status_code=500, detail=str(e))


@router.patch("/profile", response_model=ProfileResponse)
async def update_profile(data: ProfileUpdate, user=Depends(get_current_user)):
    """Update current user's profile info."""
    try:
        update_data = data.model_dump(exclude_unset=True)
        if not update_data:
            return await get_profile(user)
        
        res = supabase_admin.table("profiles").update(update_data).eq("id", user.id).execute()
        if not res.data:
            raise HTTPException(status_code=404, detail="Profile not found")
        return res.data[0]
    except Exception as e:
        if isinstance(e, HTTPException): raise e
        raise HTTPException(status_code=500, detail=str(e))


@router.get("/company", response_model=CompanyResponse)
async def get_current_company(user=Depends(get_current_user)):
    """Get current user's company info."""
    try:
        # Get company_id from profile
        profile = supabase_admin.table("profiles").select("company_id").eq("id", user.id).single().execute()
        if not profile.data or not profile.data.get("company_id"):
            raise HTTPException(status_code=400, detail="User has no company assigned")
        
        company_id = profile.data["company_id"]
        res = supabase_admin.table("company").select("*").eq("id", company_id).single().execute()
        if not res.data:
            raise HTTPException(status_code=404, detail="Company not found")
        return res.data
    except Exception as e:
        if isinstance(e, HTTPException): raise e
        raise HTTPException(status_code=500, detail=str(e))


@router.patch("/company", response_model=CompanyResponse)
async def update_current_company(data: CompanyUpdate, user=Depends(get_current_user)):
    """Update current user's company info."""
    try:
        # Get company_id from profile
        profile = supabase_admin.table("profiles").select("company_id").eq("id", user.id).single().execute()
        if not profile.data or not profile.data.get("company_id"):
            raise HTTPException(status_code=400, detail="User has no company assigned")
        
        company_id = profile.data["company_id"]
        update_data = data.model_dump(exclude_unset=True)
        if not update_data:
            return await get_current_company(user)
        
        res = supabase_admin.table("company").update(update_data).eq("id", company_id).execute()
        if not res.data:
            raise HTTPException(status_code=404, detail="Company not found")
        return res.data[0]
    except Exception as e:
        if isinstance(e, HTTPException): raise e
        raise HTTPException(status_code=500, detail=str(e))
