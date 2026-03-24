from typing import Optional, List
from .base import CamelModel

class UserLogin(CamelModel):
    email: str
    password: str

class UserRegister(CamelModel):
    email: str
    password: str
    first_name: str
    last_name: str
    company_name: str

class LoginResponse(CamelModel):
    access_token: str
    refresh_token: str
    token_type: str = "bearer"
    user_id: str
    email: str
    company_id: str
    company_name: Optional[str] = None
    first_name: Optional[str] = None
    last_name: Optional[str] = None
    role: str = "user"
    permissions: List[str] = []

class TokenRefreshRequest(CamelModel):
    refresh_token: str

class ForgotPasswordRequest(CamelModel):
    email: str

class ResetPasswordRequest(CamelModel):
    new_password: str
