from typing import Optional
from .base import CamelModel

class UserLogin(CamelModel):
    email: str
    password: str

class LoginResponse(CamelModel):
    access_token: str
    token_type: str = "bearer"
    user_id: str
    email: str
    company_id: str

class ForgotPasswordRequest(CamelModel):
    email: str

class ResetPasswordRequest(CamelModel):
    new_password: str
