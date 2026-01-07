import os
from supabase import create_client, Client
from app.core.config import settings

# Initialize Supabase client
# Using Service Role Key for Admin operations (User CRUD, Role Assignment)
supabase: Client = create_client(settings.supabase_url, settings.supabase_anon_key)
supabase_admin: Client = create_client(settings.supabase_url, settings.supabase_service_role_key)
