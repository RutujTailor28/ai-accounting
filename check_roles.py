import asyncio
from app.core.supabase import supabase_admin

async def check_roles(user_id: str):
    res = supabase_admin.table("user_roles").select("*, roles(name)").eq("user_id", user_id).execute()
    print("User Roles:", res.data)
    
if __name__ == "__main__":
    asyncio.run(check_roles("7c602240-c09c-469e-9907-95ded1abbab7"))
