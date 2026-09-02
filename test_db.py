import asyncio
from app.core.supabase import supabase_admin

async def main():
    res = supabase_admin.table("demo_requests").select("*").execute()
    print(res.data)

asyncio.run(main())
