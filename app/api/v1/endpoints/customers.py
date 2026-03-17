from fastapi import APIRouter, UploadFile, File, Form, HTTPException, Depends, BackgroundTasks
from typing import List, Optional
from app.api.deps import get_current_user
from app.core.supabase import supabase, supabase_admin
from app.api.v1.endpoints.documents import _process_document_background
from app.integrations.storage.s3_storage import s3_storage
from app.schemas.customer import CustomerResponse, CustomerCreate
import io
from uuid import UUID
import re


router = APIRouter()

@router.post("/", response_model=dict)
async def create_customer(
    background_tasks: BackgroundTasks,
    name: str = Form(...),
    aadhar_number: str = Form(...),
    pan_number: str = Form(...),
    business_type: str = Form(None),
    aadhar: UploadFile = File(...),
    pan: UploadFile = File(...),
    user=Depends(get_current_user)
):
    """
    Create a new customer (in customers table), a root folder, and upload mandatory documents.
    """
    # Validate formats
    if not re.match(r"^\d{12}$", aadhar_number):
        raise HTTPException(status_code=400, detail="Invalid Aadhar Number format. Must be 12 digits.")
    
    if not re.match(r"^[A-Z]{5}[0-9]{4}[A-Z]{1}$", pan_number.upper()):
        raise HTTPException(status_code=400, detail="Invalid PAN Number format. Must be 10 characters (e.g. ABCDE1234F).")

    try:
        # Step 0: Get user's company_id
        profile_res = supabase.table("profiles").select("company_id").eq("id", user.id).single().execute()
        if not profile_res.data or not profile_res.data.get("company_id"):
            raise HTTPException(status_code=400, detail="User profile or company assignment missing")
        
        company_id = profile_res.data["company_id"]

        # Step 1: Create record in 'customers' table
        # Check if customer already exists in 'customers' table by NAME
        existing_cust_name = supabase.table("customers") \
            .select("id") \
            .eq("name", name) \
            .eq("company_id", company_id) \
            .is_("deleted_at", "null") \
            .execute()
        
        if existing_cust_name.data:
            raise HTTPException(status_code=400, detail=f"Customer with name '{name}' already exists")

        # Check for duplicate Aadhar or PAN
        existing_cust_docs = supabase.table("customers") \
            .select("id, name") \
            .eq("company_id", company_id) \
            .is_("deleted_at", "null") \
            .or_(f"aadhar_number.eq.{aadhar_number},pan_number.eq.{pan_number}") \
            .execute()

        if existing_cust_docs.data:
            # Determine which one matched
            matched = existing_cust_docs.data[0]
            raise HTTPException(status_code=400, detail=f"Customer '{matched['name']}' already exists with this Aadhar or PAN number")

        cust_data = {
            "name": name,
            "aadhar_number": aadhar_number,
            "pan_number": pan_number,
            "company_id": company_id,
            "created_by": user.id,
            "business_type": business_type
        }
        # Using supabase_admin to bypass RLS for writes
        cust_res = supabase_admin.table("customers").insert(cust_data).execute()
        if not cust_res.data:
            raise HTTPException(status_code=500, detail="Failed to create customer record")
        
        customer_id = cust_res.data[0]["id"]
        customer_name = cust_res.data[0]["name"]

        customer_id = cust_res.data[0]["id"]
        customer_name = cust_res.data[0]["name"]

        # Step 2: Upload Aadhar and PAN (No automatic root folder in 'folders' table)

        # Step 3: Upload Aadhar and PAN
        docs_to_upload = [
            ("Aadhar Card", aadhar),
            ("PAN Card", pan)
        ]

        uploaded_files = []
        for label, upload_file in docs_to_upload:
            try:
                content = await upload_file.read()
                # Clean filename to avoid issues
                clean_name = "".join(c for c in upload_file.filename if c.isalnum() or c in "._- ")
                filename = f"{label}_{clean_name}"
                
                print(f"[DEBUG] Processing mandatory doc: {filename}")

                # S3 Upload
                s3_key = await s3_storage.upload_file(
                    file_content=content,
                    filename=filename,
                    folder_name=f"customers/{customer_id}", # Consistent S3 path
                    userId=user.id
                )
                s3_url = s3_storage.get_static_url(s3_key)
                
                print(f"[DEBUG] Uploaded to S3: {s3_key}")

                # DB Record - Using 'customer_docs' table instead of 'files'
                doc_record = {
                    "customer_id": customer_id,
                    "doc_type": label,
                    "filename": filename,
                    "s3_key": s3_key,
                    "s3_url": s3_url
                }
                doc_db_res = supabase_admin.table("customer_docs").insert(doc_record).execute()
                
                if doc_db_res.data:
                    # Optional: Background processing for OCR/RAG can still happen if required
                    # But for now we just record it in the dedicated table
                    uploaded_files.append(doc_db_res.data[0])
                    print(f"[DEBUG] Customer doc record created: {filename}")
                else:
                    print(f"[ERROR] Failed to insert customer doc record for {filename}: {doc_db_res}")

            except Exception as file_e:
                print(f"[ERROR] Failed to upload/record file {label}: {str(file_e)}")
                if hasattr(file_e, 'message'):
                    print(f"[ERROR] File Error Detail: {file_e.message}")

        return {
            "message": "Customer created successfully and documents stored in dedicated table",
            "customer": cust_res.data[0],
            "docs": uploaded_files
        }

    except Exception as e:
        print(f"[ERROR] Error creating customer: {str(e)}")
        if hasattr(e, 'message'):
            print(f"[ERROR] Supabase Error Detail: {e.message}")
        if isinstance(e, HTTPException):
            raise e
        raise HTTPException(status_code=500, detail=str(e))

@router.get("/", response_model=List[CustomerResponse])
async def list_customers(user=Depends(get_current_user)):
    """
    List all customers for the user's company from the dedicated table.
    """
    try:
        profile_res = supabase.table("profiles").select("company_id").eq("id", user.id).single().execute()
        if not profile_res.data or not profile_res.data.get("company_id"):
            raise HTTPException(status_code=400, detail="User profile or company assignment missing")
        
        company_id = profile_res.data["company_id"]

        result = supabase.table("customers") \
            .select("*") \
            .eq("company_id", company_id) \
            .is_("deleted_at", "null") \
            .order("name") \
            .execute()
        
        return result.data or []
    except Exception as e:
        print(f"[ERROR] Error listing customers: {str(e)}")
        raise HTTPException(status_code=500, detail=str(e))

@router.get("/{customer_id}/docs")
async def get_customer_docs(customer_id: UUID, user=Depends(get_current_user)):
    """
    List mandatory documents (Aadhar, PAN) for a specific customer.
    """
    try:
        # Check if customer exists and belongs to company (Security)
        profile_res = supabase.table("profiles").select("company_id").eq("id", user.id).single().execute()
        company_id = profile_res.data["company_id"]

        cust_check = supabase.table("customers") \
            .select("id") \
            .eq("id", str(customer_id)) \
            .eq("company_id", company_id) \
            .execute()
        
        if not cust_check.data:
            raise HTTPException(status_code=404, detail="Customer not found")

        result = supabase.table("customer_docs") \
            .select("*") \
            .eq("customer_id", str(customer_id)) \
            .order("created_at", desc=True) \
            .execute()
        
        return result.data or []
    except Exception as e:
        print(f"[ERROR] Error fetching customer docs: {str(e)}")
        raise HTTPException(status_code=500, detail=str(e))

@router.get("/{customer_id}", response_model=CustomerResponse)
async def get_customer(customer_id: UUID, user=Depends(get_current_user)):
    """
    Get details of a single customer.
    """
    try:
        profile_res = supabase.table("profiles").select("company_id").eq("id", user.id).single().execute()
        company_id = profile_res.data["company_id"]

        result = supabase.table("customers") \
            .select("*") \
            .eq("id", str(customer_id)) \
            .eq("company_id", company_id) \
            .is_("deleted_at", "null") \
            .single() \
            .execute()
        
        if not result.data:
            raise HTTPException(status_code=404, detail="Customer not found")
            
        return result.data
    except Exception as e:
        print(f"[ERROR] Error fetching customer: {str(e)}")
        raise HTTPException(status_code=500, detail=str(e))

@router.get("/{customer_id}/folders", response_model=List[dict])
async def list_customer_folders(customer_id: UUID, user=Depends(get_current_user)):
    """
    List all root-level folders belonging to a specific customer.
    """
    try:
        profile_res = supabase.table("profiles").select("company_id").eq("id", user.id).single().execute()
        if not profile_res.data or not profile_res.data.get("company_id"):
            raise HTTPException(status_code=400, detail="User profile or company assignment missing")
        
        company_id = profile_res.data["company_id"]

        # Fetch folders for this customer
        folders_res = supabase.table("folders") \
            .select("*") \
            .eq("customer_id", str(customer_id)) \
            .eq("company_id", company_id) \
            .is_("parent_id", "null") \
            .is_("deleted_at", "null") \
            .order("name") \
            .execute()
        
        folders = folders_res.data or []

        if not folders:
            return []

        # Fetch all files for this company to count them
        files_res = supabase.table("files") \
            .select("folder_id, size") \
            .eq("company_id", company_id) \
            .is_("deleted_at", "null") \
            .execute()
        
        files = files_res.data or []

        # Aggregate counts and sizes
        folder_stats = {}
        for f in files:
            f_id = f.get("folder_id")
            if f_id:
                if f_id not in folder_stats:
                    folder_stats[f_id] = {"count": 0, "size": 0}
                folder_stats[f_id]["count"] += 1
                folder_stats[f_id]["size"] += (f.get("size") or 0)

        # Attach stats to folders
        for folder in folders:
            stats = folder_stats.get(folder["id"], {"count": 0, "size": 0})
            folder["file_count"] = stats["count"]
            folder["total_size"] = stats["size"]
        
        return folders
    except Exception as e:
        print(f"[ERROR] Error fetching customer folders: {str(e)}")
        raise HTTPException(status_code=500, detail=str(e))

@router.delete("/{customer_id}")
async def delete_customer(customer_id: UUID, user=Depends(get_current_user)):
    """
    Soft delete a customer.
    """
    try:
        from datetime import datetime, timezone
        now = datetime.now(timezone.utc).isoformat()
        
        # Check if customer belongs to company
        profile_res = supabase.table("profiles").select("company_id").eq("id", user.id).single().execute()
        company_id = profile_res.data["company_id"]

        result = supabase.table("customers") \
            .update({"deleted_at": now}) \
            .eq("id", str(customer_id)) \
            .eq("company_id", company_id) \
            .execute()
        
        if not result.data:
            raise HTTPException(status_code=404, detail="Customer not found or access denied")
            
        print(f"[INFO] Customer {customer_id} soft-deleted at {now}")
        return {"message": "Customer soft-deleted successfully"}
    except Exception as e:
        print(f"[ERROR] Error deleting customer: {str(e)}")
        if isinstance(e, HTTPException):
            raise e
        raise HTTPException(status_code=500, detail=str(e))
