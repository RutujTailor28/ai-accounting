from fastapi import APIRouter, UploadFile, File, Form, HTTPException, Depends, BackgroundTasks
from typing import BinaryIO, Optional
import io
from app.schemas.document import UploadResponse
from app.api.deps import get_current_user
from app.integrations.ocr.pdf_parser import DocumentParser
from app.integrations.storage.s3_storage import s3_storage
from app.services.text_chunker import TextChunker
from app.services.deps import embedding_service
from app.ai.rag.retriever import vector_store
import os
import shutil

from app.core.supabase import supabase, supabase_admin
from app.schemas.folder import FileResponse
from typing import List
from datetime import datetime, timezone
import re
import gc

router = APIRouter()

document_parser = DocumentParser()
text_chunker = TextChunker()

async def _process_document_background(
    file_content: bytes,
    filename: str,
    company_id: str,
    folder_name: str,
    target_folder_id: str,
    user_id: str,
    s3_key: str,
    file_id: str = None
):
    """
    Background worker to parse, chunk, and embed a document.
    Ensures the UI doesn't time out during heavy processing.
    """
    try:
        print(f"[BG-TASK] Starting background processing for: {filename}")
        file_obj = io.BytesIO(file_content)

        text = document_parser.parse(file_obj, filename)
        if not text or not text.strip():
            print(f"[BG-TASK][ERROR] No text extracted from {filename}")
            return

        content_date_ts = None
        date_patterns = [
            r'\b(\d{4})-(0[1-9]|1[0-2])-(0[1-9]|[12]\d|3[01])\b',
            r'\b(0[1-9]|[12]\d|3[01])-(0[1-9]|1[0-2])-(\d{4})\b',
            r'\b(0[1-9]|[12]\d|3[01])/(0[1-9]|1[0-2])/(\d{4})\b'
        ]

        for pattern in date_patterns:
            match = re.search(pattern, text)
            if match:
                try:
                    date_str = match.group(0)
                    if '-' in date_str: parts = date_str.split('-')
                    else: parts = date_str.split('/')
                    if len(parts[0]) == 4: dt = datetime(int(parts[0]), int(parts[1]), int(parts[2]))
                    else: dt = datetime(int(parts[2]), int(parts[1]), int(parts[0]))
                    content_date_ts = dt.timestamp()
                    break
                except: continue

        if not content_date_ts:
            content_date_ts = datetime.now(timezone.utc).timestamp()

        chunks = text_chunker.chunk_text(text)
        if not chunks:
            print(f"[BG-TASK] No chunks created for {filename}")
            return

        total_chunks = len(chunks)
        batch_size = 50
        print(f"[BG-TASK] Processing {total_chunks} chunks in batches of {batch_size}...")

        now_ts = datetime.now(timezone.utc).timestamp()

        for i in range(0, total_chunks, batch_size):
            batch_chunks = chunks[i:i + batch_size]
            batch_embeddings = embedding_service.generate_embeddings(batch_chunks)

            batch_metadatas = [
                {
                    "company_id": company_id,
                    "folder_name": folder_name,
                    "folder_id": target_folder_id,
                    "document_name": filename,
                    "file_type": filename.split('.')[-1].lower() if '.' in filename else 'unknown',
                    "created_by": user_id,
                    "created_at": now_ts,
                    "content_date": content_date_ts,
                    "chunk_index": i + j,
                    "file_id": file_id
                }
                for j in range(len(batch_chunks))
            ]

            vector_store.add_documents(
                texts=batch_chunks,
                embeddings=batch_embeddings,
                metadatas=batch_metadatas
            )

            print(f"[BG-TASK] Processed batch {i//batch_size + 1}/{(total_chunks-1)//batch_size + 1} ({len(batch_chunks)} chunks)")

            batch_embeddings = None
            batch_chunks = None
            batch_metadatas = None
            gc.collect()

        print(f"[BG-TASK] DONE: Successfully processed {filename} ({total_chunks} chunks in total).")

    except Exception as e:
        print(f"[BG-TASK][ERROR] Critical failure for {filename}: {str(e)}")
        import traceback
        traceback.print_exc()

@router.get("/all", response_model=List[FileResponse])
async def list_all_files(user=Depends(get_current_user)):
    """List all files for the authenticated user's company."""
    try:

        profile_res = supabase.table("profiles").select("company_id").eq("id", user.id).single().execute()
        if not profile_res.data or not profile_res.data.get("company_id"):
            raise HTTPException(status_code=400, detail="User profile or company assignment missing")

        company_id = profile_res.data["company_id"]

        result = supabase.table("files") \
            .select("*") \
            .eq("company_id", company_id) \
            .is_("deleted_at", "null") \
            .order("created_at", desc=True) \
            .execute()

        files = result.data or []
        for f in files:
            if f.get("s3_key"):

                f["s3_url"] = s3_storage.generate_presigned_url(f["s3_key"], expires_in=3600)

        return files
    except Exception as e:
        print(f"[ERROR] Error listing all files: {str(e)}")
        raise HTTPException(status_code=500, detail=str(e))

@router.post("/upload", response_model=UploadResponse)
async def upload_document(
    background_tasks: BackgroundTasks,
    folder_name: str = Form(..., alias="folderName"),
    parent_id: Optional[str] = Form(None, alias="parentId"),
    customer_id: Optional[str] = Form(None, alias="customerId"),
    file: UploadFile = File(...),
    user=Depends(get_current_user)
):
    """
    Upload and process a document for RAG system with Folder management.
    """
    try:

        profile_res = supabase.table("profiles").select("company_id").eq("id", user.id).single().execute()
        if not profile_res.data or not profile_res.data.get("company_id"):
            raise HTTPException(status_code=400, detail="User profile or company assignment missing")

        company_id = profile_res.data["company_id"]

        if parent_id and (parent_id.lower() == "null" or parent_id.strip() == ""):
            parent_id = None

        if customer_id and (customer_id.lower() == "null" or customer_id.strip() == ""):
            customer_id = None

        print(f"[INFO] Received upload request: file={file.filename}, folder={folder_name}, customer={customer_id}, company_id={company_id}")

        if not file.filename:
            raise HTTPException(status_code=400, detail="Filename is required")

        extension = f".{file.filename.split('.')[-1].lower()}"
        if extension not in DocumentParser.SUPPORTED_EXTENSIONS:
            raise HTTPException(
                status_code=400,
                detail=f"Unsupported file type. Supported types: {', '.join(DocumentParser.SUPPORTED_EXTENSIONS)}"
            )

        file_content = await file.read()
        print(f"[INFO] File read complete: {len(file_content)} bytes")
        file_obj = io.BytesIO(file_content)

        print(f"[INFO] Step 1: Resolving folder '{folder_name}' (parent_id: {parent_id}, customer_id: {customer_id})...")

        target_folder_id = None

        if parent_id:
            parent_res = supabase.table("folders").select("name").eq("id", parent_id).single().execute()
            if parent_res.data and parent_res.data.get("name") == folder_name:
                print(f"[INFO] Parent folder name matches target. Using parent folder directly: {parent_id}")
                target_folder_id = parent_id

        if not target_folder_id:

            query = supabase.table("folders").select("id").eq("name", folder_name).eq("company_id", company_id)
            if parent_id:
                query = query.eq("parent_id", parent_id)
            else:
                query = query.is_("parent_id", "null")

            if customer_id:
                query = query.eq("customer_id", customer_id)
            else:
                query = query.is_("customer_id", "null")

            check_folder = query.execute()

            if check_folder.data:
                target_folder_id = check_folder.data[0]['id']
                print(f"[INFO] Using existing folder: {folder_name} (ID: {target_folder_id})")
            else:

                print(f"[INFO] Folder not found. Creating new folder: {folder_name}")
                new_folder_data = {
                    "name": folder_name,
                    "company_id": company_id,
                    "parent_id": parent_id,
                    "customer_id": customer_id,
                    "created_by": user.id if hasattr(user, 'id') else None
                }
                new_folder = supabase_admin.table("folders").insert(new_folder_data).execute()
                if not new_folder.data:
                    raise HTTPException(status_code=500, detail="Failed to create folder record")
                target_folder_id = new_folder.data[0]['id']
                print(f"[INFO] Created new folder with ID: {target_folder_id}")

        print(f"[INFO] Step 2: Uploading to S3 path: aiAccounting/{folder_name}/{file.filename}")
        s3_key = await s3_storage.upload_file(
            file_content=file_content,
            filename=file.filename,
            folder_name=folder_name,
            userId=user.id if hasattr(user, 'id') else None
        )
        s3_url = s3_storage.get_static_url(s3_key) if s3_key else None
        print(f"[INFO] DONE: Permanent S3 URL stored: {s3_url}")

        print(f"[INFO] Step 3: Saving file record to database...")
        file_record = {
            "name": file.filename,
            "folder_id": target_folder_id,
            "s3_key": s3_key,
            "s3_url": s3_url,
            "file_type": file.filename.split('.')[-1].lower() if '.' in file.filename else 'unknown',
            "company_id": company_id,
            "size": len(file_content),
            "created_by": user.id if hasattr(user, 'id') else None
        }

        file_db_res = supabase_admin.table("files").insert(file_record).execute()
        if not file_db_res.data:
            print(f"[ERROR] Failed to save file record: {file_db_res}")
            raise HTTPException(status_code=500, detail="Failed to save file metadata to database")

        print(f"[INFO] DONE: File record saved to database (Size: {len(file_content)} bytes).")

        background_tasks.add_task(
            _process_document_background,
            file_content=file_content,
            filename=file.filename,
            company_id=company_id,
            folder_name=folder_name,
            target_folder_id=target_folder_id,
            user_id=user.id if hasattr(user, 'id') else None,
            s3_key=s3_key,
            file_id=file_db_res.data[0]['id']
        )

        return UploadResponse(
            message="Document uploaded. AI processing started in background.",
            document_name=file.filename,
            chunks_created=0,
            company_id=company_id,
            s3_key=s3_key or "",
            s3_url=s3_url or ""
        )

    except HTTPException:
        raise
    except Exception as e:
        print(f"[ERROR] Error processing upload: {str(e)}")
        raise HTTPException(status_code=500, detail=f"Internal server error: {str(e)}")

@router.delete("/reset", response_model=dict)
async def reset_vector_db(user=Depends(get_current_user)):
    """
    ADMIN: Completely reset the vector database (ChromaDB).
    This removes ALL documents from the AI's memory.
    """
    print(f"[WARNING] User {user.id} requested Vector DB reset")
    success = vector_store.reset_collection()

    if not success:
        raise HTTPException(status_code=500, detail="Failed to reset vector database")

    return {"message": "Vector database has been successfully reset. All AI memory is cleared."}

@router.get("/{file_id}", response_model=FileResponse)
async def get_document(
    file_id: str,
    user=Depends(get_current_user)
):
    """
    Get detailed metadata for a specific document.
    """
    try:

        profile_res = supabase.table("profiles").select("company_id").eq("id", user.id).single().execute()
        if not profile_res.data or not profile_res.data.get("company_id"):
            raise HTTPException(status_code=400, detail="User profile or company assignment missing")

        company_id = profile_res.data["company_id"]

        result = supabase.table("files") \
            .select("*") \
            .eq("id", file_id) \
            .eq("company_id", company_id) \
            .is_("deleted_at", "null") \
            .execute()

        if not result.data:
            raise HTTPException(status_code=404, detail="File not found or access denied")

        file_data = result.data[0]

        if file_data.get("s3_key"):
            file_data["s3_url"] = s3_storage.generate_presigned_url(file_data["s3_key"], expires_in=3600)

        return file_data

    except HTTPException:
        raise
    except Exception as e:
        print(f"[ERROR] Error fetching document metadata: {str(e)}")
        raise HTTPException(status_code=500, detail=f"Internal server error: {str(e)}")

@router.get("/{file_id}/url", response_model=dict)
async def get_document_url(
    file_id: str,
    expires_in: int = 3600,
    user=Depends(get_current_user)
):
    """
    Generate a fresh temporary signed URL for a document.
    """
    try:

        profile_res = supabase.table("profiles").select("company_id").eq("id", user.id).single().execute()
        if not profile_res.data or not profile_res.data.get("company_id"):
            raise HTTPException(status_code=400, detail="User profile or company assignment missing")

        company_id = profile_res.data["company_id"]

        result = supabase.table("files") \
            .select("*") \
            .eq("id", file_id) \
            .eq("company_id", company_id) \
            .execute()

        if not result.data:
            raise HTTPException(status_code=404, detail="File not found or access denied")

        file_data = result.data[0]

        profile_res = supabase.table("profiles") \
            .select("company_id") \
            .eq("id", file_data.get("created_by")) \
            .single() \
            .execute()

        s3_key = file_data.get("s3_key")

        if not s3_key:
            raise HTTPException(status_code=404, detail="S3 key not found for this file")

        url = s3_storage.generate_presigned_url(s3_key, expires_in=expires_in)

        if not url:
            raise HTTPException(status_code=500, detail="Failed to generate presigned URL")

        return {
            "fileId": file_id,
            "documentName": file_data.get("name"),
            "presignedUrl": url,
            "expiresIn": expires_in
        }

    except HTTPException:
        raise
    except Exception as e:
        print(f"[ERROR] Error generating document URL: {str(e)}")
        raise HTTPException(status_code=500, detail=f"Internal server error: {str(e)}")

@router.delete("/{file_id}", response_model=dict)
async def delete_document(
    file_id: str,
    user=Depends(get_current_user)
):
    """
    Delete a document from:
    1. Database (files table)
    2. S3 Storage
    3. AI Vector Database (ChromaDB)
    """
    try:
        print(f"[INFO] Request to delete file {file_id} by user {user.id}")

        profile_res = supabase.table("profiles").select("company_id").eq("id", user.id).single().execute()
        if not profile_res.data or not profile_res.data.get("company_id"):
            raise HTTPException(status_code=400, detail="User profile or company assignment missing")

        company_id = profile_res.data["company_id"]

        result = supabase.table("files") \
            .select("*") \
            .eq("id", file_id) \
            .eq("company_id", company_id) \
            .execute()

        if not result.data:
            raise HTTPException(status_code=404, detail="File not found or access denied")

        file_data = result.data[0]
        s3_key = file_data.get("s3_key")
        doc_name = file_data.get("name")

        print(f"[INFO] Step 1: Soft-deleting record from Database: {file_id}")
        from datetime import datetime, timezone
        now = datetime.now(timezone.utc).isoformat()

        db_result = supabase_admin.table("files") \
            .update({"deleted_at": now}) \
            .eq("id", file_id) \
            .execute()

        print(f"[INFO] Step 2: Removing document from Vector Store: {doc_name}")
        vector_store.delete_document(company_id, doc_name)

        return {
            "message": "File soft-deleted successfully",
            "details": {
                "database": "Updated deleted_at",
                "vector_store": "Removed chunks"
            }
        }
    except HTTPException:
        raise
    except Exception as e:
        print(f"[ERROR] Error deleting file: {str(e)}")
        raise HTTPException(status_code=500, detail=f"Internal server error: {str(e)}")
