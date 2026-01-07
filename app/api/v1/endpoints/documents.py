from fastapi import APIRouter, UploadFile, File, Form, HTTPException, Depends
from typing import BinaryIO, Optional
import io
from app.schemas.document import UploadResponse
from app.api.deps import get_current_user
from app.integrations.ocr.pdf_parser import DocumentParser
from app.integrations.storage.s3_storage import s3_storage
from app.services.text_chunker import TextChunker
from app.services.embedding_service import EmbeddingService
from app.ai.rag.retriever import vector_store
import os
import shutil

router = APIRouter()

# Initialize services
document_parser = DocumentParser()
text_chunker = TextChunker()
embedding_service = EmbeddingService()


from app.core.supabase import supabase, supabase_admin

@router.post("/upload", response_model=UploadResponse)
async def upload_document(
    folder_name: str = Form(..., alias="folderName"),
    parent_id: Optional[str] = Form(None, alias="parentId"),
    file: UploadFile = File(...),
    user=Depends(get_current_user)
):
    """
    Upload and process a document for RAG system with Folder management.
    """
    try:
        # Step 0: Get user's company_id from profile (Security enforcement)
        profile_res = supabase.table("profiles").select("company_id").eq("id", user.id).single().execute()
        if not profile_res.data or not profile_res.data.get("company_id"):
            raise HTTPException(status_code=400, detail="User profile or company assignment missing")
        
        company_id = profile_res.data["company_id"]

        # Sanitize parent_id: "null" string or empty string should be None
        if parent_id and (parent_id.lower() == "null" or parent_id.strip() == ""):
            parent_id = None

        print(f"[INFO] Received upload request: file={file.filename}, folder={folder_name}, company_id={company_id}")
        
        # Validate file extension
        if not file.filename:
            raise HTTPException(status_code=400, detail="Filename is required")
        
        extension = f".{file.filename.split('.')[-1].lower()}"
        if extension not in DocumentParser.SUPPORTED_EXTENSIONS:
            raise HTTPException(
                status_code=400,
                detail=f"Unsupported file type. Supported types: {', '.join(DocumentParser.SUPPORTED_EXTENSIONS)}"
            )

        # Read file content
        file_content = await file.read()
        print(f"[INFO] File read complete: {len(file_content)} bytes")
        file_obj = io.BytesIO(file_content)
        
        # Step 1: Resolve Folder
        print(f"[INFO] Step 1: Resolving folder '{folder_name}' (parent_id: {parent_id})...")
        
        target_folder_id = None
        
        # Folder Refinement Logic:
        # If parent_id is provided, check if the parent folder's name matches the target folder_name
        if parent_id:
            parent_res = supabase.table("folders").select("name").eq("id", parent_id).single().execute()
            if parent_res.data and parent_res.data.get("name") == folder_name:
                print(f"[INFO] Parent folder name matches target. Using parent folder directly: {parent_id}")
                target_folder_id = parent_id
        
        if not target_folder_id:
            # Look for existing folder with same name and parent
            # If parent_id is None, it's a root folder
            query = supabase.table("folders").select("id").eq("name", folder_name).eq("company_id", company_id)
            if parent_id:
                query = query.eq("parent_id", parent_id)
            else:
                query = query.is_("parent_id", "null")
            
            check_folder = query.execute()
            
            if check_folder.data:
                target_folder_id = check_folder.data[0]['id']
                print(f"[INFO] Using existing folder: {folder_name} (ID: {target_folder_id})")
            else:
                # Create it
                print(f"[INFO] Folder not found. Creating new folder: {folder_name}")
                new_folder_data = {
                    "name": folder_name,
                    "company_id": company_id,
                    "parent_id": parent_id,
                    "created_by": user.id if hasattr(user, 'id') else None
                }
                new_folder = supabase.table("folders").insert(new_folder_data).execute()
                if not new_folder.data:
                    raise HTTPException(status_code=500, detail="Failed to create folder record")
                target_folder_id = new_folder.data[0]['id']
                print(f"[INFO] Created new folder with ID: {target_folder_id}")

        # Step 2: Upload to S3
        print(f"[INFO] Step 2: Uploading to S3 path: aiAccounting/{folder_name}/{file.filename}")
        s3_key = await s3_storage.upload_file(
            file_content=file_content,
            filename=file.filename,
            folder_name=folder_name,
            userId=user.id if hasattr(user, 'id') else None
        )
        s3_url = s3_storage.get_static_url(s3_key) if s3_key else None
        print(f"[INFO] DONE: Permanent S3 URL stored: {s3_url}")
        
        # Step 3: Save File Record (DB)
        print(f"[INFO] Step 3: Saving file record to database...")
        file_record = {
            "name": file.filename,
            "folder_id": target_folder_id,
            "s3_key": s3_key,
            "s3_url": s3_url,
            "file_type": file.filename.split('.')[-1].lower() if '.' in file.filename else 'unknown',
            "company_id": company_id,
            "uploaded_by": user.id if hasattr(user, 'id') else None
        }
        
        file_db_res = supabase.table("files").insert(file_record).execute()
        print(f"[INFO] DONE: File record saved to database.")

        # Step 4: Parse document
        print(f"[INFO] Step 4: Parsing document...")
        text = document_parser.parse(file_obj, file.filename)
        
        if not text or not text.strip():
            raise HTTPException(status_code=400, detail="No text could be extracted from the document")
        print(f"[INFO] DONE: Document parsed successfully ({len(text)} characters).")
        
        # Step 5: Chunk text
        print(f"[INFO] Step 5: Chunking text...")
        chunks = text_chunker.chunk_text(text)
        
        if not chunks:
            raise HTTPException(status_code=400, detail="Failed to create text chunks")
        print(f"[INFO] DONE: Text split into {len(chunks)} chunks.")
        
        print(f"[INFO] Step 6: Generating embeddings...")
        embeddings = embedding_service.generate_embeddings(chunks)
        print(f"[INFO] DONE: Generated embeddings for {len(embeddings)} chunks.")
        
        # Step 7: Prepare metadata for Vector Store
        metadatas = [
            {
                "company_id": company_id,
                "folder_name": folder_name,
                "document_name": file.filename,
                "chunk_index": i
            }
            for i in range(len(chunks))
        ]
        
        
        # Step 8: Store in vector database
        print(f"[INFO] Step 8: Storing in vector database...")
        vector_store.add_documents(
            texts=chunks,
            embeddings=embeddings,
            metadatas=metadatas
        )
        print(f"[INFO] DONE: Chunks successfully stored in Vector Database.")
        
        print(f"[INFO] Successfully processed document: {file.filename} ({len(chunks)} chunks) for group={folder_name}")
        
        return UploadResponse(
            message="Document uploaded and processed successfully",
            document_name=file.filename,
            chunks_created=len(chunks),
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
        # Step 0: Get user's company_id
        profile_res = supabase.table("profiles").select("company_id").eq("id", user.id).single().execute()
        if not profile_res.data or not profile_res.data.get("company_id"):
            raise HTTPException(status_code=400, detail="User profile or company assignment missing")
        
        company_id = profile_res.data["company_id"]

        # 1. Get File Metadata and verify ownership
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
            .eq("id", file_data.get("uploaded_by")) \
            .single() \
            .execute()
            
        s3_key = file_data.get("s3_key")
        
        if not s3_key:
            raise HTTPException(status_code=404, detail="S3 key not found for this file")
            
        # 2. Generate Presigned URL
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
        
        # Step 0: Get user's company_id
        profile_res = supabase.table("profiles").select("company_id").eq("id", user.id).single().execute()
        if not profile_res.data or not profile_res.data.get("company_id"):
            raise HTTPException(status_code=400, detail="User profile or company assignment missing")
        
        company_id = profile_res.data["company_id"]

        # 1. Get File Metadata and verify ownership
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
        
        # 1. Delete from S3
        if s3_key:
            print(f"[INFO] Deleting from S3: {s3_key}")
            s3_storage.delete_file(s3_key)
            
        # 2. Delete from Vector Store
        if company_id and doc_name:
            print(f"[INFO] Deleting from Vector Store: {doc_name} for company {company_id}")
            vector_store.delete_document(company_id=company_id, document_name=doc_name)
        
        # 3. Delete from Database
        print(f"[INFO] Step 3: Deleting record from Database: {file_id}")
        # Use supabase_admin to bypass RLS if no DELETE policy is set for admin key
        db_result = supabase_admin.table("files").delete().eq("id", file_id).execute()
        
        # Check if any rows were actually deleted
        if hasattr(db_result, 'data') and not db_result.data:
            print(f"[WARNING] No rows deleted from database for file_id: {file_id} even with Admin. Checking existence...")
            check = supabase_admin.table("files").select("id").eq("id", file_id).execute()
            print(f"[INFO] Record exist check: {check.data}")
        else:
            print(f"[INFO] DONE: File record deleted from database.")

        return {
            "message": "File deletion process completed",
            "details": {
                "s3": "Deleted" if s3_key else "Skipped (No Key)",
                "vector_db": "Deleted",
                "database": "Deleted" if db_result.data else "Failed to delete (Row not found or RLS)"
            }
        }
        
    except HTTPException:
        raise
    except Exception as e:
        print(f"[ERROR] Error deleting file: {str(e)}")
        raise HTTPException(status_code=500, detail=f"Internal server error: {str(e)}")
