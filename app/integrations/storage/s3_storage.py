"""S3 storage service for document uploads."""
import boto3
from botocore.exceptions import ClientError, BotoCoreError
from typing import Optional
import uuid
from datetime import datetime
from app.core.config import settings


class S3Storage:
    """Service for storing and retrieving files from AWS S3."""
    
    def __init__(self):
        """Initialize S3 client."""
        if not settings.s3_bucket_name:
            print(f"[WARNING] S3 bucket name not configured. S3 storage will be disabled.")
            self.client = None
            return
        
        try:
            # Initialize S3 client with credentials from settings
            self.client = boto3.client(
                's3',
                aws_access_key_id=settings.aws_access_key_id,
                aws_secret_access_key=settings.aws_secret_access_key,
                region_name=settings.aws_region
            )
            self.bucket_name = settings.s3_bucket_name
            print(f"[INFO] S3Storage initialized with bucket: {self.bucket_name}")
        except Exception as e:
            print(f"[ERROR] Failed to initialize S3 client: {str(e)}")
            self.client = None
    
    async def upload_file(
        self,
        file_content: bytes,
        filename: str,
        folder_name: str,
        userId: Optional[str] = None
    ) -> Optional[str]:
        """
        Upload a file to S3.
        
        Args:
            file_content: Binary file content
            filename: Original filename
            folder_name: Folder name for storage path
            userId: Optional user ID for metadata
            
        Returns:
            S3 object key (path) if successful, None otherwise
            
        file structure: aiAccounting/{folder_name}/{filename}
        """
        if not self.client:
            print(f"[WARNING] S3 client not initialized. Skipping file upload.")
            return None
        
        try:
            # Clean folder and file names to prevent path injection
            safe_folder = folder_name.replace(' ', '_').replace('/', '_').replace('\\', '_')
            safe_filename = filename.replace(' ', '_').replace('/', '_').replace('\\', '_')
            now = datetime.utcnow()
            
            # Create path structure: aiAccounting/{folder_name}/{filename}
            object_key = f"aiAccounting/{safe_folder}/{safe_filename}"
            
            # Upload to S3
            self.client.put_object(
                Bucket=self.bucket_name,
                Key=object_key,
                Body=file_content,
                ContentType=self._get_content_type(filename),
                Metadata={
                    'originalFilename': filename,
                    'folderName': folder_name,
                    'uploadedAt': now.isoformat(),
                    'userId': userId or 'unknown'
                }
            )
            
            print(f"[INFO] File uploaded to S3: {object_key}")
            return object_key
            
        except ClientError as e:
            print(f"[ERROR] AWS S3 error uploading file: {str(e)}")
            return None
        except Exception as e:
            print(f"[ERROR] Unexpected error uploading to S3: {str(e)}")
            return None
    
    def get_static_url(self, object_key: str) -> Optional[str]:
        """
        Get a permanent, unsigned URL for an S3 object.
        Note: This URL may not be accessible if the bucket is private.
        """
        if not self.client:
            return None
            
        region = settings.aws_region
        if settings.s3_base_url:
            return f"{settings.s3_base_url.rstrip('/')}/{object_key}"
            
        return f"https://{self.bucket_name}.s3.{region}.amazonaws.com/{object_key}"

    def generate_presigned_url(self, object_key: str, expires_in: int = 3600) -> Optional[str]:
        """
        Generate a temporary presigned URL for accessing a file.
        
        Args:
            object_key: S3 object key (path)
            expires_in: URL expiration time in seconds (default: 1 hour)
            
        Returns:
            Presigned URL or None if error
        """
        if not self.client:
            return None
        
        try:
            url = self.client.generate_presigned_url(
                'get_object',
                Params={'Bucket': self.bucket_name, 'Key': object_key},
                ExpiresIn=expires_in
            )
            return url
            
        except ClientError as e:
            print(f"[ERROR] Error generating presigned URL: {str(e)}")
            return None
    
    def delete_file(self, object_key: str) -> bool:
        """
        Delete a file from S3.
        
        Args:
            object_key: S3 object key (path)
            
        Returns:
            True if successful, False otherwise
        """
        if not self.client:
            return False
        
        try:
            self.client.delete_object(Bucket=self.bucket_name, Key=object_key)
            print(f"[INFO] File deleted from S3: {object_key}")
            return True
            
        except ClientError as e:
            print(f"[ERROR] Error deleting file from S3: {str(e)}")
            return False
    
    @staticmethod
    def _get_content_type(filename: str) -> str:
        """Get MIME type based on file extension."""
        extension = filename.split('.')[-1].lower() if '.' in filename else ''
        content_types = {
            'pdf': 'application/pdf',
            'xlsx': 'application/vnd.openxmlformats-officedocument.spreadsheetml.sheet',
            'xls': 'application/vnd.ms-excel',
            'csv': 'text/csv',
            'docx': 'application/vnd.openxmlformats-officedocument.wordprocessingml.document',
            'doc': 'application/msword',
        }
        return content_types.get(extension, 'application/octet-stream')


# Global S3 storage instance
s3_storage = S3Storage()

