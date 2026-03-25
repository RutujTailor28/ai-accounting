from typing import List
from langchain_text_splitters import RecursiveCharacterTextSplitter
from app.core.config import settings

class TextChunker:
    """Service for chunking text into smaller segments."""

    def __init__(self):
        """Initialize text chunker with configured chunk size and overlap."""
        self.splitter = RecursiveCharacterTextSplitter(
            chunk_size=settings.chunk_size,
            chunk_overlap=settings.chunk_overlap,
            length_function=len,
            separators=["\n\n", "\n", " ", ""]
        )
        print(f"[INFO] TextChunker initialized with chunk_size={settings.chunk_size}, chunk_overlap={settings.chunk_overlap}")

    def chunk_text(self, text: str) -> List[str]:
        """
        Split text into chunks.

        Args:
            text: Input text to chunk

        Returns:
            List of text chunks
        """
        if not text or not text.strip():
            print(f"[WARNING] Empty text provided for chunking")
            return []

        chunks = self.splitter.split_text(text)
        print(f"[INFO] Created {len(chunks)} chunks from {len(text)} characters")

        return chunks
