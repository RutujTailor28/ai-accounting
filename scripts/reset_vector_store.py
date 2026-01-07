import sys
import os

# Add the project root to the python path
sys.path.append(os.getcwd())

from app.ai.rag.retriever import VectorStore
from app.core.logging import app_logger

def reset_chroma():
    print("Initializing VectorStore...")
    try:
        store = VectorStore()
        print("Attempting to reset collection 'accounting_documents'...")
        success = store.reset_collection()
        
        if success:
            print("SUCCESS: Vector store has been successfully reset.")
        else:
            print("FAILURE: Could not reset vector store.")
            
    except Exception as e:
        print(f"ERROR: {str(e)}")

if __name__ == "__main__":
    reset_chroma()
