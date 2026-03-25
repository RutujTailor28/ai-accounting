from typing import List, Dict, Any
import chromadb
from chromadb.config import Settings as ChromaSettings
from app.core.config import settings

class VectorStore:
    """Service for managing vector storage with ChromaDB."""

    def __init__(self):
        """Initialize ChromaDB client and collection."""
        self.client = chromadb.PersistentClient(
            path=settings.chroma_persist_directory,
            settings=ChromaSettings(anonymized_telemetry=False)
        )

        self.collection = self.client.get_or_create_collection(
            name="accounting_documents",
            metadata={"description": "RAG system for accounting documents"}
        )

        self.feedback_collection = self.client.get_or_create_collection(
            name="accounting_feedback",
            metadata={"description": "User corrections and feedback rules"}
        )

        try:
            sample = self.collection.get(limit=1, include=["embeddings"])

            if sample is not None and sample.get("embeddings") is not None:
                embs = sample["embeddings"]
                if len(embs) > 0:
                    existing_dim = len(embs[0])
                    print(f"[DEBUG] Collection dimension check: {existing_dim}")
        except Exception as e:
            print(f"[DEBUG] Dimension check skip: {e}")

        print(f"[INFO] VectorStore initialized with persist_directory={settings.chroma_persist_directory}")

    def _build_where_filter(
        self,
        company_id: str,
        file_types: List[str] = None,
        folder_ids: List[str] = None,
        uploaded_by: List[str] = None,
        start_date: str = None,
        end_date: str = None,
        tags: List[str] = None,
        document_names: List[str] = None
    ) -> Dict[str, Any]:
        """
        Build ChromaDB 'where' filter from multiple parameters.
        """
        filters = [{"company_id": company_id}]

        if file_types:

            normalized_types = [ft.lower() for ft in file_types]
            filters.append({"file_type": {"$in": normalized_types}})

        if folder_ids:
            filters.append({"folder_id": {"$in": folder_ids}})

        if uploaded_by:
            filters.append({"created_by": {"$in": uploaded_by}})

        if document_names:
            filters.append({"document_name": {"$in": document_names}})

        if tags:

            normalized_tags = [t.lower() for t in tags]
            filters.append({"tags": {"$in": normalized_tags}})

        if len(filters) == 1:
            res = filters[0]
        else:
            res = {"$and": filters}

        print(f"[DEBUG] Built where_filter: {res}")
        return res

    def add_documents(
        self,
        texts: List[str],
        embeddings: List[List[float]],
        metadatas: List[Dict[str, Any]]
    ) -> None:
        """
        Add documents to the vector store.

        Args:
            texts: List of text chunks
            embeddings: List of embedding vectors
            metadatas: List of metadata dictionaries (must include company_id and document_name)
        """
        if not texts or not embeddings or not metadatas:
            print(f"[WARNING] Empty data provided to add_documents")
            return

        if not (len(texts) == len(embeddings) == len(metadatas)):
            raise ValueError("texts, embeddings, and metadatas must have the same length")

        ids = []
        for i in range(len(texts)):

            chunk_idx = metadatas[i].get('chunk_index', i)
            ids.append(f"{metadatas[i]['company_id']}_{metadatas[i]['document_name']}_{chunk_idx}")

        try:
            self.collection.add(
                documents=texts,
                embeddings=embeddings,
                metadatas=metadatas,
                ids=ids
            )
            print(f"[INFO] Added {len(texts)} documents to vector store for company_id={metadatas[0]['company_id']}")
        except Exception as e:
            error_msg = str(e)
            if "dimension" in error_msg or "expecting embedding with dimension" in error_msg:
                print(f"[WARNING] EMBDEDDING DIMENSION MISMATCH DETECTED: {error_msg}")
                print(f"[ACTION] Resetting collection to match new model dimension...")
                self.reset_collection()

                self.collection.add(
                    documents=texts,
                    embeddings=embeddings,
                    metadatas=metadatas,
                    ids=ids
                )
                print(f"[INFO] Successfully recovered and added documents after collection reset.")
                return

            print(f"[ERROR] Error adding documents to vector store: {str(e)}")
            raise

    def query(
        self,
        query_embedding: List[float],
        company_id: str,
        n_results: int = 5,
        **filters
    ) -> Dict[str, Any]:
        """
        Query the vector store for relevant documents with optional filters.
        """
        try:
            where_filter = self._build_where_filter(company_id=company_id, **filters)

            results = self.collection.query(
                query_embeddings=[query_embedding],
                n_results=n_results,
                where=where_filter
            )

            print(f"[INFO] Query returned {len(results['documents'][0])} results for company_id={company_id}")

            return results
        except Exception as e:
            print(f"[ERROR] Error querying vector store: {str(e)}")
            raise

    def workspace_query(
        self,
        query_embedding: List[float],
        company_id: str,
        document_names: List[str],
        n_results: int = 20,
        **filters
    ) -> Dict[str, Any]:
        """
        Query the vector store, strictly limited to a specified set of documents and optional filters.
        """
        try:
            where_filter = self._build_where_filter(
                company_id=company_id,
                document_names=document_names,
                **filters
            )

            results = self.collection.query(
                query_embeddings=[query_embedding],
                n_results=n_results,
                where=where_filter
            )

            print(f"[INFO] Workspace Query returned {len(results['documents'][0])} results for {len(document_names)} docs")
            return results
        except Exception as e:
            print(f"[ERROR] Error in workspace_query: {str(e)}")
            raise

    def get_collection_count(self, company_id: str = None) -> int:
        """
        Get the count of documents in the collection.

        Args:
            company_id: Optional company ID to filter count

        Returns:
            Number of documents
        """
        try:
            if company_id:

                results = self.collection.get(
                    where={"company_id": company_id},
                    include=[],
                    limit=1000000
                )
                count = len(results['ids'])
            else:
                count = self.collection.count()

            print(f"[DEBUG] Collection count: {count}")
            return count
        except Exception as e:
            print(f"[ERROR] Error getting collection count: {str(e)}")
            raise
    def delete_document(self, company_id: str, document_name: str) -> bool:
        """
        Delete a specific document from the vector store.

        Args:
            company_id: Company ID associated with the document
            document_name: Name of the document to delete

        Returns:
            True if successful, False otherwise
        """
        try:
            print(f"[INFO] Deleting document {document_name} for company {company_id} from vector store")
            self.collection.delete(
                where={
                    "$and": [
                        {"company_id": {"$eq": company_id}},
                        {"document_name": {"$eq": document_name}}
                    ]
                }
            )
            return True
        except Exception as e:
            print(f"[ERROR] Error deleting document from vector store: {str(e)}")
            return False

    def reset_collection(self) -> bool:
        """
        Completely delete and recreate the vector store collection.
        This is necessary to handle embedding dimension changes.
        """
        try:
            print(f"[WARNING] RESETTING VECTOR STORE COLLECTION (DELETE & RECREATE)")

            try:
                self.client.delete_collection(name="accounting_documents")
            except Exception as inner_e:
                print(f"[DEBUG] Collection deletion error (may not exist): {inner_e}")

            self.collection = self.client.get_or_create_collection(
                name="accounting_documents",
                metadata={"description": "RAG system for accounting documents"}
            )
            print(f"[INFO] DONE: Vector store collection recreated.")
            return True
        except Exception as e:
            print(f"[ERROR] Error resetting vector store: {str(e)}")
            return False

    def add_feedback(
        self,
        text: str,
        embedding: List[float],
        metadata: Dict[str, Any]
    ) -> None:
        """
        Add a user correction (feedback) to the vector store.
        """
        try:

            import uuid
            feedback_id = str(uuid.uuid4())

            self.feedback_collection.add(
                documents=[text],
                embeddings=[embedding],
                metadatas=[metadata],
                ids=[feedback_id]
            )
            print(f"[INFO] Added feedback rule to vector store: {text[:50]}...")
        except Exception as e:
            print(f"[ERROR] Error adding feedback to vector store: {str(e)}")
            raise

    def query_feedback(
        self,
        query_embedding: List[float],
        company_id: str,
        n_results: int = 3
    ) -> List[str]:
        """
        Find relevant past corrections for a given query/transaction.
        """
        try:
            results = self.feedback_collection.query(
                query_embeddings=[query_embedding],
                n_results=n_results,
                where={"company_id": company_id}
            )

            if results and results.get('documents'):
                rules = results['documents'][0] if results['documents'] else []
                print(f"[INFO] Found {len(rules)} relevant past corrections.")
                return rules

            return []
        except Exception as e:
            print(f"[ERROR] Error querying feedback: {str(e)}")
            return []

    def clear_company_feedback(self, company_id: str) -> bool:
        """
        Delete all feedback/corrections for a specific company.
        """
        try:
            print(f"[INFO] Clearing all feedback for company {company_id}")
            self.feedback_collection.delete(
                where={"company_id": company_id}
            )
            return True
        except Exception as e:
            print(f"[ERROR] Error clearing feedback: {str(e)}")
            return False

vector_store = VectorStore()
