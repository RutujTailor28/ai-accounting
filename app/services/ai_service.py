from typing import List, Dict, Any
import asyncio
from langchain_openai import ChatOpenAI
from app.core.config import settings


class LLMService:
    """Service for generating answers using OpenRouter LLM."""
    
    def __init__(self):
        """Initialize LLM service with OpenRouter model."""
        if not settings.openrouter_api_key:
            print("[WARNING] OpenRouter API key not configured. AI features may fail.")
            
        self.llm = ChatOpenAI(
            model=settings.openrouter_model,
            openai_api_key=settings.openrouter_api_key,
            openai_api_base=settings.openrouter_base_url,
            temperature=0.1,
            max_tokens=1000,  # Reduced to fit within budget limits
            default_headers={
                "HTTP-Referer": "https://localhost:8000", # Optional: Your site URL
                "X-Title": "Accounting RAG System", # Optional: Your site name
            }
        )
        print(f"[INFO] LLMService initialized with OpenRouter model={settings.openrouter_model}")
    
    async def generate_answer(
        self,
        question: str,
        context_chunks: List[str],
        source_documents: List[str]
    ) -> Dict[str, Any]:
        """
        Generate an answer based on the question and retrieved context.
        """
        if not context_chunks:
            print(f"[WARNING] No context chunks provided for answer generation")
            return {
                "answer": "Information not available in uploaded records",
                "sources": []
            }
        
        # Build context from chunks
        context = "\n\n".join([f"[Context {i+1}]\n{chunk}" for i, chunk in enumerate(context_chunks)])
        
        # Create a strict prompt to prevent hallucinations and enforce exhaustive listing
        prompt = f"""You are a data extraction assistant for an accounting company. Your task is to extract information from the provided documents.

CRITICAL RULES:
1. Answer ONLY using information from the context below.
2. **EXHAUSTIVE LISTING REQUIRED**: If the user asks for a category of items (e.g., "UPI transactions", "dates", "amounts", "names"), you must list **EVERY SINGLE OCCURRENCE** found in the context.
3. **NO SUMMARIES**: Do NOT summarize. Do NOT say "Here are a few..." or "Including X, Y, Z". You must output the COMPLETE list.
4. **NO TRUNCATION**: Do not shorten the list. If there are 50 items, list all 50.
5. **DEFAULT TO ALL**: Assume the user ALWAYS wants "ALL" data unless they specifically ask for a summary.
6. If the answer is not in the context, respond EXACTLY with: "Information not available in uploaded records".
7. Be accurate. Do NOT guess or hallucinate.

Context from uploaded documents:
{context}

Question: {question}

Answer:"""
        
        try:
            print(f"[INFO] Generating answer for question: {question[:100]}...")
            # Use ainvoke for asynchronous LLM call
            response = await self.llm.ainvoke(prompt)
            
            # Clean up the response
            answer = response.content.strip()
            
            # Check if the model couldn't find the answer
            if any(phrase in answer.lower() for phrase in [
                "not available", "not found", "cannot find", "no information",
                "not mentioned", "not provided", "don't have"
            ]):
                answer = "Information not available in uploaded records"
                sources = []
            else:
                sources = list(set(source_documents))  # Remove duplicates
            
            print(f"[INFO] Generated answer with {len(sources)} sources")
            
            return {
                "answer": answer,
                "sources": sources
            }
        except Exception as e:
            print(f"[ERROR] Error generating answer: {str(e)}")
            raise

    async def generate_exhaustive_answer(
        self,
        question: str,
        context_chunks: List[str],
        source_documents: List[str],
        batch_size: int = 15
    ) -> Dict[str, Any]:
        """
        Iteratively extract information from batches of chunks to ensure 100% recall.
        """
        if not context_chunks:
            return {
                "answer": "Information not available in uploaded records",
                "sources": []
            }

        print(f"[INFO] Starting parallel exhaustive extraction across {len(context_chunks)} chunks in batches of {batch_size}")
        
        # Create tasks for each batch
        tasks = []
        for i in range(0, len(context_chunks), batch_size):
            batch = context_chunks[i:i + batch_size]
            batch_num = (i // batch_size) + 1
            total_batches = (len(context_chunks) + batch_size - 1) // batch_size
            
            print(f"[INFO] Scheduling extraction batch {batch_num}/{total_batches} ({len(batch)} chunks)...")
            tasks.append(self.generate_answer(question, batch, source_documents))

        # Execute all batches in parallel
        results = await asyncio.gather(*tasks)
        print(f"[INFO] DONE: All {len(results)} batches processed by AI.")

        all_answers = []
        all_sources = set()

        for result in results:
            if result['answer'] != "Information not available in uploaded records":
                all_answers.append(result['answer'])
                all_sources.update(result['sources'])

        if not all_answers:
            return {
                "answer": "Information not available in uploaded records",
                "sources": []
            }

        combined_answer = "\n\n".join(all_answers)
        print(f"[INFO] Parallel exhaustive extraction complete. Combined {len(all_answers)} responses.")
        
        return {
            "answer": combined_answer,
            "sources": list(all_sources)
        }
