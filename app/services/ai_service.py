from typing import List, Dict, Any
import asyncio
from langchain_openai import ChatOpenAI
from app.core.config import settings
import random

# Helper for rate limit retries
async def retry_with_backoff(func, *args, **kwargs):
    retries = 20 # Effectively infinite retries to guarantee success
    for i in range(retries):
        try:
            return await func(*args, **kwargs)
        except Exception as e:
            if "429" in str(e) or "Rate limit" in str(e):
                if i == retries - 1:
                    raise e
                # Reduced wait time for faster retries
                # Base wait (10s) + Jitter (1-3s) to prevent thundering herd
                wait_time = 10 + random.uniform(1, 3)
                print(f"[WARNING] Rate limit hit. Waiting {wait_time:.1f}s to reset quota... (Attempt {i+1}/{retries})")
                await asyncio.sleep(wait_time)
            else:
                raise e

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
            max_tokens=50000,  # Increased to allow complete extraction of all matching records without truncation
            default_headers={
                "HTTP-Referer": "https://localhost:8000", # Optional: Your site URL
                "X-Title": "Accounting RAG System", # Optional: Your site name
            }
        )
        print(f"[INFO] LLMService initialized with OpenRouter model={settings.openrouter_model}")
    
    def _extract_json(self, text: str) -> Dict[str, Any]:
        """
        Robustly extract and parse JSON from LLM output.
        Attempts to find the largest valid JSON object or array in the text.
        """
        import json
        import re

        def clean_json_string(s):
            # Remove markdown code blocks if present
            s = re.sub(r'```json\s*', '', s)
            s = re.sub(r'```\s*', '', s)
            return s.strip()

        text = clean_json_string(text)
        
        # Step 1: Direct full parse
        try:
            parsed = json.loads(text)
            print(f"[DEBUG] JSON parsed successfully. Type: {type(parsed)}")
            
            if isinstance(parsed, list):
                print(f"[DEBUG] Parsed as list with {len(parsed)} items")
                return {"transactions": parsed}
            elif isinstance(parsed, dict):
                # Check if it's a single transaction object (has fields like date, amount, description)
                transaction_indicators = ['date', 'amount', 'description', 'transaction_id', 'type']
                if any(key in parsed for key in transaction_indicators) and 'transactions' not in parsed:
                    print(f"[DEBUG] Detected single transaction object, wrapping in array")
                    return {"transactions": [parsed]}
                print(f"[DEBUG] Parsed as dict with keys: {list(parsed.keys())}")
                return parsed
        except json.JSONDecodeError as e:
            print(f"[DEBUG] Direct JSON parse failed: {e}")

        # Step 2: Iterative search for all possible JSON objects/arrays
        all_found = []
        decoder = json.JSONDecoder()
        
        # We try to find any substring that starts with { or [ and parses as valid JSON
        for i in range(len(text)):
            if text[i] in '{[':
                try:
                    obj, end = decoder.raw_decode(text[i:])
                    all_found.append(obj)
                except (json.JSONDecodeError, ValueError):
                    pass

        print(f"[DEBUG] Iterative search found {len(all_found)} JSON objects")

        # Step 3: Parse recovery for incomplete JSON (simple closer appending)
        if not all_found:
             # Try appending closing braces/brackets
            temp_text = text
            for _ in range(3): 
                temp_text += "}"
                try:
                    match = re.search(r'(\{.*\})', temp_text, re.DOTALL)
                    if match:
                        all_found.append(json.loads(match.group(0)))
                        break
                except: pass
                
                temp_text_sq = text + "]"
                try:
                    match = re.search(r'(\[.*\])', temp_text_sq, re.DOTALL)
                    if match:
                        all_found.append(json.loads(match.group(0)))
                        break
                except: pass

        # Step 4: Logic to pick the "best" object
        if all_found:
            print(f"[DEBUG] Processing {len(all_found)} found objects")
            
            # Prioritize dicts containing "transactions" or "data"
            for item in all_found:
                if isinstance(item, dict):
                    if "transactions" in item or "data" in item:
                        print(f"[DEBUG] Found dict with transactions/data key")
                        return item
            
            # Check if we have transaction-like objects
            transaction_objects = []
            for item in all_found:
                if isinstance(item, dict):
                    transaction_indicators = ['date', 'amount', 'description', 'transaction_id', 'type']
                    if any(key in item for key in transaction_indicators):
                        transaction_objects.append(item)
            
            if transaction_objects:
                print(f"[DEBUG] Found {len(transaction_objects)} transaction objects, wrapping in array")
                return {"transactions": transaction_objects}
            
            # If we found a standalone list, wrap it
            for item in all_found:
                if isinstance(item, list):
                    print(f"[DEBUG] Found standalone list with {len(item)} items")
                    return {"transactions": item}
                    
            # Fallback: just return the first valid object found
            print(f"[DEBUG] Using fallback: returning first found object")
            return all_found[0] if isinstance(all_found[0], dict) else {"data": all_found[0]}

        # Final Fallback
        print(f"[WARNING] No valid JSON found in response, returning empty")
        return {
            "message": text.strip()[:200],  # Only keep first 200 chars to avoid bloat
            "transactions": []
        }

    async def _extract_document_metadata(self, context_chunks: List[str], source_documents: List[str]) -> Dict[str, Dict[str, str]]:
        """
        Identify the Bank Name and Account Holder for each unique document by scanning the first few chunks.
        Parallelized for speed.
        """
        unique_docs = list(set(source_documents))
        
        async def scan_single_doc(doc):
            try:
                # Find the first few chunks for this document to get a complete header
                all_doc_indices = [i for i, x in enumerate(source_documents) if x == doc]
                # Take up to 10 chunks to ensure we see the full header
                header_indices = all_doc_indices[:10]
                peek_context = "\n".join([context_chunks[i] for i in header_indices])

                prompt = f"""You are a professional bank document analyzer. Your goal is to identify the Issuing Bank Name and the Main Account Holder Name from the provided snippet.

                STRATEGY:
                1. Look for the absolute header (often the first 2-5 lines).
                2. Search for bank names (e.g., "IDBI Bank", "HDFC Bank", "State Bank of India", "Kotak", "Axis", "ICICI").
                3. Check for specific addresses or branch names that indicate the bank.
                4. Look for labels like "Account Name:", "Name of Account Holder:", or "Beneficiary Name:".

                SNIPPET:
                --- 
                {peek_context[:4000]}
                ---

                Return ONLY a JSON object: {{"bank_name": "IDENTIFIED BANK NAME", "account_holder": "IDENTIFIED NAME"}}
                If a field is truly not found, use "Unknown".
                """
                response = await self.llm.ainvoke(prompt)
                meta = self._extract_json(response.content)
                
                bank_name = meta.get("bank_name", "Unknown")
                account_holder = meta.get("account_holder", "Unknown")

                return doc, {"bank_name": bank_name, "account_holder": account_holder}
            except Exception as e:
                print(f"[ERROR] Failed to extract metadata for {doc}: {e}")
                return doc, {"bank_name": "Unknown", "account_holder": "Unknown"}

        # Extract all unique docs in parallel
        tasks = [scan_single_doc(doc) for doc in unique_docs]
        results = await asyncio.gather(*tasks)
        
        doc_metadata = dict(results)
        for doc, meta in doc_metadata.items():
            print(f"[INFO] Detected metadata for {doc}: {meta}")
            
        return doc_metadata

    async def generate_answer(
        self,
        question: str,
        context_chunks: List[str],
        source_documents: List[str],
        doc_metadata: Dict[str, Dict[str, str]] = None
    ) -> Dict[str, Any]:
        """
        Generate an answer based on the question and retrieved context using a DYNAMIC schema.
        """
        if not context_chunks:
            print(f"[WARNING] No context chunks provided for answer generation")
            return {
                "answer": "Information not available in uploaded records",
                "sources": []
            }
        
        
        # Build context from chunks WITH source document information and preset metadata
        context_parts = []
        for i, chunk in enumerate(context_chunks):
            doc_name = source_documents[i] if i < len(source_documents) else "Unknown"
            
            # Add metadata hint to the context block if available
            meta_str = ""
            if doc_metadata and doc_name in doc_metadata:
                m = doc_metadata[doc_name]
                meta_str = f" [STATEMENT_BANK: {m['bank_name']}, HOLDER: {m['account_holder']}]"
                
            context_parts.append(f"[Context {i+1} - Source: {doc_name}{meta_str}]\n{chunk}")
        
        context = "\n\n".join(context_parts)
        
        # Enhanced Financial Assistant Prompt
        prompt = f"""You are an expert financial assistant designed to analyze and extract information from uploaded bank documents such as statements, ledgers, and journals. Your task is to help users find specific transactions or records by interpreting their queries and searching through the provided documents.

**USER QUERY:** "{question}"

**DOCUMENT CONTEXT:**
{context}

**CRITICAL INSTRUCTIONS:**
1. **Understand the Query:** Carefully read the user's question to identify what they are looking for (e.g., UPI records, specific transactions, dates, amounts, payees, etc.).

2. **Document Analysis:** Scan the uploaded documents for relevant data. Focus on:
   - Transaction dates
   - Amounts (debit/credit)
   - Description/narration
   - UPI IDs, reference numbers, or payee names
   - Transaction types (UPI, NEFT, cash, etc.)

3. **COMPLETE EXTRACTION REQUIREMENT:**
   - Extract **EVERY SINGLE RECORD** that matches the user's query from the provided context
   - Include **BOTH CREDIT AND DEBIT** transactions - do NOT filter by direction
   - Do NOT summarize, truncate, or limit the number of records
   - Do NOT skip any matching transactions
   - If there are 100 matching records, return all 100
   - If there are 1000 matching records, return all 1000
   - COMPLETENESS is more important than brevity
   
   **CRITICAL EXTRACTION LOGIC**:
   - Bank statements usually have THREE numeric columns on every line: **"Withdrawal/Debit"**, **"Deposit/Credit"**, and **"Balance"**.
   - In plain text, these often appear as three consecutive numbers. 
   - **Example 1**: "22/11/24 Rent Paid 500.00 0.00 10000.00" -> 500.00 is in the first column (Withdrawal) = **DEBIT**.
   - **Example 2**: "22/11/24 Salary 0.00 50000.00 60000.00" -> 50000.00 is in the second column (Deposit) = **CREDIT**.
   - **Example 3**: "22/11/24 UPI-RECEIVED 100.00 10100.00" -> If only two numbers are present, check the balance delta. If the balance increased, it's a **CREDIT**.
   - **Example 4**: "200.00 UPI-PAID" -> If only one number is present, it's almost always a **DEBIT** for UPI payments, but check the preceding balance to be sure.
   
   - **MANDATORY RULES**:
     1. You MUST scan BOTH columns on every line.
     2. If an amount is in the **Deposit/Credit** column (often the 2nd amount column) → **direction = "CREDIT"** (Money Incoming).
     3. If an amount is in the **Withdrawal/Debit** column (often the 1st amount column) → **direction = "DEBIT"** (Money Outgoing).
     4. **DO NOT IGNORE 0.00 VALUES.** They are markers for the OTHER column. If you see "0.00 500.00", the 500.00 is a CREDIT.
     5. Look for keywords like "Deposit", "CR", "Credit", "Interest", "Amount Received", "Inward", "Received from", "Refund" to identify credits.
     6. Look for keywords like "Withdrawal", "DR", "Debit", "Amount Paid", "Outward", "Payment to", "Transfer to" to identify debits.
     7. **VERIFY WITH BALANCE (REQUIRED)**: If the record includes a "Balance" or "Closing Balance" column, verify if the amount made the balance go UP (Credit) or DOWN (Debit).
     8. **SMOOSHED NUMBERS**: If numbers appear jammed together like "100.005000.00", split them: the first is the Amount, the second is the Balance.
     9. **INVISIBLE CREDITS**: If only ONE amount is present but the word "Received", "Credit", "Interest", or "CR" appears, OR if the balance increases from the previous line, it is a **CREDIT**.
     10. **DO NOT SKIP** any transaction because it is a credit. The user wants **ALL** records matching their query.

4. **Output Format:** Return results as a STRICT valid JSON object with this structure:
   {{
     "transactions": [
       {{
         "date": "transaction date",
         "description": "transaction description/narration (include the full text for accuracy)",
         "amount": "transaction amount (ABSORLUTELY REQUIRED: Use the non-zero numeric value. NEVER use 0.00)",
         "direction": "CREDIT or DEBIT",
         "transaction_id": "UPI ID/reference number/Chq No if available",
         "type": "UPI/NEFT/CASH/etc if identifiable",
         "bank_name": "MANDATORY: Use the 'STATEMENT_BANK' name from the block header. DO NOT use names found in UPI IDs (like @oksbi or @okicici).",
         "source_document": "name of the document this transaction was found in"
       }}
     ]
   }}
   - **CRITICAL**: If you found NO records in this specific context block, return an empty list: {{"transactions": []}}.
   - **CRITICAL**: Do NOT list "0.00" as the transaction amount. Use the actual numeric value from the other column.

5. **No Data Found:** If no matching records are found in the context, return: {{ "transactions": [], "message": "No matching records found in this context." }}

**Response Guidelines:**
- Extract EVERY SINGLE piece of valid data that matches the user's request - NO EXCEPTIONS
- Return the COMPLETE dataset - do NOT summarize or provide a sample
- **ZERO AMOUNT RULE**: If you extract a record with `amount: 0.0` or `0.00`, you have FAILED. Look at the numbers on that same line again. One of them is non-zero. Use THAT one.
- **BANK NAME CONSISTENCY**: Do NOT change the `bank_name` based on the payee or UPI ID (like @oksbi). Use the bank name of the statement owner.
- Keep descriptions complete including any reference numbers found at the end of the line.

**RESPONSE (JSON ONLY):**
"""
        
        try:
            print(f"[INFO] Generating answer for question: {question[:100]}...")
            print(f"[DEBUG] Context chunks count: {len(context_chunks)}")
            print(f"[DEBUG] [Sending to AI] Input context preview: {context_chunks[0][:200] if context_chunks else 'NO CONTEXT'}...")
            
            # Use ainvoke for asynchronous LLM call
            response = await self.llm.ainvoke(prompt)
            print(f"[DEBUG] [Received from AI] Response length: {len(response.content)} chars")
            
            # Clean up the response
            answer = response.content.strip()
            
            # Use robust JSON extraction
            parsed_json = self._extract_json(answer)
            
            # Log transaction count
            tx_count = len(parsed_json.get('transactions', []))
            print(f"[INFO] Extracted {tx_count} transactions from this batch")
            
            # Normalize key to 'transactions' if 'data' is present
            if "data" in parsed_json and "transactions" not in parsed_json:
                parsed_json["transactions"] = parsed_json.pop("data")
            
            sources = list(set(source_documents))
            
            import json
            return {
                "answer": json.dumps(parsed_json),
                "sources": sources
            }
        except Exception as e:
            if "429" not in str(e) and "Rate limit" not in str(e):
                print(f"[ERROR] Error generating answer: {str(e)}")
            raise

    async def generate_exhaustive_answer(
        self,
        question: str,
        context_chunks: List[str],
        source_documents: List[str],
        batch_size: int = 40
    ) -> Dict[str, Any]:
        """
        Iteratively extract information from batches of chunks.
        """
        if not context_chunks:
            return {
                "answer": "Information not available in uploaded records",
                "sources": []
            }

        sem = asyncio.Semaphore(6)
        print(f"[INFO] Starting parallel exhaustive extraction across {len(context_chunks)} chunks")
        
        # Step 1: Pre-scan for document metadata
        doc_metadata = await self._extract_document_metadata(context_chunks, source_documents)
        
        async def process_batch_with_sem(batch, batch_source_docs):
            async with sem:
                return await retry_with_backoff(self.generate_answer, question, batch, batch_source_docs, doc_metadata)

        # Create tasks for each batch
        tasks = []
        total_batches = (len(context_chunks) + batch_size - 1) // batch_size
        print(f"[INFO] Splitting {len(context_chunks)} chunks into {total_batches} batches (batch_size={batch_size})")
        
        for i in range(0, len(context_chunks), batch_size):
            batch = context_chunks[i:i + batch_size]
            batch_source_docs = source_documents[i:i + batch_size]
            batch_num = (i // batch_size) + 1
            print(f"[INFO] Batch {batch_num}/{total_batches}: Processing {len(batch)} chunks")
            tasks.append(process_batch_with_sem(batch, batch_source_docs))

        # Execute all batches
        results = await asyncio.gather(*tasks)

        full_answer = ""
        all_transactions = []
        all_sources = set()
        seen_fingerprints = set()

        import json
        for result in results:
            full_answer += result['answer'] + "\n---\n"
            if result.get('sources'):
                all_sources.update(result['sources'])

            try:
                content = json.loads(result['answer'])
                
                # Dynamic extraction of the list
                txs = []
                if "transactions" in content: txs = content["transactions"]
                elif "data" in content: txs = content["data"]
                # Fallback scan values for list
                elif not txs:
                     for val in content.values():
                        if isinstance(val, list) and val:
                            txs = val
                            break
                
                if txs:
                    for t in txs:
                        # Create a fingerprint to avoid duplicates
                        desc = str(t.get('description', '')).strip().lower()
                        amt = str(t.get('amount', '0')).replace(',', '')
                        date = str(t.get('date', ''))
                        # Use full description for granularity to avoid missing similar legitimate transactions
                        fp = f"{date}|{amt}|{desc}"
                        
                        if fp not in seen_fingerprints:
                            seen_fingerprints.add(fp)
                            all_transactions.append(t)

            except Exception as e:
                print(f"[DEBUG] Error merging batch result: {str(e)}")

        print(f"[INFO] Exhaustive extraction complete: {len(all_transactions)} total unique records")

        return {
            "answer": json.dumps({"transactions": all_transactions}),
            "sources": list(all_sources)
        }

    async def stream_exhaustive_answer(
        self,
        question: str,
        context_chunks: List[str],
        source_documents: List[str],
        batch_size: int = 40
    ):
        """
        Stream extraction results as they are processed.
        """
        import json
        
        if not context_chunks:
            yield json.dumps({"type": "error", "message": "No context provided"})
            return

        # Step 1: Pre-scan for document metadata
        doc_metadata = await self._extract_document_metadata(context_chunks, source_documents)
        
        total_batches = (len(context_chunks) + batch_size - 1) // batch_size
        tasks = []
        sem = asyncio.Semaphore(6) # Increased for speed (Higher throughput)
        
        async def process_batch_with_sem(batch, batch_source_docs):
            async with sem:
                return await retry_with_backoff(self.generate_answer, question, batch, batch_source_docs, doc_metadata)

        print(f"[INFO] Streaming exhaustive extraction: {len(context_chunks)} chunks")

        for i in range(0, len(context_chunks), batch_size):
            batch = context_chunks[i:i + batch_size]
            batch_source_docs = source_documents[i:i + batch_size]
            tasks.append(process_batch_with_sem(batch, batch_source_docs))

        completed_count = 0
        all_transactions_count = 0
        all_sources = set()
        seen_fingerprints = set() # For streaming de-duplication

        for future in asyncio.as_completed(tasks):
            try:
                result = await future
                completed_count += 1
                
                # yield progress
                yield json.dumps({
                    "type": "progress",
                    "completed": completed_count,
                    "total": total_batches,
                    "percent": int((completed_count / total_batches) * 100)
                })
                
                # Process result
                try:
                    content = json.loads(result['answer'])
                    
                    # DYNAMIC: Find the data list
                    txs = []
                    if content.get('transactions'): txs = content['transactions']
                    elif content.get('data'): txs = content['data']
                    # Fallback scan values for list
                    elif not txs:
                         for val in content.values():
                            if isinstance(val, list) and val:
                                txs = val
                                break

                    if txs:
                        unique_txs = []
                        for t in txs:
                            # Create a fingerprint to avoid duplicates
                            desc = str(t.get('description', '')).strip().lower()
                            amt = str(t.get('amount', '0')).replace(',', '')
                            date = str(t.get('date', ''))
                            # Use full description for higher granularity
                            fp = f"{date}|{amt}|{desc}"
                            
                            if fp not in seen_fingerprints:
                                seen_fingerprints.add(fp)
                                unique_txs.append(t)

                        if unique_txs:
                            count = len(unique_txs)
                            print(f"[DEBUG] Stream batch {completed_count}: Found {count} unique items")
                            all_transactions_count += count
                            yield json.dumps({
                                "type": "data",
                                "transactions": unique_txs
                            })
                    else:
                        print(f"[DEBUG] Stream batch {completed_count}: No new transactions found")

                    if result.get('sources'):
                        all_sources.update(result['sources'])
                        
                except Exception as e:
                    print(f"[WARNING] Failed to parse batch result in stream: {e}")
                        
            except Exception as e:
                print(f"[ERROR] Stream batch failed: {e}")
                yield json.dumps({"type": "error", "message": str(e)})

        # Final summary
        yield json.dumps({
            "type": "summary",
            "total_transactions": all_transactions_count,
            "sources": list(all_sources)
        })
