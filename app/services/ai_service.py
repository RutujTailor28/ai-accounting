from typing import List, Dict, Any
import asyncio
from langchain_openai import ChatOpenAI
from app.core.config import settings
import random
from datetime import date
import re
import json

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
            temperature=0.0,
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

                prompt = f"""You are a professional financial document analyzer. Your goal is to identify the Issuing Bank Name, the Main Account Holder Name, and the DOCUMENT TYPE from the provided snippet.

                STRATEGY:
                1. Look for the absolute header (often the first few lines of the file).
                2. **BANK NAME IDENTIFICATION (CRITICAL)**:
                   - Look for the official bank name in the main header (usually first 10-20 lines).
                   - Check for bank logos, letterheads, or explicit "Bank Statement" titles at the top.
                   - **CRITICAL**: Ignore bank names found inside transaction lists or UPI IDs (e.g. "@oksbi", "to SBI account"). Only identify the ISSUING bank of the statement itself.
                   - **CRITICAL**: If you cannot find an explicit issuing bank name in the header, return "Unknown" - DO NOT guess.
                3. Identify Account Holder Name (e.g., "Name of the Assessee", "Beneficiary Name:", "Account Name:", "Customer Name:").
                4. IDENTIFY DOCUMENT TYPE:
                   - **Bank Statement**: Contains transaction lists with Date, Narration, Withdrawal/Deposit, Balance.
                   - **Balance Sheet**: Summary of Assets and Liabilities. Look for terms like "Share Capital", "Fixed Assets", "Current Liabilities", "Balance Sheet as on...".
                   - **Profit & Loss Account**: Summary of Income and Expenditure. Look for terms like "Sales", "Expenses", "Net Profit", "Profit & Loss A/c for the period...".
                   - **Computation of Income**: Detailed tax/income calculations. Often contains many sections like "Income from House Property", "Business Income", etc.

                SNIPPET:
                --- 
                {peek_context[:6000]}
                ---

                Return ONLY a JSON object: {{"bank_name": "IDENTIFIED BANK NAME", "account_holder": "IDENTIFIED NAME", "document_type": "TYPE"}}
                **CRITICAL**: For bank_name, use the FULL official name (e.g., "HDFC Bank", "IDBI Bank Ltd."). If truly not found, use "Unknown".
                """
                response = await self.llm.ainvoke(prompt)
                meta = self._extract_json(response.content)
                
                bank_name = meta.get("bank_name", "Unknown")
                account_holder = meta.get("account_holder", "Unknown")
                doc_type = meta.get("document_type", "Unknown")
                
                # Removed manual bank heuristic logic to allow LLM to find the bank name dynamically.

                return doc, {"bank_name": bank_name, "account_holder": account_holder, "document_type": doc_type}
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
        
        # Step 1: Pre-scan for document metadata if not provided
        if doc_metadata is None:
            doc_metadata = await self._extract_document_metadata(context_chunks, source_documents)
        # Build context from chunks WITH source document information and preset metadata
        context_parts = []
        for i, chunk in enumerate(context_chunks):
            doc_name = source_documents[i] if i < len(source_documents) else "Unknown"
            
            # Add metadata hint to the context block if available
            meta_str = ""
            if doc_metadata and doc_name in doc_metadata:
                m = doc_metadata[doc_name]
                meta_str = f" [TYPE: {m.get('document_type', 'Unknown')}, STATEMENT_BANK: {m['bank_name']}, HOLDER: {m['account_holder']}]"
                
            context_parts.append(f"[Context {i+1} - Source: {doc_name}{meta_str}]\n{chunk}")
        
        context = "\n\n".join(context_parts)
        
        # Get current date for relative time queries
        today = date.today()
        current_date_str = today.strftime("%d-%m-%Y")
        current_year = today.year
        
        # Enhanced Financial Assistant Prompt
        prompt = f"""You are an expert financial assistant designed to analyze and extract information from uploaded bank documents such as statements, ledgers, journals, Balance Sheets, and Profit & Loss statements. Your task is to help users find specific transactions or understand financial summaries by interpreting their queries.

**CURRENT SYSTEM DATE:** {current_date_str} (Use this to resolve "this year", "previous year", "last month", etc. Previous year = {current_year - 1})

**USER QUERY:** "{question}"

**DOCUMENT CONTEXT:**
{context}

**CRITICAL INSTRUCTIONS:**
1. **CATEGORIZATION ACCURACY (HIGHEST PRIORITY)**:
   - **EQUITY AND LIABILITIES**: Capital, Loans, Sundry Creditors, Provisions, Outstanding Expenses.
   - **ASSETS**: Fixed Assets (Cars, Land, Furniture), Sundry Debtors, Bank Balance, Cash in Hand, Deposits, Prepaid Expenses.
   - **SUNDRY DEBTORS ARE ALWAYS ASSETS**. Do NOT place them in Liabilities.
   - **SUNDRY CREDITORS ARE ALWAYS LIABILITIES**.

2. **Understand the Query:** Identify if the user is looking for SPECIFIC TRANSACTIONS or a GENERAL SUMMARY (like a Balance Sheet or P&L).

3. **Document Analysis & Identification:**
   - Scan the context to identify the DOCUMENT TYPE (e.g., Bank Statement, Balance Sheet, P&L).
   - Identify the FINANCIAL PERIOD or YEAR mentioned in the document headers.
   - If the user asks for "previous year", look for documents dated {current_year - 1}.

4. **HANDLING SUMMARY REPORTS (Balance Sheet / P&L / Computation):**
   - If the user asks for a high-level summary (e.g., "give me balance sheet", "show p&l", "financial report"), your PRIMARY goal is to find and extract that report's structural tables.
   - **MANDATORY**: Look for keywords like "Balance Sheet", "Assets", "Liabilities", "Equity", "Profit & Loss", "Capital Account", "Income", "Expenditure", "Statement of Affairs", "Financial Position" in the text.
   - **DO NOT** extract individual bank statement transactions if a summary report is requested.
   - **TABLE INTEGRITY**: You MUST produce a single continuous markdown table for each section. **DO NOT** break a table with empty lines or interleaved text. Output EVERY row for a section in one block.
   - **MANDATORY**: Cleanly format the summary data into a Markdown Table.
   - **CATEGORIZATION ACCURACY (CRITICAL)**:
     - **EQUITY AND LIABILITIES**: Capital, Loans, Sundry Creditors, Provisions, Outstanding Expenses.
     - **ASSETS**: Fixed Assets (Cars, Land, Furniture), Sundry Debtors, Bank Balance, Cash in Hand, Deposits, Prepaid Expenses.
     - **SUNDRY DEBTORS ARE ALWAYS ASSETS**. Do NOT place them in Liabilities.
   - **VERTICALIZATION RULE (CRITICAL)**: Many documents show Liabilities and Assets side-by-side in a 4-column layout. You MUST VERTICALIZE this.
     - NEVER produce a table with 4 columns (Liabilities, Amount, Assets, Amount).
     - Process the entire "LIABILITIES" column/side first.
     - Then process the entire "ASSETS" column/side first.
     - Output them as TWO SEPARATE TABLES, one after the other.
   - **BALANCE SHEET COMPLETENESS**: For a Balance Sheet, you MUST provide BOTH an "ASSETS" table AND an "EQUITY AND LIABILITIES" table. DO NOT stop after the first table.
   - **STRICT FORMATTING**: 
     - **START DIRECTLY** with the first markdown table.
     - **DO NOT** include any report titles, headers, or introductory text (e.g. NO "BALANCE SHEET").
     - **DO NOT** use markdown headers (#) or bolding (**).
     - **DO NOT** wrap the entire response in markdown code blocks (```markdown or ```).
   - **REPORT FORMAT**: 
      | Particulars | Amount |
      | :--- | :--- |
      | [Row Data] | [Amount] |
      
      {{"transactions": []}}
   - If the request is for a report and you found NO such report in the context, return:
     "{{"transactions": [], "message": "I could not find a structured [Report Type] table in the provided documents."}}"

4. **STRICT PRE-FILTERING (HIGHEST PRIORITY):**
   - Identify the SPECIFIC INTENT: Is the user asking for "Cash", "UPI", "ATM", "NEFT", etc.?
   - **CASH FILTER**: If the user asks for "Cash", you MUST EXCLUDE any transaction that contains UPI identifiers, VPA IDs, or NEFT/IMPS markers. ONLY include "CASH", "ATM", "WITHDRAWAL", "SELF", or "WDL".
   - **UPI FILTER**: If the user asks for "UPI", you MUST EXCLUDE any transaction that contains "CASH" or "ATM" identifiers.
   - **STRICT EXCLUSION**: Your goal is NOT to find "similar" things, but to find EXACT matches for the requested type. 
   - If a transaction is ambiguous or does not explicitly match the requested type, **SKIP IT**.

5. **HANDLING TRANSACTION QUERIES:**
   - Transaction dates
   - Amounts (debit/credit)
   - Description/narration
   - UPI IDs, reference numbers, or payee names
   - Transaction types (UPI, NEFT, cash, etc.)
   - **BALANCE VALUES** - These are CRITICAL for determining transaction direction

**PROCESSING WORKFLOW (FOLLOW THIS EXACT ORDER):**
1. **IDENTIFY FILTER**: Determine the specific transaction type or keyword the user is looking for (e.g., "Cash").
2. **SCAN LINE-BY-LINE**: Read through the context.
3. **APPLY FILTER**: For each line, check: "Does this line match the IDENTIFIED FILTER?"
4. **DECIDE**:
   - If NO: **STOP immediately** for this line. Do NOT extract anything. Move to the next line.
   - If YES: Proceed to step 5.
5. **EXTRACT (MATCHING ONLY)**:
   a. Extract: Date, Description, Amount(s), Balance
   b. **FIND THE PREVIOUS LINE'S BALANCE**
   c. **COMPARE**: Current Balance vs Previous Balance
   d. **DETERMINE DIRECTION** (Credit if balance increased, Debit if decreased)
   e. Extract all other fields
6. **VERIFY COMPLETENESS**: Ensure EVERY record that matches the filter is extracted.
7. **VERIFY PURITY**: Ensure NO record that fails the filter (e.g. no UPI in a Cash search) is extracted.

3. **COMPLETE EXTRACTION REQUIREMENT:**
   - Extract **EVERY SINGLE RECORD** that matches the user's query from the provided context
   - Include **BOTH CREDIT AND DEBIT** transactions - do NOT filter by direction
   - **CREDITS ARE EQUALLY IMPORTANT AS DEBITS** - Do NOT favor debits over credits
   - Do NOT summarize, truncate, or limit the number of records
   - Do NOT skip any matching transactions
   - If there are 100 matching records, return all 100
   - If there are 1000 matching records, return all 1000
   - COMPLETENESS is more important than brevity
   
   **🚨 CRITICAL: CREDIT TRANSACTION IDENTIFICATION (HIGHEST PRIORITY) 🚨**
   
   **CREDITS ARE MONEY COMING IN - THEY MUST BE EXTRACTED AND VISIBLE:**
   - **CREDIT = Money INCOMING** (Deposits, Receipts, Salary, Interest, Refunds, Dividends, Transfers Received, etc.)
   - **DEBIT = Money OUTGOING** (Withdrawals, Payments, Expenses, Transfers Sent, etc.)
   
   **BANK STATEMENT COLUMN STRUCTURE:**
   - Bank statements typically have THREE numeric columns: **[Withdrawal/Debit] [Deposit/Credit] [Balance]**
   - In plain text, these appear as three consecutive numbers separated by spaces
   - **THE SECOND NUMBER IS ALMOST ALWAYS THE CREDIT COLUMN**
   
   **CREDIT IDENTIFICATION RULES (MANDATORY - FOLLOW THIS EXACT ORDER):**
   
   **STEP 1: BALANCE COMPARISON (PRIMARY & MANDATORY METHOD - USE THIS FIRST):**
   - **FOR EVERY TRANSACTION, YOU MUST:**
     1. Identify the balance value on the current line
     2. Find the balance from the PREVIOUS transaction line
     3. Compare: Current Balance vs Previous Balance
     4. If Current Balance > Previous Balance → **CREDIT** (balance increased = money came in)
     5. If Current Balance < Previous Balance → **DEBIT** (balance decreased = money went out)
   
   - **Examples:**
     - Previous line balance: 19,773.73
     - Current line: "22/11/24 TXN-NARRATION-DATA 350.00 20,123.73"
     - Current balance: 20,123.73
     - Comparison: 20,123.73 > 19,773.73 → Balance INCREASED → **CREDIT**
     
     - Previous line balance: 20,123.73
     - Current line: "22/11/24 PAYMENT-DETAIL 500.00 19,623.73"
     - Current balance: 19,623.73
     - Comparison: 19,623.73 < 20,123.73 → Balance DECREASED → **DEBIT**
   
   - **CRITICAL**: This method works for ALL bank statement formats, even when columns are unclear
   - **CRITICAL**: If you cannot find a previous balance, look for the opening balance or use the first transaction's balance as reference
   
   **STEP 2: COLUMN POSITION CHECK (SECONDARY METHOD - USE IF BALANCE COMPARISON IS UNCLEAR):**
      - If you see: "Date Description 0.00 500.00 Balance" → The 500.00 is in the 2nd column = **CREDIT**
      - If you see: "Date Description 500.00 0.00 Balance" → The 500.00 is in the 1st column = **DEBIT**
      - **ALWAYS check BOTH columns - never assume the first number is the only transaction**
   
   **STEP 3: KEYWORD DETECTION (TERTIARY METHOD - USE AS CONFIRMATION):**
      - **CREDIT keywords**: "Deposit", "CR", "Credit", "Interest", "Received", "Refund", "Salary", "Inward", "Credit to", "Received from", "NEFT-CREDIT", "IMPS-CREDIT", "RTGS-CREDIT", "Dividend", "Bonus", "Reversal", "Reversal of", "Refund of"
      - **DEBIT keywords**: "Withdrawal", "DR", "Debit", "Payment", "Paid", "Outward", "Payment to", "Transfer to", "NEFT-DEBIT", "IMPS-DEBIT", "RTGS-DEBIT"
      - If description contains CREDIT keywords → **direction = "CREDIT"**
      - If description contains DEBIT keywords → **direction = "DEBIT"**
   
   **STEP 4: ZERO VALUE MARKERS**:
      - **DO NOT IGNORE 0.00 VALUES** - They indicate which column has the actual transaction
      - Pattern: "0.00 500.00" → 500.00 is a **CREDIT** (first column is 0, second has value)
      - Pattern: "500.00 0.00" → 500.00 is a **DEBIT** (first column has value, second is 0)
   
   **STEP 5: SINGLE AMOUNT DETECTION**:
      - If only ONE amount appears on a line (format: "Date Description Amount Balance"):
        - **FIRST**: Compare balance with previous line → If increased = **CREDIT**, if decreased = **DEBIT**
        - **THEN**: Check for CREDIT keywords in description → "Received", "Credit", "Interest", "CR" → **CREDIT**
        - **THEN**: Check for DEBIT keywords → "Paid", "Payment", "Debit", "DR" → **DEBIT**
        - **DEFAULT**: If balance increased, it's a **CREDIT** (this is the most reliable indicator)
   
   **STEP 6: SMOOSHED NUMBERS**:
      - If numbers appear together like "100.005000.00", split them
      - First number = Transaction amount
      - Second number = Balance
      - Compare with previous balance to determine direction
   
   **CREDIT EXTRACTION EXAMPLES (REAL BANK STATEMENT FORMATS):**
   
   **Example 1 - Two Number Format (Amount + Balance):**
   - Previous balance: 19,773.73
   - Line: "22/11/24 TXN-NARRATION-DATA SINGH-PAYTMQR1LJPTAZGXV@PAYTM 350.00 20,123.73"
   - Current balance: 20,123.73
   - Comparison: 20,123.73 > 19,773.73 → Balance INCREASED → **CREDIT** 
   
   **Example 2 - Three Number Format (Debit + Credit + Balance):**
   - Line: "22/11/24 Salary Credit 0.00 50000.00 60000.00"
   - Previous balance: 10,000.00
   - Current balance: 60,000.00
   - Comparison: 60,000.00 > 10,000.00 → Balance INCREASED → **CREDIT** 
   - Also: 50,000.00 is in 2nd column (Credit column) → Confirms **CREDIT** 
   
   **Example 3 - Single Amount with Balance:**
   - Previous balance: 10,000.00
   - Line: "22/11/24 RECEIVED-DATA from John 1000.00 11000.00"
   - Current balance: 11,000.00
   - Comparison: 11,000.00 > 10,000.00 → Balance INCREASED → **CREDIT** 
   - Also: Keyword "RECEIVED" → Confirms **CREDIT** 
   
   **Example 4 - Interest Payment:**
   - Previous balance: 10,000.00
   - Line: "22/11/24 Interest 500.00 10500.00"
   - Current balance: 10,500.00
   - Comparison: 10,500.00 > 10,000.00 → Balance INCREASED → **CREDIT** 
   
   **Example 5 - Refund:**
   - Previous balance: 10,000.00
   - Line: "22/11/24 Refund 0.00 2000.00 12000.00"
   - Current balance: 12,000.00
   - Comparison: 12,000.00 > 10,000.00 → Balance INCREASED → **CREDIT** 
   - Also: 2,000.00 is in 2nd column (Credit column) → Confirms **CREDIT** 
   
   **MANDATORY VALIDATION FOR EVERY TRANSACTION:**
   - **BEFORE marking direction, you MUST:**
     1.  Find the balance on the current line
     2.  Find the balance from the previous transaction line (or opening balance)
     3.  Compare: Current Balance vs Previous Balance
     4.  If Current > Previous → Mark as **CREDIT**
     5.  If Current < Previous → Mark as **DEBIT**
     6.  If Current = Previous → Check for other indicators (rare case)
   
   - **AFTER extracting each transaction, verify:**
     1. Did I compare the balance with the previous line? (REQUIRED)
     2. Did I check BOTH amount columns if present?
     3. Did I look for CREDIT/DEBIT keywords as confirmation?
     4. If balance increased, did I mark it as "CREDIT"? (DO NOT mark as DEBIT if balance increased!)
   
   - **CRITICAL RULES:**
     - **DO NOT SKIP CREDITS** - They are just as important as debits
     - **BALANCE INCREASE = CREDIT** - This is the most reliable indicator
     - **If balance increased, it CANNOT be a DEBIT** - Always mark as CREDIT
     - **If you're unsure, default to balance comparison - if balance increased, it's a CREDIT**
     - **Track balance sequentially** - Process transactions line by line, maintaining balance state

4. **Output Format:** Return results as a STRICT valid JSON object with this structure:
   {{
     "transactions": [
       {{
         "date": "transaction date",
         "description": "transaction description/narration (include the full text for accuracy)",
         "amount": "transaction amount (ABSORLUTELY REQUIRED: Use the non-zero numeric value. NEVER use 0.00)",
         "direction": "CREDIT or DEBIT",
         "transaction_id": "UPI ID/reference number/Chq No/Instrument No if available",
         "type": "UPI/NEFT/CASH/CHQ/IMPS/RTGS etc if identifiable",
         "bank_name": "MANDATORY: Use the 'STATEMENT_BANK' name from the block header exactly. DO NOT guess based on narrations.",
         "source_document": "name of the document this transaction was found in"
       }}
     ]
   }}
   
   **DIRECTION DETERMINATION (MANDATORY FOR EVERY TRANSACTION):**
   - Before setting "direction", you MUST:
     1. Identify the balance on the current transaction line
     2. Identify the balance from the previous transaction line
     3. Compare them:
        - If current balance > previous balance → "direction": "CREDIT"
        - If current balance < previous balance → "direction": "DEBIT"
     4. If you cannot find previous balance, look for opening balance or use column position/keywords
   
   - **CRITICAL**: If you found NO records in this specific context block, return an empty list: {{"transactions": []}}.
   - **CRITICAL**: Do NOT list "0.00" as the transaction amount. Use the actual numeric value from the other column.
   - **CRITICAL**: Do NOT default all transactions to "DEBIT". Many transactions are CREDITS - use balance comparison to determine.

5. **No Data Found:** If no matching records OR summary reports are found, return: {{"transactions": [], "message": "I could not find the requested information in the documents."}}

**Response Guidelines:**
- Extract EVERY SINGLE piece of valid data that matches the user's request - NO EXCEPTIONS
- Return the COMPLETE dataset - do NOT summarize or provide a sample
- **CREDIT TRANSACTIONS ARE MANDATORY**: Ensure you extract ALL credit transactions. If you see deposits, receipts, salary, interest, refunds, or any money coming IN, they MUST be included with `"direction": "CREDIT"`
- **ZERO AMOUNT RULE**: If you extract a record with `amount: 0.0` or `0.00`, you have FAILED. Look at the numbers on that same line again. One of them is non-zero. Use THAT one.
- **BALANCE-BASED VALIDATION**: Before finalizing each transaction, verify the direction by checking if the balance increased (CREDIT) or decreased (DEBIT)
- **BANK NAME CONSISTENCY**: Do NOT change the `bank_name` based on the payee or UPI ID (like @oksbi). Use the bank name of the statement owner.
- Keep descriptions complete including any reference numbers found at the end of the line.
- **FINAL CHECK**: Before returning results, count how many CREDIT vs DEBIT transactions you found. If you found significantly more DEBITS than CREDITS, you may have missed some credit transactions. Re-check the data.
- **STRICT FILTER ADHERENCE**: If the user asked for "Cash", and you see ANY transaction with "UPI", "VPA", or "@" identifiers, you HAVE FAILED. REMOVE THEM. Only matching records must remain.

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
            
            if "data" in parsed_json and "transactions" not in parsed_json:
                parsed_json["transactions"] = parsed_json.pop("data")
            
            # Applying secondary Python filter
            if "transactions" in parsed_json:
                initial_count = len(parsed_json["transactions"])
                parsed_json["transactions"] = [tx for tx in parsed_json["transactions"] if not self._should_filter_transaction(tx, question)]
                if initial_count > len(parsed_json["transactions"]):
                    print(f"[FILTER] Standard answer: Filtered out {initial_count - len(parsed_json['transactions'])} non-matching transactions.")

            sources = list(set(source_documents))

            # For summary-style questions, make the "where did it come from?" explicit in the
            # human-readable answer shown in Search UI.
            q_lower = (question or "").lower()
            is_summary = any(kw in q_lower for kw in ["balance sheet", "p&l", "profit", "loss", "report", "summary", "computation"])
            # Clean up the human-readable answer (remove raw JSON blobs)
            human_answer = answer
            # Remove ```json ... ``` blocks
            human_answer = re.sub(r'```json\s*.*?\s*```', '', human_answer, flags=re.DOTALL)
            # Remove any raw { ... } that looks like JSON if the whole line is JSON
            human_lines = []
            for line in human_answer.split('\n'):
                line_strip = line.strip()
                if (line_strip.startswith('{') and line_strip.endswith('}')) or \
                   (line_strip.startswith('[') and line_strip.endswith(']')):
                    # Check if it actually parses as JSON to be sure
                    try:
                        json.loads(line_strip)
                        continue # Skip this line
                    except:
                        pass
                human_lines.append(line)
            human_answer = "\n".join(human_lines).strip()

            # Fallback for human_answer if it was purely JSON that got scrubbed
            if not human_answer and "message" in parsed_json:
                human_answer = parsed_json["message"]
            elif not human_answer and "transactions" in parsed_json and parsed_json["transactions"]:
                human_answer = f"I found {len(parsed_json['transactions'])} matching records."
            elif not human_answer:
                human_answer = "I could not find the information you requested in the uploaded documents."

            return {
                "answer": json.dumps(parsed_json),
                "full_answer": human_answer, 
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
        batch_size: int = 5
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

        # If the user asked for a summary report (Balance Sheet / P&L / Computation),
        # aggressively focus on only those document types; otherwise the model will be
        # overwhelmed by irrelevant bank statement chunks and respond "not found".
        q_lower = (question or "").lower()
        is_summary = any(kw in q_lower for kw in ["balance sheet", "p&l", "profit", "loss", "report", "summary", "computation"])
        if is_summary and doc_metadata:
            wanted_types = set()
            if "balance sheet" in q_lower:
                wanted_types.add("balance sheet")
            if "p&l" in q_lower or "profit" in q_lower or "loss" in q_lower:
                wanted_types.add("profit")
                wanted_types.add("loss")
                wanted_types.add("p&l")
            if "computation" in q_lower:
                wanted_types.add("computation")

            def _doc_matches(doc_name: str) -> bool:
                meta = doc_metadata.get(doc_name) or {}
                doc_type = str(meta.get("document_type", "")).lower()
                name = (doc_name or "").lower()
                
                # Rule 1: Match by AI-classified type
                if wanted_types and any(w in doc_type for w in wanted_types):
                    return True
                
                # Rule 2: Match by filename hints
                name_hints = ["balance", "bl.", "bl_", "-bl", "bs.", "p&l", "pl.", "pl_", "-pl", "profit", "loss", "computation"]
                if any(h in name for h in name_hints):
                    # Only match if the hint corresponds to the WANTED type
                    if "balance" in q_lower and any(h in name for h in ["balance", "bl.", "bl_", "-bl", "bs."]):
                        return True
                    if ("p&l" in q_lower or "profit" in q_lower) and any(h in name for h in ["p&l", "pl.", "pl_", "-pl", "profit", "loss"]):
                        return True
                    if "computation" in q_lower and "computation" in name:
                        return True
                return False

            keep_docs = {d for d in set(source_documents) if _doc_matches(d)}
            if keep_docs:
                filtered = [(c, d) for c, d in zip(context_chunks, source_documents) if d in keep_docs]
                if filtered:
                    context_chunks = [c for c, _ in filtered]
                    source_documents = [d for _, d in filtered]
                    print(f"[INFO] Summary focus enabled: filtered to {len(context_chunks)} chunks from {len(keep_docs)} docs")
            else:
                print("[INFO] Summary focus (stream): No document matches found; using full context to avoid missing data.")
        elif doc_metadata:
             # Transaction search focus: exclude documents that are obviously NOT statements/ledgers
            exclude_types = ["computation", "balance sheet", "p&l", "profit", "loss", "ledger", "journal", "capital account", "tax", "computation of income"]
            skip_docs = {d for d, m in doc_metadata.items() if any(et in str(m.get("document_type", "")).lower() for et in exclude_types) or any(et in (d or "").lower() for et in ["cp.pdf", "computation", "ledger"])}
            
            if skip_docs:
                filtered = [(c, d) for c, d in zip(context_chunks, source_documents) if d not in skip_docs]
                if filtered:
                    context_chunks = [c for c, _ in filtered]
                    source_documents = [d for _, d in filtered]
                    print(f"[INFO] Transaction search focus (sync): excluded {len(skip_docs)} non-transaction documents: {list(skip_docs)}")
        
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
        seen_fingerprints = set()

        seen_answers = set()
        for result in results:
            # IMPORTANT: Use the raw LLM text (markdown/table) when available so summary
            # answers like Balance Sheets show up in Search UI.
            ans = (result.get('full_answer') or result['answer']).strip()
            
            # If it's a generic "no records" JSON, only add it to full_answer if we have NO other content yet
            is_empty_msg = '{"transactions": [], "message":' in ans or '{"transactions": []}' in ans
            if is_empty_msg:
                if not full_answer.strip():
                    full_answer = ans
            else:
                # If we have a real answer, append it
                if '{"transactions": [], "message":' in full_answer:
                    full_answer = ans # Replace "no data" message with real data
                else:
                    full_answer += ans + "\n---\n"


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
                        # Include direction to prevent credits and debits from being considered duplicates
                        desc = str(t.get('description', '')).strip().lower()
                        amt = str(t.get('amount', '0')).replace(',', '').replace('.', '')  # Normalize amount
                        date = str(t.get('date', ''))
                        direction = str(t.get('direction', '')).upper()
                        tx_id = str(t.get('transaction_id', '')).strip()
                        source_doc = str(t.get('source_document', '')).strip()
                        bank_name = str(t.get('bank_name', '')).strip()
                        
                        # Use transaction_id if available for better uniqueness
                        if tx_id:
                            fp = f"{date}|{amt}|{direction}|{tx_id}"
                        else:
                            # Include direction, source_document, and bank_name to distinguish similar transactions
                            # Use first 100 chars of description to handle very long descriptions
                            desc_short = desc[:100] if len(desc) > 100 else desc
                            fp = f"{date}|{amt}|{direction}|{desc_short}|{source_doc}|{bank_name}"
                        
                        if fp not in seen_fingerprints:
                            seen_fingerprints.add(fp)
                            all_transactions.append(t)

            except Exception as e:
                print(f"[DEBUG] Error merging batch result: {str(e)}")

        # Extract source documents ONLY from transactions that made it into final results
        all_sources = set()
        for tx in all_transactions:
            source_doc = str(tx.get('source_document', '')).strip()
            if source_doc and source_doc.lower() != 'unknown':
                all_sources.add(source_doc)
        # For summary outputs (no transactions), still expose the documents we actually scanned.
        if is_summary and not all_sources:
            all_sources = set([d for d in source_documents if d and d.lower() != "unknown"])

        # Validate credit/debit distribution
        credit_count = sum(1 for tx in all_transactions if str(tx.get('direction', '')).upper() == 'CREDIT')
        debit_count = sum(1 for tx in all_transactions if str(tx.get('direction', '')).upper() == 'DEBIT')
        print(f"[INFO] Exhaustive extraction complete: {len(all_transactions)} total unique records ({credit_count} CREDITS, {debit_count} DEBITS)")
        print(f"[INFO] Source documents with transactions: {len(all_sources)} documents")
        
        # Calculate total before deduplication for comparison
        total_before_dedup = sum(len(json.loads(r['answer']).get('transactions', [])) for r in results)
        print(f"[INFO] Total transactions before deduplication: {total_before_dedup}, after: {len(all_transactions)}")
        
        # Warn if credits seem missing
        if len(all_transactions) > 10 and credit_count == 0:
            print(f"[WARNING] Large dataset ({len(all_transactions)} transactions) with zero credits. This may indicate credit extraction issues.")
        elif len(all_transactions) > 0 and credit_count == 0:
            print(f"[WARNING] No CREDIT transactions found in final results. Please verify credit identification logic.")
        elif credit_count > 0:
            credit_percentage = (credit_count / len(all_transactions)) * 100
            print(f"[INFO] Credit transactions: {credit_count} ({credit_percentage:.1f}% of total)")

        # APPLY SECONDARY PYTHON FILTER
        initial_count = len(all_transactions)
        all_transactions = [tx for tx in all_transactions if not self._should_filter_transaction(tx, question)]
        if initial_count > len(all_transactions):
            print(f"[FILTER] Exhaustive answer (sync): Filtered out {initial_count - len(all_transactions)} non-matching transactions.")

        return {
            "answer": json.dumps({"transactions": all_transactions}),
            "full_answer": full_answer,
            "sources": list(all_sources)
        }


    async def generate_summary_report(
        self,
        question: str,
        context_chunks: List[str],
        source_documents: List[str],
    ) -> Dict[str, Any]:
        """
        Generate summary reports (Balance Sheet / P&L / Computation) for Search using LLM.
        """
        if not context_chunks:
            return {"answer": "No document content provided.", "sources": []}

        # Combine chunks into a single text block for the LLM
        combined_context = "\n\n".join(context_chunks)
        sources = sorted(set([d for d in source_documents or [] if d and d.lower() != "unknown"]))

        prompt = f"""You are a specialized financial analyst. Your task is to extract a structured Financial Report (Balance Sheet, Profit & Loss, or Computation of Income) from the provided document context.

**USER REQUEST:** "{question}"

**DOCUMENT CONTEXT:**
{combined_context}

**INSTRUCTIONS:**
1. **CATEGORIZATION ACCURACY (HIGHEST PRIORITY)**:
   - **EQUITY AND LIABILITIES**: Capital, Loans, Sundry Creditors, Provisions, Outstanding Expenses.
   - **ASSETS**: Fixed Assets (Cars, Land, Furniture), Sundry Debtors, Bank Balance, Cash in Hand, Deposits, Prepaid Expenses.
   - **SUNDRY DEBTORS ARE ALWAYS ASSETS**. Do NOT place them in Liabilities.
   - **SUNDRY CREDITORS ARE ALWAYS LIABILITIES**.

2. **Identify the Report Type**: Determine if the context contains a Balance Sheet, P&L Account, or Computation of Income.
3. **Handle OCR Artifacts**: Be aware that headers might be spaced out (e.g., "L I A B I L I T I E S") or misspelled (e.g., "ASSESTS"). Use your intelligence to map them correctly.
4. **Accuracy is Critical**:
   - For Balance Sheets: Correctly separate **Liabilities/Equity** and **Assets**. (e.g. Loans are Liabilities, Cars/Fixed Assets are Assets).
   - For P&L: Separate **Income/Revenue** and **Expenses/Expenditure**.
   - Preserve the exact names and amounts as seen in the document.
5. **TABLE INTEGRITY**: You MUST produce a single continuous markdown table for each section. **DO NOT** break a table with empty lines or interleaved text. Output EVERY row for a section in one block.
6. **VERTICALIZATION RULE (CRITICAL)**: Many documents show Liabilities and Assets side-by-side in a 4-column layout. You MUST VERTICALIZE this.
   - NEVER produce a table with 4 columns (Liabilities, Amount, Assets, Amount).
   - Process the entire "LIABILITIES" column/side first.
   - Then process the entire "ASSETS" column/side first.
   - Output them as TWO SEPARATE TABLES, one after the other.
7. **BALANCE SHEET COMPLETENESS**: For a Balance Sheet, you MUST provide BOTH an "ASSETS" section AND an "EQUITY AND LIABILITIES" section. Extract all rows for both sides.
8. **STRICT FORMATTING**: 
   - **START DIRECTLY** with the first markdown table.
   - **DO NOT** include any report titles, headers, or introductory text (e.g. NO "BALANCE SHEET").
   - **DO NOT** use markdown headers (#) or bolding (**).
   - **DO NOT** wrap the entire response in markdown code blocks (```markdown or ```).
9. **Output Format**:
   - **Textual Part**: Start directly with the first markdown table.
   - **JSON Part**: At the end, include a JSON block with the following structure:
     - If Balance Sheet: `{{"transactions": [], "balance_sheet": {{"liabilities": [{{"particulars": "...", "amount": "..."}}], "assets": [{{"particulars": "...", "amount": "..."}}]}}}}`
     - If P&L: `{{"transactions": [], "p_and_l": {{"income": [{{"particulars": "...", "amount": "..."}}], "expenses": [{{"particulars": "...", "amount": "..."}}]}}}}`

**RESPONSE (Markdown + JSON):**
"""
        try:
            print(f"[INFO] Generating LLM-based summary report for: {question[:50]}...")
            response = await self.llm.ainvoke(prompt)
            answer = response.content.strip()
            
            # Extract JSON and metadata
            parsed_json = self._extract_json(answer)
            
            # Clean up the human-readable answer (remove raw JSON blobs)
            human_answer = answer
            human_answer = re.sub(r'```json\s*.*?\s*```', '', human_answer, flags=re.DOTALL)
            human_lines = []
            for line in human_answer.split('\n'):
                line_strip = line.strip()
                if (line_strip.startswith('{') and line_strip.endswith('}')) or \
                   (line_strip.startswith('[') and line_strip.endswith(']')):
                    try:
                        json.loads(line_strip)
                        continue
                    except:
                        pass
                human_lines.append(line)
            human_answer = "\n".join(human_lines).strip()

            # Ensure the structured data is passed back in the 'answer' field for the frontend
            return {
                "answer": json.dumps(parsed_json),
                "full_answer": human_answer,
                "sources": sources
            }
        except Exception as e:
            print(f"[ERROR] LLM Summary Report generation failed: {e}")
            raise

    def _should_filter_transaction(self, t: Dict[str, Any], question: str) -> bool:
        """
        Hard-coded secondary filter to remove obvious non-matches that the LLM might leak.
        Returns True if the transaction SHOULD BE FILTERED OUT (removed).
        """
        q_lower = (question or "").lower()
        desc = str(t.get("description", "")).lower()
        t_type = str(t.get("type", "")).lower()
        
        # 1. CASH/ATM STRICT FILTER
        if any(kw in q_lower for kw in ["cash", "atm", "self", "withdrawal", "wdl"]):
            # If user asks for CASH, block anything that looks like UPI or NEFT
            if any(kw in desc for kw in ["upi", "vpa", "@", "paytm", "g-pay", "phonepe", "neft", "imps", "rtgs"]):
                return True
            if any(kw in t_type for kw in ["upi", "neft", "imps", "rtgs"]):
                return True
                
        # 2. UPI STRICT FILTER
        if any(kw in q_lower for kw in ["upi", "vpa"]):
            # If user asks for UPI, block anything that looks like Cash or ATM
            if any(kw in desc for kw in ["cash", "atm", "self", "withdrawal", "wdl"]):
                return True
            if any(kw in t_type for kw in ["cash", "atm", "withdrawal"]):
                return True
                
        return False

    async def stream_exhaustive_answer(
        self,
        question: str,
        context_chunks: List[str],
        source_documents: List[str],
        batch_size: int = 5
    ):
        """
        Stream extraction results as they are processed.
        """
        
        if not context_chunks:
            yield json.dumps({"type": "error", "message": "No context provided"})
            return

        # Step 1: Pre-scan for document metadata
        doc_metadata = await self._extract_document_metadata(context_chunks, source_documents)

        # Summary focus: restrict to Balance Sheet / P&L / Computation docs.
        q_lower = (question or "").lower()
        is_summary = any(kw in q_lower for kw in ["balance sheet", "p&l", "profit", "loss", "report", "summary", "computation"])
        filtered_sources_for_summary = None
        if is_summary and doc_metadata:
            wanted_types = set()
            if "balance sheet" in q_lower:
                wanted_types.add("balance sheet")
            if "p&l" in q_lower or "profit" in q_lower or "loss" in q_lower:
                wanted_types.add("profit")
                wanted_types.add("loss")
                wanted_types.add("p&l")
            if "computation" in q_lower:
                wanted_types.add("computation")

            def _doc_matches(doc_name: str) -> bool:
                meta = doc_metadata.get(doc_name) or {}
                doc_type = str(meta.get("document_type", "")).lower()
                name = (doc_name or "").lower()
                
                # Rule 1: Match by AI-classified type
                if wanted_types and any(w in doc_type for w in wanted_types):
                    return True
                
                # Rule 2: Match by filename hints
                if "balance" in q_lower and any(h in name for h in ["balance", "bl.", "bl_", "-bl", "bs."]):
                    return True
                if ("p&l" in q_lower or "profit" in q_lower) and any(h in name for h in ["p&l", "pl.", "pl_", "-pl", "profit", "loss"]):
                    return True
                if "computation" in q_lower and "computation" in name:
                    return True
                return False

            keep_docs = {d for d in set(source_documents) if _doc_matches(d)}
            if keep_docs:
                filtered = [(c, d) for c, d in zip(context_chunks, source_documents) if d in keep_docs]
                if filtered:
                    context_chunks = [c for c, _ in filtered]
                    source_documents = [d for _, d in filtered]
                    filtered_sources_for_summary = sorted(set(source_documents))
                    print(f"[INFO] Summary focus enabled (stream): filtered to {len(context_chunks)} chunks from {len(keep_docs)} docs")
            else:
                print("[INFO] Summary focus (stream): No document matches found; using full context to avoid missing data.")
        elif doc_metadata:
            # Transaction focus: exclude documents that are obviously NOT statements/ledgers
            exclude_types = ["computation", "balance sheet", "p&l", "profit", "loss", "ledger", "journal", "capital account", "tax", "computation of income"]
            skip_docs = {d for d, m in doc_metadata.items() if any(et in str(m.get("document_type", "")).lower() for et in exclude_types) or any(et in (d or "").lower() for et in ["cp.pdf", "computation", "ledger"])}
            
            if skip_docs:
                filtered = [(c, d) for c, d in zip(context_chunks, source_documents) if d not in skip_docs]
                if filtered:
                    context_chunks = [c for c, _ in filtered]
                    source_documents = [d for _, d in filtered]
                    print(f"[INFO] Transaction search focus: excluded {len(skip_docs)} non-transaction documents: {list(skip_docs)}")
        
        total_batches = (len(context_chunks) + batch_size - 1) // batch_size
        tasks = []
        sem = asyncio.Semaphore(6) # Increased for speed (Higher throughput)
        
        async def process_batch_with_sem(batch, batch_source_docs):
            async with sem:
                return await retry_with_backoff(self.generate_answer, question, batch, batch_source_docs, doc_metadata)

        print(f"[INFO] Streaming exhaustive extraction: {len(context_chunks)} chunks")

        # Explicitly create Tasks so we can cancel them if the stream is aborted
        running_tasks = []
        for i in range(0, len(context_chunks), batch_size):
            batch = context_chunks[i:i + batch_size]
            batch_source_docs = source_documents[i:i + batch_size]
            # Create task directly
            task = asyncio.create_task(process_batch_with_sem(batch, batch_source_docs))
            running_tasks.append(task)

        completed_count = 0
        all_transactions_count = 0
        all_credit_count = 0
        all_debit_count = 0
        all_sources = set()
        seen_fingerprints = set() # For streaming de-duplication
        all_unique_transactions = []  # Store all unique transactions to extract sources at end
        full_answers = [] # Aggregated textual responses

        try:
            for future in asyncio.as_completed(running_tasks):
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
                
                    # Collect textual answer
                    if result.get('full_answer'):
                        ans = result['full_answer'].strip()
                        # Skip if it's just boilerplate "not found" or purely JSON-like
                        is_bad = any(kw in ans.lower() for kw in ["not find", "no transactions", "not available"]) or ans.startswith('{') or ans.startswith('[')
                    
                        if is_bad:
                            # Only keep if we have nothing else yet
                            if all_transactions_count == 0 and not full_answers:
                                full_answers.append(ans)
                        else:
                            # If we have real data, remove previous placeholders and JSON
                            full_answers = [a for a in full_answers if not (any(kw in a.lower() for kw in ["not find", "no transactions", "not available"]) or a.startswith('{') or a.startswith('['))]
                            if ans not in full_answers:
                                full_answers.append(ans)

                
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
                                # Include direction to prevent credits and debits from being considered duplicates
                                desc = str(t.get('description', '')).strip().lower()
                                amt = str(t.get('amount', '0')).replace(',', '').replace('.', '')  # Normalize amount
                                date = str(t.get('date', ''))
                                direction = str(t.get('direction', '')).upper()
                                tx_id = str(t.get('transaction_id', '')).strip()
                                source_doc = str(t.get('source_document', '')).strip()
                                bank_name = str(t.get('bank_name', '')).strip()
                            
                                # Use transaction_id if available for better uniqueness
                                if tx_id:
                                    fp = f"{date}|{amt}|{direction}|{tx_id}"
                                else:
                                    # Include direction, source_document, and bank_name to distinguish similar transactions
                                    # Use first 100 chars of description to handle very long descriptions
                                    desc_short = desc[:100] if len(desc) > 100 else desc
                                    fp = f"{date}|{amt}|{direction}|{desc_short}|{source_doc}|{bank_name}"
                            
                                if fp not in seen_fingerprints:
                                    # Apply secondary hard-coded filter
                                    if self._should_filter_transaction(t, question):
                                        print(f"[FILTER] Dropped non-matching transaction: {desc[:50]}")
                                        continue
                                    
                                    seen_fingerprints.add(fp)
                                    unique_txs.append(t)
                                    all_unique_transactions.append(t)  # Store for source extraction
                                else:
                                    # Log when a transaction is being skipped as duplicate
                                    skipped_direction = str(t.get('direction', '')).upper()
                                    if skipped_direction == 'CREDIT':
                                        print(f"[DEBUG] Skipping duplicate CREDIT transaction: {date}|{amt}|{desc[:50]}")

                            if unique_txs:
                                count = len(unique_txs)
                                batch_credits = sum(1 for tx in unique_txs if str(tx.get('direction', '')).upper() == 'CREDIT')
                                batch_debits = sum(1 for tx in unique_txs if str(tx.get('direction', '')).upper() == 'DEBIT')
                                all_credit_count += batch_credits
                                all_debit_count += batch_debits
                                print(f"[DEBUG] Stream batch {completed_count}: Found {count} unique items ({batch_credits} CREDITS, {batch_debits} DEBITS)")
                                print(f"[DEBUG] Stream batch {completed_count}: Yielding {count} transactions to frontend (cumulative: {all_credit_count} CREDITS, {all_debit_count} DEBITS)")
                                all_transactions_count += count
                                yield json.dumps({
                                    "type": "data",
                                    "transactions": unique_txs
                                })
                        else:
                            print(f"[DEBUG] Stream batch {completed_count}: No new transactions found")
                        
                    except Exception as e:
                        print(f"[WARNING] Failed to parse batch result in stream: {e}")
                        
                except Exception as e:
                    print(f"[ERROR] Stream batch failed: {e}")
                    yield json.dumps({"type": "error", "message": str(e)})

        except GeneratorExit:
            print(f"[INFO] Stream interrupted by client. Cancelling {len([ t for t in running_tasks if not t.done()])} pending tasks...")
            raise
        finally:
            # Ensure all pending tasks are cancelled when the generator exits (success or error)
            cancelled_count = 0
            for task in running_tasks:
                if not task.done():
                    task.cancel()
                    cancelled_count += 1
            if cancelled_count > 0:
                print(f"[INFO] Cancelled {cancelled_count} pending background tasks.")
        # Extract source documents ONLY from transactions that made it into final results
        for tx in all_unique_transactions:
            source_doc = str(tx.get('source_document', '')).strip()
            if source_doc and source_doc.lower() != 'unknown':
                all_sources.add(source_doc)
        # For summary outputs (no transactions), still show the documents we scanned.
        if is_summary and not all_sources:
            if filtered_sources_for_summary is not None:
                # If we filtered sources for the summary, use those specific sources
                all_sources = set(filtered_sources_for_summary)
            else:
                # Otherwise, fallback to all valid source documents
                all_sources = set([d for d in source_documents if d and d.lower() != "unknown"])

        # Final summary with credit/debit breakdown
        final_full_answer = "\n---\n".join(full_answers).strip()
        # If we found transactions, scrub any remaining "not found" boilerplate from the text
        if all_transactions_count > 0:
            not_found_patterns = [
                "I could not find the requested information in the documents.",
                "No transactions found",
                "Information not available in uploaded records"
            ]
            for pat in not_found_patterns:
                final_full_answer = final_full_answer.replace(pat, "")
            final_full_answer = final_full_answer.replace('{"transactions": []}', "").strip()

        print(f"[INFO] Streaming extraction complete: {all_transactions_count} total ({all_credit_count} CREDITS, {all_debit_count} DEBITS)")
        yield json.dumps({
            "type": "summary",
            "total_transactions": all_transactions_count,
            "total_credits": all_credit_count,
            "total_debits": all_debit_count,
            "sources": list(all_sources),
            "full_answer": final_full_answer
        })
            
