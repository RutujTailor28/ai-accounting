from typing import List, Dict, Any, AsyncGenerator, Union
import json
import asyncio
import hashlib
from langchain_openai import ChatOpenAI
from app.core.config import settings
from app.services.ai_service import LLMService
from app.ai.rag.retriever import vector_store

class AccountingService:
    """Specialized service for accounting report generation (Journal Entries, Balance Sheets)."""
    
    def __init__(self, model_name: str = None):
        """Initialize with a more capable model for financial synthesis."""
        model = model_name or settings.openrouter_model_accounting
        self.llm = ChatOpenAI(
            model=model,
            openai_api_key=settings.openrouter_api_key,
            openai_api_base=settings.openrouter_base_url,
            temperature=0, # Strict for financial data
            streaming=True
        )
        self.ai_service = LLMService() # Reuse AI service for extraction helpers
        self.shared_accounts = set() # Track account names across batches for consistency
        self.extraction_cache: Dict[str, List[Dict]] = {} # Cache: content_hash -> extracted transactions
        self.structured_tables: List[Dict[str, Any]] = [] # Track structured table data for frontend
        print(f"[INFO] AccountingService initialized with model={model}")
    
    async def _get_base_doc_name(self, filename: str) -> str:
        """Strip extension and then duplicates like (1) or copy."""
        import re
        if not filename:
             return ""
        # Remove extension FIRST
        base = re.sub(r'\.\w+$', '', filename)
        # THEN remove common "duplicate" suffixes
        base = re.sub(r'\s*\(\d+\)$', '', base, flags=re.IGNORECASE)
        base = re.sub(r'\s+copy$', '', base, flags=re.IGNORECASE)
        return base.strip().lower()

    async def _normalize_narration(self, narration: str) -> str:
        """
        Normalize narration for robust deduplication.
        Strips whitespace, special characters, and lowercases the text.
        """
        import re
        if not narration:
            return ""
        # Remove special characters and extra whitespace, then lowercase
        normalized = re.sub(r'[^a-zA-Z0-9]', '', narration).lower()
        return normalized

    async def _extract_transactions_from_batch(self, batch_chunks: List[str], context_query: str = None, company_id: str = None) -> List[Dict]:
        """
        Extract structured transaction data from a small batch of chunks.
        Uses content-based caching to ensure identical chunks produce identical results.
        """
        # Create a deterministic hash of the batch content for caching
        context = "\n".join(batch_chunks)
        query_part = f"_{context_query}" if context_query else ""
        content_hash = hashlib.sha256((context + query_part).encode('utf-8')).hexdigest()
        
        # Check cache first
        if content_hash in self.extraction_cache:
            print(f"[CACHE HIT] Reusing cached extraction for batch (hash: {content_hash[:8]}...)")
            return self.extraction_cache[content_hash]
        
        # Injection of already seen accounts to prevent duplicates
        known_accounts_str = ", ".join(list(self.shared_accounts)[:50]) if self.shared_accounts else "None yet"
        
        query_instruction = ""
        feedback_context = ""

        if context_query:
            query_instruction = f"""
        USER INTENT: "{context_query}"
        
        STRICT PRE-FILTERING RULES:
        1. Identify if the user is asking for a SPECIFIC transaction filter (e.g., "only cash", "only UPI").
        2. If the user is asking for a general report (e.g., "Balance Sheet", "P&L", "summary", "journal"):
           - DO NOT SKIP ANY VALID BANK TRANSACTIONS. Extract everything.
        3. ONLY if the user explicitly requested a specific filter:
           - Skip transactions that DO NOT match the requested type.
           - CASH SEARCH: Exclude any line containing UPI, VPA, NEFT, or @ markers.
           - UPI SEARCH: Exclude any line containing CASH, ATM, or WITHDRAWAL markers.
        """
            # Agentic Feedback Injection
            if company_id:
                try:
                    # Generate embedding for the query to find similar past mistakes
                    from app.services.deps import embedding_service
                    q_emb = embedding_service.generate_embedding(context_query)
                    
                    relevant_feedback = vector_store.query_feedback(q_emb, company_id)
                    if relevant_feedback:
                        feedback_context = "\n**LESSONS LEARNED (PAST USER CORRECTIONS):**\n"
                        for i, fb in enumerate(relevant_feedback):
                            feedback_context += f"- {fb}\n"
                        print(f"[INFO] AccountingService: Injected {len(relevant_feedback)} past corrections into prompt for company {company_id}")
                except Exception as e:
                    print(f"[WARNING] AccountingService: Failed to retrieve feedback: {e}")

        prompt = f"""
        You are an expert Professional Accountant and Data Entry Clerk. Your task is to extract accounting transactions from the text below.
        {query_instruction}
        {feedback_context}

        INPUT TEXT:
        {context}
        
        INSTRUCTIONS:
        1. Extract ONLY matching transactions based on the USER INTENT.
        2. **LITERAL NARRATION**: Preserve the original narration EXACTLY.
        3. **DETERMINISTIC DATES**: Extract dates in DD/MM/YYYY format.
        4. **DOUBLE ENTRY PRINCIPLE**: Every transaction must have at least TWO entries (Debit & Credit). One side is ALWAYS "Bank Account".
           - **Bank Account**: One side is ALWAYS "Bank Account".
             - If statement says DEBIT (Money Out) -> Books: Credit "Bank Account" and Debit an **EXPENSE** or **ASSET** account. (NEVER Debit "Sales" for Money Out unless it is a refund).
             - If statement says CREDIT (Money In) -> Books: Debit "Bank Account" and Credit an **INCOME** or **LIABILITY** account.
           - **Counter Account**: Classify the other side based on narration. 
             - *Examples:* "Fuel Expense", "Office Rent", "Sales Income", "Capital", "Loan from Bank".
        5. **CLASSIFICATION RULE (Ind AS)**:
           - **Current**: Expected to be settled/realized within 12 months.
           - **Non-Current**: Held for long-term use (> 12 months).

        ### TRANSACTION INTERPRETATION RULE (MANDATORY)
        The AI must read bank narration and determine:
        1) Why the transaction happened
        2) Which accounting head it belongs to
        3) Which category it belongs to
        The AI must behave like an experienced accountant.
        The AI must interpret narration intelligently.
        The AI must determine business purpose from narration.

        ### ACCOUNT HEAD IDENTIFICATION RULE
        For every transaction the AI must determine:
        - Account Head
        - Category
        Example:
        Narration: UPI PAYMENT AMAZON
        Account Head: Purchase Expense
        Category: Indirect Expense

        ### ACCOUNT CATEGORY RULE (MANDATORY)
        Every transaction must be assigned one of the following categories ONLY:
        Direct Income
        Indirect Income
        Direct Expense
        Indirect Expense
        Current Assets
        Current Liabilities
        Equity

        No other category names are allowed.
        Income must be classified as: Direct Income or Indirect Income
        Expenses must be classified as: Direct Expense or Indirect Expense
        This fixes your Profit Loss = 0 problem.

        ### INCOME DETECTION RULE
        Credit transactions usually represent Income.
        If narration contains:
        UPI CR
        NEFT CR
        IMPS CR
        RTGS CR
        RECEIVED
        BY TRANSFER
        PAYMENT RECEIVED
        Then classify as:
        Account Head: Sales Income
        Category: Direct Income

        ### EXPENSE DETECTION RULE
        Debit transactions usually represent Expenses.
        If narration contains:
        UPI DR
        POS
        ATM
        PURCHASE
        PAYMENT
        FUEL
        PETROL
        ELECTRICITY
        BILL
        RECHARGE
        BANK CHARGES
        Then classify as:
        Category: Indirect Expense

        ### INTELLIGENT ACCOUNTING RULE
        The AI must NOT classify most transactions as Transfer.
        Transfer classification must be used ONLY if narration contains:
        SELF TRANSFER
        OWN ACCOUNT
        ACCOUNT TRANSFER
        Otherwise treat as business transaction.

        ### MANDATORY TRANSACTION CLASSIFICATION
        Every transaction must be classified.
        No transaction should remain uncategorized.
        Each transaction must belong to one category.
        
        REQUIRED JSON FORMAT:
        {{
            "transactions": [
                {{
                    "date": "DD/MM/YYYY",
                    "narration": "Original narration text",
                    "entries": [
                         {{ "account": "Fuel Expense", "type": "DEBIT", "amount": 500.00, "category": "Indirect Expense" }},
                         {{ "account": "Bank Account", "type": "CREDIT", "amount": 500.00, "category": "Current Assets" }}
                    ]
                }}
            ]
        }}
        
        OUTPUT JSON ONLY:
        """
        max_retries = 5
        base_delay = 10.0

        for attempt in range(max_retries):
            try:
                # We use non-streaming call here for simpler parsing
                response = await self.llm.ainvoke(prompt)
                data = self.ai_service._extract_json(response.content)
                transactions = data.get("transactions", [])
                
                # Cache the result for future use
                self.extraction_cache[content_hash] = transactions
                print(f"[CACHE STORE] Cached {len(transactions)} transactions for batch (hash: {content_hash[:8]}...)")
                
                return transactions
            except Exception as e:
                import traceback
                error_msg = str(e).lower()
                is_rate_limit = False
                delay = base_delay * (2 ** attempt)

                # Check if it's the exact openai RateLimitError
                if type(e).__name__ == "RateLimitError" or "429" in error_msg or "rate limit" in error_msg:
                    is_rate_limit = True
                    # Try to extract the reset header if present in the error string
                    import time
                    try:
                        if hasattr(e, 'response') and e.response is not None:
                            reset_val = e.response.headers.get('x-ratelimit-reset')
                            if reset_val:
                                reset_timestamp = int(reset_val) / 1000.0  # MS to Sec
                                current_timestamp = time.time()
                                delay = max(delay, (reset_timestamp - current_timestamp) + 1.0)
                    except Exception:
                        pass # Fallback to standard exponential backoff
                                
                if is_rate_limit:
                    if attempt < max_retries - 1:
                        print(f"[RATE LIMIT] OpenRouter rate limit hit. Retrying in {delay:0.1f}s (Attempt {attempt+1}/{max_retries})...")
                        await asyncio.sleep(delay)
                        continue
                
                print(f"[WARNING] Batch extraction failed after {attempt + 1} attempts: {e}")
                traceback.print_exc()
                return []
        
        return []

    async def _get_broad_type(self, acc_name: str, cat_name: str, balance: float) -> str:
        """Helper to categorize account nature based on AI category and balance sign."""
        cat_upper = cat_name.upper()
        if "INCOME" in cat_upper: return "INCOME"
        if "EXPENSE" in cat_upper: return "EXPENSE"
        if "ASSET" in cat_upper: return "ASSET"
        if "LIABIL" in cat_upper or "EQUITY" in cat_upper: return "LIABILITY"
        
        # Fallback based on sign (Dr=Asset/Exp, Cr=Liab/Inc)
        if balance > 0: return "ASSET_OR_EXP" 
        return "LIAB_OR_INC"

    async def stream_accounting_synthesis(
        self,
        question: str,
        context_chunks: List[str],
        context_metadatas: List[Dict[str, Any]] = None,
        company_id: str = None,
        previous_context: str = None
    ) -> AsyncGenerator[Union[str, Dict[str, str]], None]:
        """
        Stream the financial synthesis using a Map-Reduce strategy (Batch Extraction -> Final Report).
        If previous_context is provided, it refines the existing report instead of starting from scratch.
        """
        
        # --- REFINEMENT MODE (ITERATION) ---
        if previous_context:
            yield {"status": "Refining previous report based on feedback..."}
            print(f"[INFO] Starting report refinement mode. Context length: {len(previous_context)}")
            
            refinement_prompt = f"""
            You are an expert Senior Chartered Accountant. 
            You previously generated a financial report (Balance Sheet / P&L) for the user.
            The user has provided feedback or requested a change.
            
            YOUR TASK:
            1. Update the report based on the USER FEEDBACK below.
            2. **CRITICAL**: You MUST RECALCULATE all sub-totals and grand totals (Total Assets, Total Liabilities, Net Profit) to reflect the changes.
            3. Maintain the exact same Markdown table structure.
            4. Do NOT output any bolding (**) or headers (#) inside the tables, keep it plain text as before.
            
            ---
            PREVIOUS REPORT:
            {previous_context}
            
            ---
            USER FEEDBACK / REQUEST:
            "{question}"
            
            ---
            UPDATED REPORT:
            """
            
            try:
                # Stream the refined report
                async for chunk in self.llm.astream(refinement_prompt):
                    if chunk.content:
                        yield chunk.content
            except Exception as e:
                print(f"[ERROR] Refinement failed: {str(e)}")
                yield f"\n[ERROR] Refinement failed: {str(e)}"
            
            return # Exit after refinement, do not proceed to extraction

        # --- STANDARD GENERATION MODE ---

        if not context_chunks:
            yield "[ERROR] No context data provided."
            return

        
        # PHASE 1: BATCH EXTRACTION (Map Step)
        
        total_chunks = len(context_chunks)
        BATCH_SIZE = 50 # Increased for faster processing of large documents.
        
        all_transactions = []
        
        yield {"status": "Analyzing documents..."}
        print(f"[INFO] Starting batch extraction for {total_chunks} chunks...")

        # Create batches
        batches = [context_chunks[i:i + BATCH_SIZE] for i in range(0, total_chunks, BATCH_SIZE)]
        total_batches = len(batches)
        
        # Parallel Processing Setup
        # Only allow 1 concurrent request to strictly align with 8 RPM limiting.
        # OpenRouter's "free" tier is highly aggressive about concurrency.
        sem = asyncio.Semaphore(1) 
        tasks = []

        async def process_batch(index, batch_data, batch_metas):
            async with sem:
                # We enforce sequential pacing of exactly ~8.0 seconds of sleep between consecutive calls
                # to guarantee we never exceed 8 tokens/minute (60/8 = 7.5s).
                if index > 0:
                    print(f"[INFO] Sequential pacing: waiting 8.0 seconds before starting batch {index + 1}...")
                    await asyncio.sleep(8.0)
                    
                transactions = await self._extract_transactions_from_batch(batch_data, context_query=question, company_id=company_id)
                
                # Tag transactions with source metadata for robust deduplication
                if transactions and batch_metas:
                    source_indices = [m.get('chunk_index') for m in batch_metas if m.get('chunk_index') is not None]
                    doc_names = list(set(m.get('document_name') for m in batch_metas if m.get('document_name')))
                    
                    min_idx = min(source_indices) if source_indices else 0
                    max_idx = max(source_indices) if source_indices else 0
                    
                    for t in transactions:
                        t['_source_chunks_range'] = (min_idx, max_idx)
                        t['_source_docs'] = doc_names
                return transactions

        for i, batch in enumerate(batches):
            batch_metas = context_metadatas[i*BATCH_SIZE : (i+1)*BATCH_SIZE] if context_metadatas else None
            tasks.append(process_batch(i, batch, batch_metas))
        
        # Process batches concurrently and stream status as each finishes
        completed_batches = 0
        for task in asyncio.as_completed(tasks):
            try:
                result = await task
                completed_batches += 1
                yield {"status": f"Extracting transactions (Batch {completed_batches}/{total_batches} complete)..."}
                
                transactions = result
                if transactions:
                    # Add to shared accounts to help next batches stay consistent
                    for t in transactions:
                        for e in t.get('entries', []):
                            self.shared_accounts.add(str(e.get('account')).strip().title())
                    
                    all_transactions.extend(transactions)
                    print(f"[INFO] Batch completed: Extracted {len(transactions)} transactions")
            
            except Exception as e:
                completed_batches += 1
                yield {"status": f"Extracting transactions (Batch {completed_batches}/{total_batches} complete)..."}
                print(f"[WARNING] Batch extraction failed: {e}")

        if not all_transactions:
            print("[WARNING] No transactions could be extracted from the documents.")
            yield "Information: No transactions could be identified in the provided documents. Please ensure the documents contain clear financial data.\n"
            return

        print(f"[INFO] Total extracted transactions: {len(all_transactions)}")
        
        # --- DEDUPLICATION ---
        # Keep ONLY unique transactions to avoid double-counting due to overlapping text chunks.
        
        unique_transactions = []
        seen_fingerprints = set()
        
        # Sort by date, then by total amount, then by narration for consistent processing
        def get_sort_key(t):
            from datetime import datetime
            date_str = str(t.get('date', '')).strip()
            try:
                # Convert DD/MM/YYYY to datetime for correct chronological sorting
                sort_date = datetime.strptime(date_str, "%d/%m/%Y")
            except Exception:
                # Fallback to a far-future date or string if parsing fails
                sort_date = datetime.max
                
            narration = str(t.get('narration', '')).strip()
            raw_entries = t.get('entries', [])
            total_amount = sum(float(e.get('amount', 0)) for e in raw_entries if str(e.get('type')).upper() == "DEBIT")
            if total_amount == 0 and raw_entries:
                total_amount = sum(float(e.get('amount', 0)) for e in raw_entries if str(e.get('type')).upper() == "CREDIT")
            return (sort_date, total_amount, narration)
        
        all_transactions.sort(key=get_sort_key)
        print(f"[DEBUG] Transactions sorted for deterministic ordering")
        
        for t in all_transactions:
            date_str = str(t.get('date', '')).strip()
            narration = str(t.get('narration', '')).strip()
            
            raw_entries = t.get('entries', [])
            total_amount = sum(float(e.get('amount', 0)) for e in raw_entries if str(e.get('type')).upper() == "DEBIT")
            if total_amount == 0 and raw_entries:
                total_amount = sum(float(e.get('amount', 0)) for e in raw_entries if str(e.get('type')).upper() == "CREDIT")
                
            # Normalize narration for deduplication
            import re
            norm_narration = re.sub(r'[^a-zA-Z0-9]', '', narration).lower()
            
            # Simple fingerprint: Date + Amount + Alphanumeric Narration
            # This is robust to slight LLM variation in narration capitalization or spaces
            fingerprint = f"{date_str}_{total_amount:.2f}_{norm_narration}"
            
            if fingerprint not in seen_fingerprints:
                seen_fingerprints.add(fingerprint)
                unique_transactions.append(t)
            else:
                pass # Extracted duplicate dropped
                
        all_transactions = unique_transactions
        print(f"[INFO] Total transactions after deduplication: {len(all_transactions)}")
        
        # --- VALIDATION: Calculate total debits and credits for consistency check ---
        total_validation_dr = 0.0
        total_validation_cr = 0.0
        for t in all_transactions:
            entries = t.get('entries', [])
            for e in entries:
                amount = float(e.get('amount', 0))
                if str(e.get('type')).upper() == "DEBIT":
                    total_validation_dr += amount
                else:
                    total_validation_cr += amount
        
        print(f"[VALIDATION] Total Debits: ₹{total_validation_dr:,.2f}, Total Credits: ₹{total_validation_cr:,.2f}")
        if abs(total_validation_dr - total_validation_cr) > 0.01:
            print(f"[WARNING] Transaction imbalance detected: Difference of ₹{abs(total_validation_dr - total_validation_cr):,.2f}")
        
        yield {"status": f"Verifying Double-Entry Integrity for {len(all_transactions)} unique transactions..."}

        
        # PHASE 2: FINANCIAL PROCESSING (Python Core)
        
        
        # Data Structures for Financial Statements
        ledger_balances: Dict[str, float] = {}
        account_category_votes: Dict[str, List[str]] = {}
        journal_rows_data = []
        tx_count = 0
        MAX_JOURNAL_ENTRIES_SHOW = 50
        
        from datetime import date
        today = date.today().strftime("%Y-%m-%d")

        yield "1. Professional Journal Book\n\n"
        yield "| Date | Particulars (Account) | L.F. | Debit (₹) | Credit (₹) | Narration |\n"
        yield "|---|---|---|---|---|---|\n"

        # Structured data for frontend editing
        journal_headers = ["Date", "Particulars (Account)", "L.F.", "Debit (₹)", "Credit (₹)", "Narration"]

        for t in all_transactions:
            date = str(t.get('date', '')).strip()
            narration = str(t.get('narration', '')).strip()
            entries = t.get('entries', [])
            
            # Validation: Double Entry
            dr_total = sum(float(e.get('amount', 0)) for e in entries if str(e.get('type')).upper() == "DEBIT")
            cr_total = sum(float(e.get('amount', 0)) for e in entries if str(e.get('type')).upper() == "CREDIT")
            
            if abs(dr_total - cr_total) > 0.01:
                print(f"[ERROR] Transaction Imbalance: Dr {dr_total} != Cr {cr_total} for {narration}")
                continue    
                
            tx_count += 1
            
            # --- LEDGER PROCESSING (Always do this for all transactions) ---
            for entry in entries:
                account = str(entry.get('account', 'Unclassified')).strip().title()
                category = entry.get('category', 'Unclassified')
                amount = float(entry.get('amount', 0))
                etype = str(entry.get('type')).upper()
                
                if account not in ledger_balances:
                    ledger_balances[account] = 0.0
                    account_category_votes[account] = []
                
                if category and category != "Unclassified":
                    account_category_votes[account].append(category)
                
                if etype == "DEBIT":
                    ledger_balances[account] += amount
                else:
                    ledger_balances[account] -= amount

            if tx_count > MAX_JOURNAL_ENTRIES_SHOW:
                 if tx_count == MAX_JOURNAL_ENTRIES_SHOW + 1:
                     yield f"| ... | ... | | ... | ... | processed internally... |\n"
                 continue

            first_entry = True
            for entry in entries:
                acc = str(entry.get('account', 'Unclassified')).strip().title()
                etype = str(entry.get('type')).upper()
                amt = float(entry.get('amount', 0))
                
                # Format columns
                disp_acc = f"{acc} Dr." if etype == "DEBIT" else f"    To {acc}"
                disp_dr = f"{amt:,.2f}" if etype == "DEBIT" else ""
                disp_cr = f"{amt:,.2f}" if etype == "CREDIT" else ""
                disp_date = date if first_entry else ""
                disp_narr = narration if first_entry else ""
                
                # Explicit Markdown Row
                yield f"| {disp_date} | {disp_acc} | | {disp_dr} | {disp_cr} | {disp_narr} |\n"
                
                # Structured Row
                journal_rows_data.append([disp_date, disp_acc, "", disp_dr, disp_cr, disp_narr])
                first_entry = False

        self.structured_tables.append({
            "type": "journal",
            "title": "Professional Journal Book",
            "headers": journal_headers,
            "rows": journal_rows_data
        })

        
        # PHASE 3: REPORT GENERATION (LLM Synthesis with Rules)
        
        
        yield {"status": "Compiling Trial Balance..."}
        
        
        # DETERMINISTIC CLASSIFICATION (VOTING)
        
        # Resolve categories by majority vote to prevent batch-order randomness
        from collections import Counter
        account_categories: Dict[str, str] = {}
        
        for acc, votes in account_category_votes.items():
            if not votes:
                account_categories[acc] = "Unclassified"
            else:
                # Pick most common
                most_common = Counter(votes).most_common(1)[0][0]
                account_categories[acc] = most_common

        # 1. GENERATE LEDGER SUMMARY / TRIAL BALANCE (Programmatic)
        yield "2. Ledger Summary (Trial Balance)\n\n"
        yield "| Account Head | Net Balance (₹) | Type |\n"
        yield "|---|---|---|\n"
        
        tb_headers = ["Account Head", "Net Balance (₹)", "Type"]
        tb_rows_data = []
        
        trial_balance_summary = ""
        total_debits = 0.0
        total_credits = 0.0
        
        for account, balance in sorted(ledger_balances.items()):
            if abs(balance) < 0.01: continue
            
            abs_bal = abs(balance)
            # Logic: If balance > 0 -> Debit (Asset/Expense), If < 0 -> Credit (Liability/Income)
            # But we must check our sign convention from Phase 2
            # Phase 2: Debit += amount, Credit -= amount. So >0 is Debit.
            
            bal_type = "Dr" if balance > 0 else "Cr"
            
            # Table Row
            yield f"| {account} | {abs_bal:,.2f} | {bal_type} |\n"
            tb_rows_data.append([account, f"{abs_bal:,.2f}", bal_type])
            
            # Summary for LLM
            # Include category hint if available
            cat_hint = f"[{account_categories.get(account, 'Unknown')}]"
            trial_balance_summary += f"- {account} {cat_hint}: ₹ {abs_bal:,.2f} ({bal_type})\n"
            
            if balance > 0:
                total_debits += balance
            else:
                total_credits += abs(balance)
                
        if abs(total_debits - total_credits) > 0.01:
            diff = total_debits - total_credits
            suspense_acc = "Suspense Account (TB Difference)"
            abs_diff = abs(diff)
            
            # Add to ledger/categories for consistent handling
            account_categories[suspense_acc] = "Current Assets" if diff < 0 else "Current Liabilities"
            
            if diff > 0:
                # Debits > Credits -> Need a Credit in Suspense
                total_credits += abs_diff
                yield f"| {suspense_acc} | {abs_diff:,.2f} | Cr |\n"
                trial_balance_summary += f"- {suspense_acc}: ₹ {abs_diff:,.2f} (Cr)\n"
            else:
                # Credits > Debits -> Need a Debit in Suspense
                total_debits += abs_diff
                yield f"| {suspense_acc} | {abs_diff:,.2f} | Dr |\n"
                trial_balance_summary += f"- {suspense_acc}: ₹ {abs_diff:,.2f} (Dr)\n"
            
            print(f"[WARNING] TB Imbalance! Difference of {abs_diff} handled via Suspense Account.")

        yield f"| TOTAL | {total_debits:,.2f} | {total_credits:,.2f} |\n"
        tb_rows_data.append(["TOTAL", f"{total_debits:,.2f}", f"{total_credits:,.2f}"])

        self.structured_tables.append({
            "type": "trial_balance",
            "title": "Ledger Summary (Trial Balance)",
            "headers": tb_headers,
            "rows": tb_rows_data
        })

        if abs(total_debits - total_credits) > 0.01:
            yield f"\n IMBALANCE DETECTED: A mismatch of ₹ {abs(total_debits - total_credits):,.2f} was found in the data extraction. A Suspense Account has been added to balance the books.\n"

        yield "\n3. Financial Statements (Profit & Loss and Balance Sheet)\n\n"
        yield {"status": "Synthesizing Final Balance Sheet with AI..."}

        prompt = f"""
You are a Professional Accountant with 10+ years of experience. You only process bank statements as your primary book of accounts.
Prepare finalized financial statements for Harsh Tailor using the provided Trial Balance data.

### BANK STATEMENT ACCOUNTING MODE (MANDATORY)
The system will receive only bank statements.
The AI must behave like a professional accountant and prepare:
- Opening Balance
- Profit & Loss Statement
- Balance Sheet
- Closing Balance
- Balance Verification
All accounting must be derived strictly from bank transactions.
No external data will be provided.
The bank statement is the primary book of accounts.

### OPENING BALANCE RULE (CRITICAL)
Opening Balance must always be extracted from the bank statement.
Opening Balance = First available balance value.
The first balance value must always be treated as Opening Balance.
Opening Balance must NOT be included in Profit & Loss.
Opening Balance must be stored as:
Assets:
Bank Account = Opening Balance

### CLOSING BALANCE RULE (CRITICAL)
Closing Balance must be extracted from the bank statement.
Closing Balance = Last available balance value.
The last balance value must be used as Bank Balance in Balance Sheet.

### PROFIT AND LOSS RULE (MANDATORY)
Profit & Loss must always be generated from transactions.
If transactions exist, Profit & Loss must NEVER be zero.
Total Income = Sum of Direct Income + Indirect Income
Total Expenses = Sum of Direct Expense + Indirect Expense
Net Profit = Income - Expense
Opening Balance must NOT be included.

### BALANCE SHEET CALCULATION RULE (CRITICAL)
The Balance Sheet must always satisfy:
Total Assets = Total Liabilities + Total Equity
If totals do not match, the AI must recalculate values.
Balance Sheet must never be returned with mismatched totals.

### BANK BALANCE RULE (MANDATORY)
Bank Balance in Balance Sheet must always equal:
Closing Balance from bank statement.
Bank Balance must NEVER be calculated manually.
Bank Balance = Last balance value in statement.
This fixes 40% mismatch issues.

### EQUITY CALCULATION RULE (CRITICAL)
Equity must always be calculated as:
Equity = Opening Balance + Net Profit - Drawings
Where:
Opening Balance = First balance value in bank statement
Net Profit = Profit & Loss result
Drawings = Owner withdrawals if detected
Most systems forget Net Profit addition.

### ASSET RULE
Assets must include:
Bank Balance
Cash Withdrawals (if ATM withdrawals exist)
Total Assets = Sum of all Assets.

### LIABILITY RULE
Liabilities must include:
Loans detected from narration such as:
LOAN CREDIT
EMI
FINANCE
NBFC
BANK LOAN
Loan credits increase liabilities.
EMI reduces liabilities.

### BALANCE VERIFICATION RULE (MANDATORY)
The AI must verify:
Total Assets
=
Total Liabilities + Total Equity

If mismatch occurs, AI must adjust Equity so that balance matches.
If Total Assets ≠ Total Liabilities + Equity, Equity must be recalculated as:
Equity = Total Assets - Total Liabilities

### CRITICAL FINAL REMINDER
If transactions exist, Profit & Loss must never return zero values.
CRITICAL: You MUST list EVERY account marked as 'Expense' (e.g., Direct Expense, Indirect Expense, etc.) from the Trial Balance in the "Particulars (Dr)" column of the P&L Statement. Do not combine them into zero!
If the Trial Balance contains ONLY Expenses and ZERO Income, you MUST still list every Expense in the P&L table, leave Income blank (or 0.00), and calculate a Net Loss.
Net Profit / (Net Loss) = Total Income - Total Expenses
If Total Expenses > Total Income, it is a NET LOSS. Show it clearly in the P&L table (By Net Loss) and subtract it from Equity.
Returning a blank or "0.00" P&L table when Expenses or Income exist in the Trial Balance is STRICTLY FORBIDDEN.

### TRANSACTION ACCOUNTING RULE
Every transaction must affect either: Income, Expense, Asset, Liability, Equity. 
No transaction should remain unclassified.

### DEBUG OUTPUT RULE
Before generating Profit & Loss, you must internally calculate:
Total Credits, Total Debits, Income Amount, Expense Amount.
If Income > 0 or Expense > 0, Profit & Loss must not be zero.

### TRIAL BALANCE DATA
{trial_balance_summary}

---

### REQUIRED OUTPUT FORMAT (MANDATORY)
You MUST return the results EXACTLY in this 5-part structure, including the numbers. DO NOT output anything else.

1) Opening Balance

Opening Balance: [Opening Balance Amount]


2) Profit & Loss Statement

| Particulars (Dr) | Amount (₹) | Particulars (Cr) | Amount (₹) |
|---|---|---|---|
| To [Specific Expense Account 1] | [Amount] | By [Specific Income Account 1] | [Amount] |
| To [Specific Expense Account 2] | [Amount] | By [Specific Income Account 2] | [Amount] |
| To [List ALL trial balance expenses!]| [Amount] | | |
| To Net Profit (if Income > Expenses) | [Amount] | By Net Loss (if Expenses > Income) | [Amount] |
| TOTAL | [Total] | TOTAL | [Total] |


3) Balance Sheet

| Liabilities & Equity | Amount (₹) | Assets | Amount (₹) |
|---|---|---|---|
| Equity (Opening Bal) | [Amount] | Bank Balance | [Closing Balance Amount] |
| Add: Net Profit | [Amount] | [Other Asset Accounts] | [Amount] |
| Less: Net Loss | [Amount] | | |
| Less: Drawings | [Amount] | | |
| Add: Capital Intro | [Amount] | | |
| Loans (Liabilities) | [Amount] | | |
| TOTAL | [Total] | TOTAL | [Total] |
"""
        max_retries = 3
        for attempt in range(max_retries):
            try:
                if attempt > 0:
                    yield f"\n\n[INFO] Connection dropped. Retrying synthesis (Attempt {attempt + 1}/{max_retries})...\n\n"
                    
                async for chunk in self.llm.astream(prompt):
                    if chunk.content:
                        yield chunk.content
                # If we complete the stream without exception, we're done
                break
            except Exception as e:
                print(f"[WARNING] Final Synthesis failed on attempt {attempt + 1}: {str(e)}")
                if attempt == max_retries - 1:
                    print(f"[ERROR] Final Synthesis failed after {max_retries} attempts: {str(e)}")
                    yield f"\n[ERROR] Final Synthesis failed: {str(e)}"
                else:
                    await asyncio.sleep(3.0 * (attempt + 1))

    async def _parse_markdown_tables(self, text: str) -> List[Dict[str, Any]]:
        """Extract structured data from markdown tables in text."""
        import re
        tables = []
        # Regex for markdown tables
        table_regex = r"((?:\|[^\n]+\|(?:\n|$))+)"
        matches = re.finditer(table_regex, text)
        
        for match in matches:
            table_str = match.group(1).strip()
            # Split and filter out empty strings from splitting
            rows = [r.strip() for r in table_str.split("\n") if r.strip()]
            
            # Clean rows and remove separator
            parsed_rows = []
            for row in rows:
                if re.match(r"^\|?[-:| ]+\|?$", row.strip()):
                    continue
                # Split by | and filter results
                cells = [c.strip() for c in row.split("|")]
                # Filter out empty cells at edges
                if row.startswith("|"): cells = cells[1:]
                if row.endswith("|") and cells: cells = cells[:-1]
                parsed_rows.append(cells)
            
            if len(parsed_rows) < 2:
                continue
                
            headers = parsed_rows[0]
            body = parsed_rows[1:]
            
            # Detect type
            table_type = "generic"
            headers_lower = [h.lower() for h in headers]
            if any("debit" in h or "credit" in h for h in headers_lower):
                table_type = "journal"
            elif any("asset" in h or "liabilit" in h for h in headers_lower):
                table_type = "balance_sheet"
            elif any("particulars" in h and "amount" in h for h in headers_lower):
                table_type = "p_and_l"
                
            tables.append({
                "type": table_type,
                "headers": headers,
                "rows": body
            })
        return tables

    async def get_all_structured_tables(self, full_text: str) -> List[Dict[str, Any]]:
        """Combine pre-generated tables and tables parsed from LLM response."""
        parsed_tables = await self._parse_markdown_tables(full_text)
        
        # Merge logic: avoid duplicate tables if they were already captured
        # We use types as hints
        all_tables = self.structured_tables.copy()
        
        captured_types = {t["type"] for t in all_tables}
        for pt in parsed_tables:
            # If it's a journal or trial_balance and we already have it from programmatic extraction, skip
            if pt["type"] in captured_types and pt["type"] in ["journal", "trial_balance"]:
                continue
            all_tables.append(pt)
                
        return all_tables


