from typing import List, Dict, Any, AsyncGenerator, Union
import json
import asyncio
import hashlib
from langchain_openai import ChatOpenAI
from app.core.config import settings
from app.services.ai_service import LLMService
from app.ai.rag.retriever import vector_store
from app.services.accounting_rules import (
    EXTRACTOR_RULES, CLASSIFIER_RULES, JOURNAL_RULES_TEXT,
    PNL_RULES_TEXT, BALANCE_SHEET_RULES_TEXT, TALLY_RULES_TEXT,
    PNL_CONFIG, BALANCE_SHEET_CONFIG, TALLY_CONFIG, JOURNAL_CONFIG,
    AGENT_SYSTEM_PROMPTS
)

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
        {AGENT_SYSTEM_PROMPTS['extractor']}

        {EXTRACTOR_RULES}

        {query_instruction}
        {feedback_context}

        INPUT TEXT (Bank Statement):
        {context}

        TASK:
        Read the bank statement text and extract every transaction you can clearly read.
        Do NOT classify, do NOT create journal entries, do NOT decide income/expense.

        REQUIRED JSON OUTPUT FORMAT:
        {{
            "transactions": [
                {{
                    "date": "DD/MM/YYYY",
                    "narration": "Exact original narration from statement",
                    "debit": 0.00,
                    "credit": 0.00,
                    "balance": 0.00
                }}
            ]
        }}

        REMINDERS:
        - debit and credit fields are MUTUALLY EXCLUSIVE per transaction (one is 0.00)
        - balance = running balance shown in statement (0.00 if not present)
        - Skip any transaction where date or amount is missing or unclear
        - Output JSON ONLY. No explanation text.
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

    async def _classify_transaction(self, transaction: Dict) -> Dict:
        """
        Agent 2: Classification Agent (LLM)
        Determines the precise Account Head and Category for a raw transaction.
        """
        narration = transaction.get("narration", "")
        # If it's already classified well by the extractor, we might skip, but let's enforce rules here.
        prompt = f"""
        You are an expert Accountant. Classify the following bank transaction narration.
        
        NARRATION: "{narration}"
        
        ### ACCOUNT CATEGORY RULE (MANDATORY)
        Every transaction must be assigned one of the following categories ONLY:
        - Direct Income
        - Indirect Income
        - Direct Expense
        - Indirect Expense
        - Current Assets
        - Current Liabilities
        - Equity

        ### RULES
        - Money arriving (CREDIT to Bank) is usually Income or Liability.
        - Money leaving (DEBIT from Bank) is usually Expense or Asset.
        - Look for keywords: "UPI PAYMENT", "NEFT CR", "ATM", "SALARY", "RENT", etc.
        
        REQUIRED JSON FORMAT:
        {{
            "account_head": "Name of the Account (e.g., Office Rent, Sales Income)",
            "category": "One of the strict categories above"
        }}
        """
        try:
            response = await self.llm.ainvoke(prompt)
            data = self.ai_service._extract_json(response.content)
            # Update the non-bank entry in the transaction
            for entry in transaction.get("entries", []):
                if entry.get("account", "").upper() != "BANK ACCOUNT":
                    entry["account"] = data.get("account_head", entry.get("account", "Unclassified"))
                    entry["category"] = data.get("category", entry.get("category", "Unclassified"))
            return transaction
        except Exception as e:
            print(f"[WARNING] Classification failed for narration '{narration}': {e}")
            return transaction

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
        Multi-Agent Pipeline:
          Agent 1: Extractor (LLM) -> Agent 2: Classifier (LLM)
          -> Agent 3: Journal (deterministic, updates Ledger Store)
          -> Agent 4: P&L (pure math) + Agent 5: Balance Sheet (pure math)
          -> Tally Agent (audit/verify)
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

        # =====================================================================
        # --- STANDARD GENERATION MODE  (5-Agent Pipeline) ---
        # =====================================================================

        if not context_chunks:
            yield "[ERROR] No context data provided."
            return

        # ===================================================================
        # AGENT 1: EXTRACTOR POOL (LLM — N parallel extractor workers)
        # ===================================================================
        # KEY: Keep batch size SMALL so the LLM doesn't hit context window limits.
        # 50 chunks × 2000 chars = 100k chars per prompt → LLM misses transactions.
        # 8 chunks × 2000 chars = 16k chars per prompt → LLM processes all rows.
        BATCH_SIZE = 8
        all_transactions: List[Dict] = []
        batches = [context_chunks[i:i + BATCH_SIZE] for i in range(0, len(context_chunks), BATCH_SIZE)]
        total_batches = len(batches)

        # 3 parallel workers to speed up extraction.
        N_EXTRACTORS = 20
        WORKER_DELAY_SEC = 1.0
        sem = asyncio.Semaphore(N_EXTRACTORS)

        print(f"[AGENT-1] Extractor Pool: {len(context_chunks)} chunks → {total_batches} batches (size={BATCH_SIZE}) → {N_EXTRACTORS} workers")

        async def process_batch(index, batch_data, batch_metas):
            async with sem:
                slot = index % N_EXTRACTORS
                # Stagger requests so all workers don't fire at the exact same millisecond
                if index >= N_EXTRACTORS:
                    await asyncio.sleep(WORKER_DELAY_SEC * slot)
                elif slot > 0:
                    await asyncio.sleep(slot * 0.5)

                print(f"[AGENT-1][Worker-{slot+1}] → Batch {index + 1}/{total_batches} starting... ({len(batch_data)} chunks)")
                txns = await self._extract_transactions_from_batch(
                    batch_data, context_query=question, company_id=company_id
                )
                print(f"[AGENT-1][Worker-{slot+1}] ← Batch {index + 1}/{total_batches} done: {len(txns)} transactions extracted")

                if txns and batch_metas:
                    src_docs = list(set(m.get('document_name') for m in batch_metas if m.get('document_name')))
                    for t in txns:
                        t['_source_docs'] = src_docs
                return txns

        tasks = [
            process_batch(i, b, context_metadatas[i*BATCH_SIZE:(i+1)*BATCH_SIZE] if context_metadatas else None)
            for i, b in enumerate(batches)
        ]
        completed = 0
        yield {"status": f"Agent 1 (Extractor ×{N_EXTRACTORS}): processing {total_batches} batches..."}
        for coro in asyncio.as_completed(tasks):
            try:
                result = await coro
                completed += 1
                yield {"status": f"Agent 1 (Extractor ×{N_EXTRACTORS}): {completed}/{total_batches} batches done..."}
                if result:
                    all_transactions.extend(result)
            except Exception as e:
                completed += 1
                print(f"[AGENT-1][ERROR] Extractor batch {completed} failed: {e}")

        if not all_transactions:
            yield "No transactions could be identified in the provided documents. Please check the file has clear financial data.\n"
            return

        # Deduplication + chronological sort
        import re as _re
        from datetime import datetime as _dt

        def _sort_key(t):
            ds = str(t.get('date', '')).strip()
            try:
                d = _dt.strptime(ds, "%d/%m/%Y")
            except Exception:
                d = _dt.max
            narr = str(t.get('narration', '')).strip()
            entries = t.get('entries', [])
            amt = sum(float(e.get('amount', 0)) for e in entries if str(e.get('type')).upper() == 'DEBIT')
            if amt == 0 and entries:
                amt = sum(float(e.get('amount', 0)) for e in entries if str(e.get('type')).upper() == 'CREDIT')
            return (d, amt, narr)

        # Improved deduplication: include occurrence counter so that two legitimately
        # identical transactions (same date + amount + narration) are NOT dropped.
        # We only drop exact cross-batch duplicates from overlapping chunk windows.
        all_transactions.sort(key=_sort_key)  # chronological order first
        from collections import Counter as _Counter
        narr_count: _Counter = _Counter()
        seen_fps: set = set()
        unique_txns: List[Dict] = []
        for t in all_transactions:
            ds = str(t.get('date', '')).strip()
            narr = str(t.get('narration', '')).strip()
            entries = t.get('entries', [])
            amt = sum(float(e.get('amount', 0)) for e in entries if str(e.get('type')).upper() == 'DEBIT')
            if amt == 0 and entries:
                amt = sum(float(e.get('amount', 0)) for e in entries if str(e.get('type')).upper() == 'CREDIT')
            norm = _re.sub(r'[^a-zA-Z0-9]', '', narr).lower()
            base_fp = f"{ds}_{amt:.2f}_{norm}"
            # Count how many times this exact fingerprint has appeared across ALL batches.
            # Only drop if this is the 2nd+ time we've seen it (true cross-batch duplicate).
            narr_count[base_fp] += 1
            fp = f"{base_fp}#{narr_count[base_fp]}"
            if fp not in seen_fps:
                seen_fps.add(fp)
                unique_txns.append(t)
        all_transactions = unique_txns
        print(f"[DEDUP] Raw extracted: {len(seen_fps)} fingerprints → {len(all_transactions)} unique transactions kept")

        # ===================================================================
        # AGENT 2: BATCH CLASSIFIER POOL (LLM — N parallel classifier workers)
        # ===================================================================
        CLASSIFY_BATCH_SIZE = 20     # transactions per LLM call
        N_CLASSIFIERS = 20           # parallel classifier workers
        CLASSIFIER_DELAY_SEC = 1.0   # stagger delay between worker slots

        classify_batches = [
            all_transactions[i:i + CLASSIFY_BATCH_SIZE]
            for i in range(0, len(all_transactions), CLASSIFY_BATCH_SIZE)
        ]
        total_classify_batches = len(classify_batches)
        print(f"[AGENT-2] Classifier Pool: {len(all_transactions)} txns → {total_classify_batches} batches (size={CLASSIFY_BATCH_SIZE}) → {N_CLASSIFIERS} workers")
        yield {"status": f"Agent 2 (Classifier ×{N_CLASSIFIERS}): classifying {len(all_transactions)} transactions..."}

        cls_sem = asyncio.Semaphore(N_CLASSIFIERS)
        classified_results: Dict[int, List[Dict]] = {}  # bidx → list of classified transactions

        async def classify_batch(bidx: int, c_batch: List[Dict]):
            async with cls_sem:
                slot = bidx % N_CLASSIFIERS
                # Small stagger on first round to avoid burst
                if bidx >= N_CLASSIFIERS:
                    await asyncio.sleep(CLASSIFIER_DELAY_SEC * (slot % 3))
                elif slot > 0:
                    await asyncio.sleep(slot * 0.3)

                narrations_list = ""
                for j, t in enumerate(c_batch):
                    dr = float(t.get('debit', 0) or 0)
                    cr = float(t.get('credit', 0) or 0)
                    direction = f"DEBIT \u20b9{dr:,.2f}" if dr > 0 else f"CREDIT \u20b9{cr:,.2f}"
                    narrations_list += f"{j+1}. [{direction}] {t.get('narration', 'Unknown')}\n"

                classify_prompt = f"""
                {AGENT_SYSTEM_PROMPTS['classifier']}

                {CLASSIFIER_RULES}

                TRANSACTIONS TO CLASSIFY:
                {narrations_list}

                For each transaction, output the double-entry mapping.
                Return a JSON array with EXACTLY {len(c_batch)} objects (one per transaction, in order):
                [{{
                    "debit_account": "Account to be debited",
                    "credit_account": "Account to be credited",
                    "account_type": "income/expense/asset/liability/equity",
                    "category": "One of the mandatory categories",
                    "confidence": 0.85
                }}]

                Bank Account Logic (NON-NEGOTIABLE):
                - If transaction is CREDIT (money IN) \u2192 debit_account = "Bank Account"
                - If transaction is DEBIT  (money OUT) \u2192 credit_account = "Bank Account"

                Output JSON array ONLY. No explanation.
                """
                batch_classified = []
                try:
                    response = await self.llm.ainvoke(classify_prompt)
                    results = self.ai_service._extract_json(response.content)
                    if isinstance(results, dict):
                        results = list(results.values())[0] if results else []
                    if not isinstance(results, list):
                        results = []
                    print(f"[AGENT-2][Worker-{slot+1}] Batch {bidx+1}/{total_classify_batches}: {len(results)} classifications")

                    for j, t in enumerate(c_batch):
                        dr = float(t.get('debit', 0) or 0)
                        cr = float(t.get('credit', 0) or 0)
                        amt = dr if dr > 0 else cr
                        if j < len(results):
                            res = results[j]
                            debit_acc  = res.get('debit_account', 'Bank Account').strip().title()
                            credit_acc = res.get('credit_account', 'Bank Account').strip().title()
                            category   = res.get('category', 'Unclassified')
                            t['entries'] = [
                                {"account": debit_acc,  "type": "DEBIT",  "amount": amt, "category": category},
                                {"account": credit_acc, "type": "CREDIT", "amount": amt, "category": "Current Assets"},
                            ]
                            t['confidence'] = res.get('confidence', 1.0)
                        else:
                            # Fallback for missing LLM result
                            if dr > 0:
                                t['entries'] = [
                                    {"account": "Suspense Account", "type": "DEBIT",  "amount": dr, "category": "Current Assets"},
                                    {"account": "Bank Account",     "type": "CREDIT", "amount": dr, "category": "Current Assets"},
                                ]
                            else:
                                t['entries'] = [
                                    {"account": "Bank Account",     "type": "DEBIT",  "amount": cr, "category": "Current Assets"},
                                    {"account": "Suspense Account", "type": "CREDIT", "amount": cr, "category": "Current Assets"},
                                ]
                        batch_classified.append(t)

                except Exception as e:
                    print(f"[AGENT-2][Worker-{slot+1}][ERROR] Batch {bidx+1} failed: {e}")
                    for t in c_batch:
                        dr = float(t.get('debit', 0) or 0)
                        cr = float(t.get('credit', 0) or 0)
                        amt = dr if dr > 0 else cr
                        if dr > 0:
                            t['entries'] = [
                                {"account": "Suspense Account", "type": "DEBIT",  "amount": amt, "category": "Current Assets"},
                                {"account": "Bank Account",     "type": "CREDIT", "amount": amt, "category": "Current Assets"},
                            ]
                        else:
                            t['entries'] = [
                                {"account": "Bank Account",     "type": "DEBIT",  "amount": amt, "category": "Current Assets"},
                                {"account": "Suspense Account", "type": "CREDIT", "amount": amt, "category": "Current Assets"},
                            ]
                        batch_classified.append(t)

                return bidx, batch_classified

        cls_tasks = [classify_batch(i, b) for i, b in enumerate(classify_batches)]
        cls_completed = 0
        for coro in asyncio.as_completed(cls_tasks):
            bidx, batch_result = await coro
            classified_results[bidx] = batch_result
            cls_completed += 1
            if cls_completed % 5 == 0 or cls_completed == total_classify_batches:
                yield {"status": f"Agent 2 (Classifier ×{N_CLASSIFIERS}): {cls_completed}/{total_classify_batches} batches done..."}

        # Reconstruct in original order (as_completed gives out-of-order results)
        classified: List[Dict] = []
        for i in range(total_classify_batches):
            classified.extend(classified_results.get(i, []))

        print(f"[AGENT-2] Parallel classification complete: {len(classified)} transactions classified")

        # ===================================================================
        # LEDGER STORE (Shared State)
        # ===================================================================
        ledger_balances: Dict[str, float] = {}
        account_categories: Dict[str, str] = {}

        # ===================================================================
        # AGENT 3: JOURNAL AGENT (Deterministic) — also populates Ledger
        # ===================================================================
        print(f"[AGENT-3] {AGENT_SYSTEM_PROMPTS['journal']}")
        yield {"status": "Agent 3 (Journal): Writing double-entry journal & populating Ledger Store..."}
        journal_rows_data: List[List] = []
        tx_count = 0
        MAX_JOURNAL = 50

        yield "\n\n1. Professional Journal Book\n\n| Date | Particulars (Account) | L.F. | Debit (\u20b9) | Credit (\u20b9) | Narration |\n|---|---|---|---|---|---|\n"

        for t in classified:
            date_val = str(t.get('date', '')).strip()
            narration = str(t.get('narration', '')).strip()
            entries = t.get('entries', [])

            dr_total = sum(float(e.get('amount', 0)) for e in entries if str(e.get('type')).upper() == 'DEBIT')
            cr_total = sum(float(e.get('amount', 0)) for e in entries if str(e.get('type')).upper() == 'CREDIT')
            if abs(dr_total - cr_total) > 0.01:
                print(f"[JOURNAL] Skipped imbalanced tx (Dr={dr_total} Cr={cr_total}): '{narration}'")
                continue

            tx_count += 1

            # Update Ledger Store
            for entry in entries:
                acc = str(entry.get('account', 'Unclassified')).strip().title()
                cat = str(entry.get('category', 'Unclassified')).strip()
                amt = float(entry.get('amount', 0))
                etype = str(entry.get('type')).upper()
                ledger_balances.setdefault(acc, 0.0)
                if cat and cat != 'Unclassified':
                    account_categories[acc] = cat
                if etype == 'DEBIT':
                    ledger_balances[acc] += amt
                else:
                    ledger_balances[acc] -= amt

            # Yield journal rows
            if tx_count <= MAX_JOURNAL:
                first = True
                for entry in entries:
                    acc = str(entry.get('account', 'Unclassified')).strip().title()
                    etype = str(entry.get('type')).upper()
                    amt = float(entry.get('amount', 0))
                    d_acc = f"{acc} Dr." if etype == 'DEBIT' else f"    To {acc}"
                    d_dr = f"{amt:,.2f}" if etype == 'DEBIT' else ""
                    d_cr = f"{amt:,.2f}" if etype == 'CREDIT' else ""
                    d_date = date_val if first else ""
                    d_narr = narration if first else ""
                    yield f"| {d_date} | {d_acc} | | {d_dr} | {d_cr} | {d_narr} |\n"
                    journal_rows_data.append([d_date, d_acc, "", d_dr, d_cr, d_narr])
                    first = False
            elif tx_count == MAX_JOURNAL + 1:
                yield "| ... | ... | | ... | ... | (remaining processed internally) |\n"

        self.structured_tables.append({
            "type": "journal",
            "title": "Professional Journal Book",
            "headers": ["Date", "Particulars (Account)", "L.F.", "Debit (\u20b9)", "Credit (\u20b9)", "Narration"],
            "rows": journal_rows_data
        })
        print(f"[AGENT-3] Journal done: {tx_count} valid transactions written.")
        print(f"[AGENT-3] Ledger Store populated: {len(ledger_balances)} accounts.")
        for acc, bal in ledger_balances.items():
            cat = account_categories.get(acc, 'Unclassified')
            print(f"[LEDGER]   {acc:<40} | {bal:>12,.2f} | {cat}")

        # ===================================================================
        # AGENT 4: P&L AGENT (Pure Math — no LLM) — uses PNL_CONFIG rules
        # ===================================================================
        print(f"[AGENT-4] {AGENT_SYSTEM_PROMPTS['pnl']}")
        print(f"[AGENT-4] Include: {PNL_CONFIG['include_types']} | Exclude: {PNL_CONFIG['exclude_types']}")
        yield {"status": "Agent 4 (P&L Agent): Calculating Profit & Loss from Ledger..."}
        yield "\n\n2. Profit & Loss Statement\n\n"

        pnl_dr_rows: List[tuple] = []
        pnl_cr_rows: List[tuple] = []
        pnl_dr = 0.0
        pnl_cr = 0.0

        # Determine which accounts to exclude from P&L (assets, liabilities, equity, etc.)
        _pnl_exclude_upper = [x.upper() for x in PNL_CONFIG['exclude_types']]

        for acc, bal in ledger_balances.items():
            cat = account_categories.get(acc, "").upper()
            if abs(bal) < 0.01:
                continue
            # Skip if account name or category matches any PNL exclude keyword
            acc_upper = acc.upper()
            if any(excl in cat or excl in acc_upper for excl in _pnl_exclude_upper):
                print(f"[AGENT-4] Excluding from P&L: {acc} (cat={cat})")
                continue
            if "INCOME" in cat:
                pnl_cr_rows.append((f"By {acc}", abs(bal)))
                pnl_cr += abs(bal)
            elif "EXPENSE" in cat:
                pnl_dr_rows.append((f"To {acc}", abs(bal)))
                pnl_dr += abs(bal)

        net_profit = pnl_cr - pnl_dr
        if net_profit > 0:
            pnl_dr_rows.append(("To Net Profit (c/d)", net_profit))
            pnl_dr += net_profit
        elif net_profit < 0:
            pnl_cr_rows.append(("By Net Loss (c/d)", abs(net_profit)))
            pnl_cr += abs(net_profit)

        if not pnl_dr_rows and not pnl_cr_rows:
            yield "_No income or expense transactions were classified. Check that the document contains transaction data._\n"
        else:
            yield "| Particulars (Dr) | Amount (\u20b9) | Particulars (Cr) | Amount (\u20b9) |\n|---|---|---|---|\n"
            pnl_table_rows: List[List] = []
            for i in range(max(len(pnl_dr_rows), len(pnl_cr_rows))):
                d_part = pnl_dr_rows[i][0] if i < len(pnl_dr_rows) else ""
                d_amt = f"{pnl_dr_rows[i][1]:,.2f}" if i < len(pnl_dr_rows) else ""
                c_part = pnl_cr_rows[i][0] if i < len(pnl_cr_rows) else ""
                c_amt = f"{pnl_cr_rows[i][1]:,.2f}" if i < len(pnl_cr_rows) else ""
                yield f"| {d_part} | {d_amt} | {c_part} | {c_amt} |\n"
                pnl_table_rows.append([d_part, d_amt, c_part, c_amt])
            yield f"| **TOTAL** | **{pnl_dr:,.2f}** | **TOTAL** | **{pnl_cr:,.2f}** |\n"
            pnl_table_rows.append(["TOTAL", f"{pnl_dr:,.2f}", "TOTAL", f"{pnl_cr:,.2f}"])
            self.structured_tables.append({
                "type": "profit_loss",
                "title": "Profit & Loss Statement",
                "headers": ["Particulars (Dr)", "Amount (\u20b9)", "Particulars (Cr)", "Amount (\u20b9)"],
                "rows": pnl_table_rows
            })
            result_label = f"Net Profit: \u20b9{net_profit:,.2f}" if net_profit > 0 else f"Net Loss: \u20b9{abs(net_profit):,.2f}"
            print(f"[AGENT-4] P&L done \u2192 Income: \u20b9{pnl_cr:,.2f} | Expense: \u20b9{pnl_dr - (net_profit if net_profit > 0 else 0):,.2f} | {result_label}")

        # ===================================================================
        # AGENT 5: BALANCE SHEET AGENT (Pure Math — no LLM)
        # ===================================================================
        print(f"[AGENT-5] {AGENT_SYSTEM_PROMPTS['balance_sheet']}")
        yield {"status": "Agent 5 (Balance Sheet Agent): Building Balance Sheet from Ledger..."}
        yield "\n\n3. Balance Sheet\n\n"

        bs_assets: List[tuple] = []
        bs_liab: List[tuple] = []
        asset_total = 0.0
        liab_total = 0.0

        for acc, bal in ledger_balances.items():
            cat = account_categories.get(acc, "").upper()
            if abs(bal) < 0.01:
                continue
            # P&L accounts excluded — their net is captured via Net Profit/Loss
            if "INCOME" in cat or "EXPENSE" in cat:
                continue
            if bal > 0:  # Debit balance → Asset
                bs_assets.append((acc, bal))
                asset_total += bal
            else:  # Credit balance → Liability / Equity
                bs_liab.append((acc, abs(bal)))
                liab_total += abs(bal)

        # Inject Net Profit/Loss into Equity side
        if net_profit > 0:
            bs_liab.append(("Add: Net Profit", net_profit))
            liab_total += net_profit
        elif net_profit < 0:
            bs_liab.append(("Less: Net Loss", -abs(net_profit)))
            liab_total -= abs(net_profit)

        yield "| Liabilities & Equity | Amount (\u20b9) | Assets | Amount (\u20b9) |\n|---|---|---|---|\n"
        bs_table_rows: List[List] = []
        for i in range(max(len(bs_liab), len(bs_assets), 1)):
            l_part = bs_liab[i][0] if i < len(bs_liab) else ""
            l_amt = f"{bs_liab[i][1]:,.2f}" if i < len(bs_liab) else ""
            a_part = bs_assets[i][0] if i < len(bs_assets) else ""
            a_amt = f"{bs_assets[i][1]:,.2f}" if i < len(bs_assets) else ""
            yield f"| {l_part} | {l_amt} | {a_part} | {a_amt} |\n"
            bs_table_rows.append([l_part, l_amt, a_part, a_amt])
        yield f"| **TOTAL** | **{liab_total:,.2f}** | **TOTAL** | **{asset_total:,.2f}** |\n"
        bs_table_rows.append(["TOTAL", f"{liab_total:,.2f}", "TOTAL", f"{asset_total:,.2f}"])
        self.structured_tables.append({
            "type": "balance_sheet",
            "title": "Balance Sheet",
            "headers": ["Liabilities & Equity", "Amount (\u20b9)", "Assets", "Amount (\u20b9)"],
            "rows": bs_table_rows
        })
        print(f"[AGENT-5] Balance Sheet done → Assets: ₹{asset_total:,.2f} | Liabilities+Equity: ₹{liab_total:,.2f}")

        # ===================================================================
        # AGENT 6 — TALLY / AUDITOR AGENT  (uses TALLY_CONFIG rules)
        # ===================================================================
        print(f"[AGENT-6] {AGENT_SYSTEM_PROMPTS['tally']}")
        print(f"[AGENT-6] Config: {TALLY_CONFIG}")
        yield {"status": "Agent 6 (Tally Auditor): Running all 3 validation checks..."}

        # --- CHECK 1: TRIAL BALANCE ---
        tb_total_dr = sum(v for v in ledger_balances.values() if v > 0)
        tb_total_cr = sum(abs(v) for v in ledger_balances.values() if v < 0)
        tb_diff = abs(tb_total_dr - tb_total_cr)
        if TALLY_CONFIG['trial_balance_check']:
            if tb_diff > 0.01:
                print(f"[AGENT-6] Trial Balance MISMATCH: Dr=\u20b9{tb_total_dr:,.2f} Cr=\u20b9{tb_total_cr:,.2f} Diff=\u20b9{tb_diff:,.2f}")
                yield f"\n> \u26a0\ufe0f **Trial Balance FAILED**: Dr \u20b9{tb_total_dr:,.2f} \u2260 Cr \u20b9{tb_total_cr:,.2f} (diff=\u20b9{tb_diff:,.2f})\n"
            else:
                print(f"[AGENT-6] Trial Balance PASSED: \u20b9{tb_total_dr:,.2f}")
                yield f"\n> \u2705 **Trial Balance PASSED** \u2014 Total Dr = Total Cr = \u20b9{tb_total_dr:,.2f}\n"

        # --- CHECK 2: BALANCE SHEET EQUATION ---
        diff = abs(asset_total - liab_total)
        if TALLY_CONFIG['balance_sheet_check']:
            if diff > 0.01:
                if TALLY_CONFIG['auto_capital_adjustment']:
                    # Add Capital Adjustment entry under Equity to absorb gap
                    cap_adj = asset_total - liab_total
                    bs_liab.append(("Capital Adjustment", cap_adj))
                    liab_total += cap_adj
                    yield (
                        f"\n> \u26a0\ufe0f **Balance Sheet ADJUSTED**: Difference of \u20b9{abs(cap_adj):,.2f} absorbed "
                        f"via Capital Adjustment under Equity (per TALLY_RULES).\n"
                    )
                    print(f"[AGENT-6] BS adjusted by Capital Adjustment: \u20b9{cap_adj:,.2f}")
                else:
                    yield (
                        f"\n> \u26a0\ufe0f **Balance Sheet FAILED**: Out of balance by \u20b9{diff:,.2f}.\n"
                        f"> Missing Opening Balance entries or unclassified transactions.\n"
                    )
                    print(f"[AGENT-6] BS FAILED: Assets=\u20b9{asset_total:,.2f} L+E=\u20b9{liab_total:,.2f} Diff=\u20b9{diff:,.2f}")
            else:
                yield (
                    f"\n> \u2705 **Balance Sheet PASSED** \u2014 Assets = Liabilities + Equity = \u20b9{asset_total:,.2f}\n"
                )
                print(f"[AGENT-6] BS PASSED: \u20b9{asset_total:,.2f}")

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


