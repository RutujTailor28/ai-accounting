from typing import List, Dict, Any, AsyncGenerator, Union
import json
import asyncio
import hashlib
from langchain_openai import ChatOpenAI
from app.core.config import settings
from app.services.ai_service import LLMService

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

    async def _extract_transactions_from_batch(self, batch_chunks: List[str]) -> List[Dict]:
        """
        Extract structured transaction data from a small batch of chunks.
        Uses content-based caching to ensure identical chunks produce identical results.
        """
        # Create a deterministic hash of the batch content for caching
        context = "\n".join(batch_chunks)
        content_hash = hashlib.sha256(context.encode('utf-8')).hexdigest()
        
        # Check cache first
        if content_hash in self.extraction_cache:
            print(f"[CACHE HIT] Reusing cached extraction for batch (hash: {content_hash[:8]}...)")
            return self.extraction_cache[content_hash]
        
        # Injection of already seen accounts to prevent duplicates
        known_accounts_str = ", ".join(list(self.shared_accounts)[:50]) if self.shared_accounts else "None yet"
        
        prompt = f"""
        You are an expert Data Entry Clerk. Your task is to extract accounting transactions from the text below.
        
        INPUT TEXT:
        {context}
        
        KNOWN CHART OF ACCOUNTS (Use these names if they match to ensure consistency):
        {known_accounts_str}
        
        INSTRUCTIONS:
        1. Extract EVERY transaction found.
        2. **LITERAL NARRATION**: Preserve the original narration EXACTLY as it appears in the text. Do not truncate, summarize, or modify the narration.
        3. **DETERMINISTIC DATES**: Extract dates in DD/MM/YYYY format.
        4. **DOUBLE ENTRY PRINCIPLE**: Every transaction must have at least TWO entries (Debit & Credit).
           - **Bank Account**: One side is ALWAYS "Bank Account".
             - If statement says DEBIT (Money Out) -> Books: Credit "Bank Account" and Debit an **EXPENSE** or **ASSET** account. (NEVER Debit "Sales" for Money Out unless it is a refund).
             - If statement says CREDIT (Money In) -> Books: Debit "Bank Account" and Credit an **INCOME** or **LIABILITY** account.
           - **Counter Account**: Classify the other side based on narration. 
             - *Examples:* "Fuel Expense", "Office Rent", "Sales Income", "Capital", "Loan from Bank".
        5. **CLASSIFICATION RULE (Ind AS)**:
           - **Current**: Expected to be settled/realized within 12 months.
           - **Non-Current**: Held for long-term use (> 12 months).
        
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
        
        VALID CATEGORIES (Ind AS / Schedule III):
        - "Non-Current Assets" (PPE, Intangibles, Long-term Investments)
        - "Current Assets" (Inventory, Receivables, Cash, Bank, Prepaid)
        - "Equity" (Share Capital, Reserves & Surplus)
        - "Non-Current Liabilities" (Long-term Borrowings, Deferred Tax)
        - "Current Liabilities" (Payables, Short-term Borrowings, Accrued Expenses)
        - "Direct Income"
        - "Indirect Income"
        - "Direct Expense"
        - "Indirect Expense"
        
        OUTPUT JSON ONLY:
        """
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
            print(f"[WARNING] Batch extraction failed: {e}")
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
        context_metadatas: List[Dict[str, Any]] = None
    ) -> AsyncGenerator[Union[str, Dict[str, str]], None]:
        """
        Stream the financial synthesis using a Map-Reduce strategy (Batch Extraction -> Final Report).
        """
        
        if not context_chunks:
            yield "[ERROR] No context data provided."
            return

        # ---------------------------------------------------------
        # PHASE 1: BATCH EXTRACTION (Map Step)
        # ---------------------------------------------------------
        total_chunks = len(context_chunks)
        BATCH_SIZE = 50 # Increased for faster processing of large documents.
        
        all_transactions = []
        
        yield {"status": "Analyzing documents..."}
        print(f"[INFO] Starting batch extraction for {total_chunks} chunks...")

        # Create batches
        batches = [context_chunks[i:i + BATCH_SIZE] for i in range(0, total_chunks, BATCH_SIZE)]
        total_batches = len(batches)
        
        # Parallel Processing Setup
        sem = asyncio.Semaphore(10) # Allow 10 concurrent batch requests for faster extraction
        tasks = []

        async def process_batch(index, batch_data, batch_metas):
            async with sem:
                # Add a small stagger to prevent all requests hitting exactly at t=0
                await asyncio.sleep(index * 0.1) 
                transactions = await self._extract_transactions_from_batch(batch_data)
                
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
        
        # Process batches in parallel but collect results in DETERMINISTIC ORDER
        # This ensures the same chunks always produce results in the same sequence
        batch_results = await asyncio.gather(*tasks, return_exceptions=True)
        
        # Process results in order
        for i, result in enumerate(batch_results):
            # Update progress
            yield {"status": f"Extracting transactions (Batch {i+1}/{total_batches} complete)..."}
            
            # Handle exceptions
            if isinstance(result, Exception):
                print(f"[WARNING] Batch {i} extraction failed: {result}")
                continue
            
            transactions = result
            if transactions:
                # Add to shared accounts to help next batches stay consistent
                for t in transactions:
                    for e in t.get('entries', []):
                        self.shared_accounts.add(str(e.get('account')).strip().title())
                
                all_transactions.extend(transactions)
                print(f"[INFO] Batch {i} completed: Extracted {len(transactions)} transactions")

        if not all_transactions:
            print("[WARNING] No transactions could be extracted from the documents.")
            yield "Information: No transactions could be identified in the provided documents. Please ensure the documents contain clear financial data.\n"
            return

        print(f"[INFO] Total extracted transactions: {len(all_transactions)}")
        
        # --- NO DEDUPLICATION ---
        # Keep ALL transactions extracted by the AI, including any duplicates.
        # Sort for deterministic ordering only.
        
        # Sort by date, then by total amount, then by narration for consistent processing
        def get_sort_key(t):
            date = str(t.get('date', '')).strip()
            narration = str(t.get('narration', '')).strip()
            raw_entries = t.get('entries', [])
            total_amount = sum(float(e.get('amount', 0)) for e in raw_entries if str(e.get('type')).upper() == "DEBIT")
            if total_amount == 0 and raw_entries:
                total_amount = sum(float(e.get('amount', 0)) for e in raw_entries if str(e.get('type')).upper() == "CREDIT")
            return (date, total_amount, narration)
        
        all_transactions.sort(key=get_sort_key)
        print(f"[DEBUG] Transactions sorted for deterministic ordering")
        print(f"[INFO] Keeping ALL {len(all_transactions)} transactions (deduplication disabled)")
        
        # No deduplication - use all transactions as-is
        print(f"[INFO] Total transactions after sorting: {len(all_transactions)} (no deduplication applied)")
        
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

        # ---------------------------------------------------------
        # PHASE 2: FINANCIAL PROCESSING (Python Core)
        # ---------------------------------------------------------
        
        # Data Structures for Financial Statements
        ledger_balances: Dict[str, float] = {}
        account_category_votes: Dict[str, List[str]] = {}
        journal_rows_data = []
        tx_count = 0
        MAX_JOURNAL_ENTRIES_SHOW = 50
        
        yield "📒 FINANCIAL STATEMENTS (DRAFT)\n\n"
        yield "STATUS: DRAFT / UNAUDITED\n"
        yield "GENERATED BY: AI Assistant (Automated)\n"
        yield "DATE: 2026-01-21\n\n"

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

        # ---------------------------------------------------------
        # PHASE 3: REPORT GENERATION (LLM Synthesis with Rules)
        # ---------------------------------------------------------
        
        yield {"status": "Compiling Trial Balance..."}
        
        # ---------------------------------------------------------
        # DETERMINISTIC CLASSIFICATION (VOTING)
        # ---------------------------------------------------------
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
            yield f"\n⚠️ IMBALANCE DETECTED: A mismatch of ₹ {abs(total_debits - total_credits):,.2f} was found in the data extraction. A Suspense Account has been added to balance the books.\n"

        yield {"status": "Synthesizing Final Balance Sheet with AI..."}

        # ---------------------------------------------------------
        # PHASE 3: REPORT GENERATION (AI-Led Synthesis)
        # ---------------------------------------------------------
        # We provide the AI with the clean Trial Balance and strict accounting rules.
        # AI manages all calculations (P&L, Net Profit, BS Tally).
        
        prompt = f"""
You are an expert Senior Chartered Accountant. Your task is to prepare finalized financial statements (Profit & Loss Account and Balance Sheet) based on the provided Trial Balance for Harsh Tailor.

### STRICT FORMATTING RULE (CRITICAL)
- DO NOT USE ANY MARKDOWN BOLDING (DOUBLE ASTERISKS **).
- DO NOT USE MARKDOWN HEADERS (#, ##, etc.).
- ALL TEXT MUST BE CLEAN, PLAIN TEXT WITHOUT BOLD FORMATTING OR HEADERS.
- THIS APPLIES TO HEADERS, ACCOUNT NAMES, AND NUMBERS.

---

### ACCOUNTING RULES (IND AS & COMPANIES ACT 2013)

1. CORE STRUCTURE (SCHEDULE III):
   - Balance Sheet Equation: Assets = Liabilities + Equity.
   - Classification: Must classify into Current and Non-Current.
     - Current: Expected to be realized/settled within 12 months or operating cycle.
     - Non-Current: Held for long-term use (> 12 months).

2. VALUATION RULES:
   - Cash and Bank: Face value.
   - Accounts Receivable: Net Realizable Value (Gross - Allowance for doubtful debts).
   - Inventory: Lower of Cost or Net Realisable Value (Ind AS 2).
   - PPE (Property, Plant & Equipment): Cost less accumulated depreciation (Ind AS 16).
   - Intangible Assets: Cost less accumulated amortization (Ind AS 38).
   - Investments: Fair value or cost (Ind AS 109).
   - Provisions: Recognized when there is a present obligation (Ind AS 37).

3. DEPRECIATION & AMORTIZATION:
   - Apply depreciation to tangible assets (PPE) and amortization to intangible assets (Ind AS 16/38).

4. IMPAIRMENT (Ind AS 36):
   - If recoverable amount < carrying amount, recognize impairment loss in P&L.

5. DISCLOSURE REQUIREMENTS:
   - Disclose Contingent Liabilities, Commitments, and Related Party Transactions in notes if applicable.

6. PROFIT & LOSS (P&L) CALCULATION:
   - Net Profit = (Total Revenue) - (Total Expenses). Use this to update Reserves & Surplus in Equity.

7. SELF-AUDIT VERIFICATION:
   - Perform math check: Total Assets MUST equal Total Liabilities + Equity.

---

### INPUT DATA (TRIAL BALANCE)
{trial_balance_summary}

---

### REQUIRED OUTPUT FORMAT (SCHEDULE III COMPLIANT)

1. Profit & Loss Account
| Particulars | Amount (₹) |
|---|---|
| Revenue/Sales | [Amount] |
| (-) Cost of Goods Sold (COGS) | [Amount] |
| Gross Profit | [Amount] |
| (-) Operating Expenses: | |
| - Salaries & Wages | [Amount] |
| - Rent | [Amount] |
| - Utilities | [Amount] |
| - Depreciation & Amortization | [Amount] |
| - Marketing & Advertising | [Amount] |
| - Repairs & Maintenance | [Amount] |
| - Administrative Expenses | [Amount] |
| - Interest Expenses | [Amount] |
| Operating Profit | [Amount] |
| (+) Other Income | [Amount] |
| (-) Tax Expenses | [Amount] |
| Net Profit/Loss | [Amount] |

2. Balance Sheet
ASSETS
| Particulars | Amount (₹) |
|---|---|
| Non-Current Assets: | |
| Property, Plant & Equipment | [Amount] |
| Intangible Assets | [Amount] |
| Long-Term Investments | [Amount] |
| Deferred Tax Assets | [Amount] |
| Current Assets: | |
| Inventories | [Amount] |
| Trade Receivables | [Amount] |
| Cash & Bank Balances | [Amount] |
| Prepaid Expenses | [Amount] |
| Other Current Assets | [Amount] |
| Total Assets | [Amount] |

EQUITY AND LIABILITIES
| Particulars | Amount (₹) |
|---|---|
| Shareholders' Equity: | |
| Share Capital | [Amount] |
| Reserves & Surplus | [Amount] |
| Non-Current Liabilities: | |
| Long-Term Borrowings | [Amount] |
| Deferred Tax Liabilities | [Amount] |
| Current Liabilities: | |
| Trade Payables | [Amount] |
| Short-Term Borrowings | [Amount] |
| Other Current Liabilities | [Amount] |
| Total Equity and Liabilities | [Amount] |

3. Notes & Disclosures
- Contingent Liabilities: [Details and Amount]
- Related Party Transactions: [Details and Amount]
- Impairment of Assets: [Details and Amount]

4. Validation Summary:
- Total Assets: ₹ [Amount]
- Total Equity and Liabilities: ₹ [Amount]
- Status: TALLIED (or "NOT TALLIED - ERROR: [Reason]")

---
GO! Perform every calculation accurately and ensure the Balance Sheet side totals are identical. REMEMBER: NO BOLDING OR HEADERS.
"""

        try:
            async for chunk in self.llm.astream(prompt):
                if chunk.content:
                    yield chunk.content
        except Exception as e:
            print(f"[ERROR] Final Synthesis failed: {str(e)}")
            yield f"\n[ERROR] Final Synthesis failed: {str(e)}"
            
        # FINAL DISCLAIMER
        yield "\n---\n"
        yield "DISCLAIMER: This report is a computer-generated DRAFT. It adheres to Indian Accounting Standards but strictly requires human verification before filing."

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


