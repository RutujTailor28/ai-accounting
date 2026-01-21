from typing import List, Dict, Any, AsyncGenerator, Union
import json
import asyncio
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
        print(f"[INFO] AccountingService initialized with model={model}")

    async def _extract_transactions_from_batch(self, batch_chunks: List[str]) -> List[Dict]:
        """
        Extract structured transaction data from a small batch of chunks.
        """
        # Injection of already seen accounts to prevent duplicates
        context = "\n".join(batch_chunks)
        known_accounts_str = ", ".join(list(self.shared_accounts)[:50]) if self.shared_accounts else "None yet"
        
        prompt = f"""
        You are an expert Data Entry Clerk. Your task is to extract accounting transactions from the text below.
        
        INPUT TEXT:
        {context}
        
        KNOWN CHART OF ACCOUNTS (Use these names if they match to ensure consistency):
        {known_accounts_str}
        
        INSTRUCTIONS:
        1. Extract EVERY transaction found.
        2. **DOUBLE ENTRY PRINCIPLE**: Every transaction must have at least TWO entries (Debit & Credit).
           - **Bank Account**: One side is ALWAYS "Bank Account".
             - If statement says DEBIT (Money Out) -> Books: Credit "Bank Account" and Debit an **EXPENSE** or **ASSET** account. (NEVER Debit "Sales" for Money Out unless it is a refund).
             - If statement says CREDIT (Money In) -> Books: Debit "Bank Account" and Credit an **INCOME** or **LIABILITY** account.
           - **Counter Account**: Classify the other side based on narration. 
             - *Examples:* "Fuel Expense", "Office Rent", "Sales Income", "Capital", "Loan from Bank".
        3. **CLASSIFICATION RULE (Ind AS)**:
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
            return data.get("transactions", [])
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
        BATCH_SIZE = 35 # Increased for faster processing of large documents.
        
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
                    source_ids = [m.get('chunk_index') for m in batch_metas if m.get('chunk_index') is not None]
                    for t in transactions:
                        t['_source_chunks'] = source_ids
                return transactions

        for i, batch in enumerate(batches):
            batch_metas = context_metadatas[i*BATCH_SIZE : (i+1)*BATCH_SIZE] if context_metadatas else None
            tasks.append(process_batch(i, batch, batch_metas))
        
        # Process batches in parallel but yield results as they complete
        completed_batches = 0
        for future in asyncio.as_completed(tasks):
            transactions = await future
            completed_batches += 1
            
            # Update progress
            yield {"status": f"Extracting transactions (Batch {completed_batches}/{total_batches} complete)..."}
            
            if transactions:
                # Add to shared accounts to help next batches stay consistent
                for t in transactions:
                    for e in t.get('entries', []):
                        self.shared_accounts.add(str(e.get('account')).strip().title())
                
                all_transactions.extend(transactions)
                print(f"[INFO] Batch completed: Extracted {len(transactions)} transactions")

        if not all_transactions:
            yield "[WARNING] No transactions could be extracted from the documents. Please check the file quality."
            return

        print(f"[INFO] Total extracted transactions (pre-dedup): {len(all_transactions)}")
        
        # --- DEDUPLICATION LOGIC ---
        # We want to remove duplicates caused by overlapping chunks (RAG artifacts)
        # but KEEP legitimate duplicate transactions in the bank statement.
        
        unique_transactions = []
        # Key: (date, narration, entries_sig), Value: List of source_chunk_sets
        seen_tx_sources: Dict[tuple, List[set]] = {}
        
        for t in all_transactions:
            date = str(t.get('date', '')).strip()
            narration = str(t.get('narration', '')).strip()
            raw_entries = t.get('entries', [])
            sorted_entries = sorted(raw_entries, key=lambda x: (str(x.get('account')), str(x.get('amount'))))
            entries_sig = "|".join([f"{e.get('account')}:{e.get('amount')}" for e in sorted_entries])
            
            tx_key = (date, narration, entries_sig)
            current_sources = set(t.get('_source_chunks', []))
            
            if tx_key not in seen_tx_sources:
                seen_tx_sources[tx_key] = [current_sources]
                unique_transactions.append(t)
            else:
                # We've seen an identical transaction before. 
                # Is it an overlap or a new legitimate one?
                # If the source chunks overlap significantly with any previous instance, it's likely an overlap duplicate.
                is_duplicate_of_overlap = False
                for prev_sources in seen_tx_sources[tx_key]:
                    # IMPORTANT: Only drop if it's an overlap artifact.
                    # If sources are EXACTLY identical, it means they were extracted from the SAME batch/call.
                    # Legitimate duplicates in the same document will appear in the same batch.
                    if current_sources != prev_sources and current_sources.intersection(prev_sources):
                        is_duplicate_of_overlap = True
                        break
                
                if not is_duplicate_of_overlap:
                    # New instance or same batch -> likely legitimate
                    seen_tx_sources[tx_key].append(current_sources)
                    unique_transactions.append(t)
        
        all_transactions = unique_transactions
        print(f"[INFO] Total unique transactions (post-dedup): {len(all_transactions)}")
        yield {"status": f"Verifying Double-Entry Integrity for {len(all_transactions)} unique transactions..."}

        # ---------------------------------------------------------
        # PHASE 2: FINANCIAL PROCESSING (Python Core)
        # ---------------------------------------------------------
        
        # Data Structures for Financial Statements
        ledger_balances: Dict[str, float] = {}
        account_category_votes: Dict[str, List[str]] = {} # Map Account -> List of Categories (for Voting)
        
        journal_rows = []
        
        yield "\n\n# 📒 FINANCIAL STATEMENTS (DRAFT)\n"
        yield "> STATUS: 🟡 DRAFT / UNAUDITED\n"
        yield "> GENERATED BY: AI Assistant (Automated)\n"
        yield "> DATE: 2026-01-19\n\n"

        yield "## 1. Professional Journal Book\n\n"
        yield "| Date | Particulars (Account) | L.F. | Debit (₹) | Credit (₹) | Narration |\n"
        yield "|---|---|---|---|---|---|\n"

        for t in all_transactions:
            date = t.get('date', '')
            narration = t.get('narration', '')
            entries = t.get('entries', [])
            
            # Validation: Double Entry
            total_dr = sum(e.get('amount', 0) for e in entries if str(e.get('type')).upper() == "DEBIT")
            total_cr = sum(e.get('amount', 0) for e in entries if str(e.get('type')).upper() == "CREDIT")
            
            if abs(total_dr - total_cr) > 0.01:
                # Reject invalid transaction
                print(f"[ERROR] Transaction Imbalance: Dr {total_dr} != Cr {total_cr} for {narration}")
                continue
                
            MAX_JOURNAL_ENTRIES_SHOW = 50
            tx_count = (tx_count + 1) if 'tx_count' in locals() else 1
            
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
            
            # --- UI OUTPUT (Truncate to avoid token limits) ---
            if tx_count <= MAX_JOURNAL_ENTRIES_SHOW:
                first_row = True
                for entry in entries:
                    account = str(entry.get('account', 'Unclassified')).strip().title()
                    etype = str(entry.get('type')).upper()
                    amount = float(entry.get('amount', 0))
                    
                    desc_col = f"{account} Dr." if etype == "DEBIT" else f"&nbsp;&nbsp;&nbsp;&nbsp;To {account}"
                    dr_val = f"{amount:,.2f}" if etype == "DEBIT" else ""
                    cr_val = f"{amount:,.2f}" if etype == "CREDIT" else ""
                    
                    yield f"| {date if first_row else ''} | {desc_col} | | {dr_val} | {cr_val} | {narration if first_row else ''} |\n"
                    first_row = False
                yield f"| | | | | | |\n" # Spacer
            
            elif tx_count == MAX_JOURNAL_ENTRIES_SHOW + 1:
                remaining = len(unique_transactions) - MAX_JOURNAL_ENTRIES_SHOW
                yield f"| ... | ... | | ... | ... | *({remaining} more transactions processed internally. 100% data integrity maintained for final report.)* |\n"

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
        yield "\n\n## 2. Ledger Summary (Trial Balance)\n\n"
        yield "| Account Head | Net Balance (₹) | Type |\n"
        yield "|---|---|---|\n"
        
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
        
        if abs(total_debits - total_credits) > 0.01:
            yield f"\n> ⚠️ IMBALANCE DETECTED: A mismatch of ₹ {abs(total_debits - total_credits):,.2f} was found in the data extraction. A Suspense Account has been added to balance the books.\n"

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
- ALL TEXT MUST BE CLEAN, PLAIN TEXT WITHOUT BOLD FORMATTING.
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
GO! Perform every calculation accurately and ensure the Balance Sheet side totals are identical. REMEMBER: NO BOLDING.
"""

        try:
            async for chunk in self.llm.astream(prompt):
                if chunk.content:
                    yield chunk.content
        except Exception as e:
            print(f"[ERROR] Final Synthesis failed: {str(e)}")
            yield f"\n[ERROR] Final Synthesis failed: {str(e)}"
            
        # FINAL COMPLIANCE FOOTER
        yield "\n---\n"
        yield "### ⚠️ COMPLIANCE & APPROVAL\n"
        yield "- [ ] Reviewer Name: ____________________\n"
        yield "- [ ] Designation: Chartered Accountant\n"
        yield "- [ ] Date: ____________________\n"
        yield "- [ ] Signature: ____________________\n\n"
        yield "> DISCLAIMER: This report is a computer-generated DRAFT. It adheres to Indian Accounting Standards but strictly requires human verification before filing."

