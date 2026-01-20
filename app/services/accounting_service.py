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
        print(f"[INFO] AccountingService initialized with model={model}")

    async def _extract_transactions_from_batch(self, batch_chunks: List[str]) -> List[Dict]:
        """
        Extract structured transaction data from a small batch of chunks.
        """
        context = "\n".join(batch_chunks)
        
        prompt = f"""
        You are an expert Data Entry Clerk. Your task is to extract accounting transactions from the text below.
        
        INPUT TEXT:
        {context}
        
        INSTRUCTIONS:
        1. Extract EVERY transaction found.
        2. **DOUBLE ENTRY PRINCIPLE**: Every transaction must have at least TWO entries (Debit & Credit).
           - **Bank Account**: One side is ALWAYS "Bank Account".
             - If statement says DEBIT (Money Out) -> Books: Credit "Bank Account" and Debit an **EXPENSE** or **ASSET** account. (NEVER Debit "Sales" for Money Out unless it is a refund).
             - If statement says CREDIT (Money In) -> Books: Debit "Bank Account" and Credit an **INCOME** or **LIABILITY** account.
           - **Counter Account**: Classify the other side based on narration. 
             - *Examples:* "Fuel Expense", "Office Rent", "Sales Income", "Capital", "Loan from Bank".
        
        REQUIRED JSON FORMAT:
        {{
            "transactions": [
                {{
                    "date": "DD/MM/YYYY",
                    "narration": "Original narration text",
                    "entries": [
                         {{ "account": "Fuel Expense", "type": "DEBIT", "amount": 500.00, "category": "Direct Expense" }},
                         {{ "account": "Bank Account", "type": "CREDIT", "amount": 500.00, "category": "Current Assets" }}
                    ]
                }}
            ]
        }}
        
        VALID CATEGORIES:
        - "Fixed Assets" (Property, Plant, Equipment)
        - "Current Assets" (Bank, Cash, Inventory, Receivables)
        - "Equity" (Capital, Draws)
        - "Long Term Liabilities" (Loans > 1yr)
        - "Current Liabilities" (Payables, Short Loans)
        - "Direct Income" (Sales, Core Revenue)
        - "Indirect Income" (Interest, Dividends)
        - "Direct Expense" (COGS, Raw Material)
        - "Indirect Expense" (Admin, Salary, Rent, Fuel)
        
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

    async def stream_accounting_synthesis(
        self,
        question: str,
        context_chunks: List[str]
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
        BATCH_SIZE = 20 # Reduced for better accuracy. (50 was too large)
        
        all_transactions = []
        
        yield {"status": "Analyzing documents..."}
        print(f"[INFO] Starting batch extraction for {total_chunks} chunks...")

        # Create batches
        batches = [context_chunks[i:i + BATCH_SIZE] for i in range(0, total_chunks, BATCH_SIZE)]
        total_batches = len(batches)
        
        # Parallel Processing Setup
        sem = asyncio.Semaphore(5) # Allow 5 concurrent batch requests
        tasks = []

        async def process_batch(index, batch_data):
            async with sem:
                # Add a small stagger to prevent all requests hitting exactly at t=0
                await asyncio.sleep(index * 0.1) 
                return await self._extract_transactions_from_batch(batch_data)

        for i, batch in enumerate(batches):
            tasks.append(process_batch(i, batch))
        
        # Process batches in parallel but yield results as they complete
        completed_batches = 0
        for future in asyncio.as_completed(tasks):
            transactions = await future
            completed_batches += 1
            
            # Update progress
            yield {"status": f"Extracting transactions (Batch {completed_batches}/{total_batches} complete)..."}
            
            if transactions:
                all_transactions.extend(transactions)
                print(f"[INFO] Batch completed: Extracted {len(transactions)} transactions")

        if not all_transactions:
            yield "[WARNING] No transactions could be extracted from the documents. Please check the file quality."
            return

        print(f"[INFO] Total extracted transactions (pre-dedup): {len(all_transactions)}")
        
        # --- DEDUPLICATION LOGIC (Fixes Chunk Overlap Duplicates) ---
        unique_transactions = []
        seen_hashes = set()
        
        for t in all_transactions:
            # Create a stable signature for each transaction
            date = str(t.get('date', '')).strip()
            narration = str(t.get('narration', '')).strip()
            # Sort entries for stable hashing
            raw_entries = t.get('entries', [])
            sorted_entries = sorted(raw_entries, key=lambda x: (str(x.get('account')), str(x.get('amount'))))
            entries_sig = "|".join([f"{e.get('account')}:{e.get('amount')}" for e in sorted_entries])
            
            tx_hash = f"{date}#{narration}#{entries_sig}"
            
            if tx_hash not in seen_hashes:
                seen_hashes.add(tx_hash)
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
        yield "> **STATUS:** 🟡 DRAFT / UNAUDITED\n"
        yield "> **GENERATED BY:** AI Assistant (Automated)\n"
        yield "> **DATE:** 2026-01-19\n\n"

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
                
            # Process Entries
            row_span = len(entries)
            first_row = True
            
            for entry in entries:
                account = entry.get('account', 'Unclassified')
                category = entry.get('category', 'Unclassified')
                amount = float(entry.get('amount', 0))
                etype = str(entry.get('type')).upper()
                
                # Update ledger
                if account not in ledger_balances:
                    ledger_balances[account] = 0.0
                    account_category_votes[account] = []
                
                # Add category vote (ignore 'Unclassified')
                if category and category != "Unclassified":
                    account_category_votes[account].append(category)
                
                if etype == "DEBIT":
                    ledger_balances[account] += amount
                    dr_str = f"{amount:,.2f}"
                    cr_str = ""
                else:
                    ledger_balances[account] -= amount # We store Credit as negative for internal sum = 0 check
                    dr_str = ""
                    cr_str = f"{amount:,.2f}"
                
                # Table Row
                prefix = "" if first_row else ""
                desc_col = f"**{account}** Dr." if etype == "DEBIT" else f"&nbsp;&nbsp;&nbsp;&nbsp;To **{account}**"
                narration_show = narration if first_row else ""
                
                yield f"| {date if first_row else ''} | {desc_col} | | {dr_str} | {cr_str} | {narration_show} |\n"
                first_row = False
            
            yield f"| | | | | | |\n" # Spacer row

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
                
        yield f"| **TOTAL** | **{total_debits:,.2f}** | **{total_credits:,.2f}** |\n"
        
        yield {"status": "Synthesizing Final Balance Sheet with AI..."}

        # ---------------------------------------------------------
        # PYTHON MATH PRE-CALCULATION (Control Layer)
        # ---------------------------------------------------------
        # We classify based on the extracted 'category' to give the LLM a math target.
        
        calc_income = 0.0
        calc_expense = 0.0
        calc_assets = 0.0
        calc_liab_equity = 0.0
        
        # Helper to categorize if missing
        def get_broad_type(acc_name, cat_name, balance):
            cat_upper = cat_name.upper()
            if "INCOME" in cat_upper: return "INCOME"
            if "EXPENSE" in cat_upper: return "EXPENSE"
            if "ASSET" in cat_upper: return "ASSET"
            if "LIABIL" in cat_upper or "EQUITY" in cat_upper: return "LIABILITY"
            
            # Fallback based on sign (Dr=Asset/Exp, Cr=Liab/Inc)
            # This is imperfect for ambiguous accounts, but good for a hint.
            if balance > 0: return "ASSET_OR_EXP" 
            return "LIAB_OR_INC"

        for acc, bal in ledger_balances.items():
            if abs(bal) < 0.01: continue
            cat = account_categories.get(acc, "")
            broad_type = get_broad_type(acc, cat, bal)
            amt = abs(bal)
            
            if broad_type == "INCOME": calc_income += amt
            elif broad_type == "EXPENSE": calc_expense += amt
            elif broad_type == "ASSET": calc_assets += amt
            elif broad_type == "LIABILITY": calc_liab_equity += amt
            elif broad_type == "ASSET_OR_EXP":
                # Educated Guess
                if "BANK" in acc.upper() or "CASH" in acc.upper() or "RECEIVABLE" in acc.upper():
                    calc_assets += amt
                else: 
                    calc_expense += amt
            else: # LIAB_OR_INC
                if "SALES" in acc.upper() or "INTEREST" in acc.upper():
                    calc_income += amt
                else:
                    calc_liab_equity += amt
                    
        calc_net_profit = calc_income - calc_expense
        # Final Verification Equation: Assets = Liab + Equity + Profit
        # So Target BS Total = calc_assets approx? 
        # Actually, if we classify Expenses as Assets (mistake), Profit increases (Assets - Expenses), Equity Increases.
        # Assets increases. BS still balances.
        # So we just provide the classification hints.
        
        control_data = f"""
        **SYSTEM PRE-CALCULATED HINTS (DO NOT CONTRADICT THESE):**
        - Python Calculated Total Income: ₹ {calc_income:,.2f}
        - Python Calculated Total Expenses: ₹ {calc_expense:,.2f}
        - **MANDATORY NET PROFIT TO TRANSFER:** ₹ {calc_net_profit:,.2f}
        - Target Balance Sheet Side Total: ₹ {calc_assets:,.2f}
        
        *INSTRUCTIONS: You MUST use the Net Profit of ₹ {calc_net_profit:,.2f} in your Capital calculation. If your classification leads to a different tally, prioritize the double-entry integrity so that Assets = Liabilities + Equity + Net Profit.*
        """

        # 2. GENERATE P&L AND BALANCE SHEET (LLM)
        # We inject the USER'S SPECIFIC RULES here
        
        prompt = f"""
        {control_data}

You are acting as a **Junior Chartered Accountant (India)** assisting in preparation of financial statements.
You are NOT allowed to take assumptions beyond accounting rules.
Human review is mandatory.

Your task is to generate a **Profit & Loss Account** and a **Balance Sheet**
from the provided **Trial Balance / Ledger Summary**.

────────────────────────────────────
BANK STATEMENT ANALYSIS RULES (FUNDAMENTAL)
────────────────────────────────────
Since the source is primarily a Bank Statement, apply these rules to classify flows:

1. **INFLOWS (Money In)** are either:
   - **Revenue Receipt**: Sales, Service Income, Commission, Interest. -> [PROFIT & LOSS]
   - **Capital Receipt**: 
     - Loan taken (Liability). -> [BALANCE SHEET]
     - Capital Introduced (Equity). -> [BALANCE SHEET]
     - Asset Sold (Reduce Asset). -> [BALANCE SHEET]

2. **OUTFLOWS (Money Out)** are either:
   - **Revenue Expenditure**: Rent, Salary, Bill Payments, Purchases, Fuel. -> [PROFIT & LOSS]
   - **Capital Expenditure**: 
     - Asset Purchase (Machine, Computer, Vehicle). -> [BALANCE SHEET - ASSETS]
     - Loan Repayment (Reduce Liability). -> [BALANCE SHEET - LIABILITIES]
     - Drawings/Personal Use (Reduce Capital). -> [BALANCE SHEET - EQUITY]

────────────────────────────────────
CRITICAL CLASSIFICATION RULES (Dr vs Cr)
────────────────────────────────────
The Trial Balance provides the nature (Dr/Cr) for every item. You MUST respect this:

1. **(Dr) DEBIT BALANCES** can *ONLY* be:
   - **ASSETS** (Balance Sheet - Right Side)
   - **EXPENSES** (P&L - Debit Side)
   - *Drawings* (Deduction from Capital)
   - **FATAL ERROR:** Never put a (Dr) balance in Liabilities or Income.

2. **(Cr) CREDIT BALANCES** can *ONLY* be:
   - **LIABILITIES** (Balance Sheet - Left Side)
   - **INCOME** (P&L - Credit Side)
   - **EQUITY/CAPITAL** (Balance Sheet - Left Side)
   - **FATAL ERROR:** Never put a (Cr) balance in Assets or Expenses.

────────────────────────────────────
STRICT ACCOUNTING RULES (NON-NEGOTIABLE)
────────────────────────────────────

GENERAL RULES:
1. Follow **Indian Accounting Standards (Ind AS)**.
2. DO NOT invent, rename, merge, or split ledger heads.
3. Each ledger must appear in **ONLY ONE** of the following:
   - Profit & Loss Account
   - Balance Sheet (Assets / Liabilities / Equity)
4. If classification is unclear, mark the item as:
   **"REQUIRES HUMAN REVIEW"** (do NOT guess).

────────────────────────────────────
PROFIT & LOSS ACCOUNT RULES
────────────────────────────────────

INCLUDE ONLY:
• Income accounts
• Expense accounts

INCOME (Credit nature):
- Sales / Service Income
- Direct Income
- Indirect Income
- Interest Income
- Miscellaneous Income
- Remittance Income

EXPENSES (Debit nature):
- All operating, administrative, finance, and miscellaneous expenses
- Purchases, rent, salary, fuel, utilities, bank charges, donations, etc.

STRICTLY EXCLUDE FROM P&L:
- Cash
- Bank
- Loans
- Capital
- Drawings
- Receivables / Payables

NET RESULT:
• Calculate Net Profit or Net Loss
• This figure MUST be transferred to CAPITAL in Balance Sheet

────────────────────────────────────
BALANCE SHEET RULES
────────────────────────────────────

Balance Sheet shows **financial position only**, NOT performance.

### ASSETS (Debit balances only)
Include ALL Assets, including but not limited to:
• Cash
• Bank balance
• Receivables / Advances given
• Deposits
• Inventory (if clearly identified)
• Fixed assets (Vehicles, Machinery, Furniture, etc.)
• Investments / Gold / Property
• Tax Receivables (GST, TDS)

DO NOT include:
• Expenses
• Income
• Payments to parties (unless clearly receivable)

### LIABILITIES (Credit balances only)
Include ALL Liabilities, including but not limited to:
• Payables / Creditors
• Outstanding expenses
• Loans (Bank / Finance companies / Directors)
• Taxes payable (GST, Income Tax)

DO NOT include:
• Payments made
• Expense ledgers
• Transfers without liability nature

### EQUITY
Include ONLY:
• Opening Capital (if available)
• Add: Net Profit
• Less: Net Loss
• Less: Drawings (explicit drawings only)

────────────────────────────────────
BALANCE SHEET (AGGREGATION IS MANDATORY)
────────────────────────────────────
1. **DO NOT list individual party names** (e.g., "Mr. Rahul", "ABC Corp") in the final Balance Sheet.
2. SUM all (Dr) balances of parties/individuals and show as a single line item: **Sundry Debtors**.
3. SUM all (Cr) balances of parties/individuals and show as a single line item: **Sundry Creditors**.
4. This keeps the Balance Sheet professional and audit-safe.

IMPORTANT:
• Transfers to individuals are NOT capital unless explicitly stated
• ATM withdrawal, cash deposit, cheque deposit are CASH/BANK movements, not liabilities

────────────────────────────────────
CRITICAL VALIDATIONS (MANDATORY)
────────────────────────────────────

1. Total Assets MUST equal:
   Total Liabilities + Equity

2. If Balance Sheet does not tally:
   - STOP
   - Report: **"BALANCE SHEET DOES NOT TALLY – HUMAN REVIEW REQUIRED"**
   - DO NOT force-balance or adjust numbers

3. Large unexplained balances, negative capital, or mixed-purpose ledgers
   must be flagged explicitly.

────────────────────────────────────
OUTPUT FORMAT (STRICT)
────────────────────────────────────

1. Profit & Loss Account – Markdown table
2. Balance Sheet – Markdown table
3. Validation Summary:
   - Net Profit / Loss
   - Assets Total
   - Liabilities + Equity Total
   - Status: TALLIED / NOT TALLIED

4. Human Review Declaration (MANDATORY):

"THIS IS AN AI-GENERATED DRAFT.
ALL FIGURES, CLASSIFICATIONS, AND TOTALS
MUST BE VERIFIED AND APPROVED
BY A QUALIFIED HUMAN ACCOUNTANT."

────────────────────────────────────
INPUT DATA (TRIAL BALANCE)
────────────────────────────────────
{trial_balance_summary}

────────────────────────────────────────
INSTRUCTIONS
────────────────────────────────────────

1. PROFIT & LOSS ACCOUNT
   - Classify Trial Balance items into Income and Expenses.
   - Calculate:
     Net Profit / Net Loss = Total Income – Total Expenses.
   - Present the calculation clearly in the statement.

2. BALANCE SHEET
   - Use standard format:
     LIABILITIES SIDE:
       Capital (show calculation: Capital + Profit – Drawings)
       Long-Term Liabilities
       Current Liabilities (including **Sundry Creditors**)
     ASSETS SIDE:
       Non-Current Assets
       Current Assets (including **Sundry Debtors** and Bank/Cash)
   - Ensure the Balance Sheet tallies exactly.
   - **CRITICAL:** Do NOT omit any Asset or Liability ledger from the Trial Balance.

────────────────────────────────────────
OUTPUT FORMAT
────────────────────────────────────────

- Markdown tables only.
- No explanations of accounting theory.
- No assumptions beyond the provided Trial Balance.
- Do not modify or invent figures.
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
        yield "- [ ] **Reviewer Name:** ____________________\n"
        yield "- [ ] **Designation:** Chartered Accountant\n"
        yield "- [ ] **Date:** ____________________\n"
        yield "- [ ] **Signature:** ____________________\n\n"
        yield "> **DISCLAIMER:** This report is a computer-generated DRAFT. It adheres to Indian Accounting Standards but strictly requires human verification before filing."

