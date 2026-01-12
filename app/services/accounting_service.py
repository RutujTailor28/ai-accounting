from typing import List, Dict, Any, AsyncGenerator
import json
import asyncio
from langchain_openai import ChatOpenAI
from app.core.config import settings

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
        print(f"[INFO] AccountingService initialized with model={model}")

    async def stream_accounting_synthesis(
        self,
        question: str,
        context_chunks: List[str]
    ) -> AsyncGenerator[str, None]:
        """
        Stream the financial synthesis (SSE format).
        """
        context = "\n\n".join(context_chunks)
        
        prompt = f"""
(Senior Chartered Accountant / CPA | Indian Accounting Standards)

You are performing a **Financial Synthesis & Audit-Compliant Accounting Task**.
Your responsibility is to convert raw bank statement data into **correct journal entries,
ledgers, Profit & Loss Account, and Balance Sheet**.

You must strictly follow **Indian Accounting Principles and Audit Rules**.
Accuracy of **Debit / Credit placement is mandatory**.

────────────────────────
CORE ACCOUNTING RULES (NON-NEGOTIABLE)
────────────────────────

1. **Golden Rules of Accounting**
   - Personal Account → Debit the Receiver, Credit the Giver
   - Real Account → Debit what comes in, Credit what goes out
   - Nominal Account → Debit all Expenses & Losses, Credit all Incomes & Gains

2. **Account Classification (MUST IDENTIFY FIRST)**
   Every transaction MUST be classified as one of:
   - Asset
   - Liability
   - Capital
   - Income
   - Expense

   Apply posting rules ONLY after classification.

3. **Debit / Credit Enforcement**
   - Assets increase → Debit | decrease → Credit
   - Liabilities increase → Credit | decrease → Debit
   - Capital introduced → Credit
   - Drawings → Debit
   - Income → ALWAYS Credit
   - Expense → ALWAYS Debit

   ❌ Never credit an expense  
   ❌ Never debit an income  

4. **Journal Entry Rules**
   - Format: Date | Particulars | L.F. | Debit (Rs.) | Credit (Rs.)
   - Debit entry MUST appear first
   - Credit entry MUST be prefixed with **"To "**
   - Every entry MUST include a short narration in brackets
   - Amounts on both sides MUST be equal

────────────────────────
PROFIT & LOSS ACCOUNT RULES
────────────────────────

5. **P&L Construction**
   - Debit side → All expenses and losses
   - Credit side → All incomes and gains
   - Net Profit = Credit total − Debit total
   - Net Profit MUST be transferred to Capital Account

────────────────────────
BALANCE SHEET RULES (AUDIT SAFE)
────────────────────────

6. **Balance Sheet Structure**
   - Liabilities:
     - Capital (Opening + Profit − Drawings)
     - Secured Loans
     - Unsecured Loans
     - Sundry Creditors
   - Assets:
     - Fixed Assets
     - Investments
     - Sundry Debtors
     - Cash & Bank
     - Closing Stock

7. **Mandatory Validation Checks**
   - Total Assets MUST equal Total Liabilities + Capital
   - Cash/Bank balance must NOT be negative (unless overdraft explicitly stated)
   - Debtors cannot have credit balance
   - Creditors cannot have debit balance
   - Capital movement must reconcile with P&L result

   If any check fails → FLAG AS **ACCOUNTING ERROR**

────────────────────────
INFERENCE RULES
────────────────────────

8. **Controlled Inference**
   - You MAY infer account names from narration (e.g., "Mobile Recharge" → Mobile Expense)
   - You MUST NOT invent amounts or transactions
   - Only use values present in the provided data

────────────────────────
INPUT DATA
────────────────────────

Context Data:
{context}

User Task:
{question}

────────────────────────
OUTPUT FORMAT (STRICT)
────────────────────────

- Professional academic tone (CA / CPA level)
- Use Markdown tables for:
  1. Journal Entries
  2. Ledger Summary
  3. Profit & Loss Account
  4. Balance Sheet
- Clearly label Debit and Credit columns
- If unsure about classification → clearly mention **"Requires Human Review"**

Begin processing now.
"""

        try:
            # Use astream for token-by-token delivery
            async for chunk in self.llm.astream(prompt):
                if chunk.content:
                    yield chunk.content
        except Exception as e:
            yield f"\n[ERROR] Synthesis failed: {str(e)}"
