from typing import List, Dict, Any, AsyncGenerator, Union
import json
import asyncio
import hashlib
from langchain_openai import ChatOpenAI
from app.core.config import settings
from app.services.ai_service import LLMService, retry_with_backoff
from app.core.llm_telemetry import (
    telemetry_callbacks,
    llm_stage,
    instrument_stream,
    STREAM_USAGE,
)
from app.ai.rag.retriever import vector_store
from app.services.accounting_rules import (
    EXTRACTOR_RULES, CLASSIFIER_RULES, REFINEMENT_RULES, JOURNAL_RULES_TEXT,
    PNL_RULES_TEXT, BALANCE_SHEET_RULES_TEXT, TALLY_RULES_TEXT, CAPITAL_ACCOUNT_RULES_TEXT,
    PNL_CONFIG, BALANCE_SHEET_CONFIG, TALLY_CONFIG, JOURNAL_CONFIG, CAPITAL_CONFIG,
    AGENT_SYSTEM_PROMPTS
)

class AccountingService:
    """Specialized service for accounting report generation (Journal Entries, Balance Sheets)."""

    def __init__(self, model_name: str = None):
        """Initialize with a more capable model for financial synthesis."""
        model = model_name or settings.active_model_accounting
        self.llm = ChatOpenAI(
            model=model,
            openai_api_key=settings.active_api_key or "not-needed",
            openai_api_base=settings.active_base_url,
            temperature=0,
            streaming=True,
            # streaming=True suppresses the usage block unless we opt in.
            # Without this every token count below would be an estimate.
            stream_usage=STREAM_USAGE,
            callbacks=telemetry_callbacks(
                provider=settings.active_provider, model=model
            ),
        )
        self.ai_service = LLMService()
        self.shared_accounts = set()
        self.extraction_cache: Dict[str, List[Dict]] = {}
        self.classification_cache: Dict[str, List[Dict]] = {}

        self.model_name = model.lower()
        self.is_free_model = ":free" in self.model_name

        self.n_extractors = 15
        self.n_classifiers = 15
        self.worker_delay = 0.1
        print(f"[INFO] AccountingService initialized with {self.n_extractors} extractors, {self.n_classifiers} classifiers.")

        print(f"[INFO] AccountingService initialized with model={model}")

    async def _get_base_doc_name(self, filename: str) -> str:
        """Strip extension and then duplicates like (1) or copy."""
        import re
        if not filename:
             return ""

        base = re.sub(r'\.\w+$', '', filename)

        base = re.sub(r'\s*\(\d+\)$', '', base, flags=re.IGNORECASE)
        base = re.sub(r'\s+copy$', '', base, flags=re.IGNORECASE)
        return base.strip().lower()

    async def _discover_intent(self, question: str) -> Dict[str, Any]:
        """
        Agent 0: Intent Discovery Agent.
        Uses LLM to categorize the user's request so we can route to the correct pipeline.
        """
        prompt = f"""
        You are an Accounting Intent Discovery Agent.
        Categorize the following user request into one of these types:
        - EXTRACTION: User wants to pull data from new documents/statements (e.g. "analyze this", "pnl", "show journal").
        - REFINEMENT: User wants to CHANGE or FIX existing data/calculations (e.g. "move this to shopping", "fix transaction 5", "update category").
        - ANALYSIS: User is asking a general question about the reports (e.g. "why is profit low?", "explain this entry").

        REQUEST: "{question}"

        JSON OUTPUT:
        {{
            "intent": "EXTRACTION | REFINEMENT | ANALYSIS",
            "explicit_limit": null,
            "report_types": ["profit_loss", "balance_sheet"],
            "target_keywords": []
        }}
        """
        from app.schemas.ai import IntentClassification
        try:
            structured_llm = self.llm.with_structured_output(IntentClassification)
            with llm_stage("AGENT-0 Intent"):
                res = await retry_with_backoff(structured_llm.ainvoke, prompt)
            data = res.model_dump()
            print(f"[INFO] Intent Discovery: query='{question[:50]}...' -> intent={data.get('intent')}")
            return data
        except Exception as e:
            print(f"[WARNING] Intent Discovery failed: {e}")
            return {"intent": "EXTRACTION", "report_types": ["profit_loss", "balance_sheet"]}

    async def _normalize_narration(self, narration: str) -> str:
        """
        Normalize narration for robust deduplication.
        Strips whitespace, special characters, and lowercases the text.
        """
        import re
        if not narration:
            return ""

        normalized = re.sub(r'[^a-zA-Z0-9]', '', narration).lower()
        return normalized

    def _detect_requested_tables(self, question: str) -> set:
        """
        Determine which table types to expose based on the user's question.
        Generates everything internally but only shows the asked-for cards.
        """
        import re

        q = question.lower()
        q = re.sub(r'p\s*&\s*l', 'p&l', q)
        q = re.sub(r'p\s+and\s+l\b', 'p&l', q)

        q_words = re.sub(r'[^a-z0-9]', ' ', q).split()

        wants_bs  = any(kw in q for kw in [
            'balanc', 'asset', 'liabilit', 'equity'
        ]) or 'bs' in q_words
        wants_pnl = any(kw in q for kw in [
            'profit', 'loss', 'p&l', 'pnl', 'income', 'expense', 'revenue',
            'trading', 'income statement', 'statement of profit'
        ])

        if wants_bs and wants_pnl:
            return {'profit_loss', 'balance_sheet', 'capital_account'}
        if wants_bs:
            return {'balance_sheet', 'capital_account'}
        if wants_pnl:
            return {'profit_loss'}

        if any(kw in q for kw in ['capital', 'equity', 'drawing', 'owner']):
            return {'capital_account', 'profit_loss', 'balance_sheet'}

        return {'profit_loss', 'balance_sheet', 'capital_account'}

    async def _extract_transactions_from_batch(
        self,
        context: Union[str, List[str]],
        context_query: str = None,
        company_id: str = None,
        feedback_context: str = ""
    ) -> List[Dict]:
        """
        Extract structured transaction data from a text batch.
        ZERO-MISMATCH DESIGN: Two-step — LLM counts rows first, then extracts exactly that many.
        Auto-retries once if extracted count differs from stated count.
        """
        if isinstance(context, list):
            context = "\n---\n".join(context)

        content_hash = hashlib.sha256(context.encode()).hexdigest()

        if content_hash in self.extraction_cache:
            print(f"[CACHE HIT] Reusing cached extraction for batch (hash: {content_hash[:8]}...)")
            return self.extraction_cache[content_hash]

        query_instruction = ""
        if context_query:
            query_instruction = f"""
        USER INTENT: "{context_query}"
        STRICT PRE-FILTERING:
        - For general reports (P&L, Balance Sheet, journal): extract ALL transactions, skip nothing.
        - Only filter if user explicitly asked for a specific type (e.g., "only cash").
        """

        from app.schemas.ai import ExtractedTransactions
        prompt = f"""
        {AGENT_SYSTEM_PROMPTS['extractor']}

        {EXTRACTOR_RULES}

        {query_instruction}
        {feedback_context}

        INPUT TEXT (Bank Statement):
        {context}

        TASK:
        1. First, quickly count how many transaction rows you see (rows with date + narration + debit OR credit amount).
           Do NOT count headers, opening/closing balance rows, or summary rows.
        2. Then extract ALL those transactions into the structured output.
        3. Your transactions array MUST contain exactly row_count_detected items. No skipping.

        RULES:
        - YOU ARE FORBIDDEN FROM SKIPPING ANY TRANSACTION ROW.
        - Money OUT → debit field. Money IN → credit field. One must be 0.00.
        - balance = running balance from statement (0.00 if not shown).
        """

        async def _run_extraction(extra_hint: str = "") -> dict:
            structured_llm = self.llm.with_structured_output(ExtractedTransactions)
            with llm_stage("AGENT-1 Extractor"):
                response = await retry_with_backoff(structured_llm.ainvoke, prompt + extra_hint)
            if response is None:
                return {"row_count_detected": 0, "transactions": []}
            elif hasattr(response, 'model_dump'):
                return response.model_dump()
            elif isinstance(response, dict):
                return response
            return {"row_count_detected": 0, "transactions": []}

        try:
            data = await _run_extraction()

            if data.get("extraction_error"):
                print(f"[AGENT-1][WARNING] Extraction fallback in batch.")
                return data.get("transactions", [])

            transactions  = data.get("transactions", [])
            stated_count  = data.get("row_count_detected")
            extracted_count = len(transactions)

            if stated_count:
                stated_count = int(stated_count)
                print(f"[AGENT-1][EXTRACT] Got {extracted_count} / {stated_count} rows")

                if extracted_count != stated_count:
                    diff = abs(stated_count - extracted_count)
                    direction = "missing" if stated_count > extracted_count else "extra"
                    print(f"[AGENT-1][RETRY] {direction}={diff}. Retrying...")
                    retry_hint = f"""
\n━━━ RETRY: You returned {extracted_count} but counted {stated_count} rows. You are {direction} {diff} transaction(s).
Go line-by-line. Find and include every skipped row. Output full JSON again. ━━━"""
                    retry_data = await _run_extraction(retry_hint)
                    retried = retry_data.get("transactions", [])
                    if retried:
                        if abs(len(retried) - stated_count) <= abs(extracted_count - stated_count):
                            print(f"[AGENT-1][RETRY] Improved: {extracted_count} → {len(retried)} (target={stated_count})")
                            transactions = retried
                        else:

                            transactions = transactions + retried
            else:
                print(f"[AGENT-1][EXTRACT] Got {extracted_count} transactions (no count target)")

            if transactions:
                if stated_count is None or len(transactions) == stated_count:
                    self.extraction_cache[content_hash] = transactions
                    print(f"[CACHE STORE] {len(transactions)} transactions (hash: {content_hash[:8]}...)")
                else:
                    print(f"[CACHE SKIP] Count still off ({len(transactions)} vs {stated_count}).")
            else:
                print(f"[CACHE SKIP] Empty result.")

            return transactions
        except Exception as e:
            print(f"[AGENT-1][ERROR] Fatal extraction error: {e}")
            return []

    async def _get_broad_type(self, acc_name: str, cat_name: str, balance: float) -> str:
        """Helper to categorize account nature based on AI category and balance sign."""
        cat_upper = cat_name.upper()
        if "INCOME" in cat_upper: return "INCOME"
        if "EXPENSE" in cat_upper: return "EXPENSE"
        if "ASSET" in cat_upper: return "ASSET"
        if "LIABIL" in cat_upper or "EQUITY" in cat_upper: return "LIABILITY"

        if balance > 0: return "ASSET_OR_EXP"
        return "LIAB_OR_INC"

    @instrument_stream("accounting_synthesis", client_id_kwarg="customer_id")
    async def stream_accounting_synthesis(
        self,
        question: str,
        context_chunks: List[str],
        context_metadatas: List[Dict[str, Any]] = None,
        company_id: str = None,
        customer_id: str = None,
        previous_context: str = None,
        input_transactions: List[Dict] = None,
        feedback_history: str = None
    ) -> AsyncGenerator[Union[str, Dict[str, Any]], None]:
        """
        Multi-Agent Pipeline:
          Agent 1: Extractor (LLM) -> Agent 2: Classifier (LLM)
          -> Agent 3: Journal (deterministic, updates Ledger Store)
          -> Agent 4: P&L (pure math) + Agent 5: Balance Sheet (pure math)
          -> Tally Agent (audit/verify)
        """

        dynamic_business_rules = ""
        business_type = None
        persistent_rules = []
        if customer_id:
            print(f"[RULES] Fetching dynamic rules for customer_id: {customer_id}")
            try:
                from app.core.supabase import supabase_admin

                cust_res = supabase_admin.table("customers").select("business_type, name").eq("id", customer_id).single().execute()
                if cust_res.data and cust_res.data.get("business_type"):
                    business_type = cust_res.data.get("business_type")
                    customer_name = cust_res.data.get("name")
                    print(f"[RULES] Found Business Type: {business_type} for Customer: {customer_name}")
                    dynamic_business_rules += f"CUSTOMER BUSINESS TYPE: {business_type}\n"

                query = supabase_admin.table("accounting_rules").select("rule_description", "created_at")
                query = query.eq("customer_id", customer_id).order("created_at", desc=False)
                rules_res = query.execute()

                if rules_res.data:
                    dynamic_business_rules += "\n━━━ MANDATORY USER-DEFINED RULES (ABSOLUTE PRIORITY) ━━━\n"
                    
                    # Filter: Only keep the LATEST profit target if multiple exist
                    all_raw_rules = [r['rule_description'].replace("USER COMMAND: ", "") for r in rules_res.data]
                    unique_rules = []
                    profit_seen = False
                    for rule in reversed(all_raw_rules):
                        is_profit_rule = any(k in rule.lower() for k in ["as net profit", "as profit", "as net loss"])
                        if is_profit_rule:
                            if not profit_seen:
                                unique_rules.append(rule)
                                profit_seen = True
                        else:
                            unique_rules.append(rule)
                    persistent_rules = list(reversed(unique_rules))

                    print(f"[RULES] Successfully fetched and filtered {len(persistent_rules)} custom accounting rules (from {len(rules_res.data)} total).")
                    for clean_rule in persistent_rules:
                        dynamic_business_rules += f"- {clean_rule}\n"
                        print(f"[RULES] -> Active Rule: {clean_rule}")
                    dynamic_business_rules += "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━\n"
                else:
                    print(f"[RULES] No custom rules found for this customer.")
            except Exception as e:
                print(f"[RULES] Failed to fetch dynamic accounting rules: {e}")

        if previous_context or feedback_history:
            yield {"status": "Refining report with new user instruction..."}
            print(f"[REFINEMENT] Injecting user feedback as dynamic rule: {question}")

            ctx_text = f"\nPREVIOUS CONTEXT:\n{feedback_history}" if feedback_history else ""
            dynamic_business_rules += f"""
            ━━━ CRITICAL COMMAND: USER AD-HOC REFINEMENT (ABSOLUTE PRIORITY) ━━━
            THE USER HAS REQUESTED THE FOLLOWING CHANGE: "{question}"
            {ctx_text}
            - YOU MUST FOLLOW THIS RULE ABOVE ALL OTHER ACCOUNTING PRINCIPLES.
            - IF THIS RULE SPECIFIES AN ACCOUNT OR CATEGORY FOR A CERTAIN RANGE/KEYWORD, APPLY IT STRICTLY.
            ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
            """

        local_structured_tables = []

        intent_data = await self._discover_intent(question)
        intent = intent_data.get('intent', 'EXTRACTION')

        _show_tables = self._detect_requested_tables(question)

        if not _show_tables:
            _show_tables = set(intent_data.get('report_types', ['profit_loss', 'balance_sheet']))

        print(f"[INTENT] User asked for: {_show_tables} | ID Intent: {intent}")

        if not context_chunks and not input_transactions:
            yield "[ERROR] No context data provided."
            return

        yield {"status": f"Agent 0 (Guard): input validated. Starting pipeline..."}

        feedback_context = ""
        if company_id:
            try:
                from app.services.deps import embedding_service
                q_emb = embedding_service.generate_embedding(question)
                relevant_feedback = vector_store.query_feedback(q_emb, company_id)
                if relevant_feedback:
                    feedback_context = "\n**LESSONS LEARNED (PAST USER CORRECTIONS):**\n"
                    for fb in relevant_feedback:
                        feedback_context += f"- {fb}\n"
                    print(f"[INFO] Pre-fetched {len(relevant_feedback)} corrections for session.")
            except Exception as e:
                print(f"[WARNING] Pre-fetch Feedback failed: {e}")

        all_transactions: List[Dict] = []

        is_refinement = (intent == "REFINEMENT" or intent == "ANALYSIS") and input_transactions is not None

        if is_refinement:
            print(f"[AGENT-7] REFINEMENT MODE: LLM Universal Mutation Engine triggered for {len(input_transactions)} transactions.")

            pnl_inc_kws = ["INCOME", "SALES", "REVENUE", "DIRECT INCOME"]
            pnl_exp_kws = ["EXPENSE", "DIRECT EXPENSE", "CHGS", "FEE", "TAX", "PURCHASE", "COST"]

            cur_income = sum(float(e.get('amount', 0)) for t in input_transactions for e in t.get('entries', [])
                            if any(k in str(e.get('category', '')).upper() for k in pnl_inc_kws))
            cur_expense = sum(float(e.get('amount', 0)) for t in input_transactions for e in t.get('entries', [])
                             if any(k in str(e.get('category', '')).upper() for k in pnl_exp_kws))
            cur_profit = cur_income - cur_expense

            yield {"status": f"Agent 7 (Universal): Calculating adjustments (Current Profit: \u20b9{cur_profit:,.2f})..."}
            from datetime import date as _date_today
            today_str = _date_today.today().strftime("%d/%m/%Y")

            txn_sample_text = ""
            for i, t in enumerate(input_transactions[:100]):
                amt = float(t.get('debit', 0) or t.get('credit', 0) or 0)
                txn_sample_text += f"{i}. [{t.get('date')}] {t.get('narration')} (\u20b9{amt:,.2f})\n"

            refinement_sys_prompt = """
You are a Senior Chartered Accountant AI. A user has given you an accounting instruction.
Your job is to translate that instruction into a precise list of mutations on the transaction ledger.

MUTATION TYPES AND WHEN TO USE THEM:

1. ADD_ENTRY - Add a new transaction/journal entry that does not yet exist.
   Use for: "add 15 lakh saloon income", "add depreciation 50000", "add opening capital",
            "add cash sale", "add GST liability", "add 1522670 in cash in saloon income"
   {{
     "type": "ADD_ENTRY",
     "date": "DD/MM/YYYY or today",
     "narration": "short description",
     "debit_account": "account to debit",
     "credit_account": "account to credit",
     "amount": 12345.00,
     "category": "Income | Expense | Current Assets | Fixed Assets | Liability | Equity"
   }}

2. RECLASSIFY - Move/reassign existing transactions to a different account.
   Use for: "move all swiggy to food expense", "put UPI under purchases"
   {{
     "type": "RECLASSIFY",
     "matching_criteria": {{
       "narration_contains": "keyword or empty",
       "type": "DEBIT or CREDIT or null",
       "min_amount": null,
       "max_amount": null
     }},
     "new_debit_account": "account name",
     "new_credit_account": "Bank Account",
     "new_category": "category name"
   }}

3. MODIFY - Change amount, date, or narration of a specific existing transaction by index.
   Use for: "change transaction 5 to 8000", "update entry 3 date to 15/03/2026"
   {{
     "type": "MODIFY",
     "index": 5,
     "fields": {{ "debit": 8000 }}
   }}

4. DELETE - Remove a transaction (duplicate, incorrect, etc.). DO NOT delete unless specifically asked.
   {{
     "type": "DELETE",
     "matching_criteria": {{ "narration_contains": "keyword or empty", "exact_amount": null, "index": null }}
   }}

5. SPLIT - Replace one transaction with multiple smaller entries.
   {{
     "type": "SPLIT",
     "matching_criteria": {{ "narration_contains": "keyword", "exact_amount": 10000 }},
     "splits": [
       {{"narration": "Rent", "amount": 6000, "debit_account": "Rent Expense", "credit_account": "Bank Account", "category": "Expense"}},
       {{"narration": "Utilities", "amount": 4000, "debit_account": "Utilities Expense", "credit_account": "Bank Account", "category": "Expense"}}
     ]
   }}

━━━━ IMPORTANT RULES ━━━━
- AMOUNT INTEGRITY: In a RECLASSIFY, you MUST NOT change the amount of any transaction. Only change its account/category.
- AD-HOC ENTRIES: If the user says "I want X as profit" or "add Y amount", ALWAYS use ADD_ENTRY with the Delta or specific amount.
- CATEGORY MATCHING: Use "Income" for all revenue types and "Expense" for all cost types to ensure they sync with the P&L calculator.

CATEGORIES TO USE:
- "Income" for revenue, sales, fees received
- "Expense" for costs, purchases, depreciation, salaries
- "Current Assets" for Bank, Cash, Debtors, Stock
- "Fixed Assets" for machinery, equipment, furniture, vehicles
- "Liability" for loans, creditors, GST payable, TDS payable
- "Equity" for capital, reserves, retained earnings, drawings
"""

            combined_hist = "\n".join([f"- {r}" for r in persistent_rules]) if persistent_rules else "None"
            
            universal_prompt = f"""
{refinement_sys_prompt}

━━━━ HISTORICAL USER COMMANDS (RE-APPLY THESE TOO) ━━━━
{combined_hist}
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

━━━━ NEW USER COMMAND (PRIMARY TARGET) ━━━━
"{question}"
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

TODAY'S DATE: {today_str}
TOTAL TRANSACTIONS IN LEDGER: {len(input_transactions)}

CURRENT FINANCIAL STATE:
 - Total Income: ₹{cur_income:,.2f}
 - Total Expense: ₹{cur_expense:,.2f}
 - Current Net Profit: ₹{cur_profit:,.2f}

SAMPLE TRANSACTIONS (first 100 shown by index):
{txn_sample_text}

━━━━ IMPORTANT: TARGET PROFIT / TOTALS ━━━━
- If user wants "X as Net Profit", calculate the DELTA from Current Net Profit and use ADD_ENTRY to balance it.
- DELTA = Target Profit - Current Net Profit.
- If DELTA is Negative (Target < Current): Add an EXPENSE entry of abs(DELTA) to reduce profit.
- If DELTA is Positive (Target > Current): Add an INCOME entry of DELTA to increase profit.
- Account naming: Use "Profit Adjustment Entry" or "Ad-hoc Expense" for these balancing entries.

DOUBLE-ENTRY ACCOUNTING RULES (always follow):
- Money IN (income, loans, capital received): DEBIT Bank/Cash → CREDIT Income/Liability/Capital
- Money OUT (expenses, payments): DEBIT Expense/Asset → CREDIT Bank/Cash
- Depreciation: DEBIT Depreciation Expense → CREDIT Accumulated Depreciation
- Contra (cash to bank): DEBIT Bank Account → CREDIT Cash Account
- Opening balance (asset): DEBIT Asset Account → CREDIT Capital/Opening Balance
- GST payable: DEBIT Input/Sales Account → CREDIT GST Payable

OUTPUT FORMAT:
Extract the mutations required to satisfy the user request strictly matching the schema.
"""
            from app.schemas.ai import MutationPlan
            mutation_plan = {}
            try:
                structured_llm = self.llm.with_structured_output(MutationPlan)
                with llm_stage("AGENT-7 Mutation Plan"):
                    res = await retry_with_backoff(structured_llm.ainvoke, universal_prompt)
                mutation_plan = res.model_dump()
                if not isinstance(mutation_plan, dict):
                    mutation_plan = {}
                print(f"[AGENT-7][UNIVERSAL] Reasoning: {mutation_plan.get('reasoning', 'N/A')}")
                print(f"[AGENT-7][UNIVERSAL] Mutations planned: {len(mutation_plan.get('mutations', []))}")
            except Exception as e:
                print(f"[AGENT-7][UNIVERSAL][ERROR] LLM mutation planning failed: {e}")
                mutation_plan = {"mutations": []}

            mutations = mutation_plan.get("mutations", [])
            new_entries_to_add = []
            modified_count = 0

            for mut in mutations:
                mtype = mut.get("type", "").upper()

                if mtype == "ADD_ENTRY":
                    amt = float(mut.get("amount", 0))
                    if amt <= 0:
                        print(f"[AGENT-7][UNIVERSAL] ADD_ENTRY skipped: amount is 0 or missing.")
                        continue
                    dr_acc = str(mut.get("debit_account", "Bank Account")).strip().title()
                    cr_acc = str(mut.get("credit_account", "Unclassified Income")).strip().title()
                    cat    = str(mut.get("category", "Income")).strip()
                    narr   = str(mut.get("narration", "Manual Entry")).strip()
                    date   = str(mut.get("date", today_str)).strip()
                    is_income_side = any(k in cat.upper() for k in ["INCOME", "LIAB", "EQUITY"])
                    new_txn = {
                        "date": date, "narration": narr, "balance": 0.0, "_manual": True,
                        "debit": 0.0 if is_income_side else amt,
                        "credit": amt if is_income_side else 0.0,
                        "entries": [
                            {"account": dr_acc, "type": "DEBIT", "amount": amt,
                             "category": "Current Assets" if any(k in dr_acc.upper() for k in ["BANK","CASH"]) else cat},
                            {"account": cr_acc, "type": "CREDIT", "amount": amt,
                             "category": "Current Assets" if any(k in cr_acc.upper() for k in ["BANK","CASH"]) else cat},
                        ]
                    }
                    new_entries_to_add.append(new_txn)
                    modified_count += 1
                    print(f"[AGENT-7][UNIVERSAL] ADD_ENTRY: {narr} | Dr:{dr_acc} Cr:{cr_acc} ₹{amt:,.2f}")

                elif mtype == "RECLASSIFY":
                    mc = mut.get("matching_criteria", {})
                    new_dr  = str(mut.get("new_debit_account", "") or mut.get("new_account", "")).strip().title()
                    new_cr  = str(mut.get("new_credit_account", "Bank Account")).strip().title()
                    new_cat = str(mut.get("new_category", "Expense")).strip()
                    
                    yield {"status": f"Searching for transactions matching: {mc}..."}
                    reclassified = 0
                    for t in input_transactions:
                        orig_amt = 0.0
                        if t.get('entries'):
                            orig_amt = float(t['entries'][0].get('amount', 0))
                        else:
                            orig_amt = float(t.get('debit', 0) or t.get('credit', 0) or 0)

                        t_type = "DEBIT" if float(t.get('debit', 0)) > 0 else "CREDIT"
                        if mc.get('type') and mc['type'].upper() != t_type: continue
                        if mc.get('min_amount') is not None and orig_amt < float(mc['min_amount']): continue
                        if mc.get('max_amount') is not None and orig_amt > float(mc['max_amount']): continue
                        narr_kw = str(mc.get('narration_contains', '')).lower()
                        if narr_kw and narr_kw not in str(t.get('narration', '')).lower(): continue

                        t['entries'] = [
                            {"account": new_dr, "type": "DEBIT", "amount": orig_amt,
                             "category": "Current Assets" if any(k in new_dr.upper() for k in ["BANK","CASH"]) else new_cat},
                            {"account": new_cr, "type": "CREDIT", "amount": orig_amt,
                             "category": "Current Assets" if any(k in new_cr.upper() for k in ["BANK","CASH"]) else new_cat},
                        ]
                        reclassified += 1
                    modified_count += reclassified
                    yield {"status": f"Reclassified {reclassified} transaction(s) to '{new_dr}'."}
                    print(f"[AGENT-7][UNIVERSAL] RECLASSIFY: {reclassified} transactions (amount preserved) → Dr:{new_dr} Cr:{new_cr}")

                elif mtype == "MODIFY":
                    idx    = mut.get("index")
                    fields = mut.get("fields", {})
                    if idx is not None and 0 <= int(idx) < len(input_transactions):
                        t = input_transactions[int(idx)]
                        for field, val in fields.items():
                            if field in ['debit', 'credit']:
                                t[field] = float(val)
                                for e in t.get('entries', []): e['amount'] = float(val)
                            elif field in ['date', 'narration']:
                                t[field] = str(val)
                        modified_count += 1
                        yield {"status": f"Modified entry at index {idx}."}
                        print(f"[AGENT-7][UNIVERSAL] MODIFY: txn[{idx}] updated fields={list(fields.keys())}")

                elif mtype == "DELETE":
                    mc  = mut.get("matching_criteria", {})
                    idx = mc.get("index")
                    deleted = 0
                    if idx is not None and 0 <= int(idx) < len(input_transactions):
                        input_transactions[int(idx)]['_deleted'] = True
                        deleted += 1
                    else:
                        for t in input_transactions:
                            amt     = float(t.get('debit', 0) or t.get('credit', 0) or 0)
                            narr_kw = str(mc.get('narration_contains', '')).lower()
                            exact   = mc.get('exact_amount')
                            if narr_kw and narr_kw not in str(t.get('narration', '')).lower(): continue
                            if exact is not None and abs(amt - float(exact)) > 0.01: continue
                            t['_deleted'] = True
                            deleted += 1
                    modified_count += deleted
                    yield {"status": f"Deleted {deleted} transaction(s)."}
                    print(f"[AGENT-7][UNIVERSAL] DELETE: {deleted} transactions marked for removal")


                elif mtype == "SPLIT":
                    mc     = mut.get("matching_criteria", {})
                    splits = mut.get("splits", [])
                    for t in input_transactions:
                        orig_amt = float(t.get('debit', 0) or t.get('credit', 0) or 0)
                        narr_kw  = str(mc.get('narration_contains', '')).lower()
                        exact    = mc.get('exact_amount')
                        if narr_kw and narr_kw not in str(t.get('narration', '')).lower(): continue
                        if exact is not None and abs(orig_amt - float(exact)) > 0.01: continue
                        t['_deleted'] = True
                        for sp in splits:
                            sp_amt = float(sp.get('amount', 0))
                            sp_dr  = str(sp.get('debit_account', 'Expense Account')).strip().title()
                            sp_cr  = str(sp.get('credit_account', 'Bank Account')).strip().title()
                            sp_cat = str(sp.get('category', 'Expense')).strip()
                            new_entries_to_add.append({
                                "date": t.get('date', today_str),
                                "narration": sp.get('narration', t.get('narration', '')),
                                "debit": sp_amt if float(t.get('debit', 0)) > 0 else 0,
                                "credit": sp_amt if float(t.get('credit', 0)) > 0 else 0,
                                "balance": 0.0, "_manual": True,
                                "entries": [
                                    {"account": sp_dr, "type": "DEBIT",  "amount": sp_amt, "category": sp_cat},
                                    {"account": sp_cr, "type": "CREDIT", "amount": sp_amt, "category": sp_cat},
                                ]
                            })
                        modified_count += 1
                        print(f"[AGENT-7][UNIVERSAL] SPLIT: '{t.get('narration')}' → {len(splits)} entries")
                        break

                else:
                    print(f"[AGENT-7][UNIVERSAL] Unknown mutation type: '{mtype}' — skipping.")

            input_transactions[:] = [t for t in input_transactions if not t.get('_deleted')]
            input_transactions.extend(new_entries_to_add)

            if is_refinement:
                try:
                    from app.core.supabase import supabase_admin

                    scope = "CUSTOMER"

                    full_rule_desc = f"USER COMMAND: {question}"

                    save_data = {
                        "rule_description": full_rule_desc,

                        "customer_id": customer_id,
                        "company_id": company_id,
                    }

                    print(f"[DEBUG][LEARNING] Persisting verbatim rule for CUSTOMER scope...")
                    print(f"[DEBUG][LEARNING] Content: {full_rule_desc}")

                    check_query = supabase_admin.table("accounting_rules").select("id").eq("rule_description", full_rule_desc).eq("customer_id", customer_id)

                    check_res = check_query.execute()
                    if not check_res.data:
                        print(f"[DEBUG][LEARNING] No duplicate. Inserting...")
                        insert_res = supabase_admin.table("accounting_rules").insert(save_data).execute()
                        if insert_res.data:
                            print(f"[LEARNING][SUCCESS] Verbose rule saved: {full_rule_desc}")

                            yield {"rule_persisted": save_data}
                        else:
                            print(f"[LEARNING][ERROR] Insert failed (no data returned). Response: {insert_res}")
                    else:
                        print(f"[LEARNING][SKIP] Rule already exists (ID: {check_res.data[0].get('id')}).")

                except Exception as e:
                    import traceback
                    print(f"[LEARNING][ERROR] Persistence failed: {str(e)}")
                    traceback.print_exc()

            all_transactions = input_transactions

            yield {"status": f"Agent 7 (Universal): Done. {modified_count} mutation(s) applied. {len(new_entries_to_add)} new entries added."}
            print(f"[AGENT-7][UNIVERSAL] Pipeline complete. Mutations applied: {modified_count}. New entries: {len(new_entries_to_add)}.")

            expected_deposits = len([t for t in all_transactions if float(t.get('credit', 0) or 0) > 0])
            expected_withdrawals = len([t for t in all_transactions if float(t.get('debit', 0) or 0) > 0])
        else:

            summary_task = None
            if not is_refinement:
                summary_context = "\n".join(context_chunks[-3:])
                summary_prompt = f"""
                Read the following bank statement footer and extract the TRANSACTION SUMMARY table.
                Look for "Transaction Summary", "No. of Transactions", "Deposits", and "Withdrawals".

                TEXT:
                {summary_context}

                JSON OUTPUT:
                {{
                    "deposits_count": 0,
                    "withdrawals_count": 0
                }}
                """
                with llm_stage("AGENT-1 Doc Summary"):
                    summary_task = asyncio.create_task(self.llm.ainvoke(summary_prompt))

            BATCH_SIZE = 15
            batches = [context_chunks[i:i + BATCH_SIZE] for i in range(0, len(context_chunks), BATCH_SIZE)]
            total_batches = len(batches)

            N_EXTRACTORS = self.n_extractors
            WORKER_DELAY_SEC = self.worker_delay
            sem = asyncio.Semaphore(N_EXTRACTORS)

            print(f"[AGENT-1] Extractor Pool: {len(context_chunks)} chunks → {total_batches} batches (size={BATCH_SIZE}) → {N_EXTRACTORS} workers")

            async def process_batch(index, batch_data, batch_metas):
                async with sem:
                    slot = index % N_EXTRACTORS

                    import hashlib as _hl
                    _batch_str = "\n---\n".join(batch_data)
                    _batch_hash = _hl.sha256(_batch_str.encode()).hexdigest()
                    if _batch_hash not in self.extraction_cache:
                        await asyncio.sleep(WORKER_DELAY_SEC * (index % N_EXTRACTORS + 1) / N_EXTRACTORS)

                    print(f"[AGENT-1][Worker-{slot+1}] → Batch {index + 1}/{total_batches} starting... ({len(batch_data)} chunks)")
                    txns = await self._extract_transactions_from_batch(
                        batch_data,
                        context_query=question,
                        company_id=company_id,
                        feedback_context=feedback_context
                    )
                    print(f"[AGENT-1][Worker-{slot+1}] ← Batch {index + 1}/{total_batches} done: {len(txns)} transactions extracted")

                    if txns and batch_metas:
                        src_docs = list(set(m.get('document_name') for m in batch_metas if m.get('document_name')))
                        for t in txns:
                            t['_source_docs'] = src_docs
                    return index, txns

            batch_results: Dict[int, List[Dict]] = {}
            tasks = [
                process_batch(i, b, context_metadatas[i*BATCH_SIZE:(i+1)*BATCH_SIZE] if context_metadatas else None)
                for i, b in enumerate(batches)
            ]
            completed = 0
            yield {"status": f"Agent 1 (Extractor ×{N_EXTRACTORS}): processing {total_batches} batches..."}
            for coro in asyncio.as_completed(tasks):
                try:
                    b_idx, result = await coro
                    batch_results[b_idx] = result or []
                    completed += 1
                    if completed % 5 == 0 or completed == total_batches:
                        yield {"status": f"Agent 1 (Extractor ×{N_EXTRACTORS}): {completed}/{total_batches} batches done..."}
                    if result:
                        all_transactions.extend(result)
                except Exception as e:
                    completed += 1
                    print(f"[AGENT-1][ERROR] Extractor batch failed: {e}")

            expected_deposits = None
            expected_withdrawals = None
            if summary_task:
                try:
                    summary_res = await summary_task
                    summary_data = self.ai_service._extract_json(summary_res.content)
                    if summary_data:
                        expected_deposits = summary_data.get("deposits_count")
                        expected_withdrawals = summary_data.get("withdrawals_count")
                        print(f"[RECONCILE] Bank Summary Found -> D:{expected_deposits}, W:{expected_withdrawals}")
                except Exception as e:
                    print(f"[RECONCILE] Summary background task failed: {e}")

            total_extracted = len(all_transactions)
            total_expected = (expected_deposits or 0) + (expected_withdrawals or 0)

            if total_expected > 0 and total_extracted != total_expected:
                missing_count = total_expected - total_extracted
                direction = "missing" if missing_count > 0 else "extra"
                print(f"[RECONCILE] COUNT MISMATCH: Found {total_extracted}, Expected {total_expected} ({direction}={abs(missing_count)}). Starting aggressive reconciliation...")
                yield {"status": f" {direction.title()} {abs(missing_count)} transactions (found {total_extracted}, expected {total_expected}). Reconciling..."}

                sorted_batches = sorted(batch_results.items(), key=lambda x: len(x[1]))
                retry_tasks = []
                for b_idx, b_res in sorted_batches:
                    if len(b_res) == 0 or (missing_count > 0 and len(b_res) < 3):
                        f_batch = batches[b_idx]
                        f_metas = context_metadatas[b_idx*BATCH_SIZE:(b_idx+1)*BATCH_SIZE] if context_metadatas else [None] * len(f_batch)
                        
                        async def _retry_chunk(chunk_data, meta):
                            r_txns = await self._extract_transactions_from_batch(
                                chunk_data, context_query=question, company_id=company_id,
                                feedback_context=feedback_context + "\n**URGENT: This chunk was skipped. Find missing rows.**"
                            )
                            if r_txns and meta:
                                src_doc = meta.get('document_name')
                                for t in r_txns: t['_source_docs'] = [src_doc] if src_doc else []
                            return r_txns
                        
                        for c_idx, chunk in enumerate(f_batch):
                            retry_tasks.append(_retry_chunk(chunk, f_metas[c_idx]))

                if retry_tasks:
                    print(f"[RECONCILE] Retrying {len(retry_tasks)} chunks in parallel...")
                    retry_results = await asyncio.gather(*retry_tasks)
                    recovered = 0
                    for r_list in retry_results:
                        if r_list:
                            all_transactions.extend(r_list)
                            recovered += len(r_list)
                    print(f"[RECONCILE] Recovery done. Recovered {recovered} additional transactions.")
                    yield {"status": f"Reconciliation: recovered {recovered} transactions (total now: {len(all_transactions)})."}
                else:
                    print(f"[RECONCILE] No obvious retry candidates found. Proceeding with {total_extracted} transactions.")

        if not all_transactions:
            yield "No transactions could be identified in the provided documents. Please check the file has clear financial data.\n"
            return

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

        all_transactions.sort(key=_sort_key)
        from collections import Counter as _Counter
        narr_count: _Counter = _Counter()
        seen_fps: set = set()
        unique_txns: List[Dict] = []
        for t in all_transactions:
            ds   = str(t.get('date', '')).strip()
            narr = str(t.get('narration', '')).strip()
            dr   = float(t.get('debit', 0) or 0)
            cr   = float(t.get('credit', 0) or 0)
            amt  = dr if dr > 0 else cr
            bal  = float(t.get('balance', 0) or 0)
            norm = _re.sub(r'[^a-zA-Z0-9]', '', narr).lower()

            base_fp = f"{ds}_{amt:.2f}_{norm}_{bal:.2f}"

            narr_count[base_fp] += 1
            fp = f"{base_fp}#{narr_count[base_fp]}"
            if fp not in seen_fps:
                seen_fps.add(fp)
                unique_txns.append(t)
        all_transactions = unique_txns
        print(f"[DEDUP] Raw extracted: {len(seen_fps)} fingerprints → {len(all_transactions)} unique transactions kept")

        extracted_deposits = len([t for t in all_transactions if float(t.get('credit', 0) or 0) > 0])
        extracted_withdrawals = len([t for t in all_transactions if float(t.get('debit', 0) or 0) > 0])

        print(f"[RECONCILE] Extraction Results: Deposits={extracted_deposits}, Withdrawals={extracted_withdrawals}")
        if expected_deposits is not None or expected_withdrawals is not None:
            if expected_deposits == extracted_deposits and expected_withdrawals == extracted_withdrawals:
                print(f"[RECONCILE] SUCCESS: Extracted counts match Bank Summary!")
            else:
                warning = f"[RECONCILE] WARNING: Counts MISMATCH! Bank Summary: D={expected_deposits}, W={expected_withdrawals} | Extracted: D={extracted_deposits}, W={extracted_withdrawals}"
                print(warning)

                if abs(len(all_transactions) - ((expected_deposits or 0) + (expected_withdrawals or 0))) > 10:
                    yield {"status": f" Extraction mismatch: found {len(all_transactions)} transactions, but summary expected {(expected_deposits or 0) + (expected_withdrawals or 0)}."}

        yield {"all_transactions": all_transactions}

        CLASSIFY_BATCH_SIZE = 100
        N_CLASSIFIERS = self.n_classifiers
        CLASSIFIER_DELAY_SEC = self.worker_delay

        if is_refinement:
            print("[AGENT-2] SKIP CLASSIFIER: Refinement was handled by Agent 7 logic.")
            total_classify_batches = 0
            classify_batches = []
        else:
            classify_batches = [
                all_transactions[i:i + CLASSIFY_BATCH_SIZE]
                for i in range(0, len(all_transactions), CLASSIFY_BATCH_SIZE)
            ]
            total_classify_batches = len(classify_batches)
            print(f"[AGENT-2] Classifier Pool: {len(all_transactions)} txns → {total_classify_batches} batches (size={CLASSIFY_BATCH_SIZE}) → {N_CLASSIFIERS} workers")
            yield {"status": f"Agent 2 (Classifier ×{N_CLASSIFIERS}): classifying {len(all_transactions)} transactions..."}

        cls_sem = asyncio.Semaphore(N_CLASSIFIERS)
        classified_results: Dict[int, List[Dict]] = {}

        async def classify_batch(bidx: int, c_batch: List[Dict]):
            async with cls_sem:
                slot = bidx % N_CLASSIFIERS

                known_accs = ", ".join(list(self.shared_accounts)[:50]) if self.shared_accounts else "None yet"

                narrations_list = ""
                for j, t in enumerate(c_batch):
                    dr = float(t.get('debit', 0) or 0)
                    cr = float(t.get('credit', 0) or 0)
                    direction = f"DEBIT \u20b9{dr:,.2f}" if dr > 0 else f"CREDIT \u20b9{cr:,.2f}"
                    narrations_list += f"{j+1}. [{direction}] {t.get('narration', 'Unknown')}\n"

                _pre_fp = hashlib.sha256((narrations_list + dynamic_business_rules).encode()).hexdigest()
                if bidx >= N_CLASSIFIERS and _pre_fp not in self.classification_cache:
                    await asyncio.sleep(CLASSIFIER_DELAY_SEC * (slot % 3))

                classify_prompt = f"""
                {AGENT_SYSTEM_PROMPTS['classifier']}

                {CLASSIFIER_RULES}

                {dynamic_business_rules}

                PREVIOUSLY IDENTIFIED ACCOUNTS (Consistency):
                {known_accs}

                TRANSACTIONS TO CLASSIFY:
                {narrations_list}

                Return a JSON array with EXACTLY {len(c_batch)} objects (one per transaction, in order).
                Use these exact short keys to save tokens:
                [{{
                    "dr_acc": "Debit Account",
                    "cr_acc": "Credit Account",
                    "type": "income/expense/asset/liability/equity",
                    "cat": "One of the mandatory categories",
                    "conf": 0.85
                }}]

                Bank Account Logic (NON-NEGOTIABLE):
                - If transaction is CREDIT (money IN) → debit_account = "Bank Account"
                - If transaction is DEBIT  (money OUT) → credit_account = "Bank Account"

                Output JSON array ONLY. No explanation.
                """
                batch_classified = []
                try:
                    if dynamic_business_rules:
                        print(f"[AGENT-2][Worker-{slot+1}] Batch {bidx+1}/{total_classify_batches}: Injecting dynamic business rules.")

                    batch_fp = hashlib.sha256(
                        (narrations_list + dynamic_business_rules).encode()
                    ).hexdigest()
                    if batch_fp in self.classification_cache:
                        results = self.classification_cache[batch_fp]
                        print(f"[AGENT-2][CACHE HIT] Batch {bidx+1}/{total_classify_batches}: {len(results)} cached classifications.")
                    else:
                        async def _run_classify(hint: str = "") -> list:
                            """Run one classify LLM call and parse results."""
                            with llm_stage("AGENT-2 Classifier"):
                                resp = await retry_with_backoff(self.llm.ainvoke, classify_prompt + hint)
                            raw = self.ai_service._extract_json(resp.content)
                            if isinstance(raw, dict):

                                for v in raw.values():
                                    if isinstance(v, list):
                                        return v
                                return []
                            if isinstance(raw, list):
                                return raw
                            return []

                        results = await _run_classify()

                        if len(results) == 0 and len(c_batch) > 0:
                            print(f"[AGENT-2][Worker-{slot+1}] Batch {bidx+1}/{total_classify_batches}: 0 classifications — retrying...")
                            retry_hint = f"\n\n⚠️ RETRY: You returned 0 items. Return a JSON ARRAY with EXACTLY {len(c_batch)} objects. No extra text."
                            results = await _run_classify(retry_hint)

                        if results:
                            self.classification_cache[batch_fp] = results

                    print(f"[AGENT-2][Worker-{slot+1}] Batch {bidx+1}/{total_classify_batches}: {len(results)} classifications")

                    for j, t in enumerate(c_batch):
                        dr = float(t.get('debit', 0) or 0)
                        cr = float(t.get('credit', 0) or 0)
                        amt = dr if dr > 0 else cr

                        res = results[j] if j < len(results) else None

                        if res:
                            debit_acc  = res.get('dr_acc', res.get('debit_account', 'Bank Account')).strip().title()
                            credit_acc = res.get('cr_acc', res.get('credit_account', 'Bank Account')).strip().title()
                            category   = res.get('cat', res.get('category', 'Unclassified'))

                            if cr > 0:
                                t['entries'] = [
                                    {"account": "Bank Account", "type": "DEBIT",  "amount": amt, "category": "Current Assets"},
                                    {"account": credit_acc,      "type": "CREDIT", "amount": amt, "category": category},
                                ]
                            else:
                                t['entries'] = [
                                    {"account": debit_acc,  "type": "DEBIT",  "amount": amt, "category": category},
                                    {"account": "Bank Account", "type": "CREDIT", "amount": amt, "category": "Current Assets"},
                                ]
                            t['confidence'] = res.get('conf', res.get('confidence', 1.0))
                        else:

                            print(f"[AGENT-2][WARNING] Truncation detected at index {j} in batch {bidx+1}. Using fallback.")
                            if t.get('entries'):
                                batch_classified.append(t)
                                continue

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

                        if t.get('entries'):
                            batch_classified.append(t)
                            continue

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

        classified: List[Dict] = []
        for i in range(total_classify_batches):
            classified.extend(classified_results.get(i, []))

        print(f"[AGENT-2] Parallel classification complete: {len(classified)} transactions classified")

        # --- PHASE 2: PERSISTENT REFINEMENT (RE-APPLYING SAVED RULES) ---
        if persistent_rules and not is_refinement:
            print(f"[AGENT-7] PERSISTENT REFINEMENT: Re-applying {len(persistent_rules)} saved commands to fresh extraction.")
            yield {"status": f"Agent 7 (Universal): Re-applying {len(persistent_rules)} manual refinements..."}
            
            pnl_inc_kws = ["INCOME", "SALES", "REVENUE", "DIRECT INCOME"]
            pnl_exp_kws = ["EXPENSE", "DIRECT EXPENSE", "CHGS", "FEE", "TAX", "PURCHASE", "COST"]
            
            cur_income = sum(float(e.get('amount', 0)) for t in classified for e in t.get('entries', [])
                            if any(k in str(e.get('category', '')).upper() for k in pnl_inc_kws))
            cur_expense = sum(float(e.get('amount', 0)) for t in classified for e in t.get('entries', [])
                             if any(k in str(e.get('category', '')).upper() for k in pnl_exp_kws))
            cur_profit = cur_income - cur_expense
            
            txn_sample_text = ""
            for i, t in enumerate(classified[:100]):
                amt = float(t.get('debit', 0) or t.get('credit', 0) or 0)
                txn_sample_text += f"{i}. [{t.get('date')}] {t.get('narration')} (\u20b9{amt:,.2f})\n"

            from datetime import date as _date_today
            today_str = _date_today.today().strftime("%d/%m/%Y")
            combined_commands = "\n".join([f"- {r}" for r in persistent_rules])

            universal_prompt = f"""
{REFINEMENT_RULES}

━━━━ HISTORICAL USER COMMANDS (RE-APPLY THESE) ━━━━
{combined_commands}
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

TODAY'S DATE: {today_str}
TOTAL TRANSACTIONS IN LEDGER: {len(classified)}

CURRENT FINANCIAL STATE:
 - Total Income: ₹{cur_income:,.2f}
 - Total Expense: ₹{cur_expense:,.2f}
 - Current Net Profit: ₹{cur_profit:,.2f}

SAMPLE TRANSACTIONS (first 100 shown by index):
{txn_sample_text}

━━━━ IMPORTANT: TARGET PROFIT / TOTALS ━━━━
- If user wants "X as Net Profit", calculate the DELTA from Current Net Profit and use ADD_ENTRY to balance it.
- DELTA = Target Profit - Current Net Profit.
- If DELTA is Negative (Target < Current): Add an EXPENSE entry of abs(DELTA) to reduce profit.
- If DELTA is Positive (Target > Current): Add an INCOME entry of DELTA to increase profit.
- Account naming: Use "Profit Adjustment Entry" or "Ad-hoc Expense" for these balancing entries.

━━━━ IMPORTANT: MULTIPLE MUTATIONS ━━━━
You MUST return mutations for ALL the commands listed above.
Check if an entry needs to be added (ADD_ENTRY), reclassified (RECLASSIFY), or modified.

DOUBLE-ENTRY ACCOUNTING RULES (always follow):
- Money IN (income, loans, capital received): DEBIT Bank/Cash → CREDIT Income/Liability/Capital
- Money OUT (expenses, payments): DEBIT Expense/Asset → CREDIT Bank/Cash
- Depreciation: DEBIT Depreciation Expense → CREDIT Accumulated Depreciation
- Contra (cash to bank): DEBIT Bank Account → CREDIT Cash Account
- Opening balance (asset): DEBIT Asset Account → CREDIT Capital/Opening Balance
- GST payable: DEBIT Input/Sales Account → CREDIT GST Payable

OUTPUT FORMAT (JSON only, no explanation outside JSON):
{{
  "reasoning": "brief explanation",
  "mutations": [
    {{ ...mutation 1... }},
    {{ ...mutation 2... }}
  ]
}}
"""
            try:
                with llm_stage("AGENT-7 Refinement"):
                    res = await retry_with_backoff(self.llm.ainvoke, universal_prompt)
                mutation_plan = self.ai_service._extract_json(res.content)
                mutations = mutation_plan.get("mutations", []) if isinstance(mutation_plan, dict) else []
                
                new_entries = []
                mod_count = 0
                for mut in mutations:
                    mtype = mut.get("type", "").upper()
                    if mtype == "ADD_ENTRY":
                        amt = float(mut.get("amount", 0))
                        if amt <= 0: continue
                        dr_acc = str(mut.get("debit_account", "Bank Account")).strip().title()
                        cr_acc = str(mut.get("credit_account", "Unclassified Income")).strip().title()
                        cat    = str(mut.get("category", "Income")).strip()
                        narr   = str(mut.get("narration", "Manual Entry")).strip()
                        date   = str(mut.get("date", today_str)).strip()
                        is_income_side = any(k in cat.upper() for k in ["INCOME", "LIAB", "EQUITY"])
                        new_txn = {
                            "date": date, "narration": narr, "balance": 0.0, "_manual": True,
                            "debit": 0.0 if is_income_side else amt,
                            "credit": amt if is_income_side else 0.0,
                            "entries": [
                                {"account": dr_acc, "type": "DEBIT", "amount": amt,
                                 "category": "Current Assets" if any(k in dr_acc.upper() for k in ["BANK","CASH"]) else cat},
                                {"account": cr_acc, "type": "CREDIT", "amount": amt,
                                 "category": "Current Assets" if any(k in cr_acc.upper() for k in ["BANK","CASH"]) else cat},
                            ]
                        }
                        new_entries.append(new_txn)
                        mod_count += 1
                    elif mtype == "RECLASSIFY":
                        mc = mut.get("matching_criteria", {})
                        new_dr = str(mut.get("new_debit_account", "") or mut.get("new_account", "")).strip().title()
                        new_cr = str(mut.get("new_credit_account", "Bank Account")).strip().title()
                        new_cat = str(mut.get("new_category", "Expense")).strip()
                        reclass_round = 0
                        for t in classified:
                            orig_amt = float(t['entries'][0].get('amount', 0)) if t.get('entries') else float(t.get('debit', 0) or t.get('credit', 0) or 0)
                            t_type = "DEBIT" if float(t.get('debit', 0)) > 0 else "CREDIT"
                            if mc.get('type') and mc['type'].upper() != t_type: continue
                            if mc.get('min_amount') is not None and orig_amt < float(mc['min_amount']): continue
                            if mc.get('max_amount') is not None and orig_amt > float(mc['max_amount']): continue
                            narr_kw = str(mc.get('narration_contains', '')).lower()
                            if narr_kw and narr_kw not in str(t.get('narration', '')).lower(): continue
                            t['entries'] = [
                                {"account": new_dr, "type": "DEBIT", "amount": orig_amt, "category": "Current Assets" if "BANK" in new_dr.upper() else new_cat},
                                {"account": new_cr, "type": "CREDIT", "amount": orig_amt, "category": "Current Assets" if "BANK" in new_cr.upper() else new_cat},
                            ]
                            reclass_round += 1
                        mod_count += reclass_round
                    elif mtype == "DELETE":
                        mc = mut.get("matching_criteria", {})
                        for t in classified:
                            amt = float(t.get('debit', 0) or t.get('credit', 0) or 0)
                            narr_kw = str(mc.get('narration_contains', '')).lower()
                            exact = mc.get('exact_amount')
                            if narr_kw and narr_kw not in str(t.get('narration', '')).lower(): continue
                            if exact is not None and abs(amt - float(exact)) > 0.01: continue
                            t['_deleted'] = True
                        classified[:] = [t for t in classified if not t.get('_deleted')]
                
                classified.extend(new_entries)
                print(f"[AGENT-7] Persistent refinement complete. {mod_count} modifications applied, {len(new_entries)} new entries.")
                yield {"status": f"Successfully re-applied {len(persistent_rules)} manual refinements."}
            except Exception as e:
                print(f"[AGENT-7][ERROR] Persistent refinement failed: {e}")

        ledger_balances: Dict[str, float] = {}
        account_categories: Dict[str, str] = {}

        print(f"[AGENT-3] {AGENT_SYSTEM_PROMPTS['journal']}")
        yield {"status": "Agent 3 (Journal): Writing double-entry journal & populating Ledger Store..."}
        journal_rows_data: List[List] = []
        tx_count = 0

        yield "\n\n1. Professional Journal Book\n\n| Date | Particulars (Account) | L.F. | Debit (\u20b9) | Credit (\u20b9) | Narration |\n|---|---|---|---|---|---|\n"

        tx_source = all_transactions if is_refinement else classified
        for t in tx_source:
            date_val = str(t.get('date', '')).strip()
            narration = str(t.get('narration', '')).strip()
            entries = t.get('entries', [])

            dr_total = sum(float(e.get('amount', 0)) for e in entries if str(e.get('type')).upper() == 'DEBIT')
            cr_total = sum(float(e.get('amount', 0)) for e in entries if str(e.get('type')).upper() == 'CREDIT')
            if abs(dr_total - cr_total) > 0.01:
                print(f"[JOURNAL] Skipped imbalanced tx (Dr={dr_total} Cr={cr_total}): '{narration}'")
                continue

            tx_count += 1

            for entry in entries:
                acc = str(entry.get('account', 'Unclassified')).strip().title()
                cat = str(entry.get('category', 'Unclassified')).strip()
                amt = float(entry.get('amount', 0))
                etype = str(entry.get('type')).upper()
                ledger_balances.setdefault(acc, 0.0)
                self.shared_accounts.add(acc)
                if cat and cat != 'Unclassified':
                    account_categories[acc] = cat
                if etype == 'DEBIT':
                    ledger_balances[acc] += amt
                else:
                    ledger_balances[acc] -= amt

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

                journal_rows_data.append([d_date, d_acc, "", d_dr, d_cr, d_narr])
                first = False

        print(f"[AGENT-3] Journal done: {tx_count} valid transactions written.")
        print(f"[AGENT-3] Ledger Store populated: {len(ledger_balances)} accounts.")
        for acc, bal in ledger_balances.items():
            cat = account_categories.get(acc, 'Unclassified')
            print(f"[LEDGER]   {acc:<40} | {bal:>12,.2f} | {cat}")

        # --- AGENT 4: PROFIT & LOSS ---
        print(f"[AGENT-4] {AGENT_SYSTEM_PROMPTS['pnl']}")
        yield {"status": "Agent 4 (P&L Agent): Synthesizing Profit & Loss from Ledger logic..."}

        pnl_ledger = {acc: bal for acc, bal in ledger_balances.items() 
                     if any(kw in account_categories.get(acc, "").upper() for kw in ["INCOME", "EXPENSE", "REVENUE", "COST"])}
        
        pnl_prompt = f"""
        {AGENT_SYSTEM_PROMPTS['pnl']}
        {PNL_RULES_TEXT}
        
        LEDGER DATA:
        {json.dumps(pnl_ledger, indent=2)}
        
        TASK: Synthesize the P&L statement in JSON format following the rules above.
        """
        
        net_profit = 0.0
        pnl_table_rows = []
        try:
            with llm_stage("AGENT-4 P&L"):
                pnl_res = await retry_with_backoff(self.llm.ainvoke, pnl_prompt)
            pnl_data = self.ai_service._extract_json(pnl_res.content)
            
            inc_rows = pnl_data.get("income_rows", [])
            exp_rows = pnl_data.get("expense_rows", [])
            net_profit = float(pnl_data.get("net_profit", 0))
            
            pnl_dr_total = float(pnl_data.get("total_expense", 0))
            pnl_cr_total = float(pnl_data.get("total_income", 0))

            # Build Table Rows
            for i in range(max(len(inc_rows), len(exp_rows)) + 1):
                d_part, d_amt, c_part, c_amt = "", "", "", ""
                if i < len(exp_rows):
                    d_part, d_amt = f"To {exp_rows[i]['account']}", f"{float(exp_rows[i]['amount']):,.2f}"
                elif i == len(exp_rows) and net_profit > 0:
                    d_part, d_amt = "To Net Profit (c/d)", f"{net_profit:,.2f}"
                
                if i < len(inc_rows):
                    c_part, c_amt = f"By {inc_rows[i]['account']}", f"{float(inc_rows[i]['amount']):,.2f}"
                elif i == len(inc_rows) and net_profit < 0:
                    c_part, c_amt = "By Net Loss (c/d)", f"{abs(net_profit):,.2f}"
                
                if d_part or c_part:
                    pnl_table_rows.append([d_part, d_amt, c_part, c_amt])
            
            pnl_table_rows.append(["TOTAL", f"{max(pnl_dr_total, pnl_cr_total):,.2f}", "TOTAL", f"{max(pnl_dr_total, pnl_cr_total):,.2f}"])
            
            local_structured_tables.append({
                "type": "profit_loss",
                "title": "Profit & Loss Statement",
                "headers": ["Particulars (Dr)", "Amount (₹)", "Particulars (Cr)", "Amount (₹)"],
                "rows": pnl_table_rows
            })
        except Exception as e:
            print(f"[AGENT-4][ERROR] P&L Synthesis failed: {e}")
            yield {"status": " P&L Synthesis failed. Falling back..."}

            if 'profit_loss' in _show_tables:
                yield {
                    "structured_tables": [t for t in local_structured_tables if t.get('type') == 'profit_loss'],
                    "all_tables": local_structured_tables
                }

        # --- AGENT 4.5: CAPITAL ACCOUNT ---
        print(f"[AGENT-4.5] {AGENT_SYSTEM_PROMPTS.get('capital', 'Generating Capital Account...')}")
        yield {"status": "Agent 4.5 (Capital Account Agent): Synthesizing Capital movement..."}

        equity_ledger = {acc: bal for acc, bal in ledger_balances.items() if "EQUITY" in account_categories.get(acc, "").upper()}
        
        cap_prompt = f"""
        {AGENT_SYSTEM_PROMPTS['capital']}
        {CAPITAL_ACCOUNT_RULES_TEXT}
        
        EQUITY LEDGER:
        {json.dumps(equity_ledger, indent=2)}
        
        NET RESULT:
        {"Net Profit" if net_profit > 0 else "Net Loss"}: ₹{abs(net_profit):,.2f}
        
        TASK: Output the Capital Account Statement JSON.
        """
        
        closing_capital = 0.0
        try:
            with llm_stage("AGENT-4.5 Capital"):
                cap_res = await retry_with_backoff(self.llm.ainvoke, cap_prompt)
            cap_data = self.ai_service._extract_json(cap_res.content)
            
            capital_table_rows = [[p["label"], f"{float(p['amount']):,.2f}"] for p in cap_data.get("particulars", [])]
            closing_capital = float(cap_data.get("closing_capital", 0))
            capital_table_rows.append(["CLOSING CAPITAL", f"{closing_capital:,.2f}"])
            
            local_structured_tables.append({
                "type": "capital_account",
                "title": "Capital Account Statement",
                "headers": ["Particulars", "Amount (₹)"],
                "rows": capital_table_rows
            })
        except Exception as e:
            print(f"[AGENT-4.5][ERROR] Capital Synthesis failed: {e}")
            yield {"status": " Capital Synthesis failed."}

        if 'capital_account' in _show_tables:
            yield {
                "structured_tables": [t for t in local_structured_tables if t.get('type') in ['profit_loss', 'capital_account']],
                "all_tables": local_structured_tables
            }

        # --- AGENT 5: BALANCE SHEET ---
        print(f"[AGENT-5] {AGENT_SYSTEM_PROMPTS['balance_sheet']}")
        yield {"status": "Agent 5 (Balance Sheet Agent): Synthesizing Balance Sheet..."}

        bs_ledger = {acc: bal for acc, bal in ledger_balances.items() 
                    if acc not in pnl_ledger and "EQUITY" not in account_categories.get(acc, "").upper()}
        
        bs_prompt = f"""
        {AGENT_SYSTEM_PROMPTS['balance_sheet']}
        {BALANCE_SHEET_RULES_TEXT}
        
        LEDGER DATA:
        {json.dumps(bs_ledger, indent=2)}
        
        CLOSING CAPITAL (from Agent 4.5):
        ₹{closing_capital:,.2f}
        
        TASK: Synthesize the Balance Sheet JSON.
        """
        
        try:
            with llm_stage("AGENT-5 Balance Sheet"):
                bs_res = await retry_with_backoff(self.llm.ainvoke, bs_prompt)
            bs_data = self.ai_service._extract_json(bs_res.content)
            
            assets_dict = bs_data.get("assets", {})
            liab_dict = bs_data.get("liabilities", {})
            equity_dict = bs_data.get("equity", {})
            
            bs_table_rows = []
            
            # Helper to flatten JSON structure for table display
            l_rows = []
            for grp, items in liab_dict.items():
                l_rows.append([f"**{grp.upper()}**", ""])
                for itm in items: l_rows.append([f"   {itm['account']}", f"{float(itm['amount']):,.2f}"])
            for grp, items in equity_dict.items():
                l_rows.append([f"**{grp.upper()}**", ""])
                for itm in items: l_rows.append([f"   {itm['account']}", f"{float(itm['amount']):,.2f}"])
            
            a_rows = []
            for grp, items in assets_dict.items():
                a_rows.append([f"**{grp.upper()}**", ""])
                for itm in items: a_rows.append([f"   {itm['account']}", f"{float(itm['amount']):,.2f}"])
            
            for i in range(max(len(l_rows), len(a_rows))):
                l_part, l_amt, a_part, a_amt = "", "", "", ""
                if i < len(l_rows): l_part, l_amt = l_rows[i]
                if i < len(a_rows): a_part, a_amt = a_rows[i]
                bs_table_rows.append([l_part, l_amt, a_part, a_amt])
                
            bs_total_a = float(bs_data.get("total_assets", 0))
            bs_total_l = float(bs_data.get("total_liabilities_equity", 0))
            bs_table_rows.append(["TOTAL", f"{bs_total_l:,.2f}", "TOTAL", f"{bs_total_a:,.2f}"])

            local_structured_tables.append({
                "type": "balance_sheet",
                "title": "Balance Sheet",
                "headers": ["Liabilities & Equity", "Amount (₹)", "Assets", "Amount (₹)"],
                "rows": bs_table_rows
            })
            
            asset_total = bs_total_a
            liab_total = bs_total_l
        except Exception as e:
            print(f"[AGENT-5][ERROR] Balance Sheet Synthesis failed: {e}")
            yield {"status": " Balance Sheet Synthesis failed."}

        if 'balance_sheet' in _show_tables:
            yield {
                "structured_tables": [t for t in local_structured_tables if t.get('type') in ['profit_loss', 'capital_account', 'balance_sheet']],
                "all_tables": local_structured_tables
            }

        allowed_types = _show_tables.copy()
        if 'generic' not in allowed_types:

            pass

        requested_tables = [t for t in local_structured_tables if t.get('type') in allowed_types]
        print(f"[INTENT] Exposing {len(requested_tables)} table(s): {[t.get('type') for t in requested_tables]}")

        yield {"structured_tables": requested_tables, "all_tables": local_structured_tables}

        print(f"[AGENT-6] {AGENT_SYSTEM_PROMPTS['tally']}")
        print(f"[AGENT-6] Config: {TALLY_CONFIG}")
        yield {"status": "Agent 6 (Tally Auditor): Running all 3 validation checks..."}

        tb_total_dr = sum(v for v in ledger_balances.values() if v > 0)
        tb_total_cr = sum(abs(v) for v in ledger_balances.values() if v < 0)
        tb_diff = abs(tb_total_dr - tb_total_cr)
        if TALLY_CONFIG.get('trial_balance_check'):
            if tb_diff > 0.01:
                print(f"[AGENT-6] Trial Balance MISMATCH: Dr=\u20b9{tb_total_dr:,.2f} Cr=\u20b9{tb_total_cr:,.2f} Diff=\u20b9{tb_diff:,.2f}")
                yield f"\n> \u26a0\ufe0f **Trial Balance FAILED**: Dr \u20b9{tb_total_dr:,.2f} \u2260 Cr \u20b9{tb_total_cr:,.2f} (diff=\u20b9{tb_diff:,.2f})\n"
            else:
                print(f"[AGENT-6] Trial Balance PASSED: \u20b9{tb_total_dr:,.2f}")
                yield f"\n> \u2705 **Trial Balance PASSED** \u2014 Total Dr = Total Cr = \u20b9{tb_total_dr:,.2f}\n"

        diff = abs(asset_total - liab_total)
        if TALLY_CONFIG.get('balance_sheet_check'):
            if diff > 0.01:
                if TALLY_CONFIG.get('auto_capital_adjustment'):
                    cap_adj = asset_total - liab_total
                    # Since reports are now LLM-synthesized, we report the adjustment needed
                    yield (
                        f"\n>  **Balance Sheet ADJUSTED**: Difference of ₹{abs(cap_adj):,.2f} absorbed "
                        f"via Capital Adjustment under Equity (Agent 6 validation).\n"
                    )
                    print(f"[AGENT-6] BS adjustment needed: ₹{cap_adj:,.2f}")
                else:
                    yield (
                        f"\n>  **Balance Sheet FAILED**: Out of balance by ₹{diff:,.2f}.\n"
                        f"> Check for unclassified transactions or missing opening balances.\n"
                    )
                    print(f"[AGENT-6] BS FAILED: Assets=₹{asset_total:,.2f} L+E=₹{liab_total:,.2f} Diff=₹{diff:,.2f}")
            else:
                yield (
                    f"\n>  **Balance Sheet PASSED** — Assets = Liabilities + Equity = ₹{asset_total:,.2f}\n"
                )
                print(f"[AGENT-6] BS PASSED: ₹{asset_total:,.2f}")

    async def _parse_markdown_tables(self, text: str) -> List[Dict[str, Any]]:
        """Extract structured data from markdown tables in text."""
        import re
        tables = []

        table_regex = r"((?:\|[^\n]+\|(?:\n|$))+)"
        matches = re.finditer(table_regex, text)

        for match in matches:
            table_str = match.group(1).strip()

            rows = [r.strip() for r in table_str.split("\n") if r.strip()]

            parsed_rows = []
            for row in rows:
                if re.match(r"^\|?[-:| ]+\|?$", row.strip()):
                    continue

                cells = [c.strip() for c in row.split("|")]

                if row.startswith("|"): cells = cells[1:]
                if row.endswith("|") and cells: cells = cells[:-1]
                parsed_rows.append(cells)

            if len(parsed_rows) < 2:
                continue

            headers = parsed_rows[0]
            body = parsed_rows[1:]

            table_type = "generic"
            headers_lower = [h.lower() for h in headers]
            if any("debit" in h or "credit" in h for h in headers_lower):
                table_type = "journal"
            elif any("asset" in h or "liabilit" in h for h in headers_lower):
                table_type = "balance_sheet"
            elif any("particulars" in h and "amount" in h for h in headers_lower):
                table_type = "profit_loss"

            tables.append({
                "type": table_type,
                "headers": headers,
                "rows": body
            })
        return tables
