from typing import List, Dict, Any, AsyncGenerator, Union
import json
import asyncio
import hashlib
from langchain_openai import ChatOpenAI
from app.core.config import settings
from app.services.ai_service import LLMService, retry_with_backoff
from app.ai.rag.retriever import vector_store
from app.services.accounting_rules import (
    EXTRACTOR_RULES, CLASSIFIER_RULES, REFINEMENT_RULES, JOURNAL_RULES_TEXT,
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
        
        # --- SMART CONCURRENCY DETECTION ---
        # User requested 10 workers maximum. 
        # We still auto-throttle for ":free" models to avoid 429s.
        self.model_name = model.lower()
        self.is_free_model = ":free" in self.model_name
        
        if self.is_free_model:
            self.n_extractors = 5
            self.n_classifiers = 3
            self.worker_delay = 1.5
            print(f"[INFO] Free Model Detected ({model}). Throttling to {self.n_extractors} workers.")
        else:
            self.n_extractors = 10  # USER REQUESTED LIMIT
            self.n_classifiers = 10
            self.worker_delay = 0.5
            print(f"[INFO] Professional Model ({model}). Speed capped at {self.n_extractors} workers.")
            
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
        try:
            res = await self.llm.ainvoke(prompt)
            data = self.ai_service._extract_json(res.content)
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
        # Remove special characters and extra whitespace, then lowercase
        normalized = re.sub(r'[^a-zA-Z0-9]', '', narration).lower()
        return normalized

    def _detect_requested_tables(self, question: str) -> set:
        """
        Determine which table types to expose based on the user's question.
        Generates everything internally but only shows the asked-for cards.
        """
        import re
        # Normalize: collapse "p & l" → "p&l", "p & l statement" → "p&l", etc.
        q = question.lower()
        q = re.sub(r'p\s*&\s*l', 'p&l', q)  # "p & l" → "p&l"
        q = re.sub(r'p\s+and\s+l\b', 'p&l', q)  # "p and l" → "p&l"
        
        q_words = re.sub(r'[^a-z0-9]', ' ', q).split()

        wants_bs  = any(kw in q for kw in [
            'balanc', 'asset', 'liabilit', 'equity'
        ]) or 'bs' in q_words
        wants_pnl = any(kw in q for kw in [
            'profit', 'loss', 'p&l', 'pnl', 'income', 'expense', 'revenue',
            'trading', 'income statement', 'statement of profit'
        ])

        if wants_bs and wants_pnl:
            return {'profit_loss', 'balance_sheet'}
        if wants_bs:
            return {'balance_sheet'}
        if wants_pnl:
            return {'profit_loss'}
        # Default: show all financial statements (no journal)
        return {'profit_loss', 'balance_sheet'}

    async def _extract_transactions_from_batch(
        self, 
        context: Union[str, List[str]], 
        context_query: str = None, 
        company_id: str = None,
        feedback_context: str = ""
    ) -> List[Dict]:
        """
        Extract structured transaction data from a text batch.
        IMPROVED: Feedback context passed as argument to avoid redundant VDB queries.
        """
        if isinstance(context, list):
            context = "\n---\n".join(context)
            
        content_hash = hashlib.sha256(context.encode()).hexdigest()
        
        # Check cache
        if content_hash in self.extraction_cache:
            print(f"[CACHE HIT] Reusing cached extraction for batch (hash: {content_hash[:8]}...)")
            return self.extraction_cache[content_hash]
        
        if context_query:
            query_instruction = f"""
        USER INTENT: "{context_query}"
        
        STRICT PRE-FILTERING RULES:
        1. Identify if the user is asking for a SPECIFIC transaction filter (e.g., "only cash", "only UPI").
        2. If the user is asking for a general report (e.g., "Balance Sheet", "P&L", "summary", "journal"):
           - DO NOT SKIP ANY VALID BANK TRANSACTIONS. Extract everything.
        3. ONLY if the user explicitly requested a specific filter:
           - Skip transactions that DO NOT match the requested type.
        """

        prompt = f"""
        {AGENT_SYSTEM_PROMPTS['extractor']}

        {EXTRACTOR_RULES}

        {query_instruction}
        {feedback_context}

        INPUT TEXT (Bank Statement):
        {context}

        TASK:
        Read the bank statement text and extract EVERY SINGLE transaction without exception.
        - YOU ARE FORBIDDEN FROM SKIPPING ANY ROW.
        - IDENTIFY COLUMNS: Look for headers like Withdrawal/Debit and Deposit/Credit.
        - MAP DATA CORRECTLY: Money OUT goes into `debit`, Money IN goes into `credit`.
        - DO NOT put everything into `debit`. If it's a Deposit, put it in `credit`.
        - If a line looks like a transaction, it MUST be in your output.

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
        
        try:
            # Use the robust retry_with_backoff helper
            response = await retry_with_backoff(self.llm.ainvoke, prompt)
            data = self.ai_service._extract_json(response.content)
            transactions = data.get("transactions", [])
            
            # Cache the result
            self.extraction_cache[content_hash] = transactions
            print(f"[CACHE STORE] Cached {len(transactions)} transactions (hash: {content_hash[:8]}...)")
            
            return transactions
        except Exception as e:
            print(f"[AGENT-1][ERROR] Fatal extraction error in batch: {e}")
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
        
        # --- DYNAMIC RULES FETCHING ---
        dynamic_business_rules = ""
        business_type = None
        if customer_id:
            print(f"[RULES] Fetching dynamic rules for customer_id: {customer_id}")
            try:
                from app.core.supabase import supabase
                # Get business type
                cust_res = supabase.table("customers").select("business_type, name").eq("id", customer_id).single().execute()
                if cust_res.data and cust_res.data.get("business_type"):
                    business_type = cust_res.data.get("business_type")
                    customer_name = cust_res.data.get("name")
                    print(f"[RULES] Found Business Type: {business_type} for Customer: {customer_name}")
                    dynamic_business_rules += f"CUSTOMER BUSINESS TYPE: {business_type}\n"
                
                # Get specific rules (matching either customer_id or business_type)
                # Note: bypassing company_id to avoid UUID type mismatch error, customer_id is unique anyway
                query = supabase.table("accounting_rules").select("rule_description")
                if business_type:
                    query = query.or_(f"customer_id.eq.{customer_id},business_type.eq.{business_type}")
                else:
                    query = query.eq("customer_id", customer_id)
                rules_res = query.execute()
                
                if rules_res.data:
                    dynamic_business_rules += "SPECIFIC USER-TAUGHT RULES (HIGHEST PRIORITY):\n"
                    print(f"[RULES] Successfully fetched {len(rules_res.data)} custom accounting rules.")
                    for r in rules_res.data:
                        rule_text = r['rule_description']
                        dynamic_business_rules += f"- {rule_text}\n"
                        print(f"[RULES] -> Applied Rule: {rule_text}")
                else:
                    print(f"[RULES] No custom rules found for this customer.")
            except Exception as e:
                print(f"[RULES] Failed to fetch dynamic accounting rules: {e}")

        # --- REFINEMENT MODE (ITERATION) ---
        if previous_context or feedback_history:
            yield {"status": "Refining report with new user instruction..."}
            print(f"[REFINEMENT] Injecting user feedback as dynamic rule: {question}")
            # Add the user's specific refinement instruction as a CRITICAL rule
            ctx_text = f"\nPREVIOUS CONTEXT:\n{feedback_history}" if feedback_history else ""
            dynamic_business_rules += f"""
            ━━━ CRITICAL COMMAND: USER AD-HOC REFINEMENT (ABSOLUTE PRIORITY) ━━━
            THE USER HAS REQUESTED THE FOLLOWING CHANGE: "{question}"
            {ctx_text}
            - YOU MUST FOLLOW THIS RULE ABOVE ALL OTHER ACCOUNTING PRINCIPLES.
            - IF THIS RULE SPECIFIES AN ACCOUNT OR CATEGORY FOR A CERTAIN RANGE/KEYWORD, APPLY IT STRICTLY.
            ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
            """
            # We don't return here anymore; we let the pipeline re-run with this new rule.

        # =====================================================================
        # --- STANDARD GENERATION MODE  (5-Agent Pipeline) ---
        # =====================================================================

        # Local structured tables track instead of singleton variables
        local_structured_tables = []
        
        # --- AGENT 0: INTENT DISCOVERY ---
        intent_data = await self._discover_intent(question)
        intent = intent_data.get('intent', 'EXTRACTION')
        _show_tables = set(intent_data.get('report_types', ['profit_loss', 'balance_sheet']))
        
        print(f"[INTENT] User asked for: {_show_tables} | ID Intent: {intent}")

        if not context_chunks and not input_transactions:
            yield "[ERROR] No context data provided."
            return

        yield {"status": f"Agent 0 (Guard): input validated. Starting pipeline..."}

        # --- PRE-FETCH ANALYTICS FEEDBACK (Learning Agent) ---
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
        
        # Determine Routing: If it's a REFINEMENT intent AND we have data, use Agent 7.
        is_refinement = (intent == "REFINEMENT" or intent == "ANALYSIS") and input_transactions is not None
        
        if is_refinement:
            print(f"[AGENT-7] REFINEMENT MODE: LLM Refiner triggered by intent '{intent}' for {len(input_transactions)} transactions.")
            # --- AGENT 7: REFINEMENT ORCHESTRATOR ---
            yield {"status": "Agent 7 (Orchestrator): Analyzing your request and identifying targets..."}
            
            # Agent 7 sees a sample of transactions to understand the context
            sample_txns = input_transactions[:100] # First 100 for context
            txn_sample_text = ""
            for i, t in enumerate(sample_txns):
                amt = float(t.get('debit', 0) or t.get('credit', 0) or 0)
                txn_sample_text += f"{i}. [{t.get('date')}] {t.get('narration')} (₹{amt:,.2f})\n"

            refine_discovery_prompt = f"""
            {AGENT_SYSTEM_PROMPTS['refinement']}
            {REFINEMENT_RULES}

            USER COMMAND: "{question}"
            BUSINESS TYPE: {business_type or "Unknown"}
            
            SAMPLE TRANSACTIONS:
            {txn_sample_text}
            
            TASK: Identify if this is a CLASSIFICATION change (move accounts) or MATHEMATICAL change (edit totals/dates).
            Provide the 'matching_criteria' (like the old rules) so Python can find ALL targets in the full list of {len(input_transactions)} entries.
            
            Output JSON only.
            """
            
            plan = {}
            try:
                res = await self.llm.ainvoke(refine_discovery_prompt)
                plan = self.ai_service._extract_json(res.content)
                if not isinstance(plan, dict): plan = {}
                print(f"[AGENT-7] Plan Discovered: Intent={plan.get('intent')} | Reasoning: {plan.get('reasoning')}")
            except Exception as e:
                print(f"[AGENT-7][ERROR] Orchestration failed: {e}")
                plan = {"intent": "GENERAL"}

            intent = plan.get('intent', 'GENERAL')
            meta = plan.get('rule_metadata', {})
            matches_for_reclassification = []
            modified_indices = set()
            
            # Use matching criteria from Agent 7 to find targets in Python (for scalability)
            m = plan.get('matching_criteria', plan.get('match', {})) # Backward compatibility during transition
            
            if m:
                for i, t in enumerate(input_transactions):
                    amt = float(t.get('debit', 0) or t.get('credit', 0) or 0)
                    is_dr = float(t.get('debit', 0)) > 0
                    t_type = "DEBIT" if is_dr else "CREDIT"
                    
                    # Match conditions
                    if m.get('type') and m.get('type') != t_type: continue
                    if m.get('min_amount') is not None and amt < m.get('min_amount'): continue
                    if m.get('max_amount') is not None and amt > m.get('max_amount'): continue
                    
                    narr_match = m.get('narration_contains', '').lower()
                    if narr_match and narr_match not in str(t.get('narration', '')).lower(): continue
                        
                    matches_for_reclassification.append(t)
                    modified_indices.add(i)

            modified_count = len(matches_for_reclassification)
            
            # --- EXECUTION BRANCHES ---
            if intent == "CLASSIFICATION" and modified_count > 0:
                yield {"status": f"Agent 7: Identified {modified_count} targets. Agent 2 (Expert): Re-classifying..."}
                
                # Agent 2 gets the USER'S ORIGINAL COMMAND as the law
                refine_as_rule = f"\n━━━ USER REFINEMENT COMMAND (MANDATORY) ━━━\n{question}\n━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━\n"
                
                re_classify_batches = [
                    matches_for_reclassification[i:i + 50] 
                    for i in range(0, len(matches_for_reclassification), 50)
                ]
                
                total_recl = 0
                for b_idx, batch in enumerate(re_classify_batches):
                    prompt = f"""
                    {AGENT_SYSTEM_PROMPTS['classifier']}
                    {CLASSIFIER_RULES}
                    {refine_as_rule}
                    
                    TRANSACTIONS TO RE-CLASSIFY:
                    """
                    for j, t in enumerate(batch):
                        dr = float(t.get('debit', 0) or 0)
                        cr = float(t.get('credit', 0) or 0)
                        direction = f"DEBIT ₹{dr:,.2f}" if dr > 0 else f"CREDIT ₹{cr:,.2f}"
                        prompt += f"{j+1}. [{direction}] {t.get('narration', 'Unknown')}\n"
                    
                    try:
                        res = await retry_with_backoff(self.llm.ainvoke, prompt)
                        raw_recl = self.ai_service._extract_json(res.content)
                        if isinstance(raw_recl, dict) and "transactions" in raw_recl:
                            updated_results = raw_recl["transactions"]
                        elif isinstance(raw_recl, list):
                            updated_results = raw_recl
                        else:
                            updated_results = []
                        
                        if isinstance(updated_results, list) and len(updated_results) == len(batch):
                            for k, updated_cls in enumerate(updated_results):
                                orig_t = batch[k]
                                dr_acc = updated_cls.get('debit_account', 'Unclassified')
                                cr_acc = updated_cls.get('credit_account', 'Unclassified')
                                cat = updated_cls.get('category', 'Indirect Expense')
                                
                                amt = float(orig_t.get('debit', 0) or orig_t.get('credit', 0) or 0)
                                orig_t['entries'] = [
                                    {"account": dr_acc, "type": "DEBIT", "amount": amt, "category": cat if "Bank" not in dr_acc else "Current Assets"},
                                    {"account": cr_acc, "type": "CREDIT", "amount": amt, "category": cat if "Bank" not in cr_acc else "Current Assets"}
                                ]
                                total_recl += 1
                    except Exception as e:
                        print(f"[RE-CLASSIFY][ERROR] Batch {b_idx} failed: {e}")
                
            elif intent == "MATHEMATICAL":
                yield {"status": "Agent 7: Identified Mathematical updates. Applying changes..."}
                updates = plan.get('math_updates', [])
                for u in updates:
                    idx = u.get('index')
                    if idx is not None and 0 <= idx < len(input_transactions):
                        t = input_transactions[idx]
                        field = u.get('field')
                        if field in ['debit', 'credit', 'amount']:
                            val = float(u.get('new_value', 0))
                            if field == 'amount':
                                # Determine which column to update based on existing values
                                if float(t.get('debit', 0)) > 0: t['debit'] = val
                                else: t['credit'] = val
                            else:
                                t[field] = val
                            print(f"[MATH] Updated txn {idx} {field} to {val}")
                        elif field == 'date':
                            t['date'] = u.get('new_value')

            # --- PERSIST VERBATIM RULES (LEARNING) ---
            if modified_count > 0 or intent == "MATHEMATICAL":
                try:
                    from app.core.supabase import supabase
                    scope = meta.get('scope', 'CUSTOMER')
                    # Use verbatim command if available, else user's instruction
                    rule_desc = meta.get('rule_description', question)
                    verbatim = meta.get('verbatim_command', question)
                    
                    save_data = {
                        "rule_description": f"{rule_desc} (User Command: {verbatim})",
                        "business_type": business_type if scope == "INDUSTRY" else None,
                        "customer_id": customer_id if scope == "CUSTOMER" else None,
                        "is_active": True
                    }
                    
                    # Duplicate check
                    check = supabase.table("accounting_rules").select("id").eq("rule_description", save_data["rule_description"])
                    if scope == "INDUSTRY": check = check.eq("business_type", business_type).is_("customer_id", "null")
                    else: check = check.eq("customer_id", customer_id)
                    
                    if not check.execute().data:
                        supabase.table("accounting_rules").insert(save_data).execute()
                        print(f"[LEARNING] Saved {scope} rule: {save_data['rule_description']}")
                except Exception as e:
                    print(f"[LEARNING][ERROR] Failed: {e}")

            all_transactions = input_transactions # Preserve changes

            yield {"status": f"Refinement complete. {modified_count} transactions targetted and re-classified."}
            print(f"[AGENT-7] Pipeline: Filtered {modified_count} txns for Agent 2.")

            # Reconciliation logging
            expected_deposits = len([t for t in all_transactions if float(t.get('credit', 0) or 0) > 0])
            expected_withdrawals = len([t for t in all_transactions if float(t.get('debit', 0) or 0) > 0])
        else:
            # --- TRANSACTION SUMMARY EXTRACTION (Parallelized) ---
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
                summary_task = asyncio.create_task(self.llm.ainvoke(summary_prompt))

            BATCH_SIZE = 10
            batches = [context_chunks[i:i + BATCH_SIZE] for i in range(0, len(context_chunks), BATCH_SIZE)]
            total_batches = len(batches)

            # Parallel workers for speed - Auto-scaled based on model tier
            N_EXTRACTORS = self.n_extractors
            WORKER_DELAY_SEC = self.worker_delay
            sem = asyncio.Semaphore(N_EXTRACTORS)

            print(f"[AGENT-1] Extractor Pool: {len(context_chunks)} chunks → {total_batches} batches (size={BATCH_SIZE}) → {N_EXTRACTORS} workers")

            async def process_batch(index, batch_data, batch_metas):
                async with sem:
                    slot = index % N_EXTRACTORS
                    # Stagger to avoid huge initial burst
                    if index < N_EXTRACTORS:
                        await asyncio.sleep(slot * WORKER_DELAY_SEC)

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

            # Store batch results for potential reconciliation
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

            # Now finalize the summary task if it was running in background
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

            # --- RECONCILIATION LOOP (NEW) ---
            # If we missed transactions compared to the bank summary, retry empty batches meticulously
            total_extracted = len(all_transactions)
            total_expected = (expected_deposits or 0) + (expected_withdrawals or 0)
            
            if total_expected > total_extracted + 5: # 5 txn buffer for footer/header noise
                missing_count = total_expected - total_extracted
                print(f"[RECONCILE] Detected missing data: Found {total_extracted}, Expected {total_expected}. Retrying empty batches...")
                yield {"status": f"⚠️ Missing {missing_count} transactions. Agent 1 (Reconciliation): Retrying empty areas..."}
                
                # Identify batches that returned 0
                failed_indices = [idx for idx, res in batch_results.items() if not res]
                if failed_indices:
                    print(f"[RECONCILE] Retrying {len(failed_indices)} batches. Flattening chunks for parallel extraction...")
                    
                    # Flatten into a single chunk pool for maximum parallelism
                    missing_tasks = []
                    for f_idx in failed_indices:
                        f_batch = batches[f_idx]
                        f_metas = context_metadatas[f_idx*BATCH_SIZE:(f_idx+1)*BATCH_SIZE] if context_metadatas else [None] * len(f_batch)
                        for c_idx, chunk in enumerate(f_batch):
                            missing_tasks.append((chunk, f_metas[c_idx]))

                    async def retry_chunk(c_idx_global, chunk_data, meta):
                        async with sem: # Use the same extractor semaphore
                            slot = c_idx_global % N_EXTRACTORS
                            # Add stagger even for reconciliation to avoid bursts
                            await asyncio.sleep(slot * WORKER_DELAY_SEC)
                            
                            r_txns = await self._extract_transactions_from_batch(
                                chunk_data, 
                                context_query=question, 
                                company_id=company_id,
                                feedback_context=feedback_context
                            )
                            if r_txns and meta:
                                src_doc = meta.get('document_name')
                                for t in r_txns:
                                    t['_source_docs'] = [src_doc] if src_doc else []
                            return r_txns

                    reconcile_results = await asyncio.gather(*[retry_chunk(i, c, m) for i, (c, m) in enumerate(missing_tasks)])
                    
                    new_txns_count = 0
                    for r_list in reconcile_results:
                        if r_list:
                            all_transactions.extend(r_list)
                            new_txns_count += len(r_list)
                    
                    print(f"[RECONCILE] Done. Recovered {new_txns_count} missing transactions.")
                    yield {"status": f"Reconciliation successful: recovered {new_txns_count} transactions."}
                else:
                    print(f"[RECONCILE] No empty batches to retry, but count still mismatches.")

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

        # --- RECONCILIATION LOGGING ---
        extracted_deposits = len([t for t in all_transactions if float(t.get('credit', 0) or 0) > 0])
        extracted_withdrawals = len([t for t in all_transactions if float(t.get('debit', 0) or 0) > 0])
        
        print(f"[RECONCILE] Extraction Results: Deposits={extracted_deposits}, Withdrawals={extracted_withdrawals}")
        if expected_deposits is not None or expected_withdrawals is not None:
            if expected_deposits == extracted_deposits and expected_withdrawals == extracted_withdrawals:
                print(f"[RECONCILE] SUCCESS: Extracted counts match Bank Summary!")
            else:
                warning = f"[RECONCILE] WARNING: Counts MISMATCH! Bank Summary: D={expected_deposits}, W={expected_withdrawals} | Extracted: D={extracted_deposits}, W={extracted_withdrawals}"
                print(warning)
                # yield error/warning message to UI optionally if it's too bad
                if abs(len(all_transactions) - ((expected_deposits or 0) + (expected_withdrawals or 0))) > 10:
                    yield {"status": f"⚠️ Extraction mismatch: found {len(all_transactions)} transactions, but summary expected {(expected_deposits or 0) + (expected_withdrawals or 0)}."}

        # Signal the API to store these transactions for future refinement
        yield {"all_transactions": all_transactions}

        # ===================================================================
        # AGENT 2: BATCH CLASSIFIER POOL (LLM — N parallel classifier workers)
        # ===================================================================
        CLASSIFY_BATCH_SIZE = 50     # transactions per LLM call
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
        classified_results: Dict[int, List[Dict]] = {}  # bidx → list of classified transactions

        async def classify_batch(bidx: int, c_batch: List[Dict]):
            async with cls_sem:
                slot = bidx % N_CLASSIFIERS
                # Small stagger on first round to avoid burst
                if bidx >= N_CLASSIFIERS:
                    await asyncio.sleep(CLASSIFIER_DELAY_SEC * (slot % 3))
                if dynamic_business_rules:
                    print(f"[AGENT-2][Worker-{slot+1}] Injecting {len(dynamic_business_rules)} chars of custom business rules into prompt.")
                
                known_accs = ", ".join(list(self.shared_accounts)[:50]) if self.shared_accounts else "None yet"
                
                narrations_list = ""
                for j, t in enumerate(c_batch):
                    dr = float(t.get('debit', 0) or 0)
                    cr = float(t.get('credit', 0) or 0)
                    direction = f"DEBIT \u20b9{dr:,.2f}" if dr > 0 else f"CREDIT \u20b9{cr:,.2f}"
                    narrations_list += f"{j+1}. [{direction}] {t.get('narration', 'Unknown')}\n"

                classify_prompt = f"""
                {AGENT_SYSTEM_PROMPTS['classifier']}

                {CLASSIFIER_RULES}
                
                {dynamic_business_rules}

                PREVIOUSLY IDENTIFIED ACCOUNTS (Consistency):
                {known_accs}

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
                - If transaction is CREDIT (money IN) → debit_account = "Bank Account"
                - If transaction is DEBIT  (money OUT) → credit_account = "Bank Account"

                Output JSON array ONLY. No explanation.
                """
                batch_classified = []
                try:
                    if dynamic_business_rules:
                        print(f"[AGENT-2][Worker-{slot+1}] Batch {bidx+1}/{total_classify_batches}: Injecting dynamic business rules.")
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
                            
                            # Apply double-entry logic with correct category mapping
                            # If CREDIT (money IN) -> Bank is DEBITED, Income/Liability is CREDITED
                            # If DEBIT (money OUT) -> Expense/Asset is DEBITED, Bank is CREDITED
                            if cr > 0: # Money IN (Credit in Statement, Debit in Bank A/c)
                                t['entries'] = [
                                    {"account": "Bank Account", "type": "DEBIT",  "amount": amt, "category": "Current Assets"},
                                    {"account": credit_acc,      "type": "CREDIT", "amount": amt, "category": category},
                                ]
                            else: # Money OUT (Debit in Statement, Credit in Bank A/c)
                                t['entries'] = [
                                    {"account": debit_acc,  "type": "DEBIT",  "amount": amt, "category": category},
                                    {"account": "Bank Account", "type": "CREDIT", "amount": amt, "category": "Current Assets"},
                                ]
                            t['confidence'] = res.get('confidence', 1.0)
                        else:
                            # Fallback for missing LLM result: Preserve if exists!
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
                        # CRITICAL: Preserve previous classification if it exists!
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

            # Update Ledger Store and Shared Accounts
            for entry in entries:
                acc = str(entry.get('account', 'Unclassified')).strip().title()
                cat = str(entry.get('category', 'Unclassified')).strip()
                amt = float(entry.get('amount', 0))
                etype = str(entry.get('type')).upper()
                ledger_balances.setdefault(acc, 0.0)
                self.shared_accounts.add(acc) # Track for next classification batch
                if cat and cat != 'Unclassified':
                    account_categories[acc] = cat
                if etype == 'DEBIT':
                    ledger_balances[acc] += amt
                else:
                    ledger_balances[acc] -= amt

            # Yield journal rows unconditionally
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
                # journal row collected — not streamed as token
                journal_rows_data.append([d_date, d_acc, "", d_dr, d_cr, d_narr])
                first = False

        # Journal computed for internal ledger use — NOT added to structured_tables (not shown to user)
        print(f"[AGENT-3] Journal done: {tx_count} valid transactions written.")
        print(f"[AGENT-3] Ledger Store populated: {len(ledger_balances)} accounts.")
        for acc, bal in ledger_balances.items():
            cat = account_categories.get(acc, 'Unclassified')
            print(f"[LEDGER]   {acc:<40} | {bal:>12,.2f} | {cat}")

        print(f"[AGENT-4] {AGENT_SYSTEM_PROMPTS['pnl']}")
        print(f"[AGENT-4] Include: {PNL_CONFIG['include_types']} | Exclude: {PNL_CONFIG['exclude_types']}")
        yield {"status": "Agent 4 (P&L Agent): Calculating Profit & Loss from Ledger..."}

        pnl_dr_rows: List[tuple] = []
        pnl_cr_rows: List[tuple] = []
        pnl_dr = 0.0
        pnl_cr = 0.0

        for acc, bal in ledger_balances.items():
            cat = account_categories.get(acc, "").upper()
            if abs(bal) < 0.01:
                continue

            # Filter based on PNL_CONFIG rules
            should_exclude = any(excl.upper() in cat or excl.upper() in acc.upper() for excl in PNL_CONFIG['exclude_types'])
            if should_exclude:
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
            pass  # No P&L data — card will simply not appear
        else:
            pnl_table_rows: List[List] = []
            for i in range(max(len(pnl_dr_rows), len(pnl_cr_rows))):
                d_part = pnl_dr_rows[i][0] if i < len(pnl_dr_rows) else ""
                d_amt = f"{pnl_dr_rows[i][1]:,.2f}" if i < len(pnl_dr_rows) else ""
                c_part = pnl_cr_rows[i][0] if i < len(pnl_cr_rows) else ""
                c_amt = f"{pnl_cr_rows[i][1]:,.2f}" if i < len(pnl_cr_rows) else ""
                pnl_table_rows.append([d_part, d_amt, c_part, c_amt])
            pnl_table_rows.append(["TOTAL", f"{pnl_dr:,.2f}", "TOTAL", f"{pnl_cr:,.2f}"])
            local_structured_tables.append({
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

        bs_table_rows: List[List] = []
        for i in range(max(len(bs_liab), len(bs_assets), 1)):
            l_part = bs_liab[i][0] if i < len(bs_liab) else ""
            l_amt = f"{bs_liab[i][1]:,.2f}" if i < len(bs_liab) else ""
            a_part = bs_assets[i][0] if i < len(bs_assets) else ""
            a_amt = f"{bs_assets[i][1]:,.2f}" if i < len(bs_assets) else ""
            bs_table_rows.append([l_part, l_amt, a_part, a_amt])
        bs_table_rows.append(["TOTAL", f"{liab_total:,.2f}", "TOTAL", f"{asset_total:,.2f}"])
        local_structured_tables.append({
            "type": "balance_sheet",
            "title": "Balance Sheet",
            "headers": ["Liabilities & Equity", "Amount (\u20b9)", "Assets", "Amount (\u20b9)"],
            "rows": bs_table_rows
        })
        print(f"[AGENT-5] Balance Sheet done → Assets: ₹{asset_total:,.2f} | Liabilities+Equity: ₹{liab_total:,.2f}")

        # Filter local_structured_tables to only expose what the user asked for
        requested_tables = [t for t in local_structured_tables if t.get('type') in _show_tables]
        print(f"[INTENT] Exposing {len(requested_tables)} table(s): {[t.get('type') for t in requested_tables]}")

        # Yield structured tables directly over the stream.
        # 'structured_tables' is what gets shown to the user this turn.
        # 'all_tables' is cached in the DB for instant retrieval on follow-up questions.
        yield {"structured_tables": requested_tables, "all_tables": local_structured_tables}

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
        if TALLY_CONFIG.get('trial_balance_check'):
            if tb_diff > 0.01:
                print(f"[AGENT-6] Trial Balance MISMATCH: Dr=\u20b9{tb_total_dr:,.2f} Cr=\u20b9{tb_total_cr:,.2f} Diff=\u20b9{tb_diff:,.2f}")
                yield f"\n> \u26a0\ufe0f **Trial Balance FAILED**: Dr \u20b9{tb_total_dr:,.2f} \u2260 Cr \u20b9{tb_total_cr:,.2f} (diff=\u20b9{tb_diff:,.2f})\n"
            else:
                print(f"[AGENT-6] Trial Balance PASSED: \u20b9{tb_total_dr:,.2f}")
                yield f"\n> \u2705 **Trial Balance PASSED** \u2014 Total Dr = Total Cr = \u20b9{tb_total_dr:,.2f}\n"

        # --- CHECK 2: BALANCE SHEET EQUATION ---
        diff = abs(asset_total - liab_total)
        if TALLY_CONFIG.get('balance_sheet_check'):
            if diff > 0.01:
                if TALLY_CONFIG.get('auto_capital_adjustment'):
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
                table_type = "profit_loss"
                
            tables.append({
                "type": table_type,
                "headers": headers,
                "rows": body
            })
        return tables

