from typing import List, Dict, Any
import json
import asyncio
from langchain_openai import ChatOpenAI
from app.core.config import settings
import random
from datetime import date
import re
from app.ai.rag.retriever import vector_store

async def retry_with_backoff(func, *args, **kwargs):
    retries = 20
    for i in range(retries):
        try:
            return await func(*args, **kwargs)
        except Exception as e:
            if "429" in str(e) or "Rate limit" in str(e):
                if i == retries - 1:
                    raise e

                wait_time = 10 + random.uniform(1, 3)
                print(f"[WARNING] Rate limit hit. Waiting {wait_time:.1f}s to reset quota... (Attempt {i+1}/{retries})")
                await asyncio.sleep(wait_time)
            else:
                raise e

class LLMService:
    """Service for generating answers using active LLM provider."""

    def __init__(self):
        provider = settings.active_provider
        model = settings.active_model
        api_key = settings.active_api_key or "not-needed"
        base_url = settings.active_base_url

        self.llm = ChatOpenAI(
            model=model,
            openai_api_key=api_key,
            openai_api_base=base_url,
            temperature=0.0,
            max_tokens=50000,
            default_headers={
                "HTTP-Referer": "https://localhost:8000",
                "X-Title": "Fineyukt AI Accounting System",
            }
        )
        print(f"[INFO] LLMService initialized with Provider='{provider}' Model='{model}' at {base_url}")

    async def classify_query_intent(self, question: str) -> Dict[str, Any]:
        """
        Use the LLM to categorize the user's query intent.
        """
        from app.schemas.ai import IntentClassification
        prompt = f"""You are an intent discovery agent for a financial RAG system.
        Categorize the user's question into one of three intents:

        1. **SUMMARY**: The user is asking for a high-level overview, a balance, financial health, or a specific structured report (Balance Sheet, P&L, Statement, Computation).
           Examples: "what is my closing balance?", "give me balance sheet", "how are my finances?", "summary of account".

        2. **EXTRACTION**: The user is asking for specific transactional data points, a list of records, or has provided an explicit limit (e.g., "last 5 upi").
           Examples: "list upi transactions", "show last 10 records", "extract all atm withdrawals".

        3. **GENERAL**: General questions about the system or accounting practices that don't directly require document data extraction.
           Examples: "how do I upload?", "what is a balance sheet?".

        USER QUESTION: "{question}"
        """
        try:
            # Create a model instance bound to the schema
            structured_llm = self.llm.with_structured_output(IntentClassification)
            intent_data = await structured_llm.ainvoke(prompt)
            data_dict = intent_data.model_dump()
            print(f"[INFO] Intent Discovery: query='{question}' -> intent={data_dict.get('intent')}")
            return data_dict
        except Exception as e:
            print(f"[ERROR] Intent Discovery failed: {e}")
            return {"intent": "EXTRACTION", "explicit_limit": None, "report_types": [], "target_keywords": []}

    def _extract_json(self, text: str) -> dict:
        """
        Legacy fallback: Robustly extract and parse JSON from LLM output.
        """
        import re, json
        def clean_json_string(s):
            s = re.sub(r'```json\s*', '', s)
            s = re.sub(r'```\s*', '', s)
            return s.strip()

        text = clean_json_string(text)
        try:
            parsed = json.loads(text)
            if isinstance(parsed, list): return {"data": parsed}
            return parsed
        except json.JSONDecodeError:
            pass

        all_found = []
        decoder = json.JSONDecoder()
        for i in range(len(text)):
            if text[i] in '{[':
                try:
                    obj, _ = decoder.raw_decode(text[i:])
                    all_found.append(obj)
                except: pass

        if all_found:
            for item in all_found:
                if isinstance(item, dict) and ("transactions" in item or "mutations" in item or "data" in item or "deposits_count" in item or "report_name" in item):
                    return item
            return all_found[0] if isinstance(all_found[0], dict) else {"data": all_found[0]}
        
        return {}

    async def _extract_document_metadata(self, context_chunks: List[str], source_documents: List[str], company_id: str = None) -> Dict[str, Dict[str, str]]:
        """
        Identify the Bank Name and Account Holder for each unique document.
        IMPROVED: Explicitly fetches the first 3 chunks (header) from VectorDB to ensure we see the logo/bank name,
        instead of relying only on the random chunks retrieved for the query.
        """
        unique_docs = list(set([d for d in source_documents if d and d != "Unknown"]))

        async def scan_single_doc(doc):
            try:

                header_text = ""
                if company_id:
                    try:

                        v_res = vector_store.collection.get(
                            where={
                                "$and": [
                                    {"company_id": {"$eq": company_id}},
                                    {"document_name": {"$eq": doc}},
                                    {"chunk_index": {"$lt": 3}}
                                ]
                            },
                            include=["documents", "metadatas"]
                        )

                        if v_res and v_res.get("documents"):

                            sorted_chunks = sorted(
                                zip(v_res["documents"], v_res["metadatas"]),
                                key=lambda x: x[1].get("chunk_index", 999)
                            )
                            header_chunks = [c[0] for c in sorted_chunks]
                            header_text = "\n".join(header_chunks)
                            print(f"[INFO] Fetched {len(header_chunks)} header chunks for {doc} from DB")
                    except Exception as ve:
                        print(f"[WARNING] Failed to fetch headers for {doc} from DB: {ve}")

                all_doc_indices = [i for i, x in enumerate(source_documents) if x == doc]

                context_indices = all_doc_indices[:5]
                context_text = "\n".join([context_chunks[i] for i in context_indices])

                full_peek_content = f"--- DOCUMENT HEADER (Start of File) ---\n{header_text}\n\n--- RELEVANT EXTRACTS_FROM_QUERY ---\n{context_text}"

                prompt = f"""You are a professional financial document analyzer. Your goal is to identify the Issuing Bank Name, the Main Account Holder Name, and the DOCUMENT TYPE.

                I have provided two sections:
                1. THE HEADER (Start of file) - Look here for Bank Name/Logo.
                2. RELEVANT EXTRACTS - Look here for context if header is unclear.

                STRATEGY:
                1. **BANK NAME IDENTIFICATION (CRITICAL)**:
                   - Look for the official bank name in the **HEADER SECTION**.
                   - Check for bank logos, letterheads, or explicit "Bank Statement" titles at the top.
                   - **CRITICAL**: Ignore bank names found inside transaction lists (e.g. "Transfer to SBI"). Only identify the ISSUING bank.
                   - If header is empty/unclear, look at the extracts.
                   - **CRITICAL**: If you cannot find an explicit issuing bank name, return "Unknown".

                2. Identify Account Holder Name (e.g., "Name of the Assessee", "Beneficiary Name:", "Account Name:").

                3. IDENTIFY DOCUMENT TYPE:
                   - **Bank Statement**: Transaction lists, withdrawals, deposits.
                   - **Balance Sheet**: Assets, Liabilities, Equity.
                   - **Profit & Loss**: Income, Expenses, Net Profit.
                   - **Computation of Income**: Tax calculations, detailed income breakdown.

                SNIPPET:
                {full_peek_content[:15000]}
                ---

                **RULES FOR BANK NAME**:
                - Extract the **Full Official Name** (e.g. "State Bank of India", "HDFC Bank").
                - Do not abbreviate.
                """
                from app.schemas.ai import DocumentMetadataExtraction
                structured_llm = self.llm.with_structured_output(DocumentMetadataExtraction)
                meta_data = await structured_llm.ainvoke(prompt)
                meta = meta_data.model_dump()

                bank_name = str(meta.get("bank_name", "Unknown")).strip()
                account_holder = meta.get("account_holder", "Unknown")
                doc_type = meta.get("document_type", "Unknown")

                return doc, {"bank_name": bank_name, "account_holder": account_holder, "document_type": doc_type}
            except Exception as e:
                print(f"[ERROR] Failed to extract metadata for {doc}: {e}")
                return doc, {"bank_name": "Unknown", "account_holder": "Unknown"}

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
        doc_metadata: Dict[str, Dict[str, str]] = None,
        company_id: str = None,
        skip_conversational_analysis: bool = False
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

        if doc_metadata is None:
            doc_metadata = await self._extract_document_metadata(context_chunks, source_documents, company_id=company_id)

        context_parts = []
        for i, chunk in enumerate(context_chunks):
            doc_name = source_documents[i] if i < len(source_documents) else "Unknown"

            meta_str = ""
            if doc_metadata and doc_name in doc_metadata:
                m = doc_metadata[doc_name]
                meta_str = f" [TYPE: {m.get('document_type', 'Unknown')}, STATEMENT_BANK: {m['bank_name']}, HOLDER: {m['account_holder']}]"

            context_parts.append(f"[Context {i+1} - Source: {doc_name}{meta_str}]\n{chunk}")

        context = "\n\n".join(context_parts)

        today = date.today()
        current_date_str = today.strftime("%d-%m-%Y")
        current_year = today.year

        feedback_context = ""
        try:
            if company_id:

                from app.services.deps import embedding_service
                q_emb = embedding_service.generate_embedding(question)

                relevant_feedback = vector_store.query_feedback(q_emb, company_id)
                if relevant_feedback:
                    feedback_context = "\n**LESSONS LEARNED (PAST USER CORRECTIONS):**\n"
                    for i, fb in enumerate(relevant_feedback):
                        feedback_context += f"- {fb}\n"
                    print(f"[INFO] Injected {len(relevant_feedback)} past corrections into prompt for company {company_id}")
            else:
                print(f"[WARNING] No company_id provided to generate_answer; skipping feedback retrieval.")
        except Exception as e:
            print(f"[WARNING] Failed to retrieve feedback: {e}")
            feedback_context = ""

        from app.schemas.ai import ExtractedTransactions

        extraction_prompt = f"""You are an expert financial assistant.
**CURRENT SYSTEM DATE:** {current_date_str}

**USER QUERY:** "{question}"

{feedback_context}

**DOCUMENT CONTEXT:**
{context}

**CRITICAL INSTRUCTIONS:**
Extract ALL transactions from the provided context that match the user's query.
- For summary reports (Balance Sheet/P&L), extract the structural tables as transactions where 'narration' is the row particular and 'amount' is the value.
- If looking for specific transactions (e.g. "Cash"), strictly filter out non-matching rows.
- **CREDITS ARE MONEY COMING IN. DEBITS ARE MONEY OUT.**
- **MANDATORY DIRECTION CHECK:** Compare the running balance with the previous line's balance. If balance increased, it is a CREDIT. If balance decreased, it is a DEBIT.

Extract the data strictly adhering to the schema.
"""
        try:
            print(f"[INFO] Generating answer for question: {question[:100]}...")
            
            structured_llm = self.llm.with_structured_output(ExtractedTransactions)
            extracted_data = await structured_llm.ainvoke(extraction_prompt)
            if extracted_data is None:
                parsed_json = {"row_count_detected": 0, "transactions": [], "message": "No transactions found."}
            elif hasattr(extracted_data, 'model_dump'):
                parsed_json = extracted_data.model_dump()
            elif isinstance(extracted_data, dict):
                parsed_json = extracted_data
            else:
                parsed_json = {"row_count_detected": 0, "transactions": [], "message": "Failed to parse transactions."}

            for tx in parsed_json.get("transactions", []):
                narr = str(tx.get('narration') or '').strip()
                desc = str(tx.get('description') or '').strip()
                if narr and (not desc or desc.upper() in ['DR', 'CR', 'DEBIT', 'CREDIT', 'N/A', 'NONE'] or len(desc) < len(narr)):
                    tx['description'] = narr
                elif desc and not narr:
                    tx['narration'] = desc

                debit = float(tx.get('debit') or 0.0)
                credit = float(tx.get('credit') or 0.0)
                amt = float(tx.get('amount') or 0.0)

                if credit > 0:
                    tx['direction'] = 'CREDIT'
                    tx['amount'] = credit
                elif debit > 0:
                    tx['direction'] = 'DEBIT'
                    tx['amount'] = debit
                elif amt > 0:
                    tx['amount'] = amt
                    if not tx.get('direction'):
                        tx['direction'] = 'DEBIT'
            
            if skip_conversational_analysis:
                tx_count = len(parsed_json.get("transactions", []))
                human_answer = f"Extracted {tx_count} records." if tx_count else ""
            else:
                analysis_prompt = f"""
You are an expert financial analyst. 
The user asked: "{question}"
Here is the extracted data from the bank documents:
{json.dumps(parsed_json, indent=2)}

Please provide a detailed, human-readable analysis of this data.
- Use multiple bullet points.
- Start every key fact with a dash.
- Use plain English instead of technical jargon.
- DO NOT return JSON. Just return the analysis.
"""
                analysis_response = await self.llm.ainvoke(analysis_prompt)
                human_answer = analysis_response.content.strip()

            if not human_answer and "message" in parsed_json and parsed_json["message"]:
                human_answer = parsed_json["message"]
            elif not human_answer and "transactions" in parsed_json and parsed_json["transactions"]:
                human_answer = f"I found {len(parsed_json['transactions'])} matching records."
            elif not human_answer:
                human_answer = "I could not find the information you requested in the uploaded documents."

            sources = list(set(source_documents))

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
        batch_size: int = 3,
        company_id: str = None
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

        doc_metadata = await self._extract_document_metadata(context_chunks, source_documents, company_id=company_id)

        intent_data = await self.classify_query_intent(question)
        intent = intent_data.get("intent")
        is_summary = (intent == "SUMMARY")

        if is_summary and doc_metadata:
            wanted_types = set(intent_data.get("report_types", []))

            def _doc_matches(doc_name: str) -> bool:
                meta = doc_metadata.get(doc_name) or {}
                doc_type = str(meta.get("document_type", "")).lower()
                name = (doc_name or "").lower()

                if wanted_types and any(w in doc_type for w in wanted_types):
                    return True

                name_hints = ["balance", "bl.", "bl_", "-bl", "bs.", "p&l", "pl.", "pl_", "-pl", "profit", "loss", "computation", "statement"]
                if any(h in name for h in name_hints):
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

            exclude_types = ["computation", "balance sheet", "p&l", "profit", "loss", "tax", "computation of income"]
            skip_docs = {d for d, m in doc_metadata.items() if any(et in str(m.get("document_type", "")).lower() for et in exclude_types) or any(et in (d or "").lower() for et in ["cp.pdf", "computation"])}

            if skip_docs:
                filtered = [(c, d) for c, d in zip(context_chunks, source_documents) if d not in skip_docs]
                if filtered:
                    context_chunks = [c for c, _ in filtered]
                    source_documents = [d for _, d in filtered]
                    print(f"[INFO] Transaction search focus (sync): excluded {len(skip_docs)} non-transaction documents: {list(skip_docs)}")
                else:
                    print(f"[WARNING] Transaction search focus (sync): All {len(skip_docs)} documents were filtered out. Using original context to avoid empty results.")

        async def process_batch_with_sem(batch, batch_source_docs):
            async with sem:

                for attempt in range(2):
                    try:
                        return await retry_with_backoff(self.generate_answer, question, batch, batch_source_docs, doc_metadata, company_id)
                    except Exception as e:
                        if attempt == 1: raise e
                        print(f"[WARNING] Batch failed, retrying once... Error: {e}")
                        await asyncio.sleep(2)

        tasks = []

        step = batch_size - 1 if batch_size > 1 else 1

        total_batches = (len(context_chunks) + step - 1) // step
        print(f"[INFO] Splitting {len(context_chunks)} chunks into ~{total_batches} batches with overlap (step={step}, batch_size={batch_size})")

        batch_idx = 0
        for i in range(0, len(context_chunks), step):
            batch = context_chunks[i:i + batch_size]
            batch_source_docs = source_documents[i:i + batch_size]
            batch_idx += 1
            print(f"[INFO] Batch {batch_idx}: Processing {len(batch)} chunks (start_idx={i})")
            tasks.append(process_batch_with_sem(batch, batch_source_docs))

        results = await asyncio.gather(*tasks)

        full_answer = ""
        all_transactions = []
        seen_fingerprints = set()

        seen_answers = set()
        for result in results:

            ans = (result.get('full_answer') or result['answer']).strip()

            is_empty_msg = '{"transactions": [], "message":' in ans or '{"transactions": []}' in ans
            if is_empty_msg:
                if not full_answer.strip():
                    full_answer = ans
            else:

                if '{"transactions": [], "message":' in full_answer:
                    full_answer = ans
                else:
                    full_answer += ans + "\n---\n"

            try:
                content = json.loads(result['answer'])

                txs = []
                if "transactions" in content: txs = content["transactions"]
                elif "data" in content: txs = content["data"]

                elif not txs:
                     for val in content.values():
                        if isinstance(val, list) and val:
                            txs = val
                            break

                if txs:
                    for t in txs:

                        desc = str(t.get('description', '')).strip().lower()
                        def _norm(val: str) -> str:
                            try: return f"{float(val.replace(',', '').strip()):.2f}"
                            except: return val.replace(',', '').strip()

                        amt = _norm(str(t.get('amount', '0')))
                        balance = _norm(str(t.get('balance', '')))
                        date = str(t.get('date', '')).strip()
                        direction = str(t.get('direction', '')).upper().strip()
                        tx_id = str(t.get('transaction_id', '')).strip()

                        if tx_id:
                            fp = tx_id
                        elif balance and balance != '0.00':
                            fp = f"{date}|{amt}|{direction}|{balance}"
                        else:
                            fp = f"{date}|{amt}|{direction}"

                        if fp not in seen_fingerprints:
                            seen_fingerprints.add(fp)
                            all_transactions.append(t)
                        else:
                            print(f"[DEBUG] Skipping duplicate transaction in sync exhaustive: {fp[:50]}...")

            except Exception as e:
                print(f"[DEBUG] Error merging batch result: {str(e)}")

        valid_source_docs = set(source_documents)
        all_sources = set()
        for tx in all_transactions:
            source_doc = str(tx.get('source_document', '')).strip()
            if source_doc and source_doc in valid_source_docs:
                all_sources.add(source_doc)

        if not all_sources and not is_summary:
            all_sources = set([d for d in source_documents if d and d.lower() != 'unknown'])

        if is_summary and not all_sources:
            all_sources = set([d for d in source_documents if d and d.lower() != "unknown"])

        credit_count = sum(1 for tx in all_transactions if str(tx.get('direction', '')).upper() == 'CREDIT')
        debit_count = sum(1 for tx in all_transactions if str(tx.get('direction', '')).upper() == 'DEBIT')
        print(f"[INFO] Exhaustive extraction complete: {len(all_transactions)} total unique records ({credit_count} CREDITS, {debit_count} DEBITS)")
        print(f"[INFO] Source documents with transactions: {len(all_sources)} documents")

        total_before_dedup = sum(len(json.loads(r['answer']).get('transactions', [])) for r in results)
        print(f"[INFO] Total transactions before deduplication: {total_before_dedup}, after: {len(all_transactions)}")

        if len(all_transactions) > 10 and credit_count == 0:
            print(f"[WARNING] Large dataset ({len(all_transactions)} transactions) with zero credits. This may indicate credit extraction issues.")
        elif len(all_transactions) > 0 and credit_count == 0:
            print(f"[WARNING] No CREDIT transactions found in final results. Please verify credit identification logic.")
        elif credit_count > 0:
            credit_percentage = (credit_count / len(all_transactions)) * 100
            print(f"[INFO] Credit transactions: {credit_count} ({credit_percentage:.1f}% of total)")

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

        CONTEXT_CAP = 250
        if len(context_chunks) > CONTEXT_CAP:
            print(f"[WARNING] generate_summary_report: Capping context from {len(context_chunks)} to {CONTEXT_CAP} chunks for safety.")
            context_chunks = context_chunks[:CONTEXT_CAP]
            source_documents = source_documents[:CONTEXT_CAP]

        combined_context = "\n\n".join(context_chunks)
        sources = sorted(set([d for d in source_documents or [] if d and d.lower() != "unknown"]))

        prompt = f"""You are a senior financial analyst. Your task is to provide a helpful, professional, and comprehensive summary of the provided documents, or to extract a specific financial report if asked.

**USER REQUEST:** "{question}"

**DOCUMENT CONTEXT:**
{combined_context}

**INSTRUCTIONS**:
1. **HELPFUL & DETAILED ANALYST (CRITICAL)**: Act as a senior financial analyst. Provide a **comprehensive and detailed summary** of the documents. Do not be brief. Explain what the numbers mean, identify major categories, and provide a full picture of the financial state.
2. **SIMPLE LANGUAGE, DEEP INSIGHT**: Answer in **simple English**, but provide **deep detail**. **DO NOT** use difficult financial words like "liabilities" or "assets" where regular words work better (e.g., use "money you owe" or "things you own").
3. **MANDATORY BULLET POINTS**: Every important detail, fact, observation, and figure **MUST start with a dash and space** (e.g., `- The total revenue recorded is **\u20b95,00,000**`). Use multiple bullet points to organize the information clearly.
4. **CATEGORIZATION RULES**:
   - **MONEY YOU OWE (LIABILITIES)**: Capital, Loans, Sundry Creditors, Provisions, Outstanding Expenses.
   - **THINGS YOU OWN (ASSETS)**: Fixed Assets, Sundry Debtors, Bank Balance, Cash, Deposits, Prepaid Expenses.
5. **Accuracy is Critical**: Preserve exact names and amounts as seen in the documents.
6. **Output Format (MANDATORY)**:
   - **PART 1 (STRUCTURED)**: AT THE VERY BEGINNING, provide the structured data in a strict JSON code block:
     ```json
     {{
       "balance_sheet": {{ "liabilities": [...], "assets": [...] }},
       "p_and_l": {{ "income": [...], "expenses": [...] }},
       "summary_metadata": {{ "account_name": "...", "account_number": "...", "period": "..." }}
     }}
     ```
   - **PART 2 (CONVERSATIONAL)**: AFTER the JSON, provide your **detailed natural language answer**. This should be a thorough explanation of the findings. Use **MANDATORY bullet points** (`- `) for every category and major detail. Use standard markdown bolding (**text**) for all numbers.

**RESPONSE**:
"""
        try:
            print(f"[INFO] Generating LLM-based summary report for: {question[:50]}...")
            response = await self.llm.ainvoke(prompt)
            answer = response.content.strip() if response.content else ""
            print(f"[DEBUG] [Received from AI] Response length: {len(answer)} chars")
            print(f"[DEBUG] [Raw AI Response Preview]: {answer[:500]}...")

            if not answer:
                print(f"[WARNING] LLM returned empty response for summary report. Retrying with smaller context...")

                if len(context_chunks) > 100:
                    context_chunks = context_chunks[:100]
                    combined_context = "\n\n".join(context_chunks)

                    response = await self.llm.ainvoke(prompt)
                    answer = response.content.strip() if response.content else ""
                    print(f"[DEBUG] [Retry] Received {len(answer)} chars")

            parsed_json = self._extract_json(answer)

            human_answer = answer

            human_answer = re.sub(r'```(?:json)?\s*[\{\[][\s\S]*?[\}\]]\s*```', '', human_answer, flags=re.DOTALL)

            human_answer = re.sub(r'(?m)^[\{\[].*?[\}\]]$', '', human_answer, flags=re.DOTALL)

            human_lines = []
            for line in human_answer.split('\n'):
                line_strip = line.strip()

                if not line_strip:
                    human_lines.append(line)
                    continue

                if line_strip in ['{', '}', '[', ']', '},', '],']:
                    continue

                if re.match(r'^\s*"\w+"\s*:\s*.*?,?\s*$', line_strip):
                    continue

                if (line_strip.startswith('{') and line_strip.endswith('}')) or \
                   (line_strip.startswith('[') and line_strip.endswith(']')):
                    try:
                        json.loads(line_strip)
                        continue
                    except:
                        pass
                human_lines.append(line)
            human_answer = "\n".join(human_lines).strip()

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

        if any(kw in q_lower for kw in ["cash", "atm", "self", "withdrawal", "wdl"]):

            if any(kw in desc for kw in ["upi", "vpa", "@", "paytm", "g-pay", "phonepe", "neft", "imps", "rtgs"]):
                return True
            if any(kw in t_type for kw in ["upi", "neft", "imps", "rtgs"]):
                return True

        if any(kw in q_lower for kw in ["upi", "vpa"]):

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
        batch_size: int = 3,
        company_id: str = None
    ):
        """
        Stream extraction results as they are processed.
        """

        if not context_chunks:
            yield json.dumps({"type": "error", "message": "No context provided"})
            return

        doc_metadata = await self._extract_document_metadata(context_chunks, source_documents, company_id=company_id)

        intent_data = await self.classify_query_intent(question)
        intent = intent_data.get("intent")
        is_summary = (intent == "SUMMARY")

        filtered_sources_for_summary = None
        if is_summary and doc_metadata:
            wanted_types = set(intent_data.get("report_types", []))

            def _doc_matches(doc_name: str) -> bool:
                meta = doc_metadata.get(doc_name) or {}
                doc_type = str(meta.get("document_type", "")).lower()
                name = (doc_name or "").lower()

                if wanted_types and any(w in doc_type for w in wanted_types):
                    return True

                name_hints = ["balance", "bl.", "bl_", "-bl", "bs.", "p&l", "pl.", "pl_", "-pl", "profit", "loss", "computation", "statement"]
                if any(h in name for h in name_hints):
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

            exclude_types = ["computation", "balance sheet", "p&l", "profit", "loss", "tax", "computation of income"]
            skip_docs = {d for d, m in doc_metadata.items() if any(et in str(m.get("document_type", "")).lower() for et in exclude_types) or any(et in (d or "").lower() for et in ["cp.pdf", "computation"])}

            if skip_docs:
                filtered = [(c, d) for c, d in zip(context_chunks, source_documents) if d not in skip_docs]
                if filtered:
                    context_chunks = [c for c, _ in filtered]
                    source_documents = [d for _, d in filtered]
                    print(f"[INFO] Transaction search focus: excluded {len(skip_docs)} non-transaction documents: {list(skip_docs)}")
                else:
                    print(f"[WARNING] Transaction search focus: All {len(skip_docs)} documents were filtered out. Using original context to avoid empty results.")

        step = batch_size - 1 if batch_size > 1 else 1
        total_batches = (len(context_chunks) + step - 1) // step
        print(f"[INFO] Splitting {len(context_chunks)} chunks into ~{total_batches} batches with overlap (step={step}, batch_size={batch_size})")

        tasks = []
        sem = asyncio.Semaphore(4)

        async def process_batch_with_sem(batch, batch_source_docs):
            async with sem:

                for attempt in range(3):
                    try:
                        return await retry_with_backoff(self.generate_answer, question, batch, batch_source_docs, doc_metadata, company_id, skip_conversational_analysis=True)
                    except Exception as e:
                        if attempt == 2:
                            print(f"[ERROR] Batch failed after 3 attempts. Last error: {e}")
                            raise e
                        print(f"[WARNING] Stream batch failed (Attempt {attempt+1}/3), retrying... Error: {e}")
                        await asyncio.sleep(2 * (attempt + 1))

        print(f"[INFO] Streaming exhaustive extraction: {len(context_chunks)} chunks")

        running_tasks = []
        batch_idx = 0
        for i in range(0, len(context_chunks), step):
            batch = context_chunks[i:i + batch_size]
            batch_source_docs = source_documents[i:i + batch_size]
            batch_idx += 1
            print(f"[INFO] Stream Batch {batch_idx}: Processing {len(batch)} chunks (start_idx={i})")

            task = asyncio.create_task(process_batch_with_sem(batch, batch_source_docs))
            running_tasks.append(task)

        completed_count = 0
        all_transactions_count = 0
        all_credit_count = 0
        all_debit_count = 0
        all_sources = set()
        seen_fingerprints = set()
        all_unique_transactions = []
        full_answers = []

        try:

            for task in asyncio.as_completed(running_tasks):
                try:
                    result = await task
                    completed_count += 1

                    yield json.dumps({
                        "type": "progress",
                        "completed": completed_count,
                        "total": total_batches,
                        "percent": int((completed_count / total_batches) * 100)
                    })

                    if result.get('full_answer'):
                        ans = result['full_answer'].strip()

                        is_bad = any(kw in ans.lower() for kw in ["not find", "no transactions", "not available"]) or ans.startswith('{') or ans.startswith('[')

                        if is_bad:

                            if all_transactions_count == 0 and not full_answers:
                                full_answers.append(ans)
                        else:

                            full_answers = [a for a in full_answers if not (any(kw in a.lower() for kw in ["not find", "no transactions", "not available"]) or a.startswith('{') or a.startswith('['))]
                            if ans not in full_answers:
                                full_answers.append(ans)

                    try:
                        content = json.loads(result['answer'])

                        txs = []
                        if content.get('transactions'): txs = content['transactions']
                        elif content.get('data'): txs = content['data']

                        elif not txs:
                             for val in content.values():
                                if isinstance(val, list) and val:
                                    txs = val
                                    break

                        if txs:
                            unique_txs = []
                            for t in txs:
                                # Ensure description and narration are synced
                                narr = str(t.get('narration') or '').strip()
                                desc_raw = str(t.get('description') or '').strip()
                                if narr and (not desc_raw or desc_raw.upper() in ['DR', 'CR', 'DEBIT', 'CREDIT', 'N/A', 'NONE'] or len(desc_raw) < len(narr)):
                                    t['description'] = narr
                                elif desc_raw and not narr:
                                    t['narration'] = desc_raw

                                # Ensure debit, credit, amount, direction are synced
                                debit = float(t.get('debit') or 0.0)
                                credit = float(t.get('credit') or 0.0)
                                raw_amt = float(t.get('amount') or 0.0)

                                if credit > 0:
                                    t['direction'] = 'CREDIT'
                                    t['amount'] = credit
                                elif debit > 0:
                                    t['direction'] = 'DEBIT'
                                    t['amount'] = debit
                                elif raw_amt > 0:
                                    t['amount'] = raw_amt
                                    if not t.get('direction'):
                                        t['direction'] = 'DEBIT'

                                desc = str(t.get('description', '')).strip().lower()
                                def _norm(val: str) -> str:
                                    try: return f"{float(val.replace(',', '').strip()):.2f}"
                                    except: return val.replace(',', '').strip()

                                amt = _norm(str(t.get('amount', '0')))
                                balance = _norm(str(t.get('balance', '')))
                                date = str(t.get('date', '')).strip()
                                direction = str(t.get('direction', '')).upper().strip()
                                tx_id = str(t.get('transaction_id', '')).strip()

                                if tx_id:
                                    fp = tx_id
                                elif balance and balance != '0.00':
                                    fp = f"{date}|{amt}|{direction}|{balance}"
                                else:
                                    fp = f"{date}|{amt}|{direction}"

                                if fp not in seen_fingerprints:

                                    if self._should_filter_transaction(t, question):
                                        print(f"[FILTER] Dropped non-matching transaction: {desc[:50]}")
                                        continue

                                    seen_fingerprints.add(fp)
                                    unique_txs.append(t)
                                    all_unique_transactions.append(t)
                                else:

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

            cancelled_count = 0
            for task in running_tasks:
                if not task.done():
                    task.cancel()
                    cancelled_count += 1
            if cancelled_count > 0:
                print(f"[INFO] Cancelled {cancelled_count} pending background tasks.")

        valid_source_docs = set(source_documents)
        for tx in all_unique_transactions:
            source_doc = str(tx.get('source_document', '')).strip()
            if source_doc and source_doc in valid_source_docs:
                all_sources.add(source_doc)

        if not all_sources and not is_summary:
            all_sources = set([d for d in source_documents if d and d.lower() != 'unknown'])

        if is_summary and not all_sources:
            if filtered_sources_for_summary is not None:

                all_sources = set(filtered_sources_for_summary)
            else:

                all_sources = set([d for d in source_documents if d and d.lower() != "unknown"])

        final_full_answer = "\n---\n".join(full_answers).strip()

        if all_transactions_count > 0:
            not_found_patterns = [
                "I could not find the requested information in the documents.",
                "No transactions found",
                "Information not available in uploaded records"
            ]
            for pat in not_found_patterns:
                final_full_answer = final_full_answer.replace(pat, "")
            final_full_answer = final_full_answer.replace('{"transactions": []}', "").strip()
            if not final_full_answer:
                final_full_answer = "I could not find the information you requested in the uploaded documents."

        print(f"[INFO] Streaming extraction complete: {all_transactions_count} total ({all_credit_count} CREDITS, {all_debit_count} DEBITS)")
        yield json.dumps({
            "type": "summary",
            "total_transactions": all_transactions_count,
            "total_credits": all_credit_count,
            "total_debits": all_debit_count,
            "sources": list(all_sources),
            "full_answer": final_full_answer
        })
