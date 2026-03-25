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
            max_tokens=50000,
            default_headers={
                "HTTP-Referer": "https://localhost:8000",
                "X-Title": "Accounting RAG System",
            }
        )
        print(f"[INFO] LLMService initialized with OpenRouter model={settings.openrouter_model}")

    async def classify_query_intent(self, question: str) -> Dict[str, Any]:
        """
        Use the LLM to categorize the user's query intent.
        Returns: {
            "intent": "SUMMARY" | "EXTRACTION" | "GENERAL",
            "explicit_limit": int | None,
            "report_types": List[str] | None,
            "target_keywords": List[str] | None
        }
        """
        prompt = f"""You are an intent discovery agent for a financial RAG system.
        Categorize the user's question into one of three intents:

        1. **SUMMARY**: The user is asking for a high-level overview, a balance, financial health, or a specific structured report (Balance Sheet, P&L, Statement, Computation).
           Examples: "what is my closing balance?", "give me balance sheet", "how are my finances?", "summary of account".

        2. **EXTRACTION**: The user is asking for specific transactional data points, a list of records, or has provided an explicit limit (e.g., "last 5 upi").
           Examples: "list upi transactions", "show last 10 records", "extract all atm withdrawals".

        3. **GENERAL**: General questions about the system or accounting practices that don't directly require document data extraction.
           Examples: "how do I upload?", "what is a balance sheet?".

        USER QUESTION: "{question}"

        Return ONLY a JSON object:
        {{
            "intent": "SUMMARY" | "EXTRACTION" | "GENERAL",
            "explicit_limit": int_or_null,
            "report_types": ["balance_sheet", "p_and_l", "computation", "statement", "general_summary"],
            "target_keywords": ["keyword1", "keyword2"]
        }}
        """
        try:
            response = await self.llm.ainvoke(prompt)
            intent_data = self._extract_json(response.content)
            print(f"[INFO] Intent Discovery: query='{question}' -> intent={intent_data.get('intent')}")
            return intent_data
        except Exception as e:
            print(f"[ERROR] Intent Discovery failed: {e}")

            return {"intent": "EXTRACTION", "explicit_limit": None, "report_types": [], "target_keywords": []}

    def _extract_json(self, text: str) -> Dict[str, Any]:
        """
        Robustly extract and parse JSON from LLM output.
        Attempts to find the largest valid JSON object or array in the text.
        """

        def clean_json_string(s):

            s = re.sub(r'```json\s*', '', s)
            s = re.sub(r'```\s*', '', s)
            return s.strip()

        text = clean_json_string(text)

        try:
            parsed = json.loads(text)
            print(f"[DEBUG] JSON parsed successfully. Type: {type(parsed)}")

            if isinstance(parsed, list):
                print(f"[DEBUG] Parsed as list with {len(parsed)} items")
                return {"transactions": parsed}
            elif isinstance(parsed, dict):

                transaction_indicators = ['date', 'amount', 'description', 'transaction_id', 'type']
                if any(key in parsed for key in transaction_indicators) and 'transactions' not in parsed:
                    print(f"[DEBUG] Detected single transaction object, wrapping in array")
                    return {"transactions": [parsed]}
                print(f"[DEBUG] Parsed as dict with keys: {list(parsed.keys())}")
                return parsed
        except json.JSONDecodeError as e:
            print(f"[DEBUG] Direct JSON parse failed: {e}")

        all_found = []
        decoder = json.JSONDecoder()

        for i in range(len(text)):
            if text[i] in '{[':
                try:
                    obj, end = decoder.raw_decode(text[i:])
                    all_found.append(obj)
                except (json.JSONDecodeError, ValueError):
                    pass

        print(f"[DEBUG] Iterative search found {len(all_found)} JSON objects")

        if not all_found:

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

        if all_found:
            print(f"[DEBUG] Processing {len(all_found)} found objects")

            for item in all_found:
                if isinstance(item, dict):
                    if "transactions" in item or "data" in item:
                        print(f"[DEBUG] Found dict with transactions/data key")
                        return item

            transaction_objects = []
            for item in all_found:
                if isinstance(item, dict):
                    transaction_indicators = ['date', 'amount', 'description', 'transaction_id', 'type']
                    if any(key in item for key in transaction_indicators):
                        transaction_objects.append(item)

            if transaction_objects:
                print(f"[DEBUG] Found {len(transaction_objects)} transaction objects, wrapping in array")
                return {"transactions": transaction_objects}

            for item in all_found:
                if isinstance(item, list):
                    print(f"[DEBUG] Found standalone list with {len(item)} items")
                    return {"transactions": item}

            print(f"[DEBUG] Using fallback: returning first found object")
            return all_found[0] if isinstance(all_found[0], dict) else {"data": all_found[0]}

        print(f"[WARNING] No valid JSON found in response, returning empty")
        return {
            "message": text.strip()[:200],
            "transactions": []
        }

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

                Return ONLY a JSON object: {{"bank_name": "IDENTIFIED BANK NAME", "account_holder": "IDENTIFIED NAME", "document_type": "TYPE"}}

                **RULES FOR BANK NAME**:
                - Extract the **Full Official Name** (e.g. "State Bank of India", "HDFC Bank").
                - Do not abbreviate.
                """
                response = await self.llm.ainvoke(prompt)
                meta = self._extract_json(response.content)

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
        company_id: str = None
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

        prompt = f"""You are an expert financial assistant designed to analyze and extract information from uploaded bank documents such as statements, ledgers, journals, Balance Sheets, and Profit & Loss statements. Your task is to help users find specific transactions or understand financial summaries by interpreting their queries.

**CURRENT SYSTEM DATE:** {current_date_str} (Use this to resolve "this year", "previous year", "last month", etc. Previous year = {current_year - 1})

**USER QUERY:** "{question}"

{feedback_context}

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
   - **MANDATORY**: Look for keywords like "Balance Sheet", "Assets", "Liabilities", "Equity", "Profit & Loss", "p & l", "Capital Account", "Income", "Expenditure", "Statement of Affairs", "Financial Position" in the text.
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

   **DATE PARSING & FILTERING (CRITICAL):**
   - **User Input Format:** The user provides dates in **YYYY-MM-DD** format (e.g., 2024-03-01).
   - **Document Format:** Bank statements often use **DD/MM/YYYY** or **DD-MM-YYYY** (e.g., 01/03/2024).
   - **YOUR JOB:** You MUST map the user's YYYY-MM-DD request to the document's DD/MM/YYYY dates.
   - **Example:** If user asks for "2024-03-01 to 2024-03-31":
     - INCLUDE: "01/03/2024", "15/03/2024", "31/03/2024"
     - EXCLUDE: "01/01/2024" (January), "03/01/2024" (January 3rd)
   - **Ambiguity:** If a date is ambiguous (e.g., 01/02/2024 could be Jan 2nd or Feb 1st), use the context of other dates in the document to decide. Indian/UK banks use DD/MM/YYYY. US banks use MM/DD/YYYY.
   - **STRICT FILTER ADHERENCE:** Only extract transactions that fall WITHIN the requested date range.

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

4. **Output Format:**     {{
      "transactions": [
        {{
          "date": "transaction date",
          "description": "transaction description/narration (include the full text for accuracy)",
          "amount": "transaction amount (ABSORLUTELY REQUIRED: Use the non-zero numeric value. NEVER use 0.00)",
          "direction": "CREDIT or DEBIT",
          "balance": "MANDATORY: The running balance value shown on the same line as the transaction",
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

**GOAL**: Be helpful, comprehensive, and professional. Provide a **detailed and thorough analysis** of the found information. Do not give short or one-line answers. Explain the context, any trends you see, and all relevant details found in the documents.

**MANDATORY FORMATTING**:
- Your response **MUST** be structured with multiple bullet points.
- Every key fact, insight, transaction detail, or summary point **MUST start with a dash and space** (e.g., `- Detailed insight here`).
- Use standard markdown bolding (**text**) for important numbers, dates, or names.
- Use **simple, easy-to-understand English**. Avoid technical financial jargon (like "liabilities" or "assets") where possible—instead, use plain words (like "money you owe" or "things you own").
- If the user asks a question, provide a deep explanation based on the context.

**DO NOT** just return JSON; always include a **rich, detailed human-readable explanation** before the JSON. The human-readable part should feel like a complete mini-report, not just a snippet.

**RESPONSE**:
"""

        try:
            print(f"[INFO] Generating answer for question: {question[:100]}...")
            print(f"[DEBUG] Context chunks count: {len(context_chunks)}")
            print(f"[DEBUG] [Sending to AI] Input context preview: {context_chunks[0][:200] if context_chunks else 'NO CONTEXT'}...")

            response = await self.llm.ainvoke(prompt)
            print(f"[DEBUG] [Received from AI] Response length: {len(response.content)} chars")

            if not response.content or not response.content.strip():
                print(f"[WARNING] Received empty response from LLM (0 chars). Triggering retry...")
                raise ValueError("Empty response from LLM")

            answer = response.content.strip()

            parsed_json = self._extract_json(answer)

            if "message" in parsed_json and not parsed_json.get("transactions"):
                print(f"[WARNING] JSON extraction failed (Fallback triggered). Raw response preview: {answer[:100]}...")
                raise ValueError("Failed to extract valid JSON from LLM response")

            if "data" in parsed_json and "transactions" not in parsed_json:
                parsed_json["transactions"] = parsed_json.pop("data")

            if "transactions" in parsed_json:
                initial_count = len(parsed_json["transactions"])
                parsed_json["transactions"] = [tx for tx in parsed_json["transactions"] if not self._should_filter_transaction(tx, question)]
                if initial_count > len(parsed_json["transactions"]):
                    print(f"[FILTER] Standard answer: Filtered out {initial_count - len(parsed_json['transactions'])} non-matching transactions.")

            sources = list(set(source_documents))

            intent_data = await self.classify_query_intent(question)
            is_summary = (intent_data.get("intent") == "SUMMARY")

            human_answer = answer

            human_answer = re.sub(r'```(?:json)?\s*[\{\[][\s\S]*?[\}\]]\s*```', '', human_answer, flags=re.DOTALL)

            human_answer = re.sub(r'(?m)^[\{\[]\s*".*?"\s*:[\s\S]*?[\}\]]\s*$', '', human_answer, flags=re.DOTALL)

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
        sem = asyncio.Semaphore(6)

        async def process_batch_with_sem(batch, batch_source_docs):
            async with sem:

                for attempt in range(3):
                    try:
                        return await retry_with_backoff(self.generate_answer, question, batch, batch_source_docs, doc_metadata, company_id)
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
