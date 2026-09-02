from pydantic import BaseModel, Field
from typing import List, Optional, Literal, Any, Dict

class IntentClassification(BaseModel):
    intent: Literal["SUMMARY", "EXTRACTION", "GENERAL", "REFINEMENT", "ANALYSIS"] = Field(
        description="The categorized intent of the user query."
    )
    explicit_limit: Optional[int] = Field(
        None, description="Explicit limit mentioned by user (e.g. 'last 5')."
    )
    report_types: List[str] = Field(
        default_factory=list, description="Requested reports (balance_sheet, p_and_l, etc.)."
    )
    target_keywords: List[str] = Field(
        default_factory=list, description="Specific target keywords mentioned in the query."
    )

class DocumentMetadataExtraction(BaseModel):
    bank_name: str = Field(description="The full official Issuing Bank Name found in the document header. 'Unknown' if not found.")
    account_holder: str = Field(description="The account holder name. 'Unknown' if not found.")
    document_type: str = Field(description="Type of document (Bank Statement, Balance Sheet, etc.). 'Unknown' if not found.")

class TransactionData(BaseModel):
    date: str = Field(description="Transaction date in DD/MM/YYYY format")
    narration: str = Field(description="Transaction description or narration")
    description: Optional[str] = Field(None, description="Alias for narration / description")
    debit: float = Field(default=0.0, description="Amount debited/withdrawn")
    credit: float = Field(default=0.0, description="Amount credited/deposited")
    amount: Optional[float] = Field(default=0.0, description="Transaction amount")
    direction: Optional[str] = Field(default=None, description="Transaction direction: CREDIT or DEBIT")
    balance: float = Field(default=0.0, description="Running balance. 0.0 if not available")
    transaction_id: Optional[str] = Field(None, description="Reference number or UPI ID")
    type: Optional[str] = Field(None, description="Transaction type (UPI, NEFT, CASH, etc)")
    bank_name: str = Field(default="Unknown", description="Issuing bank name")
    source_document: str = Field(default="Unknown", description="Source document name")

class ExtractedTransactions(BaseModel):
    row_count_detected: int = Field(description="Number of transaction rows counted in the text")
    transactions: List[TransactionData] = Field(description="List of extracted transactions")
    message: Optional[str] = Field(None, description="Any explanatory message if extraction fails or finds nothing")

class AnswerGenerationResult(BaseModel):
    answer_data: ExtractedTransactions = Field(description="The raw extracted structured data")
    full_answer: str = Field(description="A rich, detailed human-readable explanation of the data")

class MutationAction(BaseModel):
    type: Literal["UPDATE_CATEGORY", "ADD_ENTRY", "DELETE_ENTRY", "MERGE_ENTRIES", "SPLIT_ENTRY", "BULK_UPDATE"]
    target_index: Optional[int] = None
    target_indices: Optional[List[int]] = None
    new_category: Optional[str] = None
    new_date: Optional[str] = None
    new_narration: Optional[str] = None
    new_debit: Optional[float] = None
    new_credit: Optional[float] = None
    split_amounts: Optional[List[float]] = None
    reasoning: str = Field(description="Reason for this mutation")

class MutationPlan(BaseModel):
    reasoning: str = Field(description="High-level reasoning for the entire plan")
    mutations: List[MutationAction] = Field(default_factory=list, description="List of mutations to apply")

class ReportRow(BaseModel):
    particulars: str
    amount: float

class SummaryData(BaseModel):
    report_name: str
    rows: List[ReportRow]
    total: float

class PNLData(BaseModel):
    income: List[ReportRow]
    expenses: List[ReportRow]
    net_profit: float
    total_income: float
    total_expense: float

class CapData(BaseModel):
    opening_balance: float
    net_profit_added: float
    drawings: float
    additional_capital: float
    closing_capital: float

class BSData(BaseModel):
    liabilities: List[ReportRow]
    assets: List[ReportRow]
    total_liabilities: float
    total_assets: float
