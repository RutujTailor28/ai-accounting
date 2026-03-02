from typing import List, Dict, Any, Tuple

RULES = {
    # INCOME RULES
    "NEFT": ("Sales Income", "Income"),
    "IMPS": ("Sales Income", "Income"),
    "INTEREST": ("Interest Income", "Income"),
    "COMMISSION": ("Commission Income", "Income"),

    # EXPENSE RULES
    "ATM": ("Cash Withdrawal", "Asset"), # Treated as Asset (Cash withdraws from bank to cash on hand)
    "PETROL": ("Fuel Expense", "Expense"),
    "FUEL": ("Fuel Expense", "Expense"),
    "ELECTRICITY": ("Electricity Expense", "Expense"),
    "RECHARGE": ("Communication Expense", "Expense"),
    "CHARGES": ("Bank Charges", "Expense"),
    "GST": ("Tax Expense", "Expense"),
    "RENT": ("Rent Expense", "Expense"),
}

def classify_transaction(narration: str, amount: float, is_credit: bool = False) -> Tuple[str, str]:
    narration_upper = str(narration).upper() if narration else ""
    
    # Check Transfers FIRST
    if any(k in narration_upper for k in ["SELF", "OWN ACCOUNT", "TRANSFER", "ACCOUNT TRANSFER"]):
        return ("Transfer", "Transfer")
        
    # Check Loans
    if "LOAN" in narration_upper or "EMI" in narration_upper:
        if is_credit:
            return ("Loan Received", "Liability")
        else:
            return ("Loan Repayment", "Liability") # Reduces liability
            
    # Check Capital
    if "DEPOSIT" in narration_upper and is_credit:
        return ("Capital Introduction", "Equity")
    if any(k in narration_upper for k in ["WITHDRAWAL", "DRAWINGS"]) and not is_credit and "ATM" not in narration_upper:
        return ("Drawings", "Equity")
        
    # INCOME DETECTION RULE (MANDATORY)
    income_indicators = ["UPI CR", "NEFT CR", "IMPS CR", "RTGS CR", "BY TRANSFER", "RECEIVED", "CREDIT"]
    if any(k in narration_upper for k in income_indicators):
        if is_credit:  # Double check it actually is a credit
            return ("Sales Income", "Income")
            
    # EXPENSE DETECTION RULE (MANDATORY)
    expense_indicators = ["UPI DR", "POS", "ATM", "DEBIT", "PAYMENT", "PURCHASE", "FUEL", "PETROL", "ELECTRICITY", "BILL", "BANK CHARGES"]
    if any(k in narration_upper for k in expense_indicators):
        if not is_credit:
            if "ATM" in narration_upper or "CASH" in narration_upper:
                return ("Cash Withdrawal", "Asset")
            return ("General Expense", "Expense")

    # Standard Rules Dictionary
    for key, val in RULES.items():
        if key in narration_upper:
            return val

    # TRANSACTION ACCOUNTING RULE (Default Catch-All)
    # Every transaction must affect either Income, Expense, Asset, Liability, Equity.
    # No transaction should remain unclassified.
    if is_credit:
        return ("Other Income", "Income") 
    else:
        return ("Other Expense", "Expense")

def create_journal(transactions: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    journal = []

    for tx in transactions:
        t_debit = str(tx.get("debit", "0")).strip()
        t_credit = str(tx.get("credit", "0")).strip()
        
        if not t_debit or t_debit.lower() == 'none' or t_debit == '-': t_debit = "0"
        if not t_credit or t_credit.lower() == 'none' or t_credit == '-': t_credit = "0"

        try:
            debit = float(t_debit.replace(',', ''))
        except Exception:
            debit = 0.0
            
        try:
            credit = float(t_credit.replace(',', ''))
        except Exception:
            credit = 0.0

        amount = debit if debit > 0 else credit

        if amount == 0:
            continue

        account, category = classify_transaction(tx.get("narration", ""), amount, is_credit=(credit > 0))

        if debit > 0:
            entry = {
                "date": tx.get("date", ""),
                "narration": tx.get("narration", ""),
                "balance": tx.get("balance", 0),
                "entries": [
                    {
                        "account": account,
                        "type": "DEBIT",
                        "amount": amount,
                        "category": category
                    },
                    {
                        "account": "Bank Account",
                        "type": "CREDIT",
                        "amount": amount,
                        "category": "Asset"
                    }
                ]
            }
        else:
            entry = {
                "date": tx.get("date", ""),
                "narration": tx.get("narration", ""),
                "balance": tx.get("balance", 0),
                "entries": [
                    {
                        "account": "Bank Account",
                        "type": "DEBIT",
                        "amount": amount,
                        "category": "Asset"
                    },
                    {
                        "account": account,
                        "type": "CREDIT",
                        "amount": amount,
                        "category": category
                    }
                ]
            }

        journal.append(entry)

    return journal

def parse_amt(val):
    if val is None:
        return 0.0
    try:
        if isinstance(val, str):
            val = val.replace(',', '').strip()
            if not val or val.lower() == 'none' or val == '-':
                return 0.0
        return float(val)
    except:
        return 0.0

def detect_opening_balance(transactions):
    """
    CRITICAL: 
    Opening Balance = First available balance in the statement.
    If the first transaction contains a balance column value,
    that value must be treated as Opening Balance.
    """
    if not transactions:
        return 0.0

    for tx in transactions:
        balance_val = parse_amt(tx.get("balance", 0))
        if balance_val != 0.0:
            return abs(balance_val)
            
    return 0.0

def detect_closing_balance(transactions):
    """
    CRITICAL:
    Closing Balance = Last available balance in the statement.
    """
    if not transactions:
        return 0.0
        
    for tx in reversed(transactions):
        balance_val = parse_amt(tx.get("balance", 0))
        if balance_val != 0.0:
            return abs(balance_val)
            
    return 0.0
