# Statement Formats by Entity Type

The Trading and P&L Account layout is essentially the same across entity types. What changes is the equity section — read the right subsection before building the Capital Account / Balance Sheet equity block.

## Sole Proprietorship

Simplest case, and the default when entity type is unknown.

**Capital Account** (single column):
```
Opening Capital                     XXX
Add: Capital Introduced             XXX
Add: Net Profit for the period      XXX
Less: Drawings                      (XXX)
Closing Capital                     XXX
```

Closing Capital flows to the Balance Sheet as the sole equity line.

## Partnership Firm

Same overall structure, but split by partner. Two sub-cases:

- **If the user can attribute specific inflows/outflows to specific partners** (capital introduced, drawings) from narration or their own knowledge, build one capital account per partner, each following the sole-proprietorship format above.
- **Net profit allocation**: split per the profit-sharing ratio if the user provides one. If not provided, ask — don't default to an even split silently, since that's a real assumption that changes each partner's closing capital. If they don't know/care to specify, an even split is a reasonable fallback, but say plainly that's what was assumed.

Balance Sheet equity section lists each partner's closing capital as a separate line, summing to total partners' capital.

## Private Limited Company

No "capital account" in the sole-prop/partnership sense — a company's contributed capital is fixed (share capital) and doesn't move with day-to-day owner transactions the way a proprietor's capital does. If bank data shows what look like owner drawings or capital introduced for a company, these are very likely something else — director's remuneration (an expense, not drawings), a loan to/from director (a balance sheet liability/asset, not equity), or dividend (equity, but declared formally, not just "money the promoter took out"). Flag these explicitly rather than mapping them onto capital-account logic that doesn't apply.

Equity section instead:
```
Share Capital                       XXX  (opening, unchanged unless user confirms fresh issue)
Add: Net Profit for the period      XXX  (added to Reserves & Surplus)
Reserves & Surplus (closing)        XXX
Total Equity                        XXX
```

If the business is a company, it's worth explicitly asking the user whether what looks like personal-style withdrawals are actually director's remuneration/loan account movements before classifying — this is one of the areas where getting it wrong has real tax consequences (misclassifying dividend vs. remuneration vs. loan affects both the company's and the director's tax position).

## Balance Sheet — general format (all entity types)

```
Liabilities                    Amount    |  Assets                      Amount
----------------------------------------|---------------------------------------
Capital / Equity (per above)     XXX     |  Bank Balance (closing)         XXX
Loans (identified, closing bal)  XXX     |  Cash in hand (if known)        XXX
[Creditors — not visible          —      |  [Debtors — not visible          —
  from bank data alone]                 |    from bank data alone]
                                          |  [Stock-in-hand — not visible    —
                                          |    from bank data alone]
                                          |  [Fixed assets not paid via      —
                                          |    this bank account]
----------------------------------------|---------------------------------------
Total                             XXX    |  Total                          XXX
```

The bracketed rows are not filler — keep them visible in the output (as zero/blank with the bracketed note) rather than omitting them, so the reader sees what's missing rather than a balance sheet that looks complete by omission. See `limitations.md` for the standard note to attach.