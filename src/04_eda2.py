#!/usr/bin/env python3
"""
04_eda2 v1.0 – Opportunity Dashboard Analytics: NRR + Support Cost Analysis (Ravenstack)

Purpose:
 - Generates executive-ready opportunity reports from ETL2 features: cohort_nrr_by_segment.csv, support_segment_stats_corrected.csv.
 - Focuses on high-impact business levers: Net Revenue Retention (NRR) by customer_segment, Support efficiency as % of revenue.

Core Analyses:
 - Segmented NRR Curves: Expands subscriptions to monthly ledger (capped 24 months), computes nrr_pct = mrr_monthX / mrr_month0 per SMB/Mid/Enterprise.
 - Support Cost Metrics: tickets_per_account, revenue_per_ticket across customer segments (total_tickets / total_revenue).

Key Production Safeguards:
 - Defensive loading: duplicate column removal, FileNotFoundError fallback to local files.
 - Status filtering: ['active','churned','trial','closed','canceled'] (case-insensitive).
 - Clean merges: Drop conflicting customer_segment before account join, SMB fallback.
 - Zero-division protection: revenue_per_ticket safe division, months_active minimum 1.

Business Insights Delivered:
 - cohort_nrr_by_segment.csv: Month-by-month NRR trajectory reveals expansion potential vs. churn leakage by segment.
 - support_segment_stats_corrected.csv: tickets_per_account identifies high-touch segments; revenue_per_ticket flags support drag.

Output Location: data/etl2/eda/opportunity/
Usage: python 04_eda2.py → Instant executive dashboard data (2 files, stakeholder-ready).
"""

import pandas as pd
import numpy as np
from pathlib import Path
from datetime import datetime

# ---------------------------------------------------------
# SETUP PATHS
# ---------------------------------------------------------
# Adjust this to match your project folder structure
PROJECT_ROOT = Path.cwd()
BASE = PROJECT_ROOT / "data" / "etl2"
OUT = BASE / "eda" / "opportunity"
OUT.mkdir(parents=True, exist_ok=True)

print("Loading data for Opportunity Dashboard (NRR + Support Stats)...")

# 1. Load Data
# We handle duplicates on load to be safe
try:
    subs = pd.read_csv(BASE / "subscriptions_fe.csv", parse_dates=["start_date", "end_date", "churn_date"])
    accounts = pd.read_csv(BASE / "accounts_fe.csv")
    print("Files loaded successfully.")
except FileNotFoundError:
    # Fallback for local testing if paths differ
    subs = pd.read_csv("subscriptions_fe.csv", parse_dates=["start_date", "end_date", "churn_date"])
    accounts = pd.read_csv("accounts_fe.csv")
    print("Files loaded from current directory.")

# --- FIX 1: Remove Duplicate Columns ---
subs = subs.loc[:, ~subs.columns.duplicated()]
accounts = accounts.loc[:, ~accounts.columns.duplicated()]

# --- FIX 2: Standardize Status Logic (The Case Sensitivity Fix) ---
if 'subscription_status' in subs.columns:
    subs['subscription_status'] = subs['subscription_status'].str.lower().str.strip()
    valid_status = ['active', 'churned', 'trial', 'closed', 'canceled']
    subs = subs[subs['subscription_status'].isin(valid_status)].copy()
    print(f"Filtered to {len(subs)} valid subscriptions.")

# --- FIX 3: Clean Merge ---
# Drop existing segment column from subs if it exists to avoid _x/_y columns
if 'customer_segment' in subs.columns:
    subs = subs.drop(columns=['customer_segment'])

# Merge Segment
subs = subs.merge(accounts[['account_id', 'customer_segment']], on='account_id', how='left')

# Fill missing segments
subs['customer_segment'] = subs['customer_segment'].fillna('SMB')


# =========================================================
# PART 1: GENERATE SEGMENTED NRR (The "Expansion Engine")
# =========================================================
print("\n--- Generating NRR Ledger ---")

# Define the analysis end date (Today or max date in data)
analysis_date = subs['end_date'].max()
if pd.isnull(analysis_date):
    analysis_date = datetime.now()

# Helper to calculate months active
def get_active_months(row):
    try:
        start = row['start_date']
        # If churned, use churn_date. If active, use analysis_date
        end = row['churn_date'] if pd.notnull(row['churn_date']) else analysis_date

        # Handle cases where dates might be NaT
        if pd.isnull(start) or pd.isnull(end):
            return 1

        # Calculate difference in months
        months = (end.year - start.year) * 12 + (end.month - start.month)
        return max(0, int(months) + 1) # +1 to include month 0
    except Exception:
        return 1

# Apply calculation
subs['months_active'] = subs.apply(get_active_months, axis=1)

# Cap months at 24 to keep file size small
MAX_MONTHS = 24
subs['months_active'] = subs['months_active'].clip(upper=MAX_MONTHS)

# Create an expanded dataframe (One row per month per subscription)
expanded_rows = []
for idx, row in subs.iterrows():
    mrr = row['mrr_clean']
    seg = row['customer_segment']

    if pd.isnull(mrr) or mrr == 0:
        continue

    for m in range(int(row['months_active'])):
        expanded_rows.append({
            'customer_segment': seg,
            'months_since_signup': m,
            'mrr': mrr
        })

# Convert to DataFrame
df_ledger = pd.DataFrame(expanded_rows)

if not df_ledger.empty:
    # Group by Segment and Month
    nrr_curve = df_ledger.groupby(['customer_segment', 'months_since_signup'])['mrr'].sum().reset_index()

    # Get the "Month 0" MRR (Starting Baseline) for each segment
    baseline = nrr_curve[nrr_curve['months_since_signup'] == 0][['customer_segment', 'mrr']].rename(columns={'mrr': 'starting_mrr'})

    # Merge Baseline back
    nrr_final = nrr_curve.merge(baseline, on='customer_segment', how='left')

    # Calculate Percentage
    nrr_final['nrr_pct'] = nrr_final['mrr'] / nrr_final['starting_mrr']

    # Save NRR File
    nrr_path = OUT / "cohort_nrr_by_segment.csv"
    nrr_final.to_csv(nrr_path, index=False)
    nrr_final.to_csv("cohort_nrr_by_segment.csv", index=False) # Local backup

    print(f"Success! Created: {nrr_path}")
    print(nrr_final.head(5))
else:
    print("Error: No valid subscription data found for NRR calculation.")


# =========================================================
# PART 2: GENERATE SUPPORT SEGMENT STATS (The "Hidden Cost")
# =========================================================
print("\n--- Generating Support Segment Stats ---")

try:
    # 1. Group by Segment to get raw totals from the accounts file
    # Ensure columns exist, fill with 0 if missing to prevent errors
    for col in ['ticket_count', 'total_mrr']:
        if col not in accounts.columns:
            accounts[col] = 0

    support_stats = accounts.groupby("customer_segment").agg(
        total_tickets=("ticket_count", "sum"),
        active_accounts=("account_id", "nunique"),
        total_revenue=("total_mrr", "sum")
    ).reset_index()

    # 2. Calculate Efficiency Metrics
    # Metric A: Support Intensity
    support_stats["tickets_per_account"] = support_stats["total_tickets"] / support_stats["active_accounts"]

    # Metric B: Value Efficiency
    support_stats["revenue_per_ticket"] = support_stats.apply(
        lambda x: x["total_revenue"] / x["total_tickets"] if x["total_tickets"] > 0 else 0,
        axis=1
    )

    # 3. Formatting
    support_stats["tickets_per_account"] = support_stats["tickets_per_account"].round(1)
    support_stats["revenue_per_ticket"] = support_stats["revenue_per_ticket"].round(2)

    # 4. Save Support File
    support_path = OUT / "support_segment_stats_corrected.csv"
    support_stats.to_csv(support_path, index=False)
    support_stats.to_csv("support_segment_stats_corrected.csv", index=False) # Local backup

    print(f"Success! Created: {support_path}")
    print(support_stats)

except Exception as e:
    print(f"Error generating support stats: {e}")

print("\nAll tasks complete.")
