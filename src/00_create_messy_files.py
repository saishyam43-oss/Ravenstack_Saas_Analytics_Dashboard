#!/usr/bin/env python3
"""
00_create_messy_files v1.0 – Controlled data quality corruption for ETL testing

Purpose:
 - Generate intentionally “messy” versions of clean Ravenstack base tables for QA, training, and data cleaning exercises.
 - Preserve realistic SaaS data patterns while injecting common data quality issues in a controlled, reproducible way.

Key behaviors:
 - Reads canonical CSV inputs for accounts, subscriptions, feature_usage, churn_events, and support_tickets and writes corresponding *_MESSY_ONLY.csv outputs.
 - Samples a configurable fraction of each base table (FRACTION_DIRTY) and applies table-specific corruption rules to this subset only.
 - Injects realistic issues by table, including:
   - Accounts: null industry/country, inconsistent churn_flag.
   - Subscriptions: null end_date/plan_tier/mrr_amount/arr_amount, flipped churn_flag.
   - Feature usage: negative usage_count/usage_duration_secs, null feature_name.
   - Churn events: null reason_code/feedback_text, negative refund_amount_usd.
   - Support tickets: null satisfaction_score/priority, negative resolution_time_hours.
 - Annotates each corrupted row with a messiness_description label describing which synthetic issues were applied (e.g. null_country; flipped_churn_flag), enabling targeted validation and unit testing.
 - Creates a mix of:
   - True duplicates (rows that retain original primary keys) to test de-duplication logic.
   - New records with programmatically generated primary keys that respect the original ID format (prefix + zero-padded hex suffix).

Implementation details:
 - Uses a fixed RANDOM_SEED to make corruption patterns fully reproducible across runs.
 - Ensures no primary key collisions by tracking existing IDs and generating new, unique identifiers per table.
 - Keeps the base source files unchanged and writes “messy-only” outputs, so clean vs. dirty datasets can be compared side by side in downstream ETL and data quality workflows.
"""


import pandas as pd
import numpy as np
from pathlib import Path

# ---------- CONFIG ----------
BASE_FILES = {
    "accounts": "ravenstack_accounts.csv",
    "subscriptions": "ravenstack_subscriptions.csv",
    "feature_usage": "ravenstack_feature_usage.csv",
    "churn_events": "ravenstack_churn_events.csv",
    "support_tickets": "ravenstack_support_tickets.csv",
}

PK_COL = {
    "accounts": "account_id",
    "subscriptions": "subscription_id",
    "feature_usage": "usage_id",
    "churn_events": "churn_event_id",
    "support_tickets": "ticket_id",
}

# fraction of base table to turn into messy rows
FRACTION_DIRTY = 0.10
# within messy sample, fraction that keeps original PK (true duplicates)
FRACTION_KEEP_DUP_PK = 0.30
RANDOM_SEED = 42

np.random.seed(RANDOM_SEED)


def infer_prefix_and_width(example_id: str):
    if "-" in example_id:
        prefix, suffix = example_id.split("-", 1)
        return prefix + "-", len(suffix)
    else:
        return "", len(example_id)


def generate_new_ids(existing_ids, example_id, n_new):
    prefix, width = infer_prefix_and_width(example_id)
    new_ids = []
    counter = 1
    while len(new_ids) < n_new:
        suffix = f"{counter:0{width}x}"[:width]
        candidate = f"{prefix}{suffix}"
        if candidate not in existing_ids:
            existing_ids.add(candidate)
            new_ids.append(candidate)
        counter += 1
    return new_ids


def make_messy_with_description(df: pd.DataFrame, table_name: str) -> pd.DataFrame:
    df_dirty = df.copy()
    descriptions = []

    if table_name == "subscriptions":
        for idx in df_dirty.index:
            issue_list = []
            if np.random.random() < 0.2:
                col = np.random.choice(["end_date", "plan_tier", "mrr_amount", "arr_amount"])
                df_dirty.at[idx, col] = None
                issue_list.append(f"null_{col}")
            if np.random.random() < 0.15:
                df_dirty.at[idx, "churn_flag"] = not bool(df_dirty.at[idx, "churn_flag"])
                issue_list.append("flipped_churn_flag")
            descriptions.append("; ".join(issue_list) if issue_list else "no_issue")

    elif table_name == "feature_usage":
        for idx in df_dirty.index:
            issue_list = []
            if np.random.random() < 0.15:
                col = np.random.choice(["usage_count", "usage_duration_secs"])
                df_dirty.at[idx, col] = -1
                issue_list.append(f"negative_{col}")
            if np.random.random() < 0.10:
                df_dirty.at[idx, "feature_name"] = None
                issue_list.append("null_feature_name")
            descriptions.append("; ".join(issue_list) if issue_list else "no_issue")

    elif table_name == "accounts":
        for idx in df_dirty.index:
            issue_list = []
            if np.random.random() < 0.20:
                col = np.random.choice(["industry", "country"])
                df_dirty.at[idx, col] = None
                issue_list.append(f"null_{col}")
            if np.random.random() < 0.15:
                df_dirty.at[idx, "churn_flag"] = not bool(df_dirty.at[idx, "churn_flag"])
                issue_list.append("inconsistent_churn_flag")
            descriptions.append("; ".join(issue_list) if issue_list else "no_issue")

    elif table_name == "churn_events":
        for idx in df_dirty.index:
            issue_list = []
            if np.random.random() < 0.20:
                df_dirty.at[idx, "reason_code"] = None
                issue_list.append("null_reason_code")
            if np.random.random() < 0.20:
                df_dirty.at[idx, "feedback_text"] = None
                issue_list.append("null_feedback_text")
            if np.random.random() < 0.15:
                val = float(df_dirty.at[idx, "refund_amount_usd"])
                df_dirty.at[idx, "refund_amount_usd"] = -abs(val)
                issue_list.append("negative_refund")
            descriptions.append("; ".join(issue_list) if issue_list else "no_issue")

    elif table_name == "support_tickets":
        for idx in df_dirty.index:
            issue_list = []
            if np.random.random() < 0.20:
                df_dirty.at[idx, "satisfaction_score"] = None
                issue_list.append("null_satisfaction_score")
            if np.random.random() < 0.20:
                df_dirty.at[idx, "priority"] = None
                issue_list.append("null_priority")
            if np.random.random() < 0.15:
                val = float(df_dirty.at[idx, "resolution_time_hours"])
                df_dirty.at[idx, "resolution_time_hours"] = -abs(val)
                issue_list.append("negative_resolution_time")
            descriptions.append("; ".join(issue_list) if issue_list else "no_issue")

    df_dirty["messiness_description"] = descriptions
    return df_dirty


# ---------- MAIN ----------
for name, base_path in BASE_FILES.items():
    print(f"Processing {name} from {base_path} ...")
    df_base = pd.read_csv(base_path)
    pk = PK_COL[name]

    n_base = len(df_base)
    n_dirty = max(1, int(FRACTION_DIRTY * n_base))

    # 1) sample rows to dirty
    dirty_sample = df_base.sample(n_dirty, random_state=RANDOM_SEED).copy()

    # 2) inject issues + description
    dirty_sample = make_messy_with_description(dirty_sample, name)

    # 3) choose which messy rows keep original PKs
    n_keep_dup = int(FRACTION_KEEP_DUP_PK * n_dirty)
    keep_dup_idx = dirty_sample.sample(n_keep_dup, random_state=RANDOM_SEED).index

    # 4) generate new PKs for remaining messy rows
    existing_ids = set(df_base[pk].astype(str))
    example_id = df_base[pk].dropna().astype(str).iloc[0]

    new_pk_idx = [i for i in dirty_sample.index if i not in keep_dup_idx]
    n_new_pk = len(new_pk_idx)
    new_ids = generate_new_ids(existing_ids, example_id, n_new_pk)

    for idx, new_id in zip(new_pk_idx, new_ids):
        dirty_sample.at[idx, pk] = new_id

    # 5) write messy-only output
    out_path = Path(f"ravenstack_{name}_MESSY_ONLY.csv")
    dirty_sample.to_csv(out_path, index=False)
    print(
        f"  -> wrote {out_path} with {len(dirty_sample)} rows "
        f"({n_keep_dup} true-duplicate PKs, {n_new_pk} new PKs)"
    )
