#!/usr/bin/env python3
"""
01_etl1 v1.0 – Production ETL Pipeline: Data Cleaning + Quality Framework (Ravenstack SaaS)

Purpose:
 - Comprehensive ETL pipeline processing 5 raw Ravenstack tables (accounts, subscriptions, feature_usage, churn_events, support_tickets) from messy input → production-ready clean output with full audit trail.
 - Implements industry-standard data quality framework measuring completeness, uniqueness, validity, and consistency with weighted final score.

Core Processing Pipeline:
 - Pre-imputation deduplication (keep max completeness row per PK).
 - Cross-table imputation: accounts country/industry by account_name match; subscriptions end_date from account churn; plan_tier/mrr/arr via lookup table.
 - Defensive corrections: ABS() negatives, date format validation (dd-mm-yyyy), trial subscriptions capped at 30 days.
 - Dependency-aware processing: churn_events → accounts → subscriptions → feature_usage.
 - Post-imputation duplicate detection on cleaned values (append-only flagging).

Data Quality Framework (4 Dimensions):
 - Completeness: 30% (curated required fields, Unknown=valid).
 - Uniqueness: 25% (PK-level).
 - Validity: 25% (format + range: dates dd-mm-yyyy, numerics ≥0).
 - Consistency: 20% (business logic: usage after churn, start>end dates).

Audit & Observability Outputs:
 - Cleaned tables: {table}_clean.csv (native + _clean/_imputed columns preserved).
 - DQ artifacts: data_quality_summary.csv, data_quality_long_curated.csv, data_quality_errors_curated.csv.
 - Correction logs: corrections_log.csv (PK, original→corrected, reason_code).
 - Validation: validation_summary_etl1_long.csv + dirty_reason_lookup.csv.
 - Row impact: duplicates_removed, issues_before/after, completeness_improvement_pct.

Key Decisions (Industry-Aligned):
 - Never overwrite native columns (add _clean/_imputed for auditability).
 - Flagging > deletion for priority/satisfaction (preserve signal).
 - Lookup table for plan_tier-mrr-arr (fixed SaaS pricing patterns).
 - Trial detection + 30-day cap prevents runaway ghost trials.
 - Reproducible via ETL_RUN_ID + UTC timestamps in all logs.

"""

# ================================================
# PART 1/10 — Imports, Global Config, Paths
# ================================================

import re
import json
import uuid
from datetime import datetime, timezone
from pathlib import Path
import logging
import pandas as pd
import numpy as np

# Logging (minimal per your requirement)
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s"
)

# Directories
DATA_DIR = Path("data")
RAW = DATA_DIR / "raw" / "merged"
CLEAN_DIR = DATA_DIR / "processed" / "clean"
CLEAN_DIR.mkdir(parents=True, exist_ok=True)

# Run metadata
ETL_RUN_ID = str(uuid.uuid4())
NOW_TS = lambda: datetime.now(timezone.utc).isoformat()

# Raw input locations
RAW_FILES = {
    "accounts": RAW / "accounts.csv",
    "subscriptions": RAW / "subscriptions.csv",
    "churn_events": RAW / "churn_events.csv",
    "feature_usage": RAW / "feature_usage.csv",
    "support_tickets": RAW / "support_tickets.csv",
}

# Primary keys
PKS = {
    "accounts": ["account_id"],
    "subscriptions": ["subscription_id"],
    "churn_events": ["churn_event_id"],
    "feature_usage": ["usage_id"],
    "support_tickets": ["ticket_id"],
}

# Global logs
VALIDATION_ROWS = []
CORRECTIONS = []

# Dirty reason lookup (final consolidated list)
DIRTY_REASON_LOOKUP = {
    "duplicate_raw_keep_max_completeness": "Dropped duplicate raw row",
    "blank_pk": "Primary key blank",
    "country_imputed": "Country imputed (cross-ref or Unknown)",
    "industry_imputed": "Industry imputed (cross-ref or Unknown)",
    "native_churn_flag_true_but_no_event": "Native churn flag TRUE but no churn event",
    "native_churn_flag_false_but_event_exists": "Native churn flag FALSE but churn event exists",
    "blank_reason_code": "Churn reason_code blank",
    "blank_feedback_text": "Churn feedback blank",
    "negative_refund_amount_usd": "Refund amount negative (corrected to abs)",
    "invalid_numeric_parsed": "Numeric parse failed",

    "plan_tier_imputed_from_lookup": "Plan tier imputed using lookup",
    "mrr_arr_imputed_from_lookup": "MRR/ARR imputed from lookup or formula",

    "end_date_derived": "Subscription end_date derived from account churn",
    "start_end_swapped_due_to_small_gap": "Start/end swapped (gap < 30 days)",
    "start_after_end_flagged": "Start_date > end_date and large gap",

    "feature_name_imputed": "Feature name imputed as Unknown",
    "negative_usage_count": "Usage count negative (corrected to abs)",
    "negative_usage_duration": "Usage duration negative (corrected to abs)",
    "invalid_date_format": "Date invalid (kept original)",
    "orphan_usage": "Feature usage references missing subscription",

    "negative_resolution_time": "Negative resolution hours",
    "priority_missing_flagged": "Priority missing (flagged only)",
    "satisfaction_imputed": "Satisfaction imputed as Unknown",

    "duplicate_after_cleaning_detected": "Duplicate rows after cleaning",
}

# ================================================
# PART 2/10 — Core Helpers + PK Validation + Parsing
# ================================================

# -----------------------------
# Validation logging
# -----------------------------
def add_validation(table, metric, value):
    VALIDATION_ROWS.append({
        "etl_run_id": ETL_RUN_ID,
        "timestamp": NOW_TS(),
        "table": table,
        "metric": metric,
        "value": value,
    })

# -----------------------------
# Correction logging
# -----------------------------
def record_correction(table, pk_dict, column, original, corrected, reason_code, notes=None):
    CORRECTIONS.append({
        "etl_run_id": ETL_RUN_ID,
        "timestamp": NOW_TS(),
        "table": table,
        "primary_key": json.dumps(pk_dict),
        "column": column,
        "original_value": original,
        "corrected_value": corrected,
        "reason_code": reason_code,
        "notes": notes or "",
    })

# -----------------------------
# Normalize ID values
# -----------------------------
def normalize_id_val(v):
    if pd.isna(v):
        return None
    s = str(v).strip()
    if s == "" or s.lower() in ("nan", "none", "null"):
        return None
    return s

# -----------------------------
# Safe dirty reason append
# -----------------------------
def _get_reason_cell(df, i):
    if "dirty_row_reason" not in df.columns:
        return ""
    val = df.at[i, "dirty_row_reason"]
    if val in (None, "", np.nan):
        return ""
    return str(val)

def append_reason_cell(df, i, reason):
    existing = _get_reason_cell(df, i)
    new = append_reason(existing, reason)
    df.at[i, "dirty_row_reason"] = new
    df.at[i, "dirty_row_flag"] = True  # always set flag when reason appended

# -----------------------------
# Append reason (pure function)
# -----------------------------
def append_reason(existing, reason):
    if existing in (None, "", np.nan):
        return json.dumps([reason])
    try:
        lst = json.loads(existing)
        if isinstance(lst, list):
            if reason not in lst:
                lst.append(reason)
            return json.dumps(lst)
        else:
            return json.dumps([existing, reason])
    except:
        return json.dumps([existing, reason])

# -----------------------------
# Row completeness scoring
# -----------------------------
def completeness_score(df, cols=None):
    if cols is None:
        cols = df.columns.tolist()
    return df[cols].notna().astype(int).sum(axis=1)

# -----------------------------
# Completeness before
# -----------------------------
def compute_completeness_before(df_raw):
    """
    Completeness BEFORE cleaning using all native columns.
    Missing = NaN, '', 'unknown', 'nan', 'na', 'null', 'none'
    """
    if df_raw is None or df_raw.empty:
        return 1.0

    native_cols = [
        c for c in df_raw.columns
        if not (c.endswith("_clean") or c.endswith("_imputed") or c.endswith("_norm"))
    ]

    total_cells = df_raw.shape[0] * len(native_cols)
    missing = 0

    for col in native_cols:
        s = df_raw[col]

        # TRUE missing NaN
        mask_na = s.isna()

        # After filling NA, convert to string for other checks
        s2 = s.fillna("").astype(str).str.strip().str.lower()

        # Literal missing values
        literal_missing = s2.isin(["", "unknown", "nan", "na", "null", "none"])

        missing += int((mask_na | literal_missing).sum())

    completeness_before = max(0, min(1, 1 - missing / total_cells))
    return completeness_before

# -----------------------------
# Completeness After
# -----------------------------
def compute_completeness_after(df, table_name):
    """
    Compute completeness AFTER cleaning.
    Rules:
        - Based ONLY on curated required fields
        - Unknown is VALID (not missing)
        - Derived fields like account_churn_date_clean do NOT affect completeness
        - end_date_clean missing is allowed for active subscriptions
    """

    if df is None or df.empty:
        return 1.0

    REQUIRED_FIELDS = {
        "accounts": ["country_clean", "industry_clean"],
        "subscriptions": ["start_date"],
        "feature_usage": ["usage_date", "usage_count_clean", "usage_duration_secs_clean"],
        "support_tickets": ["resolution_hours_clean"],
        "churn_events": ["churn_date"],
    }

    OPTIONAL_FIELDS = {
        "accounts": ["account_churn_date_clean"],
        "subscriptions": ["end_date_clean"],
        "feature_usage": [],
        "support_tickets": ["priority", "satisfaction_score"],
        "churn_events": ["feedback_text"],
    }

    required_cols = REQUIRED_FIELDS.get(table_name, [])
    optional_cols = OPTIONAL_FIELDS.get(table_name, [])

    nrows = len(df)
    missing_rows = set()

    for col in required_cols:

        if col not in df.columns:
            # Required column missing entirely → treat as fully missing
            missing_rows.update(range(nrows))
            continue

        series = df[col]

        # Prepare cleaned string version
        temp = series.replace({np.nan: ""})
        temp = temp.astype("string")
        s2 = temp.str.strip()

        # Unknown is valid
        mask_unknown = s2.str.lower() == "unknown"

        # Missing definitions
        mask_na = series.isna()
        mask_blank = s2 == ""
        mask_nan_token = s2.str.lower().isin(["nan", "null", "none"])

        # Missing = NA or blank or nan-token, but NOT unknown
        missing_mask = (mask_na | mask_blank | mask_nan_token) & (~mask_unknown)

        # Add row indexes where missing
        missing_idx = df.index[missing_mask].tolist()
        missing_rows.update(missing_idx)

    # Special case: subscriptions end_date_clean is OPTIONAL
    if table_name == "subscriptions" and "end_date_clean" in df.columns:
        # Remove end_date_clean from affecting completeness
        pass  # already handled by not being in required list

    completeness = 1.0 - (len(missing_rows) / nrows)
    return completeness

# -----------------------------
# Date utilities
# -----------------------------
def date_is_ddmmyyyy(s):
    if pd.isna(s):
        return False
    s = str(s).strip()
    return bool(re.match(r"^\d{2}-\d{2}-\d{4}$", s))

def parse_date_transient(s):
    if pd.isna(s):
        return None
    s = str(s).strip()
    if not date_is_ddmmyyyy(s):
        return None
    try:
        return datetime.strptime(s, "%d-%m-%Y").date()
    except:
        return None

def extract_date_ddmmyyyy(s):
    """Used for 'closed_at' style values that include time."""
    if pd.isna(s):
        return None
    s = str(s).strip()
    first = s.split()[0]
    if date_is_ddmmyyyy(first):
        try:
            return datetime.strptime(first, "%d-%m-%Y").date()
        except:
            return None
    return None

# -----------------------------
# Numeric parse (final, patched)
# -----------------------------
def parse_numeric_abs(val):
    if pd.isna(val):
        return None, None
    s = str(val).strip()
    if s == "":
        return None, None

    cleaned = (
        s.replace(",", "", 1)
         .replace("(", "-")
         .replace(")", "")
    )
    cleaned = re.sub(r"[^\d\.\-]", "", cleaned)

    try:
        num = float(cleaned)
        return abs(num), num
    except:
        return None, None

# -----------------------------
# Pre-imputation dedupe
# -----------------------------
def pre_imputation_dedupe(df, table, pk_cols):
    initial = len(df)
    if initial == 0:
        add_validation(table, "rows_in", 0)
        add_validation(table, "rows_after_raw_dedupe", 0)
        add_validation(table, "duplicates_raw_removed", 0)
        return df.copy()

    df["_raw_idx"] = range(len(df))
    df["_comp_score"] = completeness_score(df)

    keep_rows = []
    removed = 0

    for _, group in df.groupby(pk_cols, dropna=False):
        if len(group) == 1:
            keep_rows.append(group.index[0])
            continue

        max_score = group["_comp_score"].max()
        best = group[group["_comp_score"] == max_score]

        chosen = best.index[0]
        keep_rows.append(chosen)

        dropped = group.index.difference([chosen])
        removed += len(dropped)

        for idx in dropped:
            record_correction(
                table,
                {pk: df.at[idx, pk] for pk in pk_cols},
                None,
                None,
                None,
                "duplicate_raw_keep_max_completeness"
            )

    out = df.loc[keep_rows].reset_index(drop=True)
    out.drop(columns=["_raw_idx", "_comp_score"], inplace=True)

    add_validation(table, "rows_in", initial)
    add_validation(table, "rows_after_raw_dedupe", len(out))
    add_validation(table, "duplicates_raw_removed", removed)

    return out

# -----------------------------
# Primary key validation
# -----------------------------
def validate_pks(df, table, pk_cols):
    if df.empty:
        add_validation(table, "pk_null_count", 0)
        return df

    null_pk_total = 0

    for pk in pk_cols:
        if pk not in df.columns:
            continue

        norm = df[pk].apply(normalize_id_val)
        df[f"{pk}_norm"] = norm

        mask_null = norm.isna()
        null_pk_total += int(mask_null.sum())

        for idx in df[mask_null].index:
            append_reason_cell(df, idx, "blank_pk")
            record_correction(
                table,
                {pk: df.at[idx, pk]},
                pk,
                df.at[idx, pk],
                None,
                "blank_pk"
            )

    add_validation(table, "pk_null_count", null_pk_total)
    return df


pd.set_option("future.no_silent_downcasting", True)

# ================================================
# PART 3/10 — Data Quality Engine
# ================================================

# ------------------------------------------------
# Column classification
# ------------------------------------------------
def classify_column(col_name):
    c = col_name.lower()

    if c.endswith("_id") or ("id" in c and len(c) <= 25):
        return "id"
    if any(x in c for x in ["date", "timestamp", "dt"]):
        return "date"
    if any(x in c for x in ["amount", "mrr", "arr", "count", "duration", "hours", "refund", "time"]):
        return "numeric"
    if any(x in c for x in ["flag", "priority", "tier", "status", "is_"]):
        return "categorical"
    return "text"

# ------------------------------------------------
# COMPLETENESS
# ------------------------------------------------
def compute_completeness(df):
    """
    Compute completeness by favoring *_clean or *_imputed columns where present.
    Treat the string 'Unknown' (case-insensitive) as NOT missing (i.e., valid).
    """
    if df is None or df.empty:
        return 1.0

    # Build effective dataframe where *_clean or *_imputed replace native columns
    eff = pd.DataFrame(index=df.index)

    cols = df.columns.tolist()

    # Step 1: Add all *_clean and *_imputed columns mapped to their base names
    used_bases = set()
    for c in cols:
        if c.endswith("_clean") or c.endswith("_imputed"):
            base = c.rsplit("_", 1)[0]
            eff[base] = df[c]
            used_bases.add(base)

    # Step 2: Add native columns that do not have a clean/imputed replacement
    for c in cols:
        # skip columns that are clean/imputed variants
        if c.endswith("_clean") or c.endswith("_imputed"):
            continue
        if c not in used_bases and c not in eff.columns:
            eff[c] = df[c]

    if eff.empty:
        return 1.0

    # Consider Unknown (case-insensitive) as not missing
    def is_missing_series(s):
        s_str = s.astype(str).str.strip()
        missing_mask = s.isna() | (s_str == "")
        # treat 'unknown' as NOT missing
        unknown_mask = s_str.str.lower() == "unknown"
        return missing_mask & (~unknown_mask)

    missing = 0
    for col in eff.columns:
        missing += int(is_missing_series(eff[col]).sum())

    total = eff.shape[0] * eff.shape[1]
    if total == 0:
        return 1.0

    completeness = max(0.0, min(1.0, 1 - missing / total))
    return completeness

# ------------------------------------------------
# UNIQUENESS
# ------------------------------------------------
def compute_uniqueness(df, pk_cols):
    if df.empty:
        return 1.0
    if not pk_cols or any(pk not in df.columns for pk in pk_cols):
        return 1.0
    unique = df[pk_cols].drop_duplicates().shape[0]
    return unique / df.shape[0] if df.shape[0] else 1.0

# ------------------------------------------------
# VALIDITY (Format-only, per Option C)
# ------------------------------------------------
def compute_validity(df):
    """
    Compute validity using effective columns (clean/imputed preferred).
    Treat 'Unknown' as valid for categorical columns.
    """
    if df is None or df.empty:
        return 1.0

    # Build effective dataframe same as completeness
    eff = pd.DataFrame(index=df.index)
    cols = df.columns.tolist()
    used_bases = set()

    for c in cols:
        if c.endswith("_clean") or c.endswith("_imputed"):
            base = c.rsplit("_", 1)[0]
            eff[base] = df[c]
            used_bases.add(base)

    for c in cols:
        if c.endswith("_clean") or c.endswith("_imputed"):
            continue
        if c not in used_bases and c not in eff.columns:
            eff[c] = df[c]

    invalid = 0
    total = 0

    for col in eff.columns:
        col_type = classify_column(col)
        series = eff[col]

        for val in series:
            total += 1
            if val in (None, "", np.nan):
                invalid += 1
                continue

            sval = str(val).strip()

            # IDs must not be blank
            if col_type == "id":
                if sval == "" or sval.lower() in ("nan", "null", "none"):
                    invalid += 1
                continue

            # Date: must match dd-mm-yyyy
            if col_type == "date":
                if not date_is_ddmmyyyy(sval):
                    invalid += 1
                continue

            # Numeric
            if col_type == "numeric":
                cleaned = re.sub(r"[^\d\.\-]", "", sval.replace(",", ""))
                try:
                    float(cleaned)
                except:
                    invalid += 1
                continue

            # Categorical: treat 'unknown' as valid
            if col_type == "categorical":
                if sval == "" or sval.lower() in ("nan", "null", "none"):
                    invalid += 1
                else:
                    # if 'unknown' treat it as valid, so nothing to do
                    pass
                continue

            # Text: blank already handled above

    # Avoid division by zero
    if total == 0:
        return 1.0

    return max(0.0, min(1.0, 1 - invalid / total))


# ------------------------------------------------
# CONSISTENCY (Logic-based)
# ------------------------------------------------
def compute_consistency(df, table_name, pk_cols, crossmaps):
    if df.empty:
        return 1.0

    account_churn = crossmaps.get("account_churn_map", {})
    subscription_end = crossmaps.get("subscription_end_map", {})
    sub_to_acct = crossmaps.get("subscription_to_account", {})

    inconsistent = 0
    total = 0

    for i, row in df.iterrows():
        total += 1
        bad = False

        # -------- ACCOUNTS --------
        if table_name == "accounts":
            if row.get("churn_flag_clean") and not row.get("account_churn_date_clean"):
                bad = True

        # -------- SUBSCRIPTIONS --------
        if table_name == "subscriptions":
            sd = row.get("start_date")
            ed = row.get("end_date_clean")

            sdt = parse_date_transient(sd)
            edt = parse_date_transient(ed)

            if sdt and edt and sdt > edt:
                bad = True

        # -------- FEATURE_USAGE --------
        if table_name == "feature_usage":
            ud = row.get("usage_date")
            udt = parse_date_transient(ud)

            sid = row.get("subscription_id_norm")
            acct = sub_to_acct.get(sid)

            # usage after subscription end_date_clean
            if sid in subscription_end:
                edt = parse_date_transient(subscription_end[sid])
                if udt and edt and udt > edt:
                    bad = True

            # usage after account churn
            if acct in account_churn:
                cdt = parse_date_transient(account_churn[acct])
                if udt and cdt and udt > cdt:
                    bad = True

        if bad:
            inconsistent += 1

    return max(0.0, min(1.0, 1 - inconsistent / total))

# ------------------------------------------------
# FINAL WEIGHTED SCORE
# ------------------------------------------------
DQ_WEIGHTS = {
    "completeness": 0.30,
    "uniqueness": 0.25,
    "validity": 0.25,
    "consistency": 0.20,
}

def compute_final_quality_score(metrics):
    score = (
        metrics["completeness"] * DQ_WEIGHTS["completeness"]
        + metrics["uniqueness"] * DQ_WEIGHTS["uniqueness"]
        + metrics["validity"] * DQ_WEIGHTS["validity"]
        + metrics["consistency"] * DQ_WEIGHTS["consistency"]
    )
    return max(0.0, min(1.0, score))

# ------------------------------------------------
# COLUMN-LEVEL METRICS (long format)
# ------------------------------------------------
def compute_column_metrics(df, table_name):
    rows = []

    for col in df.columns:
        col_type = classify_column(col)
        s = df[col]

        missing = float((s.isna() | (s == "")).mean())
        unique_ratio = float(s.nunique(dropna=True) / len(df)) if len(df) else 1.0

        neg_ratio = 0.0
        zero_ratio = 0.0
        invalid_ratio = 0.0

        # Numeric
        if col_type == "numeric":
            cleaned = (
                s.astype(str)
                  .str.replace(",", "", regex=False)
                  .str.replace(r"[^\d\.\-]", "", regex=True)
            )
            parsed = pd.to_numeric(cleaned, errors="coerce")
            neg_ratio = float((parsed < 0).mean())
            zero_ratio = float((parsed == 0).mean())
            invalid_ratio = float(parsed.isna().mean())

        # Date
        elif col_type == "date":
            invalid_ratio = float((~s.astype(str).str.match(r"^\d{2}-\d{2}-\d{4}$", na=False)).mean())

        # ID / categorical
        elif col_type in ("id", "categorical"):
            invalid_ratio = float(((s == "") | s.isna()).mean())

        rows.append({
            "table_name": table_name,
            "column_name": col,
            "col_type": col_type,
            "percent_missing": round(missing, 6),
            "percent_invalid": round(invalid_ratio, 6),
            "percent_unique": round(unique_ratio, 6),
            "percent_negative": round(neg_ratio, 6),
            "percent_zero": round(zero_ratio, 6),
        })

    return rows

# ================================================
# PART 4/10 — Cleaning Functions (All Entities)
# ================================================


# =====================================================
# CLEAN — CHURN EVENTS
# =====================================================
def clean_churn_events(df):
    table = "churn_events"
    orig = len(df)

    df = pre_imputation_dedupe(df, table, PKS[table])
    df = validate_pks(df, table, PKS[table])

    # Ensure dirty columns exist early
    if "dirty_row_flag" not in df.columns:
        df["dirty_row_flag"] = False
    else:
        df["dirty_row_flag"] = df["dirty_row_flag"].fillna(False)

    if "dirty_row_reason" not in df.columns:
        df["dirty_row_reason"] = ""
    else:
        df["dirty_row_reason"] = df["dirty_row_reason"].fillna("")

    # Normalize account_id
    if "account_id" in df.columns:
        df["account_id_norm"] = df["account_id"].apply(normalize_id_val)
    else:
        df["account_id_norm"] = None

    # Refund amount cleaning
    df["refund_amount_clean"] = None
    colname = "refund_amount_usd" if "refund_amount_usd" in df.columns else None

    if colname:
        for i, val in df[colname].items():
            abs_val, orig_val = parse_numeric_abs(val)
            if abs_val is not None:
                df.at[i, "refund_amount_clean"] = abs_val
                if orig_val < 0:
                    append_reason_cell(df, i, "negative_refund_amount_usd")
                    record_correction(
                        table,
                        {"churn_event_id": df.at[i, "churn_event_id"]},
                        colname,
                        val,
                        abs_val,
                        "negative_refund_amount_usd"
                    )
            else:
                if val not in (None, "", np.nan):
                    append_reason_cell(df, i, "invalid_numeric_parsed")
                    record_correction(
                        table,
                        {"churn_event_id": df.at[i, "churn_event_id"]},
                        colname,
                        val,
                        None,
                        "invalid_numeric_parsed"
                    )

    # Blank reason_code
    if "reason_code" in df.columns:
        for i, val in df["reason_code"].items():
            if val in (None, "", np.nan):
                append_reason_cell(df, i, "blank_reason_code")
                record_correction(
                    table,
                    {"churn_event_id": df.at[i, "churn_event_id"]},
                    "reason_code",
                    val,
                    None,
                    "blank_reason_code"
                )

    # Blank feedback_text
    if "feedback_text" in df.columns:
        for i, val in df["feedback_text"].items():
            if val in (None, "", np.nan):
                append_reason_cell(df, i, "blank_feedback_text")
                record_correction(
                    table,
                    {"churn_event_id": df.at[i, "churn_event_id"]},
                    "feedback_text",
                    val,
                    None,
                    "blank_feedback_text"
                )

    # Build churn_map
    churn_map = {}
    if "churn_date" in df.columns:
        for _, row in df.iterrows():
            acct = row.get("account_id_norm")
            raw_dt = row.get("churn_date")
            if not acct:
                continue
            if raw_dt in (None, "", np.nan):
                continue

            parsed = parse_date_transient(raw_dt)
            if parsed:
                # keep latest churn date for the account
                prev = churn_map.get(acct)
                if prev is None or parsed > prev[0]:
                    churn_map[acct] = (parsed, raw_dt)
            else:
                # keep raw if parsed failed but no better exists
                if acct not in churn_map:
                    churn_map[acct] = (None, raw_dt)

    add_validation(table, "rows_initial", orig)
    add_validation(table, "rows_out", len(df))

    return df, churn_map


# =====================================================
# Helper: Cross-reference imputation for accounts
# =====================================================
def _impute_from_crossref(df, colname, table):
    """
    For country and industry:
    Cross-reference by account_id_norm.
    If other rows exist with non-null values → use mode.
    Else → 'Unknown'.
    """
    counts = 0
    for i, row in df.iterrows():
        val = row.get(colname)
        if val not in (None, "", np.nan):
            continue

        acct = row.get("account_id_norm")
        if not acct:
            df.at[i, colname] = "Unknown"
            append_reason_cell(df, i, f"{colname}_imputed")
            record_correction(table, {"account_id": row.get("account_id")}, colname, None, "Unknown", f"{colname}_imputed")
            counts += 1
            continue

        others = df[(df["account_id_norm"] == acct) & (df[colname].notna()) & (df[colname] != "")]
        if len(others) > 0:
            mode_val = others[colname].mode().iloc[0]
            df.at[i, colname] = mode_val
            append_reason_cell(df, i, f"{colname}_imputed")
            record_correction(table, {"account_id": row.get("account_id")}, colname, None, mode_val, f"{colname}_imputed")
            counts += 1
        else:
            df.at[i, colname] = "Unknown"
            append_reason_cell(df, i, f"{colname}_imputed")
            record_correction(table, {"account_id": row.get("account_id")}, colname, None, "Unknown", f"{colname}_imputed")
            counts += 1
    return counts


# =====================================================
# CLEAN — ACCOUNTS
# =====================================================
def clean_accounts(df_raw, churn_map):
    table = "accounts"
    orig_len = len(df_raw)

    # ---------------------------------------------------------
    # 1. Make RAW copy BEFORE dedupe for cross-reference lookups
    # ---------------------------------------------------------
    raw_ref = df_raw.copy()

    # ---------------------------------------------------------
    # 2. Now run dedupe on the dataframe to be cleaned
    # ---------------------------------------------------------
    df = pre_imputation_dedupe(df_raw.copy(), table, PKS[table])
    df = validate_pks(df, table, PKS[table])

    # ---------------------------------------------------------
    # 3. Ensure dirty columns exist
    # ---------------------------------------------------------
    # Ensure dirty_row_flag exists and is boolean
    if "dirty_row_flag" not in df.columns:
        df["dirty_row_flag"] = False
    else:
        df["dirty_row_flag"] = df["dirty_row_flag"].fillna(False).astype(bool)

    # Ensure dirty_row_reason exists and is string
    if "dirty_row_reason" not in df.columns:
        df["dirty_row_reason"] = ""
    else:
        df["dirty_row_reason"] = df["dirty_row_reason"].fillna("").astype(str)

    # ---------------------------------------------------------
    # 4. Create clean columns (native untouched)
    # ---------------------------------------------------------
    df["country_clean"] = df["country"]
    df["industry_clean"] = df["industry"]

    # Normalize IDs once
    df["account_id_norm"] = df["account_id"].apply(normalize_id_val)
    raw_ref["account_id_norm"] = raw_ref["account_id"].apply(normalize_id_val)

    # Lowercase account names for matching
    if "account_name" in df.columns:
        df["account_name_norm"] = df["account_name"].astype(str).str.lower()
        raw_ref["account_name_norm"] = raw_ref["account_name"].astype(str).str.lower()
    else:
        df["account_name_norm"] = None
        raw_ref["account_name_norm"] = None

    # ---------------------------------------------------------
    # 5. Impute country_clean & industry_clean BEFORE dedupe
    # ---------------------------------------------------------
    for idx, row in df.iterrows():
        acct = row["account_id_norm"]
        name = row["account_name_norm"]

        # COUNTRY
        if row["country"] in (None, "", np.nan):
            val = None

            # Same account_id
            if acct:
                matches = raw_ref[(raw_ref["account_id_norm"] == acct) &
                                  (raw_ref["country"].notna()) &
                                  (raw_ref["country"] != "")]
                if len(matches) > 0:
                    val = matches["country"].iloc[0]

            # Same account_name
            if val is None and name:
                matches = raw_ref[(raw_ref["account_name_norm"] == name) &
                                  (raw_ref["country"].notna()) &
                                  (raw_ref["country"] != "")]
                if len(matches) > 0:
                    val = matches["country"].iloc[0]

            if val is None:
                val = "Unknown"

            df.at[idx, "country_clean"] = val
            append_reason_cell(df, idx, "country_imputed")
            record_correction(table, {"account_id": row["account_id"]},
                              "country", None, val, "country_imputed")

        # INDUSTRY
        if row["industry"] in (None, "", np.nan):
            val = None

            # Same account_id
            if acct:
                matches = raw_ref[(raw_ref["account_id_norm"] == acct) &
                                  (raw_ref["industry"].notna()) &
                                  (raw_ref["industry"] != "")]
                if len(matches) > 0:
                    val = matches["industry"].iloc[0]

            # Same account_name
            if val is None and name:
                matches = raw_ref[(raw_ref["account_name_norm"] == name) &
                                  (raw_ref["industry"].notna()) &
                                  (raw_ref["industry"] != "")]
                if len(matches) > 0:
                    val = matches["industry"].iloc[0]

            if val is None:
                val = "Unknown"

            df.at[idx, "industry_clean"] = val
            append_reason_cell(df, idx, "industry_imputed")
            record_correction(table, {"account_id": row["account_id"]},
                              "industry", None, val, "industry_imputed")

    # ---------------------------------------------------------
    # 6. Churn Flag + Churn Date (existing logic unchanged)
    # ---------------------------------------------------------
    df["churn_flag_clean"] = False
    df["account_churn_date_clean"] = None

    for idx, row in df.iterrows():
        acct = row["account_id_norm"]
        native = bool(row.get("churn_flag", False))

        if acct in churn_map:
            df.at[idx, "churn_flag_clean"] = True
            raw = churn_map[acct][1]
            parsed = parse_date_transient(raw)
            df.at[idx, "account_churn_date_clean"] = parsed.strftime("%d-%m-%Y") if parsed else raw

            if native is False:
                append_reason_cell(df, idx, "native_churn_flag_false_but_event_exists")

        else:
            df.at[idx, "churn_flag_clean"] = False
            df.at[idx, "account_churn_date_clean"] = None

            if native is True:
                append_reason_cell(df, idx, "native_churn_flag_true_but_no_event")

    return df

# =====================================================
# CLEAN — SUBSCRIPTIONS
# =====================================================
def clean_subscriptions(df, accounts_df):
    from datetime import timedelta, datetime  # Ensure availability

    table = "subscriptions"
    orig = len(df)

    df = pre_imputation_dedupe(df, table, PKS[table])
    df = validate_pks(df, table, PKS[table])

    if "dirty_row_flag" not in df.columns:
        df["dirty_row_flag"] = False
    else:
        df["dirty_row_flag"] = df["dirty_row_flag"].fillna(False)

    if "dirty_row_reason" not in df.columns:
        df["dirty_row_reason"] = ""
    else:
        df["dirty_row_reason"] = df["dirty_row_reason"].fillna("")

    df["subscription_id_norm"] = df["subscription_id"].apply(normalize_id_val)
    df["account_id_norm"] = df["account_id"].apply(normalize_id_val)

    # Map accounts
    acct_map = accounts_df.set_index("account_id_norm").to_dict(orient="index") if not accounts_df.empty else {}

    # ---------------------------------------------
    # Lookup Table (Option C - strict normalized key)
    # ---------------------------------------------
    lookup_by_mrr_arr = {}
    lookup_by_seats_plan = {}

    tmp = df.copy()
    tmp["mrr_num"] = pd.to_numeric(tmp.get("mrr_amount"), errors="coerce")
    tmp["arr_num"] = pd.to_numeric(tmp.get("arr_amount"), errors="coerce")

    valid_rows = tmp[
        tmp["mrr_num"].notna() &
        tmp["arr_num"].notna() &
        (tmp["mrr_num"] > 0) &
        (tmp["arr_num"] > 0)
    ]

    for _, r in valid_rows.iterrows():
        seats_norm = str(r.get("seats")).strip()
        plan_norm = str(r.get("plan_tier")).strip()
        mrr = float(r["mrr_num"])
        arr = float(r["arr_num"])

        lookup_by_mrr_arr[(mrr, arr)] = plan_norm

        if seats_norm and plan_norm:
            lookup_by_seats_plan[(seats_norm, plan_norm)] = (mrr, arr)

    # ==========================================================
    # CLEANING LOOP
    # ==========================================================
    df["plan_tier_imputed"] = df.get("plan_tier")
    df["mrr_amount_clean"] = None
    df["arr_amount_clean"] = None
    df["end_date_clean"] = df.get("end_date")

    # Reference for "today" to avoid closing active new trials
    today_date = datetime.now().date()

    for i, row in df.iterrows():
        pk = {"subscription_id": row.get("subscription_id")}
        seats_norm = str(row.get("seats")).strip()

        # ------------------------------------------------------
        # FIX: IDENTIFY TRIAL STATUS EARLY
        # ------------------------------------------------------
        is_trial_val = str(row.get("is_trial", "")).lower()
        is_trial = is_trial_val in ["true", "1", "yes"]

        # ------------------------------------------------------
        # END DATE DERIVATION FROM ACCOUNT CHURN
        # ------------------------------------------------------
        acct_norm = row.get("account_id_norm")
        acct_info = acct_map.get(acct_norm)

        # FIX 1: Safer Imputation (Check churn date >= start date)
        native_end = df.at[i, "end_date_clean"] # Check current state
        if pd.isna(native_end) and acct_info:
            churn_dt_str = acct_info.get("account_churn_date_clean")
            start_dt = parse_date_transient(row.get("start_date"))
            churn_dt_obj = parse_date_transient(churn_dt_str)

            # Logic: Only impute if account churned AFTER this sub started
            if churn_dt_obj and start_dt and churn_dt_obj >= start_dt:
                df.at[i, "end_date_clean"] = churn_dt_str
                append_reason_cell(df, i, "end_date_derived")
                record_correction(table, pk, "end_date", None, churn_dt_str, "end_date_derived")

        # ------------------------------------------------------
        # FIX 2: GHOST TRIAL IMPUTATION (Cap at 30 days)
        # ------------------------------------------------------
        # Re-check end date (it might have been set by churn logic above)
        current_end = df.at[i, "end_date_clean"]

        # Only run if Trial, No End Date, and Zero MRR (to protect paid pilots)
        raw_mrr = row.get("mrr_amount")
        mrr_val, _ = parse_numeric_abs(raw_mrr)
        is_free = (mrr_val is None) or (mrr_val == 0)

        if is_trial and is_free and pd.isna(current_end):
            start_val = row.get("start_date")
            start_dt = parse_date_transient(start_val)

            if start_dt:
                cutoff = start_dt + timedelta(days=30)

                # CRITICAL CHECK: Only cap if the 30 days have already passed.
                # Do not close a trial that started yesterday.
                if cutoff < today_date:
                    cutoff_str = cutoff.strftime("%d-%m-%Y")
                    df.at[i, "end_date_clean"] = cutoff_str
                    append_reason_cell(df, i, "trial_capped_at_30_days")
                    record_correction(table, pk, "end_date", None, cutoff_str, "trial_capped_at_30_days")

        # ------------------------------------------------------
        # FIX 3: SKIP PAID IMPUTATION FOR TRIALS
        # ------------------------------------------------------
        if is_trial:
            # Force 0 revenue
            df.at[i, "mrr_amount_clean"] = 0.0
            df.at[i, "arr_amount_clean"] = 0.0

            # Handle Tier Label
            native_plan = row.get("plan_tier")
            if native_plan in (None, "", np.nan):
                df.at[i, "plan_tier_imputed"] = "Trial"
                append_reason_cell(df, i, "trial_tier_set_to_trial")
            else:
                df.at[i, "plan_tier_imputed"] = native_plan

            # SKIP the rest of the loop (Lookup Logic)
            continue

        # ------------------------------------------------------
        # PLAN TIER IMPUTATION (Paid Only)
        # ------------------------------------------------------
        native_plan = row.get("plan_tier")
        # Reuse mrr_val parsed above
        raw_arr = row.get("arr_amount")
        arr_val, _ = parse_numeric_abs(raw_arr)

        # Variables for lookup logic
        mrr_clean_val = mrr_val
        arr_clean_val = arr_val

        plan_imputed = native_plan  # default to native if present

        if native_plan in (None, "", np.nan):
            # 1) BEST: infer from (mrr,arr)
            if mrr_clean_val and arr_clean_val:
                if (mrr_clean_val, arr_clean_val) in lookup_by_mrr_arr:
                    plan_imputed = lookup_by_mrr_arr[(mrr_clean_val, arr_clean_val)]

            # 2) NEXT: infer from seats
            if plan_imputed in (None, "", np.nan):
                seat_matches = [
                    plan for (seats_key, plan) in lookup_by_seats_plan.keys()
                    if seats_key == seats_norm
                ]
                if len(seat_matches) == 1:
                    plan_imputed = seat_matches[0]

            # 3) FAILSAFE
            if plan_imputed in (None, "", np.nan):
                plan_imputed = "Unknown"

            df.at[i, "plan_tier_imputed"] = plan_imputed
            append_reason_cell(df, i, "plan_tier_imputed_from_lookup")
            record_correction(table, pk, "plan_tier", native_plan, plan_imputed, "plan_tier_imputed_from_lookup")

        # ------------------------------------------------------
        # MRR / ARR IMPUTATION (Paid Only)
        # ------------------------------------------------------
        plan_norm = str(plan_imputed).strip()
        lookup_key = (seats_norm, plan_norm)

        if lookup_key in lookup_by_seats_plan:
            mrr_lookup, arr_lookup = lookup_by_seats_plan[lookup_key]
            df.at[i, "mrr_amount_clean"] = mrr_lookup
            df.at[i, "arr_amount_clean"] = arr_lookup
            append_reason_cell(df, i, "mrr_arr_imputed_from_lookup")
            continue

        # Fallback formula logic
        final_mrr = mrr_clean_val
        final_arr = arr_clean_val

        if mrr_clean_val and not arr_clean_val:
            final_arr = mrr_clean_val * 12
        elif arr_clean_val and not mrr_clean_val:
            final_mrr = arr_clean_val / 12
        elif (mrr_clean_val == 0 and arr_clean_val is None) or (arr_clean_val == 0 and mrr_clean_val is None):
            final_mrr, final_arr = 0, 0

        df.at[i, "mrr_amount_clean"] = final_mrr
        df.at[i, "arr_amount_clean"] = final_arr

    # -------------------------------
    # Start/end swap logic (Post-processing)
    # -------------------------------
    for i, row in df.iterrows():
        sd = row.get("start_date")
        ed = df.at[i, "end_date_clean"] # Check updated value

        sdt = parse_date_transient(sd)
        edt = parse_date_transient(ed)

        if sdt and edt and sdt > edt:
            gap = (sdt - edt).days
            if gap < 30:
                df.at[i, "start_date"] = ed
                df.at[i, "end_date_clean"] = sd
                append_reason_cell(df, i, "start_end_swapped_due_to_small_gap")
            else:
                append_reason_cell(df, i, "start_after_end_flagged")

    add_validation(table, "rows_initial", orig)
    add_validation(table, "rows_out", len(df))

    return df

# =====================================================
# CLEAN — FEATURE USAGE
# =====================================================
def clean_feature_usage(df, subscriptions_df, accounts_df):
    table = "feature_usage"
    orig = len(df)

    df = pre_imputation_dedupe(df, table, PKS[table])
    df = validate_pks(df, table, PKS[table])

    if "dirty_row_flag" not in df.columns:
        df["dirty_row_flag"] = False
    if "dirty_row_reason" not in df.columns:
        df["dirty_row_reason"] = ""

    subs_map = subscriptions_df.set_index("subscription_id_norm").to_dict(orient="index")
    acct_churn_map = accounts_df.set_index("account_id_norm")["account_churn_date_clean"].to_dict()
    sub_to_acct = subscriptions_df.set_index("subscription_id_norm")["account_id_norm"].to_dict()

    df["subscription_id_norm"] = df["subscription_id"].apply(normalize_id_val)

    df["usage_count_clean"] = None
    df["usage_duration_secs_clean"] = None
    df["feature_name_imputed"] = df.get("feature_name")

    for i, row in df.iterrows():
        pk = {"usage_id": row.get("usage_id")}
        sid = row.get("subscription_id_norm")

        # Orphan usage
        if sid not in subs_map:
            append_reason_cell(df, i, "orphan_usage")
            record_correction(table, pk, "subscription_id", row.get("subscription_id"), None, "orphan_usage")
            continue

        # Feature name imputation
        fn = row.get("feature_name")
        if fn in (None, "", np.nan):
            df.at[i, "feature_name_imputed"] = "Unknown"
            append_reason_cell(df, i, "feature_name_imputed")
            record_correction(table, pk, "feature_name", None, "Unknown", "feature_name_imputed")

        # Usage count
        abs_val, orig_val = parse_numeric_abs(row.get("usage_count"))
        df.at[i, "usage_count_clean"] = abs_val
        if orig_val is not None and orig_val < 0:
            append_reason_cell(df, i, "negative_usage_count")

        # Duration
        abs_val2, orig_val2 = parse_numeric_abs(row.get("usage_duration_secs"))
        df.at[i, "usage_duration_secs_clean"] = abs_val2
        if orig_val2 is not None and orig_val2 < 0:
            append_reason_cell(df, i, "negative_usage_duration")

        # Invalid date format — flag but keep original
        ud = row.get("usage_date")
        if ud not in (None, "", np.nan) and not date_is_ddmmyyyy(str(ud)):
            append_reason_cell(df, i, "invalid_date_format")
            record_correction(table, pk, "usage_date", ud, ud, "invalid_date_format")

    add_validation(table, "rows_initial", orig)
    add_validation(table, "rows_out", len(df))
    return df


# =====================================================
# CLEAN — SUPPORT TICKETS
# =====================================================
def clean_support_tickets(df):
    table = "support_tickets"
    orig = len(df)

    df = pre_imputation_dedupe(df, table, PKS[table])
    df = validate_pks(df, table, PKS[table])

    if "dirty_row_flag" not in df.columns:
        df["dirty_row_flag"] = False
    if "dirty_row_reason" not in df.columns:
        df["dirty_row_reason"] = ""

    df["resolution_hours_clean"] = None
    df["satisfaction_imputed"] = df.get("satisfaction_score")

    for i, row in df.iterrows():
        pk = {"ticket_id": row.get("ticket_id")}

        # Resolution hours
        abs_val, orig_val = parse_numeric_abs(row.get("resolution_time_hours"))
        df.at[i, "resolution_hours_clean"] = abs_val
        if orig_val is not None and orig_val < 0:
            append_reason_cell(df, i, "negative_resolution_time")

        # Priority missing
        if row.get("priority") in (None, "", np.nan):
            append_reason_cell(df, i, "priority_missing_flagged")
            record_correction(table, pk, "priority", row.get("priority"), None, "priority_missing_flagged")

        # Satisfaction imputation
        sat = row.get("satisfaction_score")
        if sat in (None, "", np.nan):
            df.at[i, "satisfaction_imputed"] = "Unknown"
            append_reason_cell(df, i, "satisfaction_imputed")
            record_correction(table, pk, "satisfaction_score", None, "Unknown", "satisfaction_imputed")

    add_validation(table, "rows_initial", orig)
    add_validation(table, "rows_out", len(df))

    return df
# ================================================
# PART 5/10 — Duplicate Detection, Dirty Flags, DQ Writers
# ================================================

# ---------------------------------------------------------
# POST-IMPUTATION DUPLICATE DETECTION (append-only behavior)
# ---------------------------------------------------------
def post_imputation_duplicate_detection(df, table, pk_cols):
    if df.empty:
        add_validation(table, "duplicates_identical_after_cleaning_detected", 0)
        return df

    clean_cols = [
        c for c in df.columns
        if c.endswith("_clean") or c.endswith("_imputed") or c in ["end_date_clean", "account_churn_date_clean"]
    ]

    if not clean_cols:
        add_validation(table, "duplicates_identical_after_cleaning_detected", 0)
        return df

    # Signature for duplicates based on cleaned values
    df["_clean_sig"] = df[clean_cols].apply(
        lambda r: json.dumps(
            {c: (None if pd.isna(r[c]) else str(r[c])) for c in clean_cols},
            sort_keys=True
        ), axis=1
    )

    dup_count = 0

    for pk_val, group in df.groupby(pk_cols, dropna=False):
        if len(group) <= 1:
            continue

        if group["_clean_sig"].nunique() == 1:
            dup_count += len(group) - 1

            for idx in group.index:
                append_reason_cell(df, idx, "duplicate_after_cleaning_detected")
                record_correction(
                    table,
                    {pk: df.at[idx, pk] for pk in pk_cols},
                    None,
                    None,
                    None,
                    "duplicate_after_cleaning_detected"
                )

    df.drop(columns=["_clean_sig"], inplace=True)
    add_validation(table, "duplicates_identical_after_cleaning_detected", dup_count)

    return df


# ---------------------------------------------------------
# NORMALIZE DIRTY FLAGS AT END
# ---------------------------------------------------------
def _normalize_reason_cell(cell):
    if cell in (None, "", np.nan):
        return ""
    if isinstance(cell, list):
        return json.dumps(cell, ensure_ascii=False)
    try:
        parsed = json.loads(cell)
        if isinstance(parsed, list):
            return json.dumps(parsed, ensure_ascii=False)
    except:
        return json.dumps([cell], ensure_ascii=False)
    return str(cell)


def finalize_dirty_flags(cleaned_tables):
    for _, df in cleaned_tables.items():
        if df is None or df.empty:
            continue

        if "dirty_row_flag" not in df.columns:
            df["dirty_row_flag"] = False
        else:
            df["dirty_row_flag"] = df["dirty_row_flag"].fillna(False).astype(bool)

        if "dirty_row_reason" not in df.columns:
            df["dirty_row_reason"] = ""
        else:
            df["dirty_row_reason"] = df["dirty_row_reason"].fillna("").astype(str)

        df["dirty_row_reason"] = df["dirty_row_reason"].apply(_normalize_reason_cell)


# ---------------------------------------------------------
# RAW IMPACT COUNTER (before cleaning)
# ---------------------------------------------------------
def impacted_rows_raw(df, table_name):
    if df is None or df.empty:
        return 0

    issues = 0

    # blanks + nulls
    # Blank detection must be done column-wise (Series-level .str)
    # blanks + nulls (avoid double-counting)
    blank_mask = df.apply(lambda col: col.astype(str).str.strip() == "")
    # combined missing: either NaN or blank-string (count once)
    combined_missing = df.isna() | blank_mask
    issues += int(combined_missing.sum().sum())


    # PK nulls
    for pk in PKS.get(table_name, []):
        if pk in df.columns:
            issues += int(df[pk].isin([None, "", np.nan]).sum())

    # Negative numeric
    for col in df.columns:
        if any(x in col.lower() for x in ["amount", "mrr", "arr", "count", "duration", "hours", "refund"]):
            cleaned = (
                df[col].astype(str)
                      .str.replace(",", "", regex=False)
                      .str.replace(r"[^\d\.\-]", "", regex=True)
            )
            parsed = pd.to_numeric(cleaned, errors="coerce")
            issues += int((parsed < 0).sum())

    # Invalid dates
    for col in df.columns:
        if "date" in col.lower():
            col_str = df[col].astype(str)
            mask = col_str.str.match(r"^\d{2}-\d{2}-\d{4}$", na=False)
            issues += int((~mask).sum())

    return issues


# ---------------------------------------------------------
# CLEAN IMPACT COUNTER (after cleaning)
# ---------------------------------------------------------
def impacted_rows_clean(df, table_name):
    """
    Count only TRUE remaining issues after cleaning.
    Rules:
      - Missing REQUIRED fields is an issue (Unknown is valid)
      - Optional fields are ignored
      - Derived fields (account_churn_date_clean, end_date_clean) are ignored
      - Unfixable dates are counted only for required date fields
      - start_date > end_date_clean (gap >= 30 days) still flags
    """

    if df is None or df.empty:
        return 0

    # Required fields (same as completeness_after)
    REQUIRED_FIELDS = {
        "accounts": ["country_clean", "industry_clean"],
        "subscriptions": ["start_date"],
        "feature_usage": ["usage_date", "usage_count_clean", "usage_duration_secs_clean"],
        "support_tickets": ["resolution_hours_clean"],
        "churn_events": ["churn_date"],
    }

    # Optional fields = NEVER validated
    OPTIONAL_FIELDS = {
        "accounts": ["account_churn_date_clean"],
        "subscriptions": ["end_date_clean"],
        "churn_events": ["feedback_text"],
        "feature_usage": [],
        "support_tickets": ["priority", "satisfaction_score"],
    }

    required = set(REQUIRED_FIELDS.get(table_name, []))
    optional = set(OPTIONAL_FIELDS.get(table_name, []))

    bad_rows = set()

    for idx, row in df.iterrows():

        # ---------------------------------------------------
        # 1. Missing required fields
        # (Unknown is allowed, optional is ignored)
        # ---------------------------------------------------
        for col in required:
            val = row.get(col)
            if val in (None, "", np.nan):
                bad_rows.add(idx)
                break
            if isinstance(val, str) and val.strip().lower() == "unknown":
                continue

        # ---------------------------------------------------
        # 2. Subscription date inconsistency (required rule)
        # ---------------------------------------------------
        if table_name == "subscriptions":
            sd = row.get("start_date")
            ed = row.get("end_date_clean")

            sdt = parse_date_transient(sd)
            edt = parse_date_transient(ed)

            # Only required rule: gap >= 30 days
            if sdt and edt and sdt > edt:
                gap = (sdt - edt).days
                if gap >= 30:
                    bad_rows.add(idx)

        # ---------------------------------------------------
        # 3. Invalid date formats — ONLY for required date fields
        # ---------------------------------------------------
        for col in df.columns:

            # Skip optional date fields
            if col in optional:
                continue

            if "date" in col.lower():

                val = row.get(col)

                # missing is handled in required section
                if val in (None, "", np.nan):
                    continue

                if not date_is_ddmmyyyy(str(val)):
                    bad_rows.add(idx)
                    break

    return len(bad_rows)


# ---------------------------------------------------------
# SAVE CLEANED TABLE
# ---------------------------------------------------------
def save_clean(df, name):
    out_path = CLEAN_DIR / f"{name}_clean.csv"
    df.to_csv(out_path, index=False)
    logging.info(f"Saved cleaned table → {out_path}")


# ---------------------------------------------------------
# COLUMN PROFILING (for dashboard)
# ---------------------------------------------------------
def build_column_profile(df, table_name):
    rows = []

    for col in df.columns:
        s = df[col]
        col_type = classify_column(col)

        non_null = int(s.notna().sum())
        blanks = int((s == "").sum())
        nulls = int(s.isna().sum())
        distinct = int(s.nunique(dropna=True))

        sample_vals = s.dropna().astype(str).unique().tolist()[:10]

        entry = {
            "table_name": table_name,
            "column_name": col,
            "col_type": col_type,
            "non_null_count": non_null,
            "blank_count": blanks,
            "null_count": nulls,
            "distinct_count": distinct,
            "sample_values": json.dumps(sample_vals),
        }

        if col_type == "numeric":
            cleaned = (
                s.astype(str)
                 .str.replace(",", "", regex=False)
                 .str.replace(r"[^\d\.\-]", "", regex=True)
            )
            parsed = pd.to_numeric(cleaned, errors="coerce")
            entry["min"] = float(parsed.min()) if parsed.notna().any() else None
            entry["max"] = float(parsed.max()) if parsed.notna().any() else None
            entry["mean"] = float(parsed.mean()) if parsed.notna().any() else None
        else:
            entry["min"] = entry["max"] = entry["mean"] = None

        rows.append(entry)

    return rows

# ================================================
# PART 6/10 — Data Quality Metrics Builders
# ================================================

# ------------------------
# DQ: curated business fields (AFTER cleaning)
# ------------------------
CURATED_FIELDS = {
    "accounts": [
        "country_clean",
        "industry_clean",
        "churn_flag_clean",
    ],
    "subscriptions": [
        "start_date",
        "end_date_clean",          # included but missing NOT counted as issue for active subs
        "plan_tier_imputed",
        "mrr_amount_clean",
        "arr_amount_clean",
    ],
    "feature_usage": [
        "usage_date",
        "usage_count_clean",
        "usage_duration_secs_clean",
    ],
    "support_tickets": [
        "resolution_hours_clean",
    ],
    "churn_events": [
        "churn_date",
    ],
}

# Primary key mapping used by dq_errors builder
PK_MAP = {
    "accounts": "account_id",
    "subscriptions": "subscription_id",
    "feature_usage": "usage_id",
    "support_tickets": "ticket_id",
    "churn_events": "churn_event_id",
}

# ---------------------------------------------------------
# BUILD METRICS FOR A SINGLE TABLE (summary row)
# ---------------------------------------------------------
def build_table_dq_metrics(table_name, df_clean, pk_cols, crossmaps, raw_df):
    completeness_before = compute_completeness_before(raw_df)
    completeness_after  = compute_completeness_after(df_clean, table_name)

    uniqueness  = compute_uniqueness(df_clean, pk_cols)
    validity    = compute_validity(df_clean)
    consistency = compute_consistency(df_clean, table_name, pk_cols, crossmaps)

    final_score = compute_final_quality_score({
        "completeness": completeness_after,
        "uniqueness": uniqueness,
        "validity": validity,
        "consistency": consistency,
    })

    issues_before = impacted_rows_raw(raw_df, table_name)
    issues_after  = impacted_rows_clean(df_clean, table_name)

    completeness_improvement_pct = (
        (completeness_after - completeness_before) / completeness_before
        if completeness_before > 0 else None
    )

    issues_reduction_pct = (
        (issues_before - issues_after) / issues_before
        if issues_before > 0 else None
    )

    return {
        "table_name": table_name,
        "completeness_before": round(float(completeness_before), 6),
        "completeness_after": round(float(completeness_after), 6),
        "completeness_improvement_pct": round(float(completeness_improvement_pct), 6) if completeness_improvement_pct else None,
        "uniqueness": round(float(uniqueness), 6),
        "validity": round(float(validity), 6),
        "consistency": round(float(consistency), 6),
        "final_quality_score": round(float(final_score), 6),
        "rows_before": int(raw_df.shape[0]),
        "rows_after": int(df_clean.shape[0]),
        "issues_before": int(issues_before),
        "issues_after": int(issues_after),
        "issues_reduction_pct": round(float(issues_reduction_pct), 6) if issues_reduction_pct else None,
    }


# ---------------------------------------------------------
# BUILD DQ LONG
# ---------------------------------------------------------
def build_dq_long(cleaned_tables, top_n=12):
    """
    Build dq_long for curated fields (robust).
    Returns DataFrame with columns: table_name, column_name, pct_missing, pct_invalid, pct_unique
    Notes:
      - pct values returned as decimals (0.342 => 34.2%)
      - 'Unknown' is VALID (not missing)
      - 'end_date_clean' missing for active subs is NOT treated as issue
      - country_clean and industry_clean are never treated as invalid
    """
    rows = []
    invalid_tokens = {"nan", "null", "none"}

    for table, cols in CURATED_FIELDS.items():
        df = cleaned_tables.get(table)
        if df is None or df.empty:
            continue
        nrows = len(df)
        for col in cols:
            # debug trace (can be removed later)
            # print("COL_RAW_REPR:", repr(col))

            if col not in df.columns:
                rows.append({
                    "table_name": table,
                    "column_name": col,
                    "pct_missing": 1.0,
                    "pct_invalid": 1.0,
                    "pct_unique": 0.0,
                })
                continue

            series = df[col]

            # Use pandas string dtype to avoid downcasting warnings
            temp = series.fillna("").astype("string")
            s2 = temp.str.strip()
            s2_lower = s2.str.lower()

            # Missing: True NaN or blank-string or explicit nan/null/none tokens, but not 'unknown'
            mask_na = series.isna()
            mask_blank = s2 == ""
            mask_token = s2_lower.isin(list(invalid_tokens))
            mask_unknown = s2_lower == "unknown"

            missing_mask = (mask_na | mask_blank | mask_token) & (~mask_unknown)
            # Special rule: end_date_clean in subscriptions is not counted as missing
            if table == "subscriptions" and col == "end_date_clean":
                missing_count = 0
            else:
                missing_count = int(missing_mask.sum())

            # Determine invalid_count based on column type
            invalid_count = 0
            col_lower = col.lower()

            # country_clean / industry_clean: always valid
            if col_lower in ["country_clean", "industry_clean"]:
                invalid_count = 0

            # date validation
            elif "date" in col_lower:
                present_mask = ~missing_mask
                present_vals = s2[present_mask]
                invalid_mask = ~present_vals.apply(lambda x: date_is_ddmmyyyy(x))
                invalid_count = int(invalid_mask.sum())

            # numeric validation
            elif any(x in col_lower for x in ["mrr", "arr", "amount", "count", "duration", "hours", "resolution"]):
                present_mask = ~missing_mask
                svals = s2[present_mask]
                coerced = pd.to_numeric(svals.replace("", pd.NA), errors="coerce")
                invalid_count = int(coerced.isna().sum())

            # categorical validation
            else:
                present_mask = ~missing_mask
                present_vals = s2_lower[present_mask]

                # Unknown is valid for all categorical fields
                valid_unknown = present_vals == "unknown"

                # Explicit invalid tokens
                invalid_tokens = {"nan", "null", "none"}

                invalid_mask = present_vals.isin(invalid_tokens) & (~valid_unknown)
                invalid_count = int(invalid_mask.sum())

            # Unique percent - based on effective (non-null) values
            effective_vals = s2.astype("string").str.strip()
            try:
                unique_count = effective_vals.nunique(dropna=True)
            except Exception:
                unique_count = len(pd.Series(effective_vals).unique())

            pct_missing = missing_count / max(1, nrows)
            pct_invalid = invalid_count / max(1, nrows)
            pct_unique = unique_count / max(1, nrows)

            # optional debug print for country
            if col_lower == "country_clean":
                logging.debug(f"DEBUG country_clean → missing_count: {missing_count} invalid_count: {invalid_count} unique: {unique_count} sample: {series.dropna().unique()[:5]}")

            rows.append({
                "table_name": table,
                "column_name": col,
                "pct_missing": pct_missing,
                "pct_invalid": pct_invalid,
                "pct_unique": pct_unique,
            })

    df_long = pd.DataFrame(rows)
    df_long = df_long.sort_values(["pct_missing", "pct_invalid"], ascending=[False, False]).reset_index(drop=True)
    df_top = df_long.head(top_n).copy()
    out = CLEAN_DIR / "data_quality_long_curated.csv"
    df_top.to_csv(out, index=False)
    return df_top

# ---------------------------------------------------------
# BUILD DQ COL_PROFILE
# ---------------------------------------------------------
def build_dq_col_profile(cleaned_tables):
    """
    Build a compact column profile for curated fields.
    Columns: table_name, column_name, dtype, min_val, max_val, avg_val (for numeric), missing_pct, unique_pct
    """
    rows = []
    for table, cols in CURATED_FIELDS.items():
        df = cleaned_tables.get(table)
        if df is None or df.empty:
            continue
        nrows = len(df)
        for col in cols:
            if col not in df.columns:
                rows.append({
                    "table_name": table,
                    "column_name": col,
                    "dtype": "missing",
                    "min_val": None,
                    "max_val": None,
                    "avg_val": None,
                    "missing_pct": 1.0,
                    "unique_pct": 0.0,
                })
                continue

            series = df[col]
            temp = series.fillna("")
            temp = temp.astype("string")
            s_no_na = temp.str.strip()
            missing_count = int(((series.isna() | (s_no_na == "")) & (s_no_na.str.lower() != "unknown")).sum())
            unique_count = series.nunique(dropna=True)
            missing_pct = missing_count / max(1, nrows)
            unique_pct = unique_count / max(1, nrows)

            # attempt numeric profile
            numeric_cols = any(x in col.lower() for x in ["mrr", "arr", "amount", "count", "duration", "hours", "resolution"])
            date_cols = ("date" in col.lower())
            min_val = None
            max_val = None
            avg_val = None
            dtype = "text"
            if numeric_cols:
                dtype = "numeric"
                coerced = pd.to_numeric(series.replace("", pd.NA), errors="coerce")
                if coerced.notna().any():
                    min_val = coerced.min()
                    max_val = coerced.max()
                    avg_val = coerced.mean()
            elif date_cols:
                dtype = "date"
                # keep as text min/max if parseable
                parsed = series.astype(str).apply(lambda x: parse_date_transient(x) if date_is_ddmmyyyy(str(x).strip()) else None)
                parsed_valid = parsed.dropna()
                if not parsed_valid.empty:
                    mindt = min(parsed_valid)
                    maxdt = max(parsed_valid)
                    min_val = mindt.strftime("%d-%m-%Y")
                    max_val = maxdt.strftime("%d-%m-%Y")

            rows.append({
                "table_name": table,
                "column_name": col,
                "dtype": dtype,
                "min_val": min_val,
                "max_val": max_val,
                "avg_val": avg_val,
                "missing_pct": missing_pct,
                "unique_pct": unique_pct,
            })

    df_profile = pd.DataFrame(rows)
    out = CLEAN_DIR / "data_quality_col_profile_curated.csv"
    df_profile.to_csv(out, index=False)
    return df_profile

# ---------------------------------------------------------
# BUILD DQ ERRORS
# ---------------------------------------------------------
def build_dq_errors(cleaned_tables, max_errors_per_table=1000):
    """
    Build a curated dq_errors file: only record actionable errors for curated fields.
    Error types: invalid_date_format, numeric_parse_failed, negative_value_when_not_allowed, unexpected_token
    Returns dataframe and writes CSV.
    """
    rrows = []
    for table, cols in CURATED_FIELDS.items():
        df = cleaned_tables.get(table)
        if df is None or df.empty:
            continue
        pk_col = PK_MAP.get(table)
        nrows = len(df)
        errors_added = 0

        for idx, row in df.iterrows():
            if errors_added >= max_errors_per_table:
                break
            pk_val = row.get(pk_col) if pk_col in df.columns else None

            for col in cols:
                if col not in df.columns:
                    continue
                val = row.get(col)
                sval = "" if val in (None, "", np.nan) else str(val).strip()
                sval_l = sval.lower()

                # Skip blank/unknown as not errors (per rules)
                if sval == "" or sval_l == "unknown":
                    continue

                err_type = None
                cleaned_value = None
                if "date" in col.lower():
                    if not date_is_ddmmyyyy(sval):
                        err_type = "invalid_date_format"
                elif any(x in col.lower() for x in ["mrr", "arr", "amount", "count", "duration", "hours", "resolution"]):
                    # numeric checks
                    try:
                        num = float(str(sval).replace(",", ""))
                    except Exception:
                        err_type = "numeric_parse_failed"
                    else:
                        # negative check - only flag if negative and not expected
                        if num < 0 and "refund" not in col.lower():
                            err_type = "negative_value"
                else:
                    # categorical tokens that are explicit invalid
                    if sval_l in ("nan", "null", "none"):
                        err_type = "unexpected_token"

                if err_type:
                    rrows.append({
                        "table_name": table,
                        "primary_key": pk_val,
                        "column_name": col,
                        "error_type": err_type,
                        "original_value": val,
                        "cleaned_value": cleaned_value,
                        "row_index": idx,
                    })
                    errors_added += 1
                    if errors_added >= max_errors_per_table:
                        break

    df_err = pd.DataFrame(rrows)
    out = CLEAN_DIR / "data_quality_errors_curated.csv"
    df_err.to_csv(out, index=False)
    return df_err

# ---------------------------------------------------------
# BUILD DQ VALIDATION
# ---------------------------------------------------------
def build_dq_validation(raw_tables, cleaned_tables, corrections_list=None):
    """
    Build a compact validation summary for the dashboard showing counts and corrections.
    corrections_list - optional list/dict of correction records (if you have CORRECTIONS, pass it)
    """
    rows = []
    total_duplicates_removed = 0
    total_corrections = 0
    total_imputations = 0
    total_invalid_dates_remaining = 0

    # basic row comparisons
    for table in CURATED_FIELDS.keys():
        raw_df = raw_tables.get(table)
        clean_df = cleaned_tables.get(table)
        rcount = len(raw_df) if raw_df is not None else 0
        ccount = len(clean_df) if clean_df is not None else 0
        duplicates_removed = max(0, rcount - ccount)
        total_duplicates_removed += duplicates_removed

        # corrections_list: try to count corrections and imputations for this table
        if corrections_list:
            table_corr = [c for c in corrections_list if c.get("table_name") == table]
            total_corrections += len(table_corr)
            total_imputations += sum(1 for c in table_corr if "imput" in (c.get("reason_code", "") or "").lower() or "impute" in (c.get("reason_code", "") or "").lower())

        # count remaining invalid dates (use impacted_rows_clean for table)
        try:
            rem_invalid = impacted_rows_clean(clean_df, table)
        except Exception:
            rem_invalid = 0
        total_invalid_dates_remaining += rem_invalid

        rows.append({
            "table_name": table,
            "rows_before": rcount,
            "rows_after": ccount,
            "duplicates_removed": duplicates_removed,
            "remaining_issues": rem_invalid,
        })

    # totals
    summary = {
        "metric": "totals",
        "duplicates_removed": total_duplicates_removed,
        "corrections_applied": total_corrections,
        "imputations_applied": total_imputations,
        "invalid_dates_remaining": total_invalid_dates_remaining,
    }

    df_val = pd.DataFrame(rows)
    out_rows = CLEAN_DIR / "dq_validation_summary_by_table.csv"
    df_val.to_csv(out_rows, index=False)

    df_summary = pd.DataFrame([summary])
    out_summary = CLEAN_DIR / "dq_validation_summary_totals.csv"
    df_summary.to_csv(out_summary, index=False)

    return df_val, df_summary


# ================================================
# PART 7/10 — Run Coordination Utilities
# ================================================

# ---------------------------------------------------------
# LOAD RAW TABLES
# ---------------------------------------------------------
def load_raw_tables():
    raw_tables = {}
    raw_counts = {}

    for name, path in RAW_FILES.items():
        if path.exists():
            try:
                df = pd.read_csv(path, dtype=str)
                raw_tables[name] = df
                raw_counts[name] = len(df)
                logging.info(f"Loaded raw table '{name}' with {len(df)} rows.")
            except Exception as e:
                logging.error(f"Error loading raw table '{name}': {e}")
                raw_tables[name] = pd.DataFrame()
                raw_counts[name] = 0
        else:
            logging.warning(f"Raw file missing for table '{name}'.")
            raw_tables[name] = pd.DataFrame()
            raw_counts[name] = 0

    return raw_tables, raw_counts


# ---------------------------------------------------------
# ASSEMBLE CLEANED TABLE DICT
# ---------------------------------------------------------
def assemble_cleaned_tables(accounts, subs, churn, usage, tickets):
    return {
        "accounts": accounts,
        "subscriptions": subs,
        "churn_events": churn,
        "feature_usage": usage,
        "support_tickets": tickets,
    }


# ---------------------------------------------------------
# BUILD CROSSMAPS FOR CONSISTENCY LOGIC
# ---------------------------------------------------------
def build_crossmaps(accounts_df, subscriptions_df):
    # account_id_norm → churn_date_clean
    acct_churn_map = {}
    if (
        accounts_df is not None
        and not accounts_df.empty
        and "account_id_norm" in accounts_df.columns
        and "account_churn_date_clean" in accounts_df.columns
    ):
        acct_churn_map = accounts_df.set_index("account_id_norm")["account_churn_date_clean"].to_dict()

    # subscription_id_norm → end_date_clean
    sub_end_map = {}
    if (
        subscriptions_df is not None
        and not subscriptions_df.empty
        and "subscription_id_norm" in subscriptions_df.columns
        and "end_date_clean" in subscriptions_df.columns
    ):
        sub_end_map = subscriptions_df.set_index("subscription_id_norm")["end_date_clean"].to_dict()

    # subscription_id_norm → account_id_norm
    sub_to_acct = {}
    if (
        subscriptions_df is not None
        and not subscriptions_df.empty
        and "subscription_id_norm" in subscriptions_df.columns
        and "account_id_norm" in subscriptions_df.columns
    ):
        sub_to_acct = subscriptions_df.set_index("subscription_id_norm")["account_id_norm"].to_dict()

    return {
        "account_churn_map": acct_churn_map,
        "subscription_end_map": sub_end_map,
        "subscription_to_account": sub_to_acct,
    }


# ---------------------------------------------------------
# ENSURE ALL REQUIRED CLEANED TABLES HAVE FINALIZED FLAGS
# ---------------------------------------------------------
def finalize_all_dirty_flags(cleaned_tables):
    finalize_dirty_flags(cleaned_tables)
    # Safety check — if reason exists but dirty_flag is False, force dirty_flag True
    for name, df in cleaned_tables.items():
        if df.empty:
            continue
        mask = (df["dirty_row_reason"].astype(str).str.strip() != "") & (~df["dirty_row_flag"])
        if mask.any():
            df.loc[mask, "dirty_row_flag"] = True
            logging.warning(f"[{name}] Fixed rows with reason but dirty_flag=False → forced to True.")

# ================================================
# PART 8/10 — Data Quality Integration Layer
# ================================================


# ---------------------------------------------------------
# BUILD ALL DQ ARTIFACTS (summary, long, profiles, errors)
# ---------------------------------------------------------
def build_all_dq_outputs(raw_tables, cleaned_tables, crossmaps):
    dq_summary = []
    for tname, df in cleaned_tables.items():
        raw_df = raw_tables.get(tname, pd.DataFrame())
        pk_cols = PKS.get(tname, [])
        row = build_table_dq_metrics(tname, df, pk_cols, crossmaps, raw_df)
        dq_summary.append(row)

    dq_summary = pd.DataFrame(dq_summary)

    dq_long = build_dq_long(cleaned_tables, top_n=12)
    dq_profiles = build_dq_col_profile(cleaned_tables)
    dq_errors = build_dq_errors(cleaned_tables)
    dq_val_by_table, dq_val_totals = build_dq_validation(raw_tables, cleaned_tables, CORRECTIONS)

    return dq_summary, dq_long, dq_profiles, dq_errors

# ---------------------------------------------------------
# SAVE ALL DQ ARTIFACTS
# ---------------------------------------------------------
def save_dq_outputs(dq_summary, dq_long, dq_profiles, dq_errors):
    dq_summary.to_csv(CLEAN_DIR / "data_quality_summary.csv", index=False)

    # Save curated outputs with curated names
    dq_long.to_csv(CLEAN_DIR / "data_quality_long_curated.csv", index=False)
    dq_profiles.to_csv(CLEAN_DIR / "data_quality_col_profile_curated.csv", index=False)
    dq_errors.to_csv(CLEAN_DIR / "data_quality_errors_curated.csv", index=False)

    logging.info("Curated DQ artifacts saved → summary + curated long/col_profile/errors.")

# ---------------------------------------------------------
# SAVE VALIDATION + CORRECTIONS + DIRTY_REASON_LOOKUP
# ---------------------------------------------------------
def save_validation_and_corrections():
    pd.DataFrame(VALIDATION_ROWS).to_csv(
        CLEAN_DIR / "validation_summary_etl1_long.csv", index=False
    )

    pd.DataFrame(CORRECTIONS).to_csv(
        CLEAN_DIR / "corrections_log.csv", index=False
    )

    dr = [{"code": code, "description": desc} for code, desc in DIRTY_REASON_LOOKUP.items()]
    pd.DataFrame(dr).to_csv(
        CLEAN_DIR / "dirty_reason_lookup.csv", index=False
    )

    logging.info("Validation + correction logs saved.")
# ================================================
# PART 9/10 — Save Cleaned Tables + Final Artifact Writer
# ================================================


# ---------------------------------------------------------
# SAVE ALL CLEANED TABLES
# ---------------------------------------------------------
def save_all_cleaned(cleaned_tables):
    for name, df in cleaned_tables.items():
        out = CLEAN_DIR / f"{name}_clean.csv"

        # Remove *_norm columns
        norm_cols = [c for c in df.columns if c.endswith("_norm")]
        df_out = df.drop(columns=norm_cols, errors="ignore")

        df_out.to_csv(out, index=False)
        logging.info(f"Saved cleaned table: {out}")


# ---------------------------------------------------------
# FINAL ARTIFACT WRITER (CLEAN + VALIDATION + DQ)
# ---------------------------------------------------------
def write_all_artifacts(raw_tables, cleaned_tables, crossmaps):
    """
    Writes:
        - Cleaned CSVs
        - Validation summary (long)
        - Corrections log
        - Dirty reason lookup
        - All DQ artifacts (summary, long, profiling, errors)
    """

    # 1. Save cleaned tables
    save_all_cleaned(cleaned_tables)

    # 2. Save validation + corrections + reason lookup
    save_validation_and_corrections()

    # 3. Build DQ outputs
    dq_summary, dq_long, dq_profiles, dq_errors = build_all_dq_outputs(
        raw_tables, cleaned_tables, crossmaps
    )

    # 4. Save DQ outputs
    save_dq_outputs(dq_summary, dq_long, dq_profiles, dq_errors)

    logging.info("All ETL1 artifacts saved successfully.")
# ================================================
# PART 10/10 — Full ETL1 Orchestration
# ================================================

def run_etl1():
    logging.info("=== ETL1 RUN STARTED ===")

    # -----------------------------------------------------
    # 1. Load raw tables
    # -----------------------------------------------------
    raw_tables, raw_counts = load_raw_tables()

    # -----------------------------------------------------
    # 2. Clean tables in dependency order
    # -----------------------------------------------------
    churn_clean, churn_map = clean_churn_events(raw_tables.get("churn_events", pd.DataFrame()))
    logging.info("Churn events cleaned.")

    tickets_clean = clean_support_tickets(raw_tables.get("support_tickets", pd.DataFrame()))
    logging.info("Support tickets cleaned.")

    accounts_clean = clean_accounts(raw_tables.get("accounts", pd.DataFrame()), churn_map)
    logging.info("Accounts cleaned.")

    subscriptions_clean = clean_subscriptions(raw_tables.get("subscriptions", pd.DataFrame()), accounts_clean)
    logging.info("Subscriptions cleaned.")

    feature_usage_clean = clean_feature_usage(raw_tables.get("feature_usage", pd.DataFrame()), subscriptions_clean, accounts_clean)
    logging.info("Feature usage cleaned.")

    # -----------------------------------------------------
    # 3. Post-imputation duplicate detection
    # -----------------------------------------------------
    churn_clean = post_imputation_duplicate_detection(churn_clean, "churn_events", PKS["churn_events"])
    accounts_clean = post_imputation_duplicate_detection(accounts_clean, "accounts", PKS["accounts"])
    subscriptions_clean = post_imputation_duplicate_detection(subscriptions_clean, "subscriptions", PKS["subscriptions"])
    feature_usage_clean = post_imputation_duplicate_detection(feature_usage_clean, "feature_usage", PKS["feature_usage"])
    tickets_clean = post_imputation_duplicate_detection(tickets_clean, "support_tickets", PKS["support_tickets"])

    # -----------------------------------------------------
    # 4. Assemble cleaned_tables dictionary
    # -----------------------------------------------------
    cleaned_tables = assemble_cleaned_tables(
        accounts_clean,
        subscriptions_clean,
        churn_clean,
        feature_usage_clean,
        tickets_clean
    )

    # -----------------------------------------------------
    # 5. Finalize dirty flags
    # -----------------------------------------------------
    finalize_all_dirty_flags(cleaned_tables)
    logging.info("Dirty flags finalized.")

    # -----------------------------------------------------
    # 6. Build crossmaps for consistency checks
    # -----------------------------------------------------
    crossmaps = build_crossmaps(accounts_clean, subscriptions_clean)

    # -----------------------------------------------------
    # 7. Write all artifacts (clean + validation + DQ)
    # -----------------------------------------------------
    write_all_artifacts(raw_tables, cleaned_tables, crossmaps)

    # -----------------------------------------------------
    # 8. Print final ETL summary
    # -----------------------------------------------------
    summary = {
        "etl_run_id": ETL_RUN_ID,
        "timestamp": NOW_TS(),
        "accounts_out": len(accounts_clean),
        "subscriptions_out": len(subscriptions_clean),
        "churn_events_out": len(churn_clean),
        "feature_usage_out": len(feature_usage_clean),
        "support_tickets_out": len(tickets_clean),
    }

    print(json.dumps(summary, indent=2))
    logging.info("=== ETL1 RUN COMPLETED SUCCESSFULLY ===")

if __name__ == "__main__":
    run_etl1()
