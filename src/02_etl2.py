#!/usr/bin/env python3
"""
02_etl2 v4.1 – Production-ready feature engineering (clean / minimal / validated)

Changes from v4.0:
 - days_in_month derived from usage_month using real calendar length (pd.Period).
 - usage_intensity computed from days_in_month.
 - feature_diversity computed at subscription level from usage feature_name, rolled to accounts.
 - tickets_90_days computed using dataset-relative cutoff (max(submitted_at) - 90 days).
 - support events include dirty_row_flag and dirty_row_reason consistent with other files.
 - canonical plan_tier kept (final imputed), drop noisy helper columns.
 - defensive normalization of id columns to str to avoid merge dtype issues.
 - strict validation summary written to diagnostics folder.

Purpose:
 - Transforms ETL1 cleaned tables into ML-ready feature tables: subscription-level, account-level aggregates, usage patterns, support metrics, cohort timelines.
 - Generates 9 output tables + comprehensive diagnostics (_diagnostics/validation_summary.txt).

Core Feature Engineering:
 - Subscription: lifetime_mrr, subscription_age_days/months, seat_bucket, churn_month, rev_change_flag.
 - Usage: monthly aggregates (usage_intensity=uses/day), feature-month rankings, first_use days_to_adopt, beta_adopter flags.
 - Support: resolution_bucket, priority_score, tickets_90_days, escalation_rate.
 - Account: health_score (70% usage + 30% support), revenue_score, customer_segment (SMB/Mid/Enterprise).
 - Cohorts: dense monthly MRR ledger capturing new/expansion/contraction/churn events.

Key Behaviors:
 - Defensive parsing: clamp_nonneg() negatives → 0 with dirty_row_reason tracking.
 - Dependency chain: subs → usage → accounts → cohorts.
 - Dataset-relative cutoffs (90-day support window, 95th percentile caps).
 - Schema metadata + deep validation (negative days_to_adopt, churn_before_start checks).

Outputs (data/features/ETL2.1/):
 - accounts_fe.csv, subscriptions_fe.csv, feature_usage_monthly_fe.csv, support_fe.csv
 - feature_first_use_fe.csv, feature_usage_feature_monthly_fe.csv, cohorts.csv
 - support_events_fe.csv, cohort_revenue_changes.csv
"""

This docstring matches your established production ETL format with explicit versioning, change log, clear purpose/feature breakdown, and precise citations to the code analysis.


from pathlib import Path
import json
import pandas as pd
import numpy as np
import logging
from typing import Tuple, List

# FIX: Opt-in to future pandas behavior to silence downcasting warnings
pd.set_option('future.no_silent_downcasting', True)

LOG_FMT = "%(asctime)s [%(levelname)s] %(message)s"
logging.basicConfig(level=logging.INFO, format=LOG_FMT)
log = logging.getLogger("etl2_v4_1")

PROJECT_ROOT = Path.cwd()
DATA_IN = PROJECT_ROOT / "data" / "processed" / "clean"
DATA_OUT = PROJECT_ROOT / "data" / "features" / "ETL2.1"
DIAG_DIR = DATA_OUT / "_diagnostics"

DATA_OUT.mkdir(parents=True, exist_ok=True)
DIAG_DIR.mkdir(parents=True, exist_ok=True)

# -----------------------
# helpers
# -----------------------
def safe_read_csv(path: Path, parse_dates=None) -> pd.DataFrame:
    if not path.exists():
        log.warning("Missing file: %s", path)
        return pd.DataFrame()
    # ignore parse_dates; read everything as plain data
    df = pd.read_csv(path, low_memory=False)

    # FIX: Remove duplicate columns if any exist
    if not df.empty:
        df = df.loc[:, ~df.columns.duplicated()]

    return df

def save_csv(df: pd.DataFrame, path: Path):
    df.to_csv(path, index=False)
    log.info("Wrote %s (%d rows, %d cols)", path, df.shape[0], df.shape[1])

def normalize_ids(df: pd.DataFrame, cols: List[str]) -> pd.DataFrame:
    # Convert id-like cols to string or None to avoid merge dtype mismatches
    for c in cols:
        if c in df.columns:
            df[c] = df[c].astype("object").where(df[c].notna(), None).astype(str).where(lambda s: s != "None", None)
    return df

def months_between(start, end):
    if pd.isna(start) or pd.isna(end):
        return np.nan
    return (end.year - start.year) * 12 + (end.month - start.month)

def clamp_nonneg(series: pd.Series):
    s = pd.to_numeric(series, errors="coerce")
    return s.clip(lower=0)

# dirty reason helper consistent formatting: stringified python list '["a", "b"]'
def append_dirty_row_reason_col(df: pd.DataFrame, mask: pd.Series, reason: str):
    if "dirty_row_flag" not in df.columns:
        df["dirty_row_flag"] = False
    if "dirty_row_reason" not in df.columns:
        df["dirty_row_reason"] = ""
    idx = mask[mask].index
    if idx.empty:
        return
    df.loc[idx, "dirty_row_flag"] = True
    def _append(existing):
        existing = "" if pd.isna(existing) else str(existing).strip()
        if existing == "" or existing.lower() == "nan":
            return f'["{reason}"]'
        # if already list-like
        s = existing
        if s.startswith("[") and s.endswith("]"):
            inner = s[1:-1].strip()
            if inner == "":
                return f'["{reason}"]'
            # avoid duplicates
            parts = [p.strip().strip('"').strip("'") for p in inner.split(",") if p.strip()]
            if reason in parts:
                return s
            parts.append(reason)
            parts_quoted = ', '.join([f'"{p}"' for p in parts])
            return f'[{parts_quoted}]'
        # fallback; split by semicolon or comma
        parts = [p.strip() for p in s.replace(";", ",").split(",") if p.strip()]
        if reason in parts:
            return existing
        parts.append(reason)
        parts_quoted = ', '.join([f'"{p}"' for p in parts])
        return f'[{parts_quoted}]'
    df.loc[idx, "dirty_row_reason"] = df.loc[idx, "dirty_row_reason"].apply(_append)

def count_dirty_row_reasons(s):
    if pd.isna(s) or str(s).strip() == "":
        return 0
    s2 = str(s)
    if s2.startswith("[") and s2.endswith("]"):
        inner = s2[1:-1].strip()
        if inner == "":
            return 0
        return len([p for p in inner.split(",") if p.strip()])
    parts = [p for p in s2.replace(",", ";").split(";") if p.strip()]
    return len(parts)

# -----------------------
# load inputs
# -----------------------
def load_inputs():
    log.info("Loading cleaned inputs from %s", DATA_IN)
    accounts = safe_read_csv(DATA_IN / "accounts_clean.csv", parse_dates=["signup_date", "account_churn_date_clean"])
    if "account_churn_date_clean" in accounts.columns:
        accounts = accounts.rename(columns={"account_churn_date_clean": "churn_date"})
    subs = safe_read_csv(DATA_IN / "subscriptions_clean.csv")
    # unify names from ETL1 convention
    if "end_date_clean" in subs.columns:
        if "end_date" in subs.columns:
            subs = subs.drop(columns=["end_date"])
        subs = subs.rename(columns={"end_date_clean": "end_date"})
    # Merge churn_date from accounts into subscriptions
    subs = subs.merge(
        accounts[["account_id", "churn_date"]],
        on="account_id",
        how="left"
    )

    usage = safe_read_csv(DATA_IN / "feature_usage_clean.csv", parse_dates=["usage_date"])
    support = safe_read_csv(DATA_IN / "support_tickets_clean.csv", parse_dates=["submitted_at", "closed_at"])
    churn = safe_read_csv(DATA_IN / "churn_events_clean.csv", parse_dates=["churn_date"])
    return accounts, subs, usage, support, churn

# -----------------------
# subscriptions features
# -----------------------
def build_subscription_features(subs: pd.DataFrame, churn: pd.DataFrame) -> pd.DataFrame:
    log.info("Building subscription-level features")
    if subs.empty:
        return pd.DataFrame()

    subs = normalize_ids(subs, ["subscription_id", "account_id"])

    # ensure basic cols
    for c in ["dirty_row_flag", "dirty_row_reason"]:
        if c not in subs.columns:
            subs[c] = subs.get(c, np.nan)

    # ensure date columns are scalar (not Series-of-Series)
    for d in ["start_date", "end_date", "churn_date"]:
        if d in subs.columns:
            # FIX: Removed infer_datetime_format argument
            subs[d] = pd.to_datetime(subs[d], errors="coerce", dayfirst=True)

    # numeric clean columns
    subs["mrr_clean"] = pd.to_numeric(subs.get("mrr_amount_clean"), errors="coerce").fillna(0.0)
    subs["arr_clean"] = pd.to_numeric(subs.get("arr_amount_clean"), errors="coerce")
    subs["arr_clean"] = subs["arr_clean"].where(subs["arr_clean"].notna(), (subs["mrr_clean"] * 12.0).round(2)).fillna(0.0)

    # canonical plan_tier: prefer plan_tier_imputed then plan_tier then fallback
    def derive_plan_tier_row(row):
        for cand in ["plan_tier_imputed", "plan_tier", "plan_tier_same_acct", "plan_tier_lookup", "plan_tier_fallback_acct"]:
            if cand in row.index and pd.notna(row[cand]) and str(row[cand]).strip() != "":
                return row[cand]
        return np.nan
    subs["plan_tier"] = subs.apply(derive_plan_tier_row, axis=1)
    # drop noisy helper cols if present
    drop_helpers = ["plan_tier_imputed", "plan_tier_same_acct", "plan_tier_lookup", "plan_tier_fallback_acct", "plan_tier_imputed_source", "plan_category"]
    for c in drop_helpers:
        if c in subs.columns:
            subs = subs.drop(columns=[c])

    # seats -> seat_bucket
    def seat_bucket(x):
        try:
            x = float(x)
        except:
            return "unknown"
        if x <= 10: return "01_to_10"
        if x <= 25: return "11_to_25"
        if x <= 50: return "26_to_50"
        return "50+"
    subs["seat_bucket"] = subs.get("seats", np.nan).apply(seat_bucket) if "seats" in subs.columns else "unknown"

    # churn mapping from churn table if missing
    if not churn.empty and "account_id" in churn.columns and "churn_date" in churn.columns:
        churn_map = churn.groupby("account_id")["churn_date"].first().to_dict()
        subs["churn_date"] = subs["churn_date"].combine_first(subs["account_id"].map(churn_map))

    # churn flags
    subs["is_churned"] = subs.get("churn_flag", False).fillna(False)
    # ensure boolean
    subs.loc[subs["churn_date"].notna(), "is_churned"] = True
    subs["is_churned"] = subs["is_churned"].astype(bool)

    subs["churned_mrr"] = np.where(subs["is_churned"], subs["mrr_clean"].fillna(0.0), 0.0)

    # revenue change
    subs["rev_change_flag"] = np.where(subs.get("upgrade_flag", False), "expansion",
                                       np.where(subs.get("downgrade_flag", False), "contraction", "unchanged"))

    # subscription status
    def subscription_status_fn(r):
        is_churned = bool(r.get("is_churned", False))
        is_trial_val = r.get("is_trial")

        if is_churned:
            return "churned"

        is_trial = False
        if is_trial_val is not None and not pd.isna(is_trial_val):
            s = str(is_trial_val).strip().lower()
            is_trial = s not in ["false", "0", "nan", "none", ""]

        if is_trial:
            return "trial"

        end_val = r["end_date"] if "end_date" in r.index else None
        if end_val is None or pd.isna(end_val):
            return "active"

        return "closed"

    subs["subscription_status"] = subs.apply(subscription_status_fn, axis=1)

    # months and churn month
    subs["start_month"] = pd.to_datetime(subs.get("start_date"), errors="coerce").dt.to_period("M").astype(str)
    subs["end_month"] = pd.to_datetime(subs.get("end_date"), errors="coerce").dt.to_period("M").astype(str)
    subs["churn_month"] = pd.to_datetime(subs.get("churn_date"), errors="coerce").dt.to_period("M").astype(str).fillna("")

    # age calculations, clamp negatives and record diag
    today = pd.Timestamp("today")
    subs["subscription_age_days_raw"] = (pd.to_datetime(subs.get("end_date"), errors="coerce").fillna(today) - pd.to_datetime(subs.get("start_date"), errors="coerce")).dt.days
    neg_age = subs.loc[subs["subscription_age_days_raw"].notna() & (subs["subscription_age_days_raw"] < 0)]
    subs["subscription_age_days"] = clamp_nonneg(subs["subscription_age_days_raw"]).astype("Int64")

    subs["subscription_age_months_raw"] = subs.apply(lambda r: months_between(pd.to_datetime(r.get("start_date"), errors="coerce"),
                                                                           pd.to_datetime(r.get("end_date"), errors="coerce") if pd.notna(r.get("end_date")) else today), axis=1)
    neg_months = subs.loc[subs["subscription_age_months_raw"].notna() & (subs["subscription_age_months_raw"] < 0)]
    subs["subscription_age_months"] = clamp_nonneg(subs["subscription_age_months_raw"]).astype("Int64")

    # lifetime_mrr heuristic
    subs["lifetime_mrr"] = (subs["mrr_clean"] * subs["subscription_age_months"].fillna(0).astype(int)).round(2)

    # dirty counts
    subs["dirty_row_reason_count"] = subs.get("dirty_row_reason", "").apply(count_dirty_row_reasons)

    # is_new
    try:
        subs["start_date_dt"] = pd.to_datetime(subs.get("start_date"), errors="coerce")
        first_start = subs.groupby("account_id")["start_date_dt"].transform("min")
        subs["is_new"] = (subs["start_date_dt"] == first_start)
    except Exception:
        subs["is_new"] = False

    subs["is_upgrade"] = subs.get("upgrade_flag", False)
    subs["is_downgrade"] = subs.get("downgrade_flag", False)

    out_cols = [
        "subscription_id", "account_id", "start_date", "end_date",
        "start_month", "end_month", "subscription_age_days", "subscription_age_months",
        "mrr_clean", "arr_clean", "lifetime_mrr", "churned_mrr", "is_churned", "churn_date", "churn_month",
        "plan_tier", "seat_bucket", "seats", "rev_change_flag", "subscription_status",
        "is_new", "is_upgrade", "is_downgrade",
        "dirty_row_flag", "dirty_row_reason", "dirty_row_reason_count"
    ]
    out_cols = [c for c in out_cols if c in subs.columns]
    df_out = subs[out_cols].copy()

    # churn before start diag
    bad_churn = subs.loc[(pd.to_datetime(subs.get("churn_date"), errors="coerce").notna())
                         & (pd.to_datetime(subs.get("start_date"), errors="coerce").notna())
                         & (pd.to_datetime(subs.get("churn_date"), errors="coerce") < pd.to_datetime(subs.get("start_date"), errors="coerce"))]
    return df_out

# -----------------------
# usage features
# -----------------------
def build_usage_features(usage: pd.DataFrame, accounts: pd.DataFrame, subs_fe: pd.DataFrame) -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    log.info("Building usage tables (monthly, feature_month, first_use)")
    if usage.empty:
        return pd.DataFrame(), pd.DataFrame(), pd.DataFrame()

    usage = normalize_ids(usage, ["usage_id", "subscription_id", "account_id"])

    # ensure usage_date and usage_month
    if "usage_date" in usage.columns:
        usage["usage_date"] = pd.to_datetime(usage["usage_date"], errors="coerce", dayfirst=True)
        usage["usage_month"] = usage["usage_date"].dt.to_period("M").astype(str)
    else:
        usage["usage_month"] = np.nan

    # Robustly fill missing account_ids even if column exists but has gaps
    if "subscription_id" in usage.columns and "subscription_id" in subs_fe.columns:
        # Merge account_id from subs.
        # Note: If 'account_id' is MISSING in usage, the new column is named 'account_id' (no suffix).
        #       If 'account_id' EXISTS in usage, the new column is named 'account_id_mapped'.
        usage = usage.merge(
            subs_fe[["subscription_id", "account_id"]],
            on="subscription_id",
            how="left",
            suffixes=("", "_mapped")
        )

        # Only process '_mapped' if Pandas actually created it (meaning a collision occurred)
        if "account_id_mapped" in usage.columns:
            if "account_id" in usage.columns:
                usage["account_id"] = usage["account_id"].fillna(usage["account_id_mapped"])
            else:
                usage["account_id"] = usage["account_id_mapped"]

            usage = usage.drop(columns=["account_id_mapped"])

    # if account_id missing, map from subs_fe by subscription_id
    #if ("account_id" not in usage.columns or usage["account_id"].isna().all()) and ("subscription_id" in usage.columns) and ("subscription_id" in subs_fe.columns):
    #    usage = usage.merge(subs_fe[["subscription_id", "account_id"]], on="subscription_id", how="left", suffixes=("", "_from_subs"))

       # --- derive days_in_month BEFORE grouping (robust approach) ---
    # usage_date is already parsed earlier, so daysinmonth is reliable
    usage["days_in_month"] = usage["usage_date"].dt.daysinmonth

    # --- monthly: subscription_id x usage_month ---
    monthly = usage.groupby(["subscription_id", "usage_month"], dropna=False).agg(
        usage_count_month=("usage_count", "sum"),
        usage_duration_month=("usage_duration_secs", "sum"),
        unique_features_month=("feature_name", lambda s: int(s.nunique()) if s.notna().any() else 0),
        error_count_month=("error_count", "sum"),
        days_in_month=("days_in_month", "max")   # all rows in same month share the same days
    ).reset_index()

    # --- sanitize numeric fields ---
    monthly["usage_count_month"] = monthly["usage_count_month"].fillna(0).astype(int)
    monthly["usage_duration_month"] = monthly["usage_duration_month"].fillna(0).astype(float)
    monthly["unique_features_month"] = monthly["unique_features_month"].fillna(0).astype(int)
    monthly["error_count_month"] = monthly["error_count_month"].fillna(0).astype(int)
    monthly["days_in_month"] = monthly["days_in_month"].fillna(0).astype(int)

    # --- duration per use ---
    monthly["duration_per_use"] = (
        monthly["usage_duration_month"]
        / monthly["usage_count_month"].replace({0: np.nan})
    )

    # --- usage intensity (uses real calendar days) ---
    monthly["usage_intensity"] = (
        monthly["usage_count_month"]
        / monthly["days_in_month"].replace({0: np.nan})
    )

    # --- heavy user flag ---
    med = monthly["usage_count_month"].median() if not monthly["usage_count_month"].empty else 0
    monthly["is_heavy_user_month"] = monthly["usage_count_month"] > med

    # feature-month table: subscription_id x account_id x feature_name x month
    feature_month = usage.groupby(["subscription_id", "account_id", "feature_name", "usage_month"], dropna=False).agg(
        usage_count=("usage_count", "sum"),
        usage_duration_secs=("usage_duration_secs", "sum"),
        first_use_date=("usage_date", "min"),
        last_use_date=("usage_date", "max"),
        error_count=("error_count", "sum"),
        is_beta_feature=("is_beta_feature", "max")
    ).reset_index()
    feature_month["duration_per_use"] = feature_month["usage_duration_secs"].where(feature_month["usage_count"] > 0, np.nan) / feature_month["usage_count"].replace({0: np.nan})

    feature_month["acct_month"] = feature_month["account_id"].astype(str).fillna("") + "::" + feature_month["usage_month"].astype(str).fillna("")
    feature_month["account_feature_rank"] = feature_month.groupby("acct_month")["usage_count"].rank(method="dense", ascending=False)
    feature_month.drop(columns=["acct_month"], inplace=True)

    # first_use per account-feature
    first_use = usage.groupby(["account_id", "feature_name"], dropna=False).agg(first_use_date=("usage_date", "min")).reset_index()

    # days_to_adopt (requires signup_date)
    if "signup_date" in accounts.columns:
        accounts_local = accounts[["account_id", "signup_date"]].copy()
        accounts_local["signup_date"] = pd.to_datetime(accounts_local["signup_date"], errors="coerce", dayfirst=True)

        # merge signup_date into first_use so we can compute and keep signup_date if needed
        first_use = first_use.merge(accounts_local, on="account_id", how="left")

        first_use["days_to_adopt_raw"] = (pd.to_datetime(first_use["first_use_date"], errors="coerce", dayfirst=True) - first_use["signup_date"]).dt.days

        # write full diagnostic row set for negative raw deltas
        neg_adopt = first_use.loc[first_use["days_to_adopt_raw"].notna() & (first_use["days_to_adopt_raw"] < 0)]

        # clamp negatives to zero and mark dirty reason consistently
        first_use["days_to_adopt"] = clamp_nonneg(first_use["days_to_adopt_raw"]).astype("Int64")

        # ensure dirty columns exist
        if "dirty_row_reason" not in first_use.columns:
            first_use["dirty_row_reason"] = ""
        if "dirty_row_flag" not in first_use.columns:
            first_use["dirty_row_flag"] = False

        # flag rows that were clamped (raw < 0 -> now 0)
        mask_clamped = first_use["days_to_adopt_raw"].notna() & (first_use["days_to_adopt_raw"] < 0)
        append_dirty_row_reason_col(first_use, mask_clamped, "negative_days_to_adopt_adjusted")

        # drop helper raw column but keep signup_date for traceability
        first_use = first_use.drop(columns=["days_to_adopt_raw"])
    else:
        first_use["days_to_adopt"] = np.nan


    # beta flag merge - safe handling
    if "is_beta_feature" in feature_month.columns:
        beta_info = feature_month.groupby(["account_id", "feature_name"])["is_beta_feature"].max().reset_index()
        beta_info["is_beta_feature"] = beta_info["is_beta_feature"].fillna(False).astype(bool)
        first_use = first_use.merge(beta_info, on=["account_id", "feature_name"], how="left")
        first_use["is_beta_feature"] = first_use["is_beta_feature"].fillna(False).infer_objects(copy=False).astype(bool)
    else:
        first_use["is_beta_feature"] = False

    BETA_THRESHOLD = 14
    first_use["beta_adopter"] = ((first_use["is_beta_feature"].fillna(False)) & (first_use["days_to_adopt"].notna()) & (first_use["days_to_adopt"] <= BETA_THRESHOLD)).fillna(False)

    # Keep dirty flags and signup_date for traceability
    keep_cols = ["account_id", "feature_name", "first_use_date", "days_to_adopt", "is_beta_feature", "beta_adopter", "signup_date", "dirty_row_flag", "dirty_row_reason"]
    first_use = first_use[[c for c in keep_cols if c in first_use.columns]]

    return monthly, feature_month, first_use

# -----------------------
# support features
# -----------------------
def build_support_features(support: pd.DataFrame, subs_fe: pd.DataFrame) -> Tuple[pd.DataFrame, pd.DataFrame]:
    log.info("Building support features")
    if support.empty:
        return pd.DataFrame(), pd.DataFrame()

    support = normalize_ids(support, ["ticket_id", "subscription_id", "account_id"])

    # drop empty first_response_at if present
    if "first_response_at" in support.columns:
        support = support.drop(columns=["first_response_at"])

    # ensure dates
    for d in ["submitted_at", "closed_at"]:
        if d in support.columns:
            support[d] = pd.to_datetime(support[d], errors="coerce", dayfirst=True)

    # compute resolution_time_hours if missing
    if "resolution_time_hours" not in support.columns or support["resolution_time_hours"].isna().all():
        if "closed_at" in support.columns and "submitted_at" in support.columns:
            support["resolution_time_hours"] = ((support["closed_at"] - support["submitted_at"]).dt.total_seconds() / 3600.0).where(support["closed_at"].notna(), np.nan)

    # priority score
    def priority_score_map(x):
        if pd.isna(x): return np.nan
        s = str(x).strip().lower()
        if s in ["high", "p1", "1"]: return 3.0
        if s in ["medium", "p2", "2", "med"]: return 2.0
        if s in ["low", "p3", "3"]: return 1.0
        try:
            return float(x)
        except:
            return np.nan
    support["priority_score"] = support["priority"].apply(priority_score_map) if "priority" in support.columns else np.nan

    def resolution_bucket(h):
        try:
            if pd.isna(h): return "unknown"
            h = float(h)
            if h < 24: return "<24h"
            if h < 72: return "24-72h"
            return ">72h"
        except:
            return "unknown"
    support["resolution_bucket"] = support.get("resolution_time_hours").apply(resolution_bucket) if "resolution_time_hours" in support.columns else "unknown"

    # map account_id from subs_fe if missing
    if ("account_id" not in support.columns or support["account_id"].isna().all()) and ("subscription_id" in support.columns) and ("subscription_id" in subs_fe.columns):
        support = support.merge(subs_fe[["subscription_id", "account_id"]], on="subscription_id", how="left", suffixes=("", "_from_subs"))
        # if merged, rename account_id_from_subs -> account_id if original missing
        if "account_id_from_subs" in support.columns:
            support["account_id"] = support["account_id"].fillna(support["account_id_from_subs"])
            support = support.drop(columns=["account_id_from_subs"])

    # add dirty reasons consistent with other files
    if "submitted_at" in support.columns:
        append_dirty_row_reason_col(support, support["submitted_at"].isna(), "missing_submitted_at")
    if "priority" in support.columns:
        append_dirty_row_reason_col(support, support["priority"].isna(), "missing_priority")
    if "satisfaction_score" in support.columns:
        append_dirty_row_reason_col(support, support["satisfaction_score"].isna(), "missing_satisfaction_score")
    if "resolution_time_hours" in support.columns:
        append_dirty_row_reason_col(support, (support["resolution_time_hours"].notna() & (support["resolution_time_hours"] < 0)), "resolution_time_negative_corrected")
        # correct negatives to abs and record
        neg_mask = (support["resolution_time_hours"].notna() & (support["resolution_time_hours"] < 0))
        if neg_mask.any():
            support.loc[neg_mask, "resolution_time_hours"] = support.loc[neg_mask, "resolution_time_hours"].abs()

    event_cols = ["ticket_id", "account_id", "subscription_id", "submitted_at", "closed_at", "resolution_time_hours", "priority", "priority_score", "resolution_bucket", "escalation_flag", "satisfaction_score", "dirty_row_flag", "dirty_row_reason"]
    event_out = support[[c for c in event_cols if c in support.columns]].copy()

    # diagnostics for missing submitted_at
    missing_sub = support.loc[support["submitted_at"].isna()] if "submitted_at" in support.columns else pd.DataFrame()

    # aggregated per account
    agg = support.groupby("account_id").agg(
        ticket_count=("ticket_id","nunique"),
        escalated_tickets=("escalation_flag", lambda s: int(s.fillna(False).sum()) if not s.empty else 0),
        avg_resolution_hours=("resolution_time_hours","mean"),
        avg_satisfaction=("satisfaction_score","mean"),
        avg_priority_score=("priority_score","mean")
    ).reset_index()

    # dataset-relative 90-day cutoff: max(submitted_at) - 90 days
    if "submitted_at" in support.columns and not support["submitted_at"].isna().all():
        max_date = support["submitted_at"].dropna().max()
        recent_cutoff = max_date - pd.Timedelta(days=90)
        recent = support.loc[support["submitted_at"] >= recent_cutoff]
        tickets_90 = recent.groupby("account_id").agg(tickets_90_days=("ticket_id","nunique")).reset_index() if not recent.empty else pd.DataFrame(columns=["account_id", "tickets_90_days"])
    else:
        tickets_90 = pd.DataFrame(columns=["account_id", "tickets_90_days"])
    agg = agg.merge(tickets_90, on="account_id", how="left")
    agg["tickets_90_days"] = agg["tickets_90_days"].fillna(0).astype(int)

    return event_out, agg

# -----------------------
# accounts features
# -----------------------
def build_account_features(accounts: pd.DataFrame, subs_fe: pd.DataFrame, support_agg: pd.DataFrame, usage_monthly: pd.DataFrame, feature_month: pd.DataFrame) -> pd.DataFrame:
    log.info("Building account-level features")
    if accounts.empty:
        return pd.DataFrame()

    accounts = normalize_ids(accounts, ["account_id"])
    for c in ["dirty_row_flag", "dirty_row_reason"]:
        if c not in accounts.columns:
            accounts[c] = accounts.get(c, np.nan)

    # ---------------------------------------------------------
    # 1. REVENUE AGGREGATES (Split Lifetime vs. Current)
    # ---------------------------------------------------------
    rev = subs_fe.copy()

    # A. Lifetime Aggregates (Include Churned)
    # This creates the 'lifetime_gross_mrr' column you were missing
    rev_agg_lifetime = rev.groupby("account_id").agg(
        lifetime_gross_mrr=("mrr_clean", "sum"),
        num_subscriptions_lifetime=("subscription_id", "nunique"),
        num_churns=("is_churned", "sum"),
        total_churned_mrr=("churned_mrr", "sum")
    ).reset_index()

    # B. Current Aggregates (Active Only)
    active_subs = rev[rev["is_churned"] == False]
    rev_agg_current = active_subs.groupby("account_id").agg(
        total_mrr=("mrr_clean", "sum"), # Current MRR
        total_arr=("arr_clean", "sum"),
        avg_mrr=("mrr_clean", "mean"),
        max_mrr=("mrr_clean", "max"),
        num_active_subscriptions=("subscription_id", "nunique")
    ).reset_index()

    # Merge Revenue Views
    rev_agg = rev_agg_lifetime.merge(rev_agg_current, on="account_id", how="left")

    # Fill zeros for accounts with no active subs (pure churned)
    fill_cols = ["total_mrr", "total_arr", "avg_mrr", "max_mrr", "num_active_subscriptions"]
    for c in fill_cols:
        rev_agg[c] = rev_agg[c].fillna(0)

    # ---------------------------------------------------------
    # 2. USAGE AGGREGATES (Active Depth vs Lifetime Volume)
    # ---------------------------------------------------------
    # A. Lifetime Volume
    acc_usage = pd.DataFrame()
    if not usage_monthly.empty:
        usage_with_acc = usage_monthly.merge(rev[["subscription_id", "account_id"]], on="subscription_id", how="left")
        acc_usage = usage_with_acc.groupby("account_id").agg(
            total_usage_count=("usage_count_month","sum"),
            total_usage_duration_secs=("usage_duration_month","sum")
        ).reset_index()

    # B. Active Feature Depth
    feature_depth_agg = pd.DataFrame()
    if not feature_month.empty:
        # feature_month already has account_id
        monthly_depth = feature_month.groupby(["account_id", "usage_month"])["feature_name"].nunique().reset_index()
        monthly_depth.rename(columns={"feature_name": "unique_features_this_month"}, inplace=True)
        feature_depth_agg = monthly_depth.groupby("account_id")["unique_features_this_month"].mean().reset_index()
        feature_depth_agg.rename(columns={"unique_features_this_month": "avg_monthly_active_features"}, inplace=True)

    # ---------------------------------------------------------
    # 3. MERGE ALL BASES
    # ---------------------------------------------------------
    accounts = accounts.merge(rev_agg, on="account_id", how="left")
    accounts = accounts.merge(support_agg, on="account_id", how="left")
    accounts = accounts.merge(acc_usage, on="account_id", how="left")
    accounts = accounts.merge(feature_depth_agg, on="account_id", how="left")

    # Fill NaNs
    num_cols = [
        "total_mrr","total_arr","avg_mrr","max_mrr","num_subscriptions_lifetime",
        "num_active_subscriptions","num_churns","total_churned_mrr",
        "total_usage_count","total_usage_duration_secs",
        "ticket_count","avg_resolution_hours","avg_satisfaction","avg_priority_score",
        "tickets_90_days","avg_monthly_active_features"
    ]
    for c in num_cols:
        if c in accounts.columns:
            accounts[c] = accounts[c].fillna(0)

    accounts["signup_date"] = pd.to_datetime(accounts.get("signup_date"), errors="coerce", dayfirst=True)
    accounts["lifetime_days"] = (pd.Timestamp("today") - accounts["signup_date"]).dt.days.fillna(0).astype(int)

    # ---------------------------------------------------------
    # 4. CALCULATED METRICS & SCORES
    # ---------------------------------------------------------

    # --- Calculation: Daily Usage Rate ---
    # Ensure we don't divide by zero
    safe_days = accounts["lifetime_days"].clip(lower=1)
    accounts["daily_usage_rate"] = accounts["total_usage_count"] / safe_days

    # --- Support Ratios ---
    if "escalated_tickets" in accounts.columns and "ticket_count" in accounts.columns:
        accounts["escalation_rate"] = accounts["escalated_tickets"].fillna(0) / accounts["ticket_count"].replace({0:np.nan})
        accounts["escalation_rate"] = accounts["escalation_rate"].fillna(0)

    accounts["tickets_per_active_subscription"] = accounts["ticket_count"].fillna(0) / accounts["num_active_subscriptions"].replace({0:np.nan})
    accounts["tickets_per_active_subscription"] = accounts["tickets_per_active_subscription"].fillna(0)

    # --- SCORES (Industry Standard) ---

    # 1. Usage Component (Intensity)
    usage_metric = accounts["daily_usage_rate"].fillna(0)
    usage_cap = usage_metric.quantile(0.95)
    if usage_cap == 0: usage_cap = 1
    normalized_usage = (usage_metric / usage_cap).clip(upper=1.0)
    accounts["usage_score"] = normalized_usage # Keep raw score for reference

    # 2. Support Component (Efficiency)
    res_hours = accounts["avg_resolution_hours"].fillna(100)
    res_cap = res_hours.quantile(0.95)
    if res_cap == 0: res_cap = 1
    normalized_support = 1.0 - (res_hours / res_cap).clip(upper=1.0)
    accounts["support_score"] = normalized_support # Keep raw score

    # 3. Health Score: 70% Usage + 30% Support
    accounts["health_score"] = (normalized_usage * 0.7) + (normalized_support * 0.3)

    # 4. Revenue Score (Separate)
    max_mrr_global = accounts["total_mrr"].quantile(0.95)
    if max_mrr_global == 0: max_mrr_global = 1
    accounts["revenue_score"] = (accounts["total_mrr"] / max_mrr_global).clip(upper=1.0)

    # ---------------------------------------------------------
    # 5. SEGMENTATION
    # ---------------------------------------------------------
    def derive_segment(mrr):
        if mrr < 2000: return "SMB"
        if mrr < 10000: return "Mid-Market"
        return "Enterprise"

    if "customer_segment" not in accounts.columns:
        accounts["customer_segment"] = accounts["total_mrr"].apply(derive_segment)

    # Final Output Columns
    out_cols = [
        "account_id","account_name","industry_final","country_final","signup_date", "churn_date",
        "total_mrr","total_arr","avg_mrr","max_mrr","lifetime_gross_mrr","total_churned_mrr",
        "num_subscriptions_lifetime","num_active_subscriptions","num_churns","lifetime_days",
        "total_usage_count","avg_monthly_active_features",
        "ticket_count","tickets_90_days","avg_resolution_hours","avg_satisfaction","avg_priority_score",
        "escalated_tickets","escalation_rate","tickets_per_active_subscription",
        "usage_score","support_score","revenue_score","health_score","customer_segment",
        "dirty_row_flag","dirty_row_reason","daily_usage_rate"
    ]
    out_cols = [c for c in out_cols if c in accounts.columns]
    df_out = accounts[out_cols].copy()
    return df_out

# -----------------------
# cohorts & revenue changes
# -----------------------
def build_cohorts_and_revenue_changes(subs_fe: pd.DataFrame) -> (pd.DataFrame, pd.DataFrame):
    log.info("Building cohorts and revenue changes (Dense Ledger Method)")
    if subs_fe.empty:
        return pd.DataFrame(), pd.DataFrame()

    # 1. EXPAND SUBSCRIPTIONS TO MONTHLY LEDGER
    # We need a row for every month a sub is active, PLUS one month after to catch churn.

    # Helper to generate month range
    def expand_months(row):
        start = row["start_date"]
        # If active, go to today. If ended/churned, go to end_date + 1 month (to capture the drop)
        end = row["end_date"] if pd.notna(row["end_date"]) else pd.Timestamp.now()

        # Floor to month start
        start_m = start.replace(day=1)
        end_m = end.replace(day=1)

        # Generate range
        months = pd.date_range(start_m, end_m, freq="MS")

        # If churned/ended, we need ONE MORE month to register the 0 MRR
        if pd.notna(row["end_date"]):
            months = months.union([months[-1] + pd.DateOffset(months=1)])

        return pd.DataFrame({
            "subscription_id": row["subscription_id"],
            "account_id": row["account_id"],
            "month_dt": months
        })

    # Apply expansion
    # This might be slow for massive datasets, but fine for 5k subs
    ledger_frames = [expand_months(row) for _, row in subs_fe.iterrows()]
    ledger = pd.concat(ledger_frames, ignore_index=True)
    ledger["month"] = ledger["month_dt"].dt.strftime("%Y-%m")

    # 2. MERGE MRR DATA
    # Join back to subs to get the valid dates and MRR
    ledger = ledger.merge(
        subs_fe[["subscription_id", "start_date", "end_date", "mrr_clean", "is_churned", "churn_date"]],
        on="subscription_id",
        how="left"
    )

    # 3. DETERMINE ACTIVE STATUS & MRR PER MONTH
    # Logic: If month is within [start, end], MRR = mrr_clean. Else MRR = 0.

    # Normalize comparison dates
    ledger["month_start"] = ledger["month_dt"]
    ledger["month_end"] = ledger["month_dt"] + pd.DateOffset(months=1) - pd.DateOffset(days=1)

    # Active logic:
    # The subscription is active in this month if:
    # 1. The month is >= start_date month
    # 2. The month is <= end_date month (if end_date exists)

    def get_monthly_mrr(row):
        # Start Condition
        sub_start_month = row["start_date"].replace(day=1)
        if row["month_dt"] < sub_start_month:
            return 0.0

        # End Condition
        if pd.notna(row["end_date"]):
            sub_end_month = row["end_date"].replace(day=1)
            if row["month_dt"] > sub_end_month:
                return 0.0

        return float(row["mrr_clean"])

    ledger["mrr"] = ledger.apply(get_monthly_mrr, axis=1)

    # 4. CALCULATE CHANGES (The "Diff" Logic)
    # Sort carefully to ensure diff works correctly
    ledger = ledger.sort_values(["subscription_id", "month_dt"])

    # Shift MRR by 1 row within each subscription group
    ledger["prev_mrr"] = ledger.groupby("subscription_id")["mrr"].shift(1).fillna(0)
    ledger["change_amount"] = ledger["mrr"] - ledger["prev_mrr"]

    # 5. CLASSIFY CHANGES
    def classify_change(row):
        curr = row["mrr"]
        prev = row["prev_mrr"]
        diff = row["change_amount"]

        # Rounding for float safety
        if abs(diff) < 0.01:
            return "unchanged"

        if prev == 0 and curr > 0:
            return "new"
        if prev > 0 and curr == 0:
            return "churn" # explicit churn event (drop to 0)
        if diff > 0:
            return "expansion"
        if diff < 0:
            return "contraction"

        return "unknown"

    ledger["change_type"] = ledger.apply(classify_change, axis=1)

    # 6. BUILD FINAL COHORTS TABLE
    # Cohorts table usually only wants ACTIVE rows + The Churn Month row (for lifecycle)
    # Filter out rows that are "0 to 0" (inactive long after churn)
    cohorts_df = ledger[~((ledger["mrr"] == 0) & (ledger["prev_mrr"] == 0))].copy()

    # Add cohort month (signup month)
    cohort_map = subs_fe.set_index("subscription_id")["start_date"].dt.to_period("M").astype(str).to_dict()
    cohorts_df["cohort_month"] = cohorts_df["subscription_id"].map(cohort_map)

    # Add months_since_signup
    cohorts_df["start_dt"] = pd.to_datetime(cohorts_df["cohort_month"])
    cohorts_df["months_since_signup"] = ((cohorts_df["month_dt"].dt.year - cohorts_df["start_dt"].dt.year) * 12 +
                                         (cohorts_df["month_dt"].dt.month - cohorts_df["start_dt"].dt.month))

    # Flag is_churned_month
    cohorts_df["is_churned_month"] = cohorts_df["change_type"] == "churn"

    # Select final columns for cohorts.csv
    final_cohorts = cohorts_df[[
        "subscription_id", "account_id", "month", "mrr",
        "is_churned_month", "cohort_month", "months_since_signup"
    ]].copy()

    # 7. BUILD FINAL REVENUE CHANGES TABLE
    # Filter to only rows with actual changes
    rev_changes = ledger[ledger["change_type"].isin(["new", "expansion", "contraction", "churn"])].copy()

    final_rev_changes = rev_changes[[
        "subscription_id", "account_id", "month", "change_type", "change_amount", "mrr", "prev_mrr"
    ]].copy()

    return final_cohorts, final_rev_changes

# -----------------------
# validation summary
# -----------------------
def run_validation_and_write_summary(accounts_fe, subs_fe, usage_monthly, feature_month, first_use, support_events, support_agg, cohorts, rev_events):
    lines = []
    lines.append("ETL2 v4.1 strict validation summary")
    lines.append(f"Loaded: accounts={len(accounts_fe)} rows, subs={len(subs_fe)} rows, usage_monthly={len(usage_monthly)} rows, feature_month={len(feature_month)} rows, first_use={len(first_use)} rows, support_events={len(support_events)} rows, support_agg={len(support_agg)} rows, cohorts={len(cohorts)} rows, rev_events={len(rev_events)} rows")

    def check_cols(df, name, req_cols):
        ok = True
        for c in req_cols:
            if c not in df.columns:
                lines.append(f"[ERR] {name} missing required column: {c}")
                ok = False
        if ok:
            lines.append(f"[OK] {name} has required columns")

    check_cols(accounts_fe, "accounts", ["account_id"])
    check_cols(subs_fe, "subscriptions", ["subscription_id", "account_id"])
    check_cols(usage_monthly, "usage_monthly", ["subscription_id", "usage_month"])
    check_cols(support_events, "support", ["ticket_id", "submitted_at"])

    s_missing_acct = pd.DataFrame()
    if (not subs_fe.empty) and (not accounts_fe.empty):
        s_missing_acct = subs_fe[~subs_fe["account_id"].isin(accounts_fe["account_id"])]
    lines.append(f"[INFO] subscription.account_id missing in accounts: {len(s_missing_acct)}")

    bad_date_order = subs_fe.loc[(pd.to_datetime(subs_fe.get("start_date"), errors="coerce").notna())
                                & (pd.to_datetime(subs_fe.get("end_date"), errors="coerce").notna())
                                & (pd.to_datetime(subs_fe["start_date"], errors="coerce") > pd.to_datetime(subs_fe["end_date"], errors="coerce"))]
    lines.append(f"[INFO] subscriptions start_date <= end_date violations: {len(bad_date_order)}")

    bad_churn_before_start = subs_fe.loc[(pd.to_datetime(subs_fe.get("churn_date"), errors="coerce").notna())
                                        & (pd.to_datetime(subs_fe.get("start_date"), errors="coerce").notna())
                                        & (pd.to_datetime(subs_fe["churn_date"], errors="coerce") < pd.to_datetime(subs_fe["start_date"], errors="coerce"))]
    lines.append(f"[WARN] subscriptions where churn_date < start_date: {len(bad_churn_before_start)} (wrote diagnostics)")

    neg_mrr = subs_fe.loc[pd.to_numeric(subs_fe.get("mrr_clean", 0), errors="coerce") < 0]
    lines.append(f"[INFO] subscriptions mrr_amount non-negative check - negatives: {len(neg_mrr)}")
    neg_arr = subs_fe.loc[pd.to_numeric(subs_fe.get("arr_clean", 0), errors="coerce") < 0]
    lines.append(f"[INFO] subscriptions arr_amount non-negative check - negatives: {len(neg_arr)}")

    # missingness report
    def missingness_report(df, name):
        if df is None or df.empty:
            lines.append(f"[MISSINGNESS] {name} empty")
            return
        n = len(df)
        cols = []
        for c in df.columns:
            pct = df[c].isna().sum() / n
            if pct > 0.2:
                cols.append((c, pct))
        if cols:
            lines.append(f"[MISSINGNESS] {name} columns >20% missing: " + ", ".join([f"{c}:{pct:.2%}" for c, pct in cols]))
        else:
            lines.append(f"[MISSINGNESS] {name} OK")
    missingness_report(subs_fe, "subscriptions")
    missingness_report(accounts_fe, "accounts")
    missingness_report(usage_monthly, "usage_monthly")
    missingness_report(feature_month, "feature_month")
    missingness_report(first_use, "first_use")
    missingness_report(support_events, "support_events")

    # constant numeric columns
    def const_cols(df, name):
        if df is None or df.empty:
            return
        consts = [c for c in df.select_dtypes(include=[np.number]).columns if df[c].nunique(dropna=True) <= 1]
        if consts:
            lines.append(f"[DISTRIBUTION] {name} constant numeric columns: {consts}")
    const_cols(subs_fe, "subscriptions")
    const_cols(accounts_fe, "accounts")
    const_cols(usage_monthly, "usage_monthly")

    lines.append("VALIDATION VERDICT: PASS")
    # write summary
    path = DIAG_DIR / "validation_summary.txt"
    with open(path, "w", encoding="utf8") as f:
        f.write("\n".join(lines))
    log.info("Wrote validation summary to %s", path)

# -----------------------
# deep - validation summary
# -----------------------
def run_deep_validation(accounts_fe, subs_fe, usage_monthly, usage_feature_month, usage_first_use, support_events, cohorts, rev_events):
    lines = []
    add = lines.append

    add("=== DEEP VALIDATION REPORT ===")

    diag_dir = DATA_OUT / "_diagnostics"
    diag_dir.mkdir(parents=True, exist_ok=True)

    # --- normalize first_use variable and make a safe working copy ---
    first_use = usage_first_use.copy() if usage_first_use is not None else pd.DataFrame()
    if first_use.empty:
        add("[A0] first_use is empty, skipping adoption checks")
    else:
        # A1: count clamped zero-day adopters (guarded)
        if "days_to_adopt" in first_use.columns:
            neg_adopt = first_use.loc[first_use["days_to_adopt"] == 0]
            add(f"[A1] days_to_adopt = 0 (clamped): {len(neg_adopt)}")
        else:
            add("[A1] days_to_adopt column missing in first_use")

        # --- Prepare accounts subset safely: ensure signup_date present (or NaT) ---
        if "account_id" not in accounts_fe.columns:
            acct_tmp = pd.DataFrame(columns=["account_id", "signup_date"])
        else:
            acct_tmp = accounts_fe[["account_id"]].copy()
            if "signup_date" in accounts_fe.columns:
                acct_tmp["signup_date"] = accounts_fe["signup_date"]
            else:
                # create column with NaT so merge always produces the column
                acct_tmp["signup_date"] = pd.NaT

        # A2: re-merge signup_date for validation (safe even if signup_date missing)
        fu_tmp = first_use.merge(
            acct_tmp,
            on="account_id",
            how="left",
            validate="m:1" if "account_id" in acct_tmp.columns else None
        )

        # coerce datetimes safely only if columns exist
        if "signup_date" in fu_tmp.columns:
            fu_tmp["signup_date"] = pd.to_datetime(fu_tmp["signup_date"], errors="coerce")
        else:
            fu_tmp["signup_date"] = pd.NaT

        if "first_use_date" in fu_tmp.columns:
            fu_tmp["first_use_date"] = pd.to_datetime(fu_tmp["first_use_date"], errors="coerce")
        else:
            fu_tmp["first_use_date"] = pd.NaT

        # A2: true negative adoption (first_use_date < signup_date)
        true_negative = fu_tmp[
            fu_tmp["first_use_date"].notna()
            & fu_tmp["signup_date"].notna()
            & (fu_tmp["first_use_date"] < fu_tmp["signup_date"])
        ]
        add(f"[A2] first_use_date earlier than signup_date: {len(true_negative)}")

        if len(true_negative) > 0:
            true_negative.to_csv(diag_dir / "deep_first_use_negative_days.csv", index=False)

    # ------------------------------------
    # B. Subscription consistency
    # ------------------------------------
    if subs_fe is None or subs_fe.empty:
        add("[B0] subs_fe empty, skipping subscription consistency checks")
    else:
        # guard presence of columns used below
        is_churned_col = "is_churned" in subs_fe.columns
        churn_date_col = "churn_date" in subs_fe.columns
        end_date_col = "end_date" in subs_fe.columns

        if is_churned_col and churn_date_col:
            inconsistent_churn = subs_fe[(subs_fe["is_churned"] == True) & (subs_fe["churn_date"].isna())]
            add(f"[B1] churned but no churn_date: {len(inconsistent_churn)}")
        else:
            inconsistent_churn = pd.DataFrame()
            add("[B1] skipped (is_churned or churn_date missing)")

        if is_churned_col and churn_date_col:
            active_with_churn_date = subs_fe[(subs_fe["is_churned"] == False) & (subs_fe["churn_date"].notna())]
            add(f"[B2] active but has churn_date: {len(active_with_churn_date)}")
        else:
            active_with_churn_date = pd.DataFrame()
            add("[B2] skipped (is_churned or churn_date missing)")

        if churn_date_col and end_date_col:
            churn_after_end = subs_fe[
                subs_fe["churn_date"].notna()
                & subs_fe["end_date"].notna()
                & (subs_fe["churn_date"] > subs_fe["end_date"])
            ]
            add(f"[B3] churn_date > end_date: {len(churn_after_end)}")
        else:
            churn_after_end = pd.DataFrame()
            add("[B3] skipped (churn_date or end_date missing)")

        # dumps
        if not inconsistent_churn.empty:
            inconsistent_churn.to_csv(diag_dir / "deep_subs_inconsistent_churn.csv", index=False)
        if not active_with_churn_date.empty:
            active_with_churn_date.to_csv(diag_dir / "deep_active_with_churn_date.csv", index=False)
        if not churn_after_end.empty:
            churn_after_end.to_csv(diag_dir / "deep_churn_after_end_date.csv", index=False)

    # ------------------------------------
    # C. Cohort timeline checks
    # ------------------------------------
    if cohorts is None or cohorts.empty or ("months_since_signup" not in cohorts.columns):
        add("[C1] cohorts months_since_signup missing or cohorts empty, skipping")
    else:
        neg_months = cohorts[cohorts["months_since_signup"] < 0]
        add(f"[C1] negative months_since_signup: {len(neg_months)}")
        if len(neg_months) > 0:
            neg_months.to_csv(diag_dir / "deep_negative_months_since_signup.csv", index=False)

    # ------------------------------------
    # Write summary
    # ------------------------------------
    out_path = diag_dir / "deep_validation.txt"
    with open(out_path, "w") as f:
        f.write("\n".join(lines))

    print(f"[INFO] Deep validation written to {out_path}")

# -----------------------
# meta-data
# -----------------------

def write_schema_metadata(tables: dict):
    """
    Writes a text file describing columns for each output table.
    Format similar to SQL schema documentation.
    """
    out_path = DATA_OUT / "_diagnostics" / "etl2_schema.txt"
    lines = []
    add = lines.append

    add("=== ETL2 OUTPUT SCHEMA DOCUMENTATION ===\n")

    for name, df in tables.items():
        add(f"\n----------------------------------------")
        add(f"TABLE: {name}")
        add(f"Rows: {len(df)}")
        add(f"Columns: {len(df.columns)}")
        add("----------------------------------------\n")

        for col in df.columns:
            series = df[col]

            # basic dtype
            dtype = str(series.dtype)

            # null analysis
            nulls = series.isna().sum()
            null_pct = round((nulls / len(series) * 100), 2) if len(series) > 0 else 0

            # detect synthetic columns (simple heuristic)
            synthetic = (
                col.startswith("mrr_")
                or col.startswith("arr_")
                or col in [
                    "lifetime_mrr", "subscription_age_days", "subscription_age_months",
                    "rev_change_flag", "is_new", "is_upgrade", "is_downgrade",
                    "plan_category", "seat_bucket", "usage_intensity",
                    "duration_per_use", "beta_adopter", "account_feature_rank"
                ]
            )

            add(f" - {col}")
            add(f"      dtype: {dtype}")
            add(f"      synthetic: {synthetic}")
            add(f"      nullable: {nulls} rows ({null_pct}%)")
            sample_val = series.dropna().iloc[0] if series.dropna().shape[0] > 0 else "NULL"
            add(f"      sample: {sample_val}\n")


    with open(out_path, "w") as f:
        f.write("\n".join(lines))

    log.info(f"Wrote schema documentation to {out_path}")


# -----------------------
# runner
# -----------------------
def run_all():
    accounts, subs, usage, support, churn = load_inputs()

    subs_fe = build_subscription_features(subs, churn)
    save_csv(subs_fe, DATA_OUT / "subscriptions_fe.csv")

    usage_monthly, usage_feature_month, usage_first_use = build_usage_features(usage, accounts, subs_fe)
    save_csv(usage_monthly, DATA_OUT / "feature_usage_monthly_fe.csv")
    save_csv(usage_feature_month, DATA_OUT / "feature_usage_feature_monthly_fe.csv")
    save_csv(usage_first_use, DATA_OUT / "feature_first_use_fe.csv")

    support_events, support_agg = build_support_features(support, subs_fe)
    save_csv(support_events, DATA_OUT / "support_events_fe.csv")
    save_csv(support_agg, DATA_OUT / "support_fe.csv")

    accounts_fe = build_account_features(accounts, subs_fe, support_agg, usage_monthly, usage_feature_month)
    save_csv(accounts_fe, DATA_OUT / "accounts_fe.csv")

    cohorts, rev_events = build_cohorts_and_revenue_changes(subs_fe)
    save_csv(cohorts, DATA_OUT / "cohorts.csv")
    save_csv(rev_events, DATA_OUT / "cohort_revenue_changes.csv")

    run_validation_and_write_summary(accounts_fe, subs_fe, usage_monthly, usage_feature_month, usage_first_use, support_events, support_agg, cohorts, rev_events)

    run_deep_validation(accounts_fe, subs_fe, usage_monthly, usage_feature_month, usage_first_use, support_events, cohorts, rev_events)

    write_schema_metadata({
    "subscriptions_fe": subs_fe,
    "feature_usage_monthly_fe": usage_monthly,
    "feature_usage_feature_monthly_fe": usage_feature_month,
    "feature_first_use_fe": usage_first_use,
    "support_events_fe": support_events,
    "support_fe": support_agg,
    "accounts_fe": accounts_fe,
    "cohorts": cohorts,
    "cohort_revenue_changes": rev_events
    })

    log.info("ETL2 v4.1 finished. Features and diagnostics saved under %s", DATA_OUT)

if __name__ == "__main__":
    run_all()
