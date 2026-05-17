"""
Advanced preprocessing pipeline for the 1999 Czech Financial / PKDD'99 / Berka banking dataset.

Goal
----
Create a leakage-aware, one-row-per-loan classification matrix.

Designed for:
- Kaggle versions with CSV files

Core rule:
For a loan granted at `loan_date`, transaction and card features are computed only from
events that happened before or on that date. Transactions strictly use trans_date < loan_date
by default because balances after loan start can leak repayment outcome.
"""

from __future__ import annotations

import os
import re
import warnings
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Literal, Mapping, Optional, Sequence, Tuple

import numpy as np
import pandas as pd
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.compose import ColumnTransformer, make_column_selector
from sklearn.ensemble import HistGradientBoostingClassifier, RandomForestClassifier
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    average_precision_score,
    balanced_accuracy_score,
    classification_report,
    confusion_matrix,
    roc_auc_score,
)
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler

TargetMode = Literal["finished_binary", "all_binary", "multiclass"]
SplitMode = Literal["time", "stratified_cv", "time_cv"]


@dataclass(frozen=True)
class CzechFinancialConfig:
    """Configuration for feature matrix construction.

    Parameters
    ----------
    data_dir:
        Directory containing account/client/disp/district/loan/order/trans/card files.
    target_mode:
        - "finished_binary": use only A/B loans. target_bad=1 for B.
        - "all_binary": use A/B/C/D loans. target_bad=1 for B or D.
        - "multiclass": target is A/B/C/D.
    transaction_windows_days:
        Rolling windows before loan date for transaction aggregates.
    include_full_history:
        Add transaction aggregates over all available pre-loan history.
    include_orders:
        Include permanent-order features. The order table has no timestamp, so these
        features are useful but not perfectly time-auditable.
    include_cards:
        Include only cards issued on or before loan_date.
    include_district:
        Include account/client district demographics.
    include_owner_client:
        Include owner/disponent demographics through the disp table.
    strict_pre_loan_transactions:
        True => transactions must satisfy trans_date < loan_date.
        False => trans_date <= loan_date.
    min_category_count:
        Rare categories below this frequency are grouped into "__rare__" before ML encoding.
    random_state:
        Deterministic seed for downstream models/splits.
    """

    data_dir: str | os.PathLike
    target_mode: TargetMode = "all_binary"
    transaction_windows_days: Tuple[int, ...] = (30, 90, 180, 365)
    include_full_history: bool = True
    include_orders: bool = True
    include_cards: bool = True
    include_district: bool = True
    include_owner_client: bool = True
    strict_pre_loan_transactions: bool = True
    min_category_count: int = 5
    random_state: int = 42


# ---------------------------------------------------------------------
# Generic utilities
# ---------------------------------------------------------------------


def clean_column_name(name: object) -> str:
    """Normalize messy column names to lowercase snake_case."""
    s = str(name).strip().lower()
    s = s.replace(".", "_").replace("-", "_").replace("/", "_")
    s = re.sub(r"[^0-9a-zA-Z_]+", "_", s)
    s = re.sub(r"_+", "_", s).strip("_")
    return s


def clean_frame(df: pd.DataFrame) -> pd.DataFrame:
    """Lightweight dataframe cleanup: columns, blank strings, object stripping."""
    out = df.copy()
    out.columns = [clean_column_name(c) for c in out.columns]

    for col in out.select_dtypes(include=["object", "string"]).columns:
        out[col] = out[col].astype("string").str.strip()
        out[col] = out[col].replace(
            {
                "": pd.NA,
                " ": pd.NA,
                "?": pd.NA,
                "nan": pd.NA,
                "None": pd.NA,
                "NULL": pd.NA,
                "null": pd.NA,
            }
        )
    return out


def as_numeric(df: pd.DataFrame, columns: Sequence[str]) -> pd.DataFrame:
    """Coerce selected columns to numeric if they exist."""
    out = df.copy()
    for col in columns:
        if col in out.columns:
            out[col] = pd.to_numeric(out[col], errors="coerce")
    return out


def normalize_code(s: pd.Series) -> pd.Series:
    """Normalize categorical/code series."""
    return s.astype("string").str.strip().str.upper().replace({"": pd.NA, "NAN": pd.NA, "NONE": pd.NA, "NULL": pd.NA})


def coalesce_columns(df: pd.DataFrame, candidates: Sequence[str], target: str) -> pd.DataFrame:
    """Create/rename target from the first available candidate column."""
    out = df.copy()
    if target in out.columns:
        return out

    for col in candidates:
        if col in out.columns:
            out = out.rename(columns={col: target})
            return out
    return out


def parse_mixed_date(series: pd.Series) -> pd.Series:
    """Parse common Berka date encodings.

    Supports:
    - ISO: YYYY-MM-DD
    - YYMMDD used by original .asc files
    - YYYYMMDD, if present
    """
    s = series.astype("string").str.strip()
    parsed = pd.to_datetime(s, errors="coerce")

    mask_yymmdd = parsed.isna() & s.str.match(r"^\d{6}$", na=False)
    if mask_yymmdd.any():
        parsed.loc[mask_yymmdd] = pd.to_datetime(s.loc[mask_yymmdd], format="%y%m%d", errors="coerce")
        # pandas maps 68-99 to 1968-1999 and 00-67 to 2000-2067.
        # PKDD dates are in the 1990s, so move accidental 20xx dates back 100 years.
        future_mask = mask_yymmdd & (parsed.dt.year > 1999)
        parsed.loc[future_mask] = parsed.loc[future_mask] - pd.DateOffset(years=100)

    mask_yyyymmdd = parsed.isna() & s.str.match(r"^\d{8}$", na=False)
    if mask_yyyymmdd.any():
        parsed.loc[mask_yyyymmdd] = pd.to_datetime(s.loc[mask_yyyymmdd], format="%Y%m%d", errors="coerce")

    return parsed


def parse_czech_birth_number(birth_number: pd.Series) -> pd.DataFrame:
    """Extract birth_date and gender from Czech rodne cislo-like YYMMDD[xxxx].

    In the original Berka files, women have month + 50. The data is from the
    20th century, so years are interpreted as 1900 + YY.
    """
    s = birth_number.astype("string").str.extract(r"(?P<yy>\d{2})(?P<mm>\d{2})(?P<dd>\d{2})")
    yy = pd.to_numeric(s["yy"], errors="coerce")
    mm_raw = pd.to_numeric(s["mm"], errors="coerce")
    dd = pd.to_numeric(s["dd"], errors="coerce")

    gender = np.where(mm_raw > 50, "F", "M")
    mm = np.where(mm_raw > 50, mm_raw - 50, mm_raw)

    dates = pd.to_datetime(
        pd.DataFrame(
            {
                "year": 1900 + yy,
                "month": mm,
                "day": dd,
            }
        ),
        errors="coerce",
    )
    return pd.DataFrame({"birth_date": dates, "gender": gender})


def safe_divide(numerator: pd.Series | float, denominator: pd.Series | float) -> pd.Series:
    """Division helper that returns NaN on zero/invalid denominator."""
    with np.errstate(divide="ignore", invalid="ignore"):
        out = numerator / denominator
    if isinstance(out, pd.Series):
        return out.replace([np.inf, -np.inf], np.nan)
    return pd.Series(out).replace([np.inf, -np.inf], np.nan)


def add_prefix_except(df: pd.DataFrame, prefix: str, keep: Sequence[str]) -> pd.DataFrame:
    """Prefix all columns except key columns."""
    keep = set(keep)
    return df.rename(columns={c: f"{prefix}{c}" for c in df.columns if c not in keep})


def rare_category_grouping(df: pd.DataFrame, min_count: int = 5) -> pd.DataFrame:
    """Group rare categorical levels before one-hot encoding."""
    out = df.copy()
    cat_cols = out.select_dtypes(include=["object", "string", "category"]).columns
    for col in cat_cols:
        vc = out[col].astype("string").value_counts(dropna=True)
        rare = vc[vc < min_count].index
        out[col] = out[col].astype("string").where(~out[col].astype("string").isin(rare), "__rare__")
    return out


# ---------------------------------------------------------------------
# Loading and schema normalization
# ---------------------------------------------------------------------


def find_table_file(data_dir: str | os.PathLike, table: str) -> Path:
    """Find a table file by name across common Kaggle/original/GitHub layouts."""
    base = Path(data_dir)
    patterns = [
        f"**/{table}.csv",
        f"**/{table}.tsv",
        f"**/{table}.asc",
        f"**/{table}.txt",
        f"**/fin_{table}.csv",
        f"**/fin_{table}.tsv",
        f"**/*{table}*.csv",
        f"**/*{table}*.tsv",
        f"**/*{table}*.asc",
    ]

    hits: List[Path] = []
    for pattern in patterns:
        hits.extend(base.glob(pattern))

    hits = sorted({p for p in hits if p.is_file()})
    if not hits:
        raise FileNotFoundError(f"Could not find a file for table '{table}' under {base}")

    # Prefer exact names over broad wildcard hits.
    exact_names = {
        f"{table}.csv",
        f"{table}.tsv",
        f"{table}.asc",
        f"fin_{table}.tsv",
        f"fin_{table}.csv",
    }
    exact = [p for p in hits if p.name.lower() in exact_names]
    return exact[0] if exact else hits[0]


def read_table_file(path: Path) -> pd.DataFrame:
    """Read CSV/TSV/ASC files robustly."""
    suffix = path.suffix.lower()
    if suffix == ".tsv":
        df = pd.read_csv(path, sep="\t", engine="python")
    elif suffix == ".asc":
        # Original Berka .asc files are semicolon-separated.
        df = pd.read_csv(path, sep=";", engine="python")
    else:
        # Let pandas infer comma/semicolon/tab where possible.
        df = pd.read_csv(path, sep=None, engine="python")
    return clean_frame(df)


def load_raw_tables(data_dir: str | os.PathLike) -> Dict[str, pd.DataFrame]:
    """Load the eight dataset tables."""
    tables = {}
    for table in [
        "account",
        "client",
        "disp",
        "district",
        "loan",
        "order",
        "trans",
        "card",
    ]:
        path = find_table_file(data_dir, table)
        tables[table] = read_table_file(path)
    return tables


def normalize_account(account: pd.DataFrame) -> pd.DataFrame:
    out = clean_frame(account)
    out = coalesce_columns(out, ["date", "create_date", "account_date"], "account_date")
    out = as_numeric(out, ["account_id", "district_id"])
    if "account_date" in out.columns:
        out["account_date"] = parse_mixed_date(out["account_date"])
    if "frequency" in out.columns:
        out["frequency"] = normalize_code(out["frequency"])
    return out


def normalize_client(client: pd.DataFrame) -> pd.DataFrame:
    out = clean_frame(client)
    out = as_numeric(out, ["client_id", "district_id"])

    if "birth_date" in out.columns:
        out["birth_date"] = parse_mixed_date(out["birth_date"])
    elif "birth_number" in out.columns:
        parsed = parse_czech_birth_number(out["birth_number"])
        out["birth_date"] = parsed["birth_date"]
        out["gender"] = parsed["gender"]
    elif "birthdate" in out.columns:
        out = out.rename(columns={"birthdate": "birth_date"})
        out["birth_date"] = parse_mixed_date(out["birth_date"])

    if "gender" in out.columns:
        out["gender"] = normalize_code(out["gender"])
    return out


def normalize_disp(disp: pd.DataFrame) -> pd.DataFrame:
    out = clean_frame(disp)
    out = coalesce_columns(out, ["type", "disp_type"], "disp_type")
    out = as_numeric(out, ["disp_id", "client_id", "account_id"])
    if "disp_type" in out.columns:
        mapping = {
            "OWNER": "O",
            "DISPONENT": "D",
            "O": "O",
            "D": "D",
        }
        out["disp_type"] = normalize_code(out["disp_type"]).map(mapping).fillna(normalize_code(out["disp_type"]))
    return out


def normalize_loan(loan: pd.DataFrame) -> pd.DataFrame:
    out = clean_frame(loan)
    out = coalesce_columns(out, ["date", "granted_date", "loan_date"], "loan_date")
    out = as_numeric(out, ["loan_id", "account_id", "amount", "duration", "payments"])
    if "loan_date" in out.columns:
        out["loan_date"] = parse_mixed_date(out["loan_date"])
    if "status" in out.columns:
        out["status"] = normalize_code(out["status"])
    return out


def normalize_trans(trans: pd.DataFrame) -> pd.DataFrame:
    out = clean_frame(trans)
    out = coalesce_columns(out, ["date", "trans_date"], "trans_date")
    out = coalesce_columns(out, ["type", "trans_type"], "trans_type")
    out = coalesce_columns(out, ["k_symbol", "category"], "category")
    out = as_numeric(out, ["trans_id", "account_id", "amount", "balance", "other_account_id"])
    if "trans_date" in out.columns:
        out["trans_date"] = parse_mixed_date(out["trans_date"])

    # Normalize Czech/original and English/Teradata codes.
    if "trans_type" in out.columns:
        type_map = {
            "PRIJEM": "C",
            "VYDAJ": "D",
            "VYBER": "P",
            "CREDIT": "C",
            "DEBIT": "D",
            "WITHDRAWAL": "P",
            "C": "C",
            "D": "D",
            "P": "P",
        }
        out["trans_type"] = normalize_code(out["trans_type"]).map(type_map).fillna(normalize_code(out["trans_type"]))

    if "operation" in out.columns:
        operation_map = {
            "VYBER KARTOU": "CCW",
            "VKLAD": "CIC",
            "PREVOD Z UCTU": "COB",
            "VYBER": "WIC",
            "PREVOD NA UCET": "ROB",
            "CCW": "CCW",
            "CIC": "CIC",
            "COB": "COB",
            "WIC": "WIC",
            "ROB": "ROB",
        }
        out["operation"] = normalize_code(out["operation"]).map(operation_map).fillna(normalize_code(out["operation"]))

    if "category" in out.columns:
        category_map = {
            "UROK": "IC",
            "SANKC. UROK": "IO",
            "DUCHOD": "PE",
            "UVER": "LO",
            "SIPO": "HH",
            "SLUZBY": "ST",
            "POJISTNE": "IN",
            "IC": "IC",
            "IO": "IO",
            "PE": "PE",
            "LO": "LO",
            "HH": "HH",
            "ST": "ST",
            "IN": "IN",
        }
        out["category"] = normalize_code(out["category"]).map(category_map).fillna(normalize_code(out["category"]))

    return out


def normalize_order(order: pd.DataFrame) -> pd.DataFrame:
    out = clean_frame(order)
    out = coalesce_columns(out, ["k_symbol", "category"], "category")
    out = as_numeric(out, ["order_id", "account_id", "account_to", "amount"])

    if "category" in out.columns:
        category_map = {
            "SIPO": "HH",
            "POJISTNE": "IN",
            "UVER": "LO",
            "LEASING": "LE",
            "HH": "HH",
            "IN": "IN",
            "LO": "LO",
            "LE": "LE",
        }
        out["category"] = normalize_code(out["category"]).map(category_map).fillna(normalize_code(out["category"]))

    if "bank_to" in out.columns:
        out["bank_to"] = normalize_code(out["bank_to"])
    return out


def normalize_card(card: pd.DataFrame) -> pd.DataFrame:
    out = clean_frame(card)
    out = coalesce_columns(out, ["type", "card_type"], "card_type")
    out = coalesce_columns(out, ["issued", "issued_date"], "issued_date")
    out = as_numeric(out, ["card_id", "disp_id"])
    if "issued_date" in out.columns:
        out["issued_date"] = parse_mixed_date(out["issued_date"])

    if "card_type" in out.columns:
        card_map = {
            "JUNIOR": "J",
            "CLASSIC": "C",
            "GOLD": "G",
            "J": "J",
            "C": "C",
            "G": "G",
        }
        out["card_type"] = normalize_code(out["card_type"]).map(card_map).fillna(normalize_code(out["card_type"]))
    return out


def normalize_district(district: pd.DataFrame) -> pd.DataFrame:
    """Normalize district table; supports A1-A16 original names and named schemas."""
    out = clean_frame(district)

    # Original Berka column names are often A1, A2, ..., A16 after cleaning => a1, a2...
    original_map = {
        "a1": "district_id",
        "a2": "district_name",
        "a3": "region",
        "a4": "num_inhabitants",
        "a5": "num_municipalities_gt499",
        "a6": "num_municipalities_500to1999",
        "a7": "num_municipalities_2000to9999",
        "a8": "num_municipalities_gt10000",
        "a9": "num_cities",
        "a10": "ratio_urban",
        "a11": "average_salary",
        "a12": "unemployment_rate95",
        "a13": "unemployment_rate96",
        "a14": "num_entrep_per1000",
        "a15": "num_crimes95",
        "a16": "num_crimes96",
    }
    out = out.rename(columns={k: v for k, v in original_map.items() if k in out.columns})

    numeric_cols = [
        "district_id",
        "num_inhabitants",
        "num_municipalities_gt499",
        "num_municipalities_500to1999",
        "num_municipalities_2000to9999",
        "num_municipalities_gt10000",
        "num_cities",
        "ratio_urban",
        "average_salary",
        "unemployment_rate95",
        "unemployment_rate96",
        "num_entrep_per1000",
        "num_crimes95",
        "num_crimes96",
    ]
    out = as_numeric(out, numeric_cols)
    return out


def load_and_normalize_tables(data_dir: str | os.PathLike) -> Dict[str, pd.DataFrame]:
    """Load and normalize all dataset tables."""
    raw = load_raw_tables(data_dir)
    return {
        "account": normalize_account(raw["account"]),
        "client": normalize_client(raw["client"]),
        "disp": normalize_disp(raw["disp"]),
        "district": normalize_district(raw["district"]),
        "loan": normalize_loan(raw["loan"]),
        "order": normalize_order(raw["order"]),
        "trans": normalize_trans(raw["trans"]),
        "card": normalize_card(raw["card"]),
    }


# ---------------------------------------------------------------------
# Feature engineering
# ---------------------------------------------------------------------


def make_target(loan: pd.DataFrame, mode: TargetMode) -> pd.DataFrame:
    """Return normalized loan table with target column."""
    out = loan.copy()
    if mode == "finished_binary":
        out = out[out["status"].isin(["A", "B"])].copy()
        out["target"] = (out["status"] == "B").astype("int8")
    elif mode == "all_binary":
        out = out[out["status"].isin(["A", "B", "C", "D"])].copy()
        out["target"] = out["status"].isin(["B", "D"]).astype("int8")
    elif mode == "multiclass":
        out = out[out["status"].isin(["A", "B", "C", "D"])].copy()
        out["target"] = out["status"].astype("string")
    else:
        raise ValueError(f"Unknown target mode: {mode}")
    return out


def build_base_loan_account_features(
    loan: pd.DataFrame,
    account: pd.DataFrame,
) -> pd.DataFrame:
    """Loan and account static features."""
    df = loan.merge(account, on="account_id", how="left", suffixes=("", "_account"))

    df["loan_amount"] = df["amount"]
    df["loan_duration_months"] = df["duration"]
    df["loan_monthly_payment"] = df["payments"]
    df["loan_amount_log1p"] = np.log1p(df["amount"])
    df["payment_to_amount_ratio"] = safe_divide(df["payments"], df["amount"])
    df["payment_x_duration"] = df["payments"] * df["duration"]
    df["loan_total_payment_to_amount"] = safe_divide(df["payment_x_duration"], df["amount"])

    if "account_date" in df.columns:
        df["account_age_days_at_loan"] = (df["loan_date"] - df["account_date"]).dt.days
        df["account_age_months_at_loan"] = df["account_age_days_at_loan"] / 30.4375

    if "frequency" in df.columns:
        df = df.rename(columns={"frequency": "account_statement_frequency"})

    if "district_id" in df.columns:
        df = df.rename(columns={"district_id": "account_district_id"})

    return df


def build_client_features(
    loans: pd.DataFrame,
    disp: pd.DataFrame,
    client: pd.DataFrame,
    district: Optional[pd.DataFrame] = None,
) -> pd.DataFrame:
    """Aggregate owner/disponent/client information to loan_id level."""
    link = disp.merge(client, on="client_id", how="left", suffixes=("", "_client"))
    link = loans[["loan_id", "account_id", "loan_date"]].merge(link, on="account_id", how="left")

    if "birth_date" in link.columns:
        link["client_age_years_at_loan"] = (link["loan_date"] - link["birth_date"]).dt.days / 365.25

    link["is_owner"] = (link["disp_type"] == "O").astype("int8") if "disp_type" in link.columns else np.nan
    link["is_disponent"] = (link["disp_type"] == "D").astype("int8") if "disp_type" in link.columns else np.nan
    link["is_female"] = (link["gender"] == "F").astype("int8") if "gender" in link.columns else np.nan

    agg_dict = {
        "client_id": ["count", "nunique"],
    }
    if "is_owner" in link.columns:
        agg_dict["is_owner"] = ["sum"]
    if "is_disponent" in link.columns:
        agg_dict["is_disponent"] = ["sum"]
    if "is_female" in link.columns:
        agg_dict["is_female"] = ["mean", "sum"]
    if "client_age_years_at_loan" in link.columns:
        agg_dict["client_age_years_at_loan"] = ["mean", "min", "max", "std"]

    out = link.groupby("loan_id").agg(agg_dict)
    out.columns = ["client_" + "_".join(c).strip("_") for c in out.columns.to_flat_index()]
    out = out.reset_index()

    # Owner-specific features.
    owners = link[link.get("disp_type", pd.Series(index=link.index, dtype="string")).eq("O")].copy()
    if not owners.empty:
        owner_cols = ["loan_id"]
        for c in ["client_age_years_at_loan", "gender", "district_id"]:
            if c in owners.columns:
                owner_cols.append(c)
        owner = owners[owner_cols].drop_duplicates("loan_id")
        owner = owner.rename(
            columns={
                "client_age_years_at_loan": "owner_age_years_at_loan",
                "gender": "owner_gender",
                "district_id": "owner_district_id",
            }
        )
        out = out.merge(owner, on="loan_id", how="left")

    if district is not None and "owner_district_id" in out.columns:
        district_pref = add_prefix_except(district, "owner_district_", keep=["district_id"])
        out = out.merge(
            district_pref,
            left_on="owner_district_id",
            right_on="district_id",
            how="left",
        ).drop(columns=["district_id"], errors="ignore")

    return out


def build_district_features(
    frame: pd.DataFrame,
    district: pd.DataFrame,
) -> pd.DataFrame:
    """Join account district demographics."""
    if "account_district_id" not in frame.columns:
        return frame

    district_pref = add_prefix_except(district, "account_district_", keep=["district_id"])
    return frame.merge(
        district_pref,
        left_on="account_district_id",
        right_on="district_id",
        how="left",
    ).drop(columns=["district_id"], errors="ignore")


def transaction_pre_loan_join(
    loans: pd.DataFrame,
    trans: pd.DataFrame,
    strict_before: bool = True,
) -> pd.DataFrame:
    """Join transactions to loans and filter to pre-loan period."""
    needed = ["loan_id", "account_id", "loan_date"]
    missing = [c for c in needed if c not in loans.columns]
    if missing:
        raise ValueError(f"Loan table missing required columns: {missing}")

    tx = trans.merge(loans[needed], on="account_id", how="inner")
    if strict_before:
        tx = tx[tx["trans_date"] < tx["loan_date"]].copy()
    else:
        tx = tx[tx["trans_date"] <= tx["loan_date"]].copy()

    tx["days_before_loan"] = (tx["loan_date"] - tx["trans_date"]).dt.days
    tx["is_credit"] = tx["trans_type"].eq("C").astype("int8") if "trans_type" in tx.columns else 0
    tx["is_debit"] = tx["trans_type"].isin(["D", "P"]).astype("int8") if "trans_type" in tx.columns else 0
    tx["signed_amount"] = np.where(tx["is_credit"].eq(1), tx["amount"], -tx["amount"])
    tx["credit_amount"] = np.where(tx["is_credit"].eq(1), tx["amount"], 0.0)
    tx["debit_amount"] = np.where(tx["is_debit"].eq(1), tx["amount"], 0.0)
    tx["is_negative_balance"] = (tx["balance"] < 0).astype("int8") if "balance" in tx.columns else 0
    return tx


def _aggregate_transactions(tx: pd.DataFrame, prefix: str) -> pd.DataFrame:
    """Aggregate a transaction slice already filtered to the relevant window."""
    if tx.empty:
        return pd.DataFrame(columns=["loan_id"])

    base = (
        tx.groupby("loan_id")
        .agg(
            **{
                f"{prefix}tx_count": ("trans_id", "count") if "trans_id" in tx.columns else ("amount", "count"),
                f"{prefix}tx_active_days": ("trans_date", "nunique"),
                f"{prefix}tx_amount_sum": ("amount", "sum"),
                f"{prefix}tx_amount_mean": ("amount", "mean"),
                f"{prefix}tx_amount_median": ("amount", "median"),
                f"{prefix}tx_amount_std": ("amount", "std"),
                f"{prefix}tx_amount_min": ("amount", "min"),
                f"{prefix}tx_amount_max": ("amount", "max"),
                f"{prefix}net_cashflow_sum": ("signed_amount", "sum"),
                f"{prefix}credit_amount_sum": ("credit_amount", "sum"),
                f"{prefix}debit_amount_sum": ("debit_amount", "sum"),
                f"{prefix}credit_count": ("is_credit", "sum"),
                f"{prefix}debit_count": ("is_debit", "sum"),
                f"{prefix}negative_balance_count": ("is_negative_balance", "sum"),
                f"{prefix}days_since_last_tx": ("days_before_loan", "min"),
                f"{prefix}oldest_tx_days_before_loan": ("days_before_loan", "max"),
            }
        )
        .reset_index()
    )

    if "balance" in tx.columns:
        bal = (
            tx.groupby("loan_id")
            .agg(
                **{
                    f"{prefix}balance_min": ("balance", "min"),
                    f"{prefix}balance_max": ("balance", "max"),
                    f"{prefix}balance_mean": ("balance", "mean"),
                    f"{prefix}balance_median": ("balance", "median"),
                    f"{prefix}balance_std": ("balance", "std"),
                }
            )
            .reset_index()
        )
        base = base.merge(bal, on="loan_id", how="left")

        sorted_tx = tx.sort_values(["loan_id", "trans_date"] + (["trans_id"] if "trans_id" in tx.columns else []))
        first_last = (
            sorted_tx.groupby("loan_id")
            .agg(
                **{
                    f"{prefix}first_balance": ("balance", "first"),
                    f"{prefix}last_balance": ("balance", "last"),
                    f"{prefix}first_tx_date": ("trans_date", "first"),
                    f"{prefix}last_tx_date": ("trans_date", "last"),
                }
            )
            .reset_index()
        )
        first_last[f"{prefix}balance_change"] = first_last[f"{prefix}last_balance"] - first_last[f"{prefix}first_balance"]
        days_span = (first_last[f"{prefix}last_tx_date"] - first_last[f"{prefix}first_tx_date"]).dt.days.replace(0, np.nan)
        first_last[f"{prefix}balance_change_per_day"] = safe_divide(first_last[f"{prefix}balance_change"], days_span)
        first_last = first_last.drop(columns=[f"{prefix}first_tx_date", f"{prefix}last_tx_date"])
        base = base.merge(first_last, on="loan_id", how="left")

    base[f"{prefix}credit_debit_amount_ratio"] = safe_divide(
        base[f"{prefix}credit_amount_sum"], base[f"{prefix}debit_amount_sum"]
    )
    base[f"{prefix}credit_debit_count_ratio"] = safe_divide(base[f"{prefix}credit_count"], base[f"{prefix}debit_count"])
    base[f"{prefix}negative_balance_rate"] = safe_divide(base[f"{prefix}negative_balance_count"], base[f"{prefix}tx_count"])
    base[f"{prefix}tx_per_active_day"] = safe_divide(base[f"{prefix}tx_count"], base[f"{prefix}tx_active_days"])

    # Count operation/category levels. These are intentionally compact and model-friendly.
    for col in ["operation", "category", "trans_type"]:
        if col in tx.columns:
            counts = (
                pd.crosstab(tx["loan_id"], tx[col].fillna("__missing__")).add_prefix(f"{prefix}{col}_count_").reset_index()
            )
            base = base.merge(counts, on="loan_id", how="left")

    return base


def build_transaction_features(
    loans: pd.DataFrame,
    trans: pd.DataFrame,
    windows_days: Sequence[int] = (30, 90, 180, 365),
    include_full_history: bool = True,
    strict_before: bool = True,
) -> pd.DataFrame:
    """Build leakage-aware transaction features for each loan."""
    tx = transaction_pre_loan_join(loans, trans, strict_before=strict_before)
    features = loans[["loan_id"]].copy()

    if include_full_history:
        full = _aggregate_transactions(tx, prefix="hist_")
        features = features.merge(full, on="loan_id", how="left")

    for days in windows_days:
        window_tx = tx[tx["days_before_loan"].between(1 if strict_before else 0, days)].copy()
        window = _aggregate_transactions(window_tx, prefix=f"w{days}_")
        features = features.merge(window, on="loan_id", how="left")

    return features


def build_order_features(order: pd.DataFrame) -> pd.DataFrame:
    """Aggregate permanent order features to account_id level.

    The order table has no timestamp in the classic dataset. Treat these as
    static features and validate their value with an ablation experiment.
    """
    if order.empty:
        return pd.DataFrame(columns=["account_id"])

    base = (
        order.groupby("account_id")
        .agg(
            order_count=("order_id", "count") if "order_id" in order.columns else ("amount", "count"),
            order_amount_sum=("amount", "sum"),
            order_amount_mean=("amount", "mean"),
            order_amount_median=("amount", "median"),
            order_amount_max=("amount", "max"),
            order_recipient_bank_nunique=("bank_to", "nunique") if "bank_to" in order.columns else ("account_id", "size"),
            order_recipient_account_nunique=("account_to", "nunique")
            if "account_to" in order.columns
            else ("account_id", "size"),
        )
        .reset_index()
    )

    if "category" in order.columns:
        counts = pd.crosstab(order["account_id"], order["category"].fillna("__missing__")).add_prefix("order_category_count_")
        sums = pd.pivot_table(
            order,
            index="account_id",
            columns=order["category"].fillna("__missing__"),
            values="amount",
            aggfunc="sum",
            fill_value=0,
        ).add_prefix("order_category_amount_sum_")
        cat = counts.join(sums, how="outer").reset_index()
        base = base.merge(cat, on="account_id", how="left")

    base["order_amount_to_count_ratio"] = safe_divide(base["order_amount_sum"], base["order_count"])
    return base


def build_card_features(
    loans: pd.DataFrame,
    card: pd.DataFrame,
    disp: pd.DataFrame,
) -> pd.DataFrame:
    """Build card features only for cards issued on/before the loan date."""
    if card.empty:
        return loans[["loan_id"]].assign(card_count_before_loan=0, has_card_before_loan=0)

    cards = card.merge(disp[["disp_id", "account_id"]], on="disp_id", how="left")
    cards = cards.merge(loans[["loan_id", "account_id", "loan_date"]], on="account_id", how="inner")
    cards = cards[cards["issued_date"].notna() & (cards["issued_date"] <= cards["loan_date"])].copy()

    if cards.empty:
        return loans[["loan_id"]].assign(card_count_before_loan=0, has_card_before_loan=0)

    base = (
        cards.groupby("loan_id")
        .agg(
            card_count_before_loan=("card_id", "count") if "card_id" in cards.columns else ("disp_id", "count"),
            first_card_date=("issued_date", "min"),
            last_card_date=("issued_date", "max"),
        )
        .reset_index()
    )

    base["has_card_before_loan"] = (base["card_count_before_loan"] > 0).astype("int8")
    loan_dates = loans[["loan_id", "loan_date"]]
    base = base.merge(loan_dates, on="loan_id", how="left")
    base["days_since_first_card"] = (base["loan_date"] - base["first_card_date"]).dt.days
    base["days_since_last_card"] = (base["loan_date"] - base["last_card_date"]).dt.days
    base = base.drop(columns=["loan_date", "first_card_date", "last_card_date"])

    if "card_type" in cards.columns:
        counts = (
            pd.crosstab(cards["loan_id"], cards["card_type"].fillna("__missing__"))
            .add_prefix("card_type_count_")
            .reset_index()
        )
        base = base.merge(counts, on="loan_id", how="left")

    return (
        loans[["loan_id"]]
        .merge(base, on="loan_id", how="left")
        .fillna({"card_count_before_loan": 0, "has_card_before_loan": 0})
    )


def final_cleanup_features(
    df: pd.DataFrame,
    config: CzechFinancialConfig,
) -> Tuple[pd.DataFrame, pd.Series, pd.DataFrame]:
    """Separate X/y/metadata, clean infinities, group rare categories."""
    metadata_cols = [
        "loan_id",
        "account_id",
        "loan_date",
        "status",
        "target",
        "amount",
        "duration",
        "payments",
        "account_date",
        "birth_date",
    ]

    metadata_cols = [c for c in metadata_cols if c in df.columns]
    metadata = df[metadata_cols].copy()
    y = df["target"].copy()

    drop_cols = set(metadata_cols)
    # Avoid duplicating raw fields after engineered versions; keep district IDs as categorical features.
    raw_drop = {"birth_number"}
    drop_cols |= {c for c in raw_drop if c in df.columns}

    X = df.drop(columns=list(drop_cols), errors="ignore").copy()

    # Do not train on raw datetime columns.
    datetime_cols = X.select_dtypes(include=["datetime64[ns]", "datetimetz"]).columns.tolist()
    X = X.drop(columns=datetime_cols, errors="ignore")

    # Normalize infinities.
    X = X.replace([np.inf, -np.inf], np.nan)

    # Treat ID-like district columns as categorical, not ordinal magnitudes.
    for col in X.columns:
        if col.endswith("_district_id") or col in {
            "account_district_id",
            "owner_district_id",
        }:
            X[col] = X[col].astype("string")

    X = rare_category_grouping(X, min_count=config.min_category_count)

    return X, y, metadata


def build_classification_dataset(
    config: CzechFinancialConfig,
    tables: Optional[Mapping[str, pd.DataFrame]] = None,
) -> Tuple[pd.DataFrame, pd.Series, pd.DataFrame]:
    """Build a final classification dataset.

    Returns
    -------
    X:
        Feature matrix, one row per loan.
    y:
        Target vector.
    metadata:
        Loan IDs, dates, statuses, and other audit columns.
    """
    if tables is None:
        tables = load_and_normalize_tables(config.data_dir)
    else:
        # Assume caller might pass raw tables, so normalize again.
        tables = {
            "account": normalize_account(tables["account"]),
            "client": normalize_client(tables["client"]),
            "disp": normalize_disp(tables["disp"]),
            "district": normalize_district(tables["district"]),
            "loan": normalize_loan(tables["loan"]),
            "order": normalize_order(tables["order"]),
            "trans": normalize_trans(tables["trans"]),
            "card": normalize_card(tables["card"]),
        }

    loan = make_target(tables["loan"], config.target_mode)
    df = build_base_loan_account_features(loan, tables["account"])

    if config.include_owner_client:
        client_features = build_client_features(
            loans=loan,
            disp=tables["disp"],
            client=tables["client"],
            district=tables["district"] if config.include_district else None,
        )
        df = df.merge(client_features, on="loan_id", how="left")

    if config.include_district:
        df = build_district_features(df, tables["district"])

    tx_features = build_transaction_features(
        loans=loan,
        trans=tables["trans"],
        windows_days=config.transaction_windows_days,
        include_full_history=config.include_full_history,
        strict_before=config.strict_pre_loan_transactions,
    )
    df = df.merge(tx_features, on="loan_id", how="left")

    if config.include_orders:
        warnings.warn(
            "Including order features. The order table has no timestamp in the classic dataset, "
            "so run a robustness check with include_orders=False.",
            UserWarning,
            stacklevel=2,
        )
        order_features = build_order_features(tables["order"])
        df = df.merge(order_features, on="account_id", how="left")

    if config.include_cards:
        card_features = build_card_features(loan, tables["card"], tables["disp"])
        df = df.merge(card_features, on="loan_id", how="left")

    return final_cleanup_features(df, config)


# ---------------------------------------------------------------------
# Modeling pipeline helpers
# ---------------------------------------------------------------------


class NumericWinsorizer(BaseEstimator, TransformerMixin):
    """Clip numeric features to learned quantiles.

    Useful before logistic regression and SMOTE-like methods. Tree ensembles
    generally do not need this, but it is harmless when extreme outliers exist.
    """

    def __init__(self, lower: float = 0.01, upper: float = 0.99):
        self.lower = lower
        self.upper = upper

    def fit(self, X, y=None):
        X_df = pd.DataFrame(X)
        self.feature_names_in_ = getattr(X, "columns", None)

        self.lower_bounds_ = X_df.quantile(self.lower, numeric_only=True)
        self.upper_bounds_ = X_df.quantile(self.upper, numeric_only=True)
        return self

    def transform(self, X):
        X_df = pd.DataFrame(X).copy()
        return X_df.clip(self.lower_bounds_, self.upper_bounds_, axis=1)

    def get_feature_names_out(self, input_features=None):
        if input_features is not None:
            return np.asarray(input_features, dtype=object)

        if getattr(self, "feature_names_in_", None) is not None:
            return np.asarray(self.feature_names_in_, dtype=object)

        n_features = len(getattr(self, "lower_bounds_", []))
        return np.asarray([f"x{i}" for i in range(n_features)], dtype=object)


def make_preprocessor(
    scale_numeric: bool = False,
    winsorize_numeric: bool = False,
    onehot_min_frequency: Optional[int] = None,
) -> ColumnTransformer:
    """Create a sklearn ColumnTransformer for the engineered features."""
    numeric_steps: List[Tuple[str, object]] = [
        ("imputer", SimpleImputer(strategy="median", add_indicator=True)),
    ]
    if winsorize_numeric:
        # Winsorize after imputation so clipping has complete numeric input.
        numeric_steps.append(("winsorizer", NumericWinsorizer(lower=0.01, upper=0.99)))
    if scale_numeric:
        numeric_steps.append(("scaler", StandardScaler()))

    categorical_steps: List[Tuple[str, object]] = [
        ("imputer", SimpleImputer(strategy="constant", fill_value="__missing__")),
        (
            "onehot",
            OneHotEncoder(
                handle_unknown="ignore",
                min_frequency=onehot_min_frequency,
                sparse_output=False,
            ),
        ),
    ]

    return ColumnTransformer(
        transformers=[
            (
                "num",
                Pipeline(numeric_steps),
                make_column_selector(dtype_include=np.number),
            ),
            (
                "cat",
                Pipeline(categorical_steps),
                make_column_selector(dtype_include=["object", "string", "category", "bool"]),
            ),
        ],
        remainder="drop",
        verbose_feature_names_out=True,
    )


def make_model_pipeline(
    model_name: Literal["logistic", "random_forest", "hist_gradient_boosting"] = "hist_gradient_boosting",
    random_state: int = 42,
) -> Pipeline:
    """Preprocessor + classifier pipeline.

    Defaults to HistGradientBoosting because it performs well on tabular data and
    is robust without heavy tuning. For highly imbalanced targets, compare with
    RandomForestClassifier(class_weight="balanced_subsample") and calibrated
    logistic regression.
    """
    if model_name == "logistic":
        model = LogisticRegression(
            max_iter=5000,
            class_weight="balanced",
            solver="lbfgs",
            random_state=random_state,
        )
        preprocessor = make_preprocessor(scale_numeric=True, winsorize_numeric=True, onehot_min_frequency=5)

    elif model_name == "random_forest":
        model = RandomForestClassifier(
            n_estimators=600,
            min_samples_leaf=5,
            class_weight="balanced_subsample",
            n_jobs=-1,
            random_state=random_state,
        )
        preprocessor = make_preprocessor(scale_numeric=False, winsorize_numeric=False, onehot_min_frequency=5)

    elif model_name == "hist_gradient_boosting":
        model = HistGradientBoostingClassifier(
            learning_rate=0.04,
            max_iter=400,
            l2_regularization=0.1,
            random_state=random_state,
        )
        preprocessor = make_preprocessor(scale_numeric=False, winsorize_numeric=False, onehot_min_frequency=5)

    else:
        raise ValueError(f"Unknown model_name: {model_name}")

    return Pipeline(
        steps=[
            ("preprocess", preprocessor),
            ("model", model),
        ]
    )


def temporal_train_test_split(
    X: pd.DataFrame,
    y: pd.Series,
    metadata: pd.DataFrame,
    test_size: float = 0.2,
) -> Tuple[pd.DataFrame, pd.DataFrame, pd.Series, pd.Series, pd.DataFrame, pd.DataFrame]:
    """Sort by loan_date and use the last period as test set."""
    if "loan_date" not in metadata.columns:
        raise ValueError("metadata must include loan_date for temporal split.")

    order = metadata.sort_values("loan_date").index
    split = int(len(order) * (1 - test_size))
    train_idx = order[:split]
    test_idx = order[split:]

    return (
        X.loc[train_idx],
        X.loc[test_idx],
        y.loc[train_idx],
        y.loc[test_idx],
        metadata.loc[train_idx],
        metadata.loc[test_idx],
    )


def evaluate_binary_classifier(
    pipeline: Pipeline,
    X_test: pd.DataFrame,
    y_test: pd.Series,
    threshold: float = 0.5,
) -> Dict[str, object]:
    """Evaluate a binary classifier with threshold-sensitive and ranking metrics."""
    if hasattr(pipeline, "predict_proba"):
        proba = pipeline.predict_proba(X_test)[:, 1]
    else:
        # Fallback for estimators with decision_function only.
        scores = pipeline.decision_function(X_test)
        proba = (scores - scores.min()) / (scores.max() - scores.min())

    pred = (proba >= threshold).astype(int)

    results: Dict[str, object] = {
        "threshold": threshold,
        "balanced_accuracy": balanced_accuracy_score(y_test, pred),
        "confusion_matrix": confusion_matrix(y_test, pred),
        "classification_report": classification_report(y_test, pred, digits=4),
        "average_precision": average_precision_score(y_test, proba),
    }

    if len(pd.Series(y_test).unique()) == 2:
        results["roc_auc"] = roc_auc_score(y_test, proba)

    return results


def threshold_table(
    y_true: pd.Series,
    y_score: np.ndarray,
    thresholds: Sequence[float] = tuple(np.round(np.arange(0.1, 0.91, 0.05), 2)),
) -> pd.DataFrame:
    """Compare operating thresholds for risk screening."""
    rows = []
    y_true = pd.Series(y_true).astype(int)
    for t in thresholds:
        pred = (y_score >= t).astype(int)
        tn, fp, fn, tp = confusion_matrix(y_true, pred, labels=[0, 1]).ravel()
        precision = safe_divide(pd.Series([tp]), pd.Series([tp + fp])).iloc[0]
        recall = safe_divide(pd.Series([tp]), pd.Series([tp + fn])).iloc[0]
        fpr = safe_divide(pd.Series([fp]), pd.Series([fp + tn])).iloc[0]
        rows.append(
            {
                "threshold": t,
                "tp": tp,
                "fp": fp,
                "tn": tn,
                "fn": fn,
                "precision_bad": precision,
                "recall_bad": recall,
                "false_positive_rate": fpr,
            }
        )
    return pd.DataFrame(rows)


def quick_data_quality_report(X: pd.DataFrame, y: pd.Series, metadata: pd.DataFrame) -> Dict[str, object]:
    """Small report to run before modeling."""
    return {
        "n_rows": len(X),
        "n_features": X.shape[1],
        "target_counts": y.value_counts(dropna=False).to_dict(),
        "loan_date_min": metadata["loan_date"].min() if "loan_date" in metadata else None,
        "loan_date_max": metadata["loan_date"].max() if "loan_date" in metadata else None,
        "missingness_top_20": X.isna().mean().sort_values(ascending=False).head(20).to_dict(),
        "duplicate_loan_ids": int(metadata["loan_id"].duplicated().sum()) if "loan_id" in metadata else None,
    }


def get_feature_names(pipeline: Pipeline) -> List[str]:
    """Return transformed feature names after fitting the preprocessing pipeline."""
    preprocess = pipeline.named_steps["preprocess"]
    return list(preprocess.get_feature_names_out())


# ---------------------------------------------------------------------
# Example runner
# ---------------------------------------------------------------------


def run_example(data_dir: str | os.PathLike, target_mode: TargetMode = "all_binary") -> None:
    """Example end-to-end run."""
    config = CzechFinancialConfig(data_dir=data_dir, target_mode=target_mode)
    X, y, meta = build_classification_dataset(config)

    print("DATA QUALITY REPORT")
    report = quick_data_quality_report(X, y, meta)
    for k, v in report.items():
        print(f"{k}: {v}")

    X_train, X_test, y_train, y_test, meta_train, meta_test = temporal_train_test_split(X, y, meta, test_size=0.2)

    clf = make_model_pipeline("random_forest", random_state=config.random_state)
    clf.fit(X_train, y_train)

    results = evaluate_binary_classifier(clf, X_test, y_test, threshold=0.35)
    print("\nEVALUATION")
    print("ROC-AUC:", results.get("roc_auc"))
    print("Average precision:", results["average_precision"])
    print("Balanced accuracy:", results["balanced_accuracy"])
    print("Confusion matrix:\n", results["confusion_matrix"])
    print(results["classification_report"])

    proba = clf.predict_proba(X_test)[:, 1]
    print("\nTHRESHOLD TABLE")
    print(threshold_table(y_test, proba).to_string(index=False))


if __name__ == "__main__":
    # Kaggle example:
    # run_example("/kaggle/input/1999-czech-financial-dataset", target_mode="all_binary")
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument("--data-dir", required=True, help="Path to the dataset directory")
    parser.add_argument(
        "--target-mode",
        default="all_binary",
        choices=["finished_binary", "all_binary", "multiclass"],
    )
    args = parser.parse_args()
    run_example(args.data_dir, target_mode=args.target_mode)
