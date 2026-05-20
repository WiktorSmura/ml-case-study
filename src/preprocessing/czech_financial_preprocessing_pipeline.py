"""
Clean classification preprocessing pipeline for the 1999 Czech Financial / PKDD'99
/ Berka banking dataset.

The goal is not only to produce a good feature matrix, but also to make every
preprocessing decision easy to explain in a project report.

Main modeling unit
------------------
One row = one loan. The target comes from `loan.status`.

Default target
--------------
`finished_binary`, meaning:
    A -> 0, finished loan with no repayment problems
    B -> 1, finished loan with repayment problems

The alternative `all_binary` is available, but it mixes finished and still-running
contracts. Use it as a secondary experiment, not as the cleanest main result.

Leakage rules
-------------
1. Transaction features use only transactions strictly before the loan date:
       trans_date < loan_date
   Same-day transactions are excluded by default because daily timestamps cannot tell
   whether the transaction happened before or after the loan was granted.

2. Card features use only cards issued strictly before the loan date by default.

3. Permanent-order features are disabled by default. The classic `order` table has no
   date column, so those features are not time-auditable.

Redundancy rules
----------------
1. Do not create repeated rolling-window point-in-time features such as
   w30_last_balance, w90_last_balance, w365_last_balance. The last transaction before
   a loan is often the same in all windows, so these features become duplicated.

2. Remove known deterministic loan-feature duplicates, zero-variance columns, and
   exact duplicate columns from the dataset-level feature matrix.

3. Avoid creating highly redundant features in the first place. We use curated
   domain features, non-overlapping transaction windows, and compact categorical
   summaries instead of broad crosstabs.
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
class ClassificationDataset:
    """Container returned by `build_dataset_bundle`.

    Use `build_classification_dataset` when you prefer the classic `(X, y, metadata)` tuple.
    """

    X: pd.DataFrame
    y: pd.Series
    metadata: pd.DataFrame

    def quality_report(self) -> Dict[str, object]:
        return quick_data_quality_report(self.X, self.y, self.metadata)


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
        Include permanent-order features. Default is False because the classic order
        table has no timestamp, so the features are not time-auditable and may leak.
    include_cards:
        Include card features.
    strict_pre_loan_cards:
        True => cards must satisfy issued_date < loan_date.
        False => issued_date <= loan_date.
    include_district:
        Include account/client district demographics.
    include_owner_client:
        Include owner/disponent demographics through the disp table.
    strict_pre_loan_transactions:
        True => transactions must satisfy trans_date < loan_date.
        False => trans_date <= loan_date.
    min_category_count:
        Rare categories below this frequency are grouped into "__rare__" before ML encoding.
    drop_static_redundant_features:
        Drop deterministic loan/account features that duplicate other columns, such as
        payment_x_duration and loan_total_payment_to_amount.
    drop_constant_features:
        Drop zero-variance columns from X before modeling.
    drop_duplicate_features:
        Drop exact duplicate feature columns, keeping the first occurrence.
    random_state:
        Deterministic seed for downstream models/splits.
    """

    data_dir: str | os.PathLike
    target_mode: TargetMode = "finished_binary"
    transaction_windows_days: Tuple[int, ...] = (30, 90, 180, 365)
    include_full_history: bool = True
    include_orders: bool = False
    include_cards: bool = True
    strict_pre_loan_cards: bool = True
    include_district: bool = True
    include_owner_client: bool = True
    strict_pre_loan_transactions: bool = True
    min_category_count: int = 5
    drop_static_redundant_features: bool = True
    drop_constant_features: bool = True
    drop_duplicate_features: bool = True
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
    """Parse common Berka date encodings without guessing when possible.

    Supports:
    - YYMMDD used by original .asc files
    - YYYYMMDD, if present
    - ISO/common string dates as a fallback
    """
    s = series.astype("string").str.strip()
    parsed = pd.Series(pd.NaT, index=series.index, dtype="datetime64[ns]")

    mask_yymmdd = s.str.match(r"^\d{6}$", na=False)
    if mask_yymmdd.any():
        parsed.loc[mask_yymmdd] = pd.to_datetime(s.loc[mask_yymmdd], format="%y%m%d", errors="coerce")
        # pandas maps 00-67 to 2000-2067. PKDD dates are 1990s, so move accidental
        # future years back 100 years.
        future_mask = mask_yymmdd & (parsed.dt.year > 1999)
        parsed.loc[future_mask] = parsed.loc[future_mask] - pd.DateOffset(years=100)

    mask_yyyymmdd = parsed.isna() & s.str.match(r"^\d{8}$", na=False)
    if mask_yyyymmdd.any():
        parsed.loc[mask_yyyymmdd] = pd.to_datetime(s.loc[mask_yyyymmdd], format="%Y%m%d", errors="coerce")

    remaining = parsed.isna() & s.notna()
    if remaining.any():
        # Fallback for Kaggle variants that already use ISO-like date strings.
        with warnings.catch_warnings():
            warnings.simplefilter("ignore", UserWarning)
            parsed.loc[remaining] = pd.to_datetime(s.loc[remaining], errors="coerce")

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


def drop_static_redundant_features(df: pd.DataFrame) -> pd.DataFrame:
    """Drop deterministic feature-engineering duplicates known for this dataset.

    These columns are not target leakage, but they are mathematically redundant and
    create perfect correlations that make diagnostics and coefficient inspection noisy.
    """
    out = df.copy()
    redundant = [
        # Kept here for backward compatibility if the caller passes externally
        # engineered columns into cleanup. The current feature builder no longer
        # creates these by default.
        "payment_x_duration",
        "loan_total_payment_to_amount",
        "payment_to_amount_ratio",
    ]
    to_drop = [c for c in redundant if c in out.columns]
    out = out.drop(columns=to_drop, errors="ignore")
    out.attrs["dropped_static_redundant_columns"] = to_drop
    return out


def drop_constant_features(df: pd.DataFrame) -> pd.DataFrame:
    """Drop columns with a single value, treating NaN as a value."""
    out = df.copy()
    to_drop = [c for c in out.columns if out[c].nunique(dropna=False) <= 1]
    out = out.drop(columns=to_drop, errors="ignore")
    out.attrs["dropped_constant_columns"] = to_drop
    return out


def drop_duplicate_features(df: pd.DataFrame) -> pd.DataFrame:
    """Drop exact duplicate columns, keeping the first column encountered.

    This removes structural duplicates such as `has_card_before_loan` when it is
    identical to `card_count_before_loan` in a particular target subset, and duplicated
    transaction crosstab columns such as credit_count vs trans_type_count_C.
    """
    out = df.copy()
    seen: Dict[Tuple[object, ...], str] = {}
    to_drop: List[str] = []

    for col in out.columns:
        s = out[col].astype("object")
        # Normalize all missing values to the same sentinel so NaN/NA compare equal.
        signature = tuple("__MISSING__" if pd.isna(v) else v for v in s.to_numpy())
        if signature in seen:
            to_drop.append(col)
        else:
            seen[signature] = col

    out = out.drop(columns=to_drop, errors="ignore")
    out.attrs["dropped_exact_duplicate_columns"] = to_drop
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
    """Create explainable static loan/account features.

    Design choice:
    - Keep loan amount and duration.
    - Do not keep monthly payment by default because in this dataset it is a
      deterministic transformation of amount and duration for most loans.
    - Keep only one account-age unit, not days and months.
    """
    df = loan.merge(account, on="account_id", how="left", suffixes=("", "_account"))

    df["loan_amount"] = df["amount"]
    df["loan_duration_months"] = df["duration"]

    if "account_date" in df.columns:
        df["account_age_days_at_loan"] = (df["loan_date"] - df["account_date"]).dt.days

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
    """Create compact account-owner features.

    Design choice:
    The loan is requested by the account owner. Therefore, the main client features
    should describe the owner and the account composition. We intentionally avoid
    broad aggregates such as age_mean/age_min/age_max/age_std because they are often
    identical to owner age in this dataset and create artificial correlations.
    """
    link = disp.merge(client, on="client_id", how="left", suffixes=("", "_client"))
    link = loans[["loan_id", "account_id", "loan_date"]].merge(link, on="account_id", how="left")

    if "birth_date" in link.columns:
        link["client_age_years_at_loan"] = (link["loan_date"] - link["birth_date"]).dt.days / 365.25

    if "disp_type" in link.columns:
        link["is_disponent"] = link["disp_type"].eq("D").astype("int8")
    else:
        link["is_disponent"] = 0

    composition = (
        link.groupby("loan_id")
        .agg(
            client_count=("client_id", "nunique"),
            disponent_count=("is_disponent", "sum"),
        )
        .reset_index()
    )
    composition["has_disponent"] = (composition["disponent_count"] > 0).astype("int8")

    owners = link[link.get("disp_type", pd.Series(index=link.index, dtype="string")).eq("O")].copy()
    owner_cols = ["loan_id"]
    for c in ["client_age_years_at_loan", "gender", "district_id"]:
        if c in owners.columns:
            owner_cols.append(c)

    if owners.empty:
        owner = loans[["loan_id"]].copy()
    else:
        owner = owners[owner_cols].drop_duplicates("loan_id")
        owner = owner.rename(
            columns={
                "client_age_years_at_loan": "owner_age_years_at_loan",
                "gender": "owner_gender",
                "district_id": "owner_district_id",
            }
        )

    return composition.merge(owner, on="loan_id", how="left")


def build_district_features(
    frame: pd.DataFrame,
    district: pd.DataFrame,
) -> pd.DataFrame:
    """Join a curated set of account-district demographic features.

    Design choice:
    The raw district table contains many related counts from neighboring years.
    Instead of joining all A1-A16 columns, keep a compact socioeconomic profile:
    region, salary, unemployment, urbanization, entrepreneurship, population scale,
    and crime rate. This is easier to explain and reduces redundant correlations.
    """
    if "account_district_id" not in frame.columns:
        return frame

    d = district.copy()
    if {"num_crimes96", "num_inhabitants"}.issubset(d.columns):
        d["crime_rate96_per_1000"] = safe_divide(d["num_crimes96"], d["num_inhabitants"]) * 1000
    if "num_inhabitants" in d.columns:
        d["num_inhabitants_log1p"] = np.log1p(d["num_inhabitants"])

    keep_cols = [
        "district_id",
        "region",
        "average_salary",
        "unemployment_rate96",
        "ratio_urban",
        "num_entrep_per1000",
        "num_inhabitants_log1p",
        "crime_rate96_per_1000",
    ]
    keep_cols = [c for c in keep_cols if c in d.columns]
    d = d[keep_cols].copy()
    d = add_prefix_except(d, "account_district_", keep=["district_id"])

    return frame.merge(
        d,
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


def _aggregate_transactions(
    tx: pd.DataFrame,
    prefix: str,
    include_point_in_time: bool = False,
) -> pd.DataFrame:
    """Aggregate a transaction slice into a compact, non-redundant summary.

    Design choice:
    Use a small set of interpretable cash-flow, balance, and behavior features.
    Avoid broad one-hot crosstabs and avoid both count and ratio versions of the
    same idea unless each has a separate interpretation.
    """
    if tx.empty:
        return pd.DataFrame(columns=["loan_id"])

    work = tx.copy()

    # Meaningful operation behavior as rates.
    if "operation" in work.columns:
        work["is_cash_withdrawal"] = work["operation"].isin(["WIC", "CCW"]).astype("int8")
        work["is_cash_deposit"] = work["operation"].eq("CIC").astype("int8")
        work["is_bank_transfer"] = work["operation"].isin(["ROB", "COB"]).astype("int8")
    else:
        work["is_cash_withdrawal"] = 0
        work["is_cash_deposit"] = 0
        work["is_bank_transfer"] = 0

    # A few domain-specific category amount totals. No full category crosstab.
    if "category" in work.columns:
        work["pension_income_amount"] = np.where(work["category"].eq("PE"), work["amount"], 0.0)
        work["household_payment_amount"] = np.where(work["category"].eq("HH"), work["amount"], 0.0)
        work["insurance_payment_amount"] = np.where(work["category"].eq("IN"), work["amount"], 0.0)
        work["loan_related_payment_amount"] = np.where(work["category"].eq("LO"), work["amount"], 0.0)
    else:
        work["pension_income_amount"] = 0.0
        work["household_payment_amount"] = 0.0
        work["insurance_payment_amount"] = 0.0
        work["loan_related_payment_amount"] = 0.0

    agg_spec = {
        f"{prefix}tx_count": ("trans_id", "count") if "trans_id" in work.columns else ("amount", "count"),
        f"{prefix}active_days": ("trans_date", "nunique"),
        f"{prefix}credit_amount_sum": ("credit_amount", "sum"),
        f"{prefix}debit_amount_sum": ("debit_amount", "sum"),
        f"{prefix}net_cashflow_sum": ("signed_amount", "sum"),
        f"{prefix}negative_balance_count_tmp": ("is_negative_balance", "sum"),
        f"{prefix}credit_count_tmp": ("is_credit", "sum"),
        f"{prefix}cash_withdrawal_count_tmp": ("is_cash_withdrawal", "sum"),
        f"{prefix}cash_deposit_count_tmp": ("is_cash_deposit", "sum"),
        f"{prefix}bank_transfer_count_tmp": ("is_bank_transfer", "sum"),
        f"{prefix}pension_income_amount_sum": ("pension_income_amount", "sum"),
        f"{prefix}household_payment_amount_sum": ("household_payment_amount", "sum"),
        f"{prefix}insurance_payment_amount_sum": ("insurance_payment_amount", "sum"),
        f"{prefix}loan_related_payment_amount_sum": ("loan_related_payment_amount", "sum"),
    }

    if "balance" in work.columns:
        agg_spec.update(
            {
                f"{prefix}balance_min": ("balance", "min"),
                f"{prefix}balance_mean": ("balance", "mean"),
            }
        )

    base = work.groupby("loan_id").agg(**agg_spec).reset_index()

    # Rates are more comparable across accounts than duplicated raw count columns.
    base[f"{prefix}negative_balance_rate"] = safe_divide(
        base[f"{prefix}negative_balance_count_tmp"], base[f"{prefix}tx_count"]
    )
    base[f"{prefix}credit_tx_rate"] = safe_divide(base[f"{prefix}credit_count_tmp"], base[f"{prefix}tx_count"])
    base[f"{prefix}cash_withdrawal_rate"] = safe_divide(base[f"{prefix}cash_withdrawal_count_tmp"], base[f"{prefix}tx_count"])
    base[f"{prefix}cash_deposit_rate"] = safe_divide(base[f"{prefix}cash_deposit_count_tmp"], base[f"{prefix}tx_count"])
    base[f"{prefix}bank_transfer_rate"] = safe_divide(base[f"{prefix}bank_transfer_count_tmp"], base[f"{prefix}tx_count"])

    base = base.drop(
        columns=[
            f"{prefix}negative_balance_count_tmp",
            f"{prefix}credit_count_tmp",
            f"{prefix}cash_withdrawal_count_tmp",
            f"{prefix}cash_deposit_count_tmp",
            f"{prefix}bank_transfer_count_tmp",
        ],
        errors="ignore",
    )

    if include_point_in_time and "balance" in work.columns:
        sorted_tx = work.sort_values(["loan_id", "trans_date"] + (["trans_id"] if "trans_id" in work.columns else []))
        point = (
            sorted_tx.groupby("loan_id")
            .agg(
                **{
                    f"{prefix}days_since_last_tx": ("days_before_loan", "min"),
                    f"{prefix}history_span_days": ("days_before_loan", lambda s: s.max() - s.min()),
                    f"{prefix}first_balance": ("balance", "first"),
                    f"{prefix}last_balance": ("balance", "last"),
                }
            )
            .reset_index()
        )
        point[f"{prefix}balance_change"] = point[f"{prefix}last_balance"] - point[f"{prefix}first_balance"]
        point[f"{prefix}balance_change_per_day"] = safe_divide(
            point[f"{prefix}balance_change"], point[f"{prefix}history_span_days"].replace(0, np.nan)
        )
        # Keep last balance and trend; drop first balance because it is mainly an intermediate.
        point = point.drop(columns=[f"{prefix}first_balance"], errors="ignore")
        base = base.merge(point, on="loan_id", how="left")

    return base


def _window_bands(windows_days: Sequence[int], strict_before: bool = True) -> List[Tuple[int, int]]:
    """Convert cumulative cutoffs like (30, 90, 180) into non-overlapping bands."""
    cutoffs = sorted({int(w) for w in windows_days if int(w) > 0})
    bands: List[Tuple[int, int]] = []
    start = 1 if strict_before else 0
    for end in cutoffs:
        if end >= start:
            bands.append((start, end))
            start = end + 1
    return bands


def build_transaction_features(
    loans: pd.DataFrame,
    trans: pd.DataFrame,
    windows_days: Sequence[int] = (30, 90, 180, 365),
    include_full_history: bool = True,
    strict_before: bool = True,
) -> pd.DataFrame:
    """Build leakage-aware transaction features for each loan.

    Design choice:
    Rolling windows are non-overlapping bands: 1-30, 31-90, 91-180, 181-365.
    This avoids the common problem where cumulative windows create almost-identical
    columns such as w90_balance_min and w365_balance_min.
    """
    tx = transaction_pre_loan_join(loans, trans, strict_before=strict_before)
    features = loans[["loan_id"]].copy()

    if include_full_history:
        full = _aggregate_transactions(tx, prefix="hist_", include_point_in_time=True)
        features = features.merge(full, on="loan_id", how="left")

    for start, end in _window_bands(windows_days, strict_before=strict_before):
        window_tx = tx[tx["days_before_loan"].between(start, end)].copy()
        window = _aggregate_transactions(window_tx, prefix=f"tx_{start}_{end}d_", include_point_in_time=False)
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
    strict_before: bool = True,
) -> pd.DataFrame:
    """Build compact card features using only cards available before loan approval.

    Design choice:
    Keep the number of existing cards, the latest card type, and the age of the
    latest card. Do not create both `has_card` and `card_count`, and do not create
    first-card and last-card ages because they are identical for one-card accounts.
    """
    base = loans[["loan_id"]].copy()
    if card.empty:
        base["card_count_before_loan"] = 0
        base["latest_card_type_before_loan"] = "NO_CARD"
        base["days_since_latest_card"] = np.nan
        return base

    cards = card.merge(disp[["disp_id", "account_id"]], on="disp_id", how="left")
    cards = cards.merge(loans[["loan_id", "account_id", "loan_date"]], on="account_id", how="inner")
    if strict_before:
        cards = cards[cards["issued_date"].notna() & (cards["issued_date"] < cards["loan_date"])].copy()
    else:
        cards = cards[cards["issued_date"].notna() & (cards["issued_date"] <= cards["loan_date"])].copy()

    if cards.empty:
        base["card_count_before_loan"] = 0
        base["latest_card_type_before_loan"] = "NO_CARD"
        base["days_since_latest_card"] = np.nan
        return base

    counts = cards.groupby("loan_id").size().rename("card_count_before_loan").reset_index()
    latest = cards.sort_values(["loan_id", "issued_date"]).groupby("loan_id").tail(1)
    latest = latest[["loan_id", "loan_date", "issued_date"] + (["card_type"] if "card_type" in latest.columns else [])].copy()
    latest["days_since_latest_card"] = (latest["loan_date"] - latest["issued_date"]).dt.days
    if "card_type" in latest.columns:
        latest = latest.rename(columns={"card_type": "latest_card_type_before_loan"})
    else:
        latest["latest_card_type_before_loan"] = "UNKNOWN"
    latest = latest[["loan_id", "latest_card_type_before_loan", "days_since_latest_card"]]

    out = base.merge(counts, on="loan_id", how="left").merge(latest, on="loan_id", how="left")
    out["card_count_before_loan"] = out["card_count_before_loan"].fillna(0)
    out["latest_card_type_before_loan"] = out["latest_card_type_before_loan"].fillna("NO_CARD")
    return out


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
    raw_drop = {"birth_number", "account_district_id", "owner_district_id"}
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

    if config.drop_static_redundant_features:
        X = drop_static_redundant_features(X)
    if config.drop_constant_features:
        X = drop_constant_features(X)
    if config.drop_duplicate_features:
        X = drop_duplicate_features(X)

    return X, y, metadata


def validate_feature_matrix(X: pd.DataFrame, metadata: pd.DataFrame) -> None:
    """Raise an error if obvious leakage or row-grain problems remain.

    This intentionally checks only rules that should never be violated. It does not
    fail on ordinary missingness or high correlation, because those are handled by
    the model pipeline.
    """
    forbidden = {
        "target",
        "status",
        "loan_status",
        "loan_id",
        "account_id",
        "client_id",
        "amount",
        "duration",
        "payments",
    }
    present = sorted(forbidden.intersection(X.columns))
    if present:
        raise ValueError(f"Forbidden leakage/identifier columns found in X: {present}")

    if "loan_id" in metadata.columns and metadata["loan_id"].duplicated().any():
        raise ValueError("metadata contains duplicated loan_id values; row grain is not one-row-per-loan.")

    if any(pd.api.types.is_datetime64_any_dtype(X[c]) for c in X.columns):
        raise ValueError("Raw datetime columns found in X. Convert them to age/window features first.")


def audit_transaction_time_filter(
    loans: pd.DataFrame,
    trans: pd.DataFrame,
    strict_before: bool = True,
) -> Dict[str, object]:
    """Check that transaction joins respect the configured loan-date cutoff."""
    joined = transaction_pre_loan_join(loans, trans, strict_before=strict_before)
    if joined.empty:
        return {"n_joined_transactions": 0, "violations": 0, "ok": True}
    if strict_before:
        violations = int((joined["trans_date"] >= joined["loan_date"]).sum())
    else:
        violations = int((joined["trans_date"] > joined["loan_date"]).sum())
    return {
        "n_joined_transactions": int(len(joined)),
        "violations": violations,
        "ok": violations == 0,
    }


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
        if {"owner_district_id", "account_district_id"}.issubset(df.columns):
            df["owner_same_as_account_district"] = (
                df["owner_district_id"].astype("string") == df["account_district_id"].astype("string")
            ).astype("int8")

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
        card_features = build_card_features(
            loan,
            tables["card"],
            tables["disp"],
            strict_before=config.strict_pre_loan_cards,
        )
        df = df.merge(card_features, on="loan_id", how="left")

    X, y, metadata = final_cleanup_features(df, config)
    validate_feature_matrix(X, metadata)
    return X, y, metadata


def build_dataset_bundle(
    config: CzechFinancialConfig,
    tables: Optional[Mapping[str, pd.DataFrame]] = None,
) -> ClassificationDataset:
    """Build the dataset and return a named container instead of a bare tuple."""
    X, y, metadata = build_classification_dataset(config, tables=tables)
    return ClassificationDataset(X=X, y=y, metadata=metadata)


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
    """Create a sklearn ColumnTransformer for the curated feature matrix.

    There is intentionally no correlation-pruning transformer here. The dataset builder is designed
    to create compact, meaningful features from the start. The model pipeline only
    performs standard ML preprocessing: imputation, optional winsorization/scaling,
    and one-hot encoding.
    """
    numeric_steps: List[Tuple[str, object]] = [
        ("imputer", SimpleImputer(strategy="median", add_indicator=True)),
    ]
    if winsorize_numeric:
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
    """Preprocessor + classifier pipeline for the curated feature matrix."""
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
    """Strict date-based temporal split.

    Unlike a row-count split, this keeps every loan from the same loan_date on the
    same side of the split, so the maximum train date is strictly before the minimum
    test date.
    """
    if "loan_date" not in metadata.columns:
        raise ValueError("metadata must include loan_date for temporal split.")

    unique_dates = metadata["loan_date"].dropna().drop_duplicates().sort_values()
    if len(unique_dates) < 2:
        raise ValueError("Need at least two unique loan dates for a temporal split.")

    cutoff_pos = int(len(unique_dates) * (1 - test_size))
    cutoff_pos = min(max(cutoff_pos, 1), len(unique_dates) - 1)
    cutoff_date = unique_dates.iloc[cutoff_pos]

    train_idx = metadata.index[metadata["loan_date"] < cutoff_date]
    test_idx = metadata.index[metadata["loan_date"] >= cutoff_date]

    if len(train_idx) == 0 or len(test_idx) == 0:
        raise ValueError("Temporal split produced an empty train or test set.")

    return (
        X.loc[train_idx],
        X.loc[test_idx],
        y.loc[train_idx],
        y.loc[test_idx],
        metadata.loc[train_idx],
        metadata.loc[test_idx],
    )


def temporal_train_valid_test_split(
    X: pd.DataFrame,
    y: pd.Series,
    metadata: pd.DataFrame,
    valid_size: float = 0.2,
    test_size: float = 0.2,
) -> Tuple[
    pd.DataFrame, pd.DataFrame, pd.DataFrame, pd.Series, pd.Series, pd.Series, pd.DataFrame, pd.DataFrame, pd.DataFrame
]:
    """Strict date-based train/validation/test split.

    Use this when selecting thresholds, models, or hyperparameters. The test set
    remains untouched until the final evaluation.
    """
    if "loan_date" not in metadata.columns:
        raise ValueError("metadata must include loan_date for temporal split.")
    if valid_size <= 0 or test_size <= 0 or valid_size + test_size >= 1:
        raise ValueError("valid_size and test_size must be positive and sum to less than 1.")

    unique_dates = metadata["loan_date"].dropna().drop_duplicates().sort_values()
    if len(unique_dates) < 3:
        raise ValueError("Need at least three unique loan dates for train/valid/test split.")

    valid_cutoff_pos = int(len(unique_dates) * (1 - valid_size - test_size))
    test_cutoff_pos = int(len(unique_dates) * (1 - test_size))
    valid_cutoff_pos = min(max(valid_cutoff_pos, 1), len(unique_dates) - 2)
    test_cutoff_pos = min(max(test_cutoff_pos, valid_cutoff_pos + 1), len(unique_dates) - 1)

    valid_cutoff = unique_dates.iloc[valid_cutoff_pos]
    test_cutoff = unique_dates.iloc[test_cutoff_pos]

    train_idx = metadata.index[metadata["loan_date"] < valid_cutoff]
    valid_idx = metadata.index[(metadata["loan_date"] >= valid_cutoff) & (metadata["loan_date"] < test_cutoff)]
    test_idx = metadata.index[metadata["loan_date"] >= test_cutoff]

    if len(train_idx) == 0 or len(valid_idx) == 0 or len(test_idx) == 0:
        raise ValueError("Temporal split produced an empty train, validation, or test set.")

    return (
        X.loc[train_idx],
        X.loc[valid_idx],
        X.loc[test_idx],
        y.loc[train_idx],
        y.loc[valid_idx],
        y.loc[test_idx],
        metadata.loc[train_idx],
        metadata.loc[valid_idx],
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


__all__ = [
    "CzechFinancialConfig",
    "ClassificationDataset",
    "load_raw_tables",
    "load_and_normalize_tables",
    "build_classification_dataset",
    "build_dataset_bundle",
    "quick_data_quality_report",
    "temporal_train_test_split",
    "temporal_train_valid_test_split",
    "make_preprocessor",
    "make_model_pipeline",
    "evaluate_binary_classifier",
    "threshold_table",
    "get_feature_names",
    "validate_feature_matrix",
    "audit_transaction_time_filter",
    "run_example",
]


# ---------------------------------------------------------------------
# Example runner
# ---------------------------------------------------------------------


def run_example(data_dir: str | os.PathLike, target_mode: TargetMode = "finished_binary") -> None:
    """Example end-to-end run."""
    config = CzechFinancialConfig(data_dir=data_dir, target_mode=target_mode)
    X, y, meta = build_classification_dataset(config)

    print("DATA QUALITY REPORT")
    report = quick_data_quality_report(X, y, meta)
    for k, v in report.items():
        print(f"{k}: {v}")

    X_train, X_valid, X_test, y_train, y_valid, y_test, meta_train, meta_valid, meta_test = temporal_train_valid_test_split(
        X, y, meta, valid_size=0.2, test_size=0.2
    )

    clf = make_model_pipeline("random_forest", random_state=config.random_state)
    clf.fit(X_train, y_train)

    # Choose/report thresholds on validation data. The test set is reserved for final metrics.
    valid_proba = clf.predict_proba(X_valid)[:, 1]
    print("\nVALIDATION THRESHOLD TABLE")
    print(threshold_table(y_valid, valid_proba).to_string(index=False))

    chosen_threshold = 0.35
    results = evaluate_binary_classifier(clf, X_test, y_test, threshold=chosen_threshold)
    print("\nFINAL TEST EVALUATION")
    print("ROC-AUC:", results.get("roc_auc"))
    print("Average precision:", results["average_precision"])
    print("Balanced accuracy:", results["balanced_accuracy"])
    print("Confusion matrix:\n", results["confusion_matrix"])
    print(results["classification_report"])


if __name__ == "__main__":
    # Kaggle example:
    # run_example("/kaggle/input/1999-czech-financial-dataset", target_mode="finished_binary")
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument("--data-dir", required=True, help="Path to the dataset directory")
    parser.add_argument(
        "--target-mode",
        default="finished_binary",
        choices=["finished_binary", "all_binary", "multiclass"],
    )
    args = parser.parse_args()
    run_example(args.data_dir, target_mode=args.target_mode)
