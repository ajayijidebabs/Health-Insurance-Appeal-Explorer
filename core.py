"""
Core logic for the IMR appeal-outcome app (no Streamlit code in here, so it can be tested on its own).

- load_data():      reads the DMHC Independent Medical Review CSV and prepares features
- build_model():    trains a tuned Gradient Boosting model on structured, leakage-safe fields only
- predict_case():   returns the model's probability that a denial is overturned
- group_stats():    historical overturn rate for a diagnosis x age group (with case count)

The free-text 'Findings' column is NEVER used as a model input: it is a post-decision
justification and leaks the outcome (see the project report, Section 4.1).
"""
from pathlib import Path

import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.metrics import accuracy_score, roc_auc_score
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder

DATA_PATH = Path(__file__).parent / "data" / "Independent_Medical_Reviews.csv.gz"

CAT_COLS = ["Diagnosis Category", "Treatment Category", "Type", "Age Range",
            "Patient Gender", "Diag_Age"]
DX_SUB, TX_SUB = "Diagnosis Sub Category", "Treatment Sub Category"
FREQ_COLS = [DX_SUB + "_freq", TX_SUB + "_freq"]   # how common the sub-category is in the training data
NUM_COLS = ["Report Year"] + FREQ_COLS
FEATURES = CAT_COLS + NUM_COLS
NOT_SPECIFIED = "Not specified"
OVERTURNED = "Overturned Decision of Health Plan"
AGE_ORDER = ["0-10", "11_20", "21-30", "31-40", "41-50", "51-64", "65+"]
RANDOM_STATE = 42


def load_data(path=DATA_PATH):
    df = pd.read_csv(path)
    df["target"] = (df["Determination"] == OVERTURNED).astype(int)
    for col in ["Diagnosis Category", "Treatment Category", "Type", "Age Range", "Patient Gender"]:
        df[col] = df[col].fillna("Unknown")
    df["Diag_Age"] = df["Diagnosis Category"] + "_" + df["Age Range"]
    for col in (DX_SUB, TX_SUB):
        df[col] = df[col].fillna("Unknown")
    return df


def _freq_maps(train_df):
    """Share of training cases having each sub-category (computed on training rows only)."""
    return {c: train_df[c].value_counts(normalize=True).to_dict() for c in (DX_SUB, TX_SUB)}


def _add_freq(frame, maps):
    out = frame.copy()
    for c in (DX_SUB, TX_SUB):
        out[c + "_freq"] = out[c].map(maps[c]).fillna(0.0)
    return out


def build_model(df):
    """Train the tuned Gradient Boosting model on a stratified 75% split; report honest metrics
    on the other 25%. Hyperparameters come from a randomized search with 3-fold CV on the
    training data only (best CV AUC-ROC about 0.70)."""
    train_idx, test_idx = train_test_split(
        df.index, test_size=0.25, stratify=df["target"], random_state=RANDOM_STATE
    )
    maps = _freq_maps(df.loc[train_idx])
    X = _add_freq(df, maps)[FEATURES]
    y = df["target"]
    pre = ColumnTransformer(
        [("cat", OneHotEncoder(handle_unknown="ignore"), CAT_COLS)], remainder="passthrough"
    )
    clf = GradientBoostingClassifier(
        n_estimators=300, max_depth=5, learning_rate=0.05, subsample=0.85,
        min_samples_leaf=40, random_state=RANDOM_STATE,
    )
    pipe = Pipeline([("prep", pre), ("clf", clf)])
    pipe.fit(X.loc[train_idx], y.loc[train_idx])
    pipe.freq_maps = maps
    proba = pipe.predict_proba(X.loc[test_idx])[:, 1]
    metrics = {
        "auc_roc": roc_auc_score(y.loc[test_idx], proba),
        "accuracy": accuracy_score(y.loc[test_idx], (proba >= 0.5).astype(int)),
        "n_train": len(train_idx),
        "n_test": len(test_idx),
    }
    return pipe, metrics, set(train_idx), set(test_idx)


def make_row(model, diagnosis, treatment, case_type, age_range, gender, year,
             dx_sub=None, tx_sub=None):
    row = pd.DataFrame([{
        "Diagnosis Category": diagnosis,
        "Treatment Category": treatment,
        "Type": case_type,
        "Age Range": age_range,
        "Patient Gender": gender,
        "Diag_Age": f"{diagnosis}_{age_range}",
        "Report Year": year,
        DX_SUB: dx_sub if dx_sub and dx_sub != NOT_SPECIFIED else "Unknown",
        TX_SUB: tx_sub if tx_sub and tx_sub != NOT_SPECIFIED else "Unknown",
    }])
    return _add_freq(row, model.freq_maps)[FEATURES]


def predict_case(model, diagnosis, treatment, case_type, age_range, gender, year,
                 dx_sub=None, tx_sub=None):
    """Probability (0-1) that the health plan's denial is overturned."""
    row = make_row(model, diagnosis, treatment, case_type, age_range, gender, year, dx_sub, tx_sub)
    return float(model.predict_proba(row)[0, 1])


def sub_options(df, parent_col, parent_value, sub_col, min_cases=5):
    """Sub-categories seen with a given category (common ones only), for the optional dropdowns."""
    sub = df[df[parent_col] == parent_value][sub_col]
    counts = sub[sub != "Unknown"].value_counts()
    return sorted(counts[counts >= min_cases].index)


def group_stats(df, diagnosis, age_range):
    """Historical overturn rate and case count for a diagnosis x age group."""
    sub = df[(df["Diagnosis Category"] == diagnosis) & (df["Age Range"] == age_range)]
    if len(sub) == 0:
        return None, 0
    return float(sub["target"].mean()), int(len(sub))


def lookup_case(df, reference_id):
    """Find a case by Reference ID (case-insensitive, whitespace-tolerant). Returns a row or None."""
    key = str(reference_id).strip().upper()
    hit = df[df["Reference ID"].str.upper() == key]
    return None if hit.empty else hit.iloc[0]
