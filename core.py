"""
Core logic for the IMR appeal-outcome app (no Streamlit code in here, so it can be tested on its own).

- load_data():      reads the DMHC Independent Medical Review CSV and prepares features
- build_model():    trains a Gradient Boosting model on structured, leakage-safe fields only
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
NUM_COLS = ["Report Year"]
FEATURES = CAT_COLS + NUM_COLS
OVERTURNED = "Overturned Decision of Health Plan"
AGE_ORDER = ["0-10", "11_20", "21-30", "31-40", "41-50", "51-64", "65+"]
RANDOM_STATE = 42


def load_data(path=DATA_PATH):
    df = pd.read_csv(path)
    df["target"] = (df["Determination"] == OVERTURNED).astype(int)
    for col in ["Diagnosis Category", "Treatment Category", "Type", "Age Range", "Patient Gender"]:
        df[col] = df[col].fillna("Unknown")
    df["Diag_Age"] = df["Diagnosis Category"] + "_" + df["Age Range"]
    return df


def build_model(df):
    """Train on a stratified 75% split; report honest held-out metrics on the other 25%."""
    X, y = df[FEATURES], df["target"]
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.25, stratify=y, random_state=RANDOM_STATE
    )
    pre = ColumnTransformer(
        [("cat", OneHotEncoder(handle_unknown="ignore"), CAT_COLS)], remainder="passthrough"
    )
    clf = GradientBoostingClassifier(
        n_estimators=300, max_depth=4, learning_rate=0.1, subsample=0.85,
        min_samples_leaf=10, random_state=RANDOM_STATE,
    )
    pipe = Pipeline([("prep", pre), ("clf", clf)])
    pipe.fit(X_train, y_train)
    proba = pipe.predict_proba(X_test)[:, 1]
    metrics = {
        "auc_roc": roc_auc_score(y_test, proba),
        "accuracy": accuracy_score(y_test, (proba >= 0.5).astype(int)),
        "n_train": len(X_train),
        "n_test": len(X_test),
    }
    return pipe, metrics, set(X_train.index), set(X_test.index)


def make_row(diagnosis, treatment, case_type, age_range, gender, year):
    return pd.DataFrame([{
        "Diagnosis Category": diagnosis,
        "Treatment Category": treatment,
        "Type": case_type,
        "Age Range": age_range,
        "Patient Gender": gender,
        "Diag_Age": f"{diagnosis}_{age_range}",
        "Report Year": year,
    }])[FEATURES]


def predict_case(model, diagnosis, treatment, case_type, age_range, gender, year):
    """Probability (0-1) that the health plan's denial is overturned."""
    row = make_row(diagnosis, treatment, case_type, age_range, gender, year)
    return float(model.predict_proba(row)[0, 1])


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
