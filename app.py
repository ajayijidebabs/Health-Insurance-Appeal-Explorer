"""
Streamlit app: Independent Medical Review (IMR) appeal explorer + outcome predictor.

Tab 1 - Look up an existing case by Reference ID (details fill in automatically).
Tab 2 - Enter a brand-new client's case details and get a predicted chance of overturn.

Run locally:   streamlit run app.py
"""
import pandas as pd
import streamlit as st

import core

st.set_page_config(page_title="IMR Appeal Explorer", page_icon="🩺", layout="wide")


# ---------------------------------------------------------------------------
# Load data + train model once (cached across reruns and users)
# ---------------------------------------------------------------------------
@st.cache_data(show_spinner="Loading case records...")
def get_data():
    return core.load_data()


@st.cache_resource(show_spinner="Training the model (first start only, a few seconds)...")
def get_model(_df):
    return core.build_model(_df)


df = get_data()
model, metrics, train_idx, test_idx = get_model(df)

# ---------------------------------------------------------------------------
# Header + sidebar
# ---------------------------------------------------------------------------
st.title("🩺 Health Insurance Appeal Explorer")
st.write(
    "Based on **19,245 real California Independent Medical Review (IMR) cases (2001-2016)**. "
    "When a health plan denies a treatment, an independent reviewer can uphold the denial or overturn it. "
    "Look up a past case, or enter a new case to see how often similar denials were overturned."
)

with st.sidebar:
    st.header("About the model")
    st.metric("AUC-ROC on unseen cases", f"{metrics['auc_roc']:.2f}")
    st.metric("Accuracy on unseen cases", f"{metrics['accuracy']:.0%}")
    st.caption(
        f"Tuned Gradient Boosting trained on {metrics['n_train']:,} cases and tested on "
        f"{metrics['n_test']:,} cases it never saw."
    )
    st.warning(
        "**Not a decision tool.** The model only sees broad labels (diagnosis, treatment type, age, "
        "gender, year, plus how common the diagnosis/treatment detail is), not the medical evidence, so its "
        "predictions are only modestly better than chance (AUC-ROC about 0.7). "
        "It is for research and exploration."
    )
    st.caption(
        "The written 'Findings' text is never used for prediction: it is the reviewer's explanation "
        "written after the decision, so it gives the answer away."
    )

tab_lookup, tab_new = st.tabs(["🔎 Look up an existing case", "➕ New client / new case"])

# ---------------------------------------------------------------------------
# Tab 1: lookup by Reference ID
# ---------------------------------------------------------------------------
with tab_lookup:
    st.subheader("Look up a case by Reference ID")
    examples = df["Reference ID"].sample(3, random_state=1).tolist()
    ref_id = st.text_input(
        "Reference ID",
        placeholder=f"e.g. {examples[0]}",
        help="Case IDs look like MN16-22639 (MN = Medical Necessity, EI = Experimental/Investigational).",
    )
    st.caption("Example IDs you can try: " + ", ".join(examples))

    if ref_id.strip():
        case = core.lookup_case(df, ref_id)
        if case is None:
            st.error(f"No case found with ID '{ref_id.strip()}'. Check the spelling (format like MN16-22639).")
        else:
            st.success(f"Case {case['Reference ID']} found.")
            c1, c2, c3, c4 = st.columns(4)
            c1.metric("Report year", int(case["Report Year"]))
            c2.metric("Age range", case["Age Range"])
            c3.metric("Gender", case["Patient Gender"])
            c4.metric("Case type", case["Type"])

            d1, d2 = st.columns(2)
            d1.markdown(f"**Diagnosis:** {case['Diagnosis Category']}")
            d1.markdown(f"**Diagnosis detail:** {case['Diagnosis Sub Category']}")
            d2.markdown(f"**Treatment:** {case['Treatment Category']}")
            d2.markdown(f"**Treatment detail:** {case['Treatment Sub Category']}")

            actual = case["Determination"]
            if case["target"] == 1:
                st.info(f"**Actual decision:** {actual}. The health plan had to authorize the treatment.")
            else:
                st.info(f"**Actual decision:** {actual}. The health plan's denial stood.")

            p = core.predict_case(
                model, case["Diagnosis Category"], case["Treatment Category"], case["Type"],
                case["Age Range"], case["Patient Gender"], int(case["Report Year"]),
                case["Diagnosis Sub Category"], case["Treatment Sub Category"],
            )
            idx = case.name
            if idx in test_idx:
                provenance = "This case was **not** used to train the model, so this is a genuine prediction."
            else:
                provenance = ("This case **was part of the model's training data**, so treat the prediction "
                              "as an illustration, not a fair test.")
            st.markdown(f"**Model's estimated chance of overturn:** {p:.0%}")
            st.progress(p)
            st.caption(provenance)

            rate, n = core.group_stats(df, case["Diagnosis Category"], case["Age Range"])
            if rate is not None:
                st.markdown(
                    f"**Similar cases** ({case['Diagnosis Category']}, age {case['Age Range']}): "
                    f"overturned **{rate:.0%}** of the time across {n:,} cases."
                )

            with st.expander("Reviewer's written findings (for reading only, never used by the model)"):
                st.warning("This text was written after the decision and usually states or implies the outcome.")
                st.write(case["Findings"] if pd.notna(case["Findings"]) else "No findings text recorded.")

# ---------------------------------------------------------------------------
# Tab 2: new client / new case
# ---------------------------------------------------------------------------
with tab_new:
    st.subheader("Enter a new client's case")
    st.write("Fill in the details, then press **Predict**. Nothing here is stored on the server. "
             "Please do not enter real names or identifying information.")

    diagnoses = sorted(d for d in df["Diagnosis Category"].unique() if d != "Unknown")
    treatments = sorted(t for t in df["Treatment Category"].unique() if t != "Unknown")
    case_types = sorted(df["Type"].dropna().unique())
    genders = sorted(g for g in df["Patient Gender"].unique() if g != "Unknown")

    st.session_state.setdefault("new_cases", [])

    client_name = st.text_input(
        "Label for your own records (optional)",
        help="Please do not type a real name or other identifying details. This is only a note for your download.",
    )
    f1, f2 = st.columns(2)
    diagnosis = f1.selectbox("Diagnosis category", diagnoses)
    treatment = f2.selectbox("Treatment category", treatments)
    g1, g2 = st.columns(2)
    dx_sub = g1.selectbox(
        "Diagnosis detail (optional)",
        [core.NOT_SPECIFIED] + core.sub_options(df, "Diagnosis Category", diagnosis, core.DX_SUB),
    )
    tx_sub = g2.selectbox(
        "Treatment detail (optional)",
        [core.NOT_SPECIFIED] + core.sub_options(df, "Treatment Category", treatment, core.TX_SUB),
    )
    f3, f4, f5 = st.columns(3)
    age_range = f3.selectbox("Age range", core.AGE_ORDER)
    gender = f4.selectbox("Gender", genders)
    case_type = f5.selectbox("Case type", case_types)
    year = st.number_input(
        "Report year", min_value=2001, max_value=2016, value=2016, step=1,
        help="The model learned from 2001-2016, so years beyond 2016 are not offered.",
    )
    submitted = st.button("Predict", type="primary")

    if submitted:
        p = core.predict_case(model, diagnosis, treatment, case_type, age_range, gender, int(year),
                              dx_sub, tx_sub)
        rate, n = core.group_stats(df, diagnosis, age_range)

        st.markdown(f"### Estimated chance this denial is overturned: {p:.0%}")
        st.progress(p)
        if p >= 0.65:
            st.success("Leans toward **overturned** (similar cases usually won on appeal).")
        elif p <= 0.35:
            st.error("Leans toward **upheld** (similar cases usually lost on appeal).")
        else:
            st.info("**Uncertain**: the case details alone do not point strongly either way.")

        if rate is not None:
            st.markdown(
                f"**What actually happened to similar cases** ({diagnosis}, age {age_range}): "
                f"overturned **{rate:.0%}** of the time across {n:,} past cases."
            )
            if n < 30:
                st.warning(f"Only {n} past cases match this group, so treat both numbers cautiously.")
        else:
            st.warning("No past cases match this exact diagnosis and age group, so there is no history to compare.")

        if diagnosis == "Autism Spectrum" and age_range == "0-10":
            st.info(
                "Pediatric autism-treatment denials were overturned far more often than any other group "
                "in this dataset (the rate has varied by year, peaking around 2010-2012). The model's "
                "percentage can sit below the historical rate for this group, so read both numbers."
            )

        st.session_state["new_cases"].append({
            "Client": client_name or "(unlabelled)",
            "Diagnosis": diagnosis, "Diagnosis detail": dx_sub,
            "Treatment": treatment, "Treatment detail": tx_sub, "Age range": age_range,
            "Gender": gender, "Case type": case_type, "Year": int(year),
            "Predicted chance of overturn": round(p, 3),
            "Past overturn rate (similar cases)": None if rate is None else round(rate, 3),
            "Similar past cases": n,
        })

    if st.session_state["new_cases"]:
        st.markdown("#### Cases entered this session")
        history = pd.DataFrame(st.session_state["new_cases"])
        st.dataframe(history)
        st.download_button(
            "Download as CSV", history.to_csv(index=False).encode("utf-8"),
            file_name="new_client_predictions.csv", mime="text/csv",
        )
        st.caption("This list resets when the page is refreshed. Download the CSV to keep it.")
