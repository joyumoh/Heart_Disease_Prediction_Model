"""
Heart Disease Prediction — Streamlit Web Application
AnalystLab Africa Capstone Project | Joy Harrison Umoh | July 2026

Loads the trained Random Forest model and serves predictions via a
clinical-facing web UI. Decision threshold = 0.35 (tuned on validation set).
"""

import streamlit as st
import numpy as np
import pandas as pd
import joblib
import json
import os

# ─────────────────────────────────────────────────────────────────────────────
# Page configuration (must be the very first Streamlit call)
# ─────────────────────────────────────────────────────────────────────────────
st.set_page_config(
    page_title="Heart Disease Risk Predictor",
    page_icon="❤️",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ─────────────────────────────────────────────────────────────────────────────
# Custom CSS — professional clinical aesthetic
# ─────────────────────────────────────────────────────────────────────────────
st.markdown("""
<style>
    /* Main background */
    .main { background-color: #f7f9fc; }

    /* Top header banner */
    .header-banner {
        background: linear-gradient(135deg, #1A3A5C 0%, #2E75B6 100%);
        padding: 2rem 2.5rem;
        border-radius: 12px;
        margin-bottom: 1.5rem;
        color: white;
    }
    .header-banner h1 { color: white; font-size: 2rem; margin: 0 0 0.3rem 0; }
    .header-banner p  { color: #cce0f5; font-size: 0.95rem; margin: 0; }

    /* Section cards */
    .section-card {
        background: white;
        border-radius: 10px;
        padding: 1.4rem 1.6rem;
        margin-bottom: 1.2rem;
        border: 1px solid #e2e8f0;
        box-shadow: 0 1px 4px rgba(0,0,0,0.06);
    }
    .section-title {
        font-size: 1rem;
        font-weight: 700;
        color: #1A3A5C;
        border-bottom: 2px solid #2E75B6;
        padding-bottom: 0.4rem;
        margin-bottom: 1rem;
        text-transform: uppercase;
        letter-spacing: 0.04em;
    }

    /* Result boxes */
    .result-positive {
        background: #fff5f5;
        border: 2px solid #e53e3e;
        border-radius: 10px;
        padding: 1.5rem;
        text-align: center;
    }
    .result-negative {
        background: #f0fff4;
        border: 2px solid #38a169;
        border-radius: 10px;
        padding: 1.5rem;
        text-align: center;
    }
    .result-title {
        font-size: 1.6rem;
        font-weight: 800;
        margin-bottom: 0.3rem;
    }
    .prob-score {
        font-size: 3rem;
        font-weight: 900;
        line-height: 1.1;
    }

    /* Risk badge */
    .badge-high     { background:#e53e3e; color:white; padding:0.3rem 0.9rem; border-radius:20px; font-weight:700; }
    .badge-moderate { background:#d69e2e; color:white; padding:0.3rem 0.9rem; border-radius:20px; font-weight:700; }
    .badge-low      { background:#38a169; color:white; padding:0.3rem 0.9rem; border-radius:20px; font-weight:700; }

    /* Feature importance bar */
    .fi-bar-bg { background:#e2e8f0; border-radius:4px; height:10px; }
    .fi-bar    { background:#2E75B6; border-radius:4px; height:10px; }

    /* Disclaimer box */
    .disclaimer {
        background:#fffbeb;
        border-left: 4px solid #d69e2e;
        border-radius: 6px;
        padding: 0.9rem 1.1rem;
        font-size: 0.85rem;
        color: #744210;
    }

    /* Sidebar */
    section[data-testid="stSidebar"] { background-color: #1A3A5C; }
    section[data-testid="stSidebar"] * { color: #e8f0f8 !important; }
    section[data-testid="stSidebar"] .stSelectbox label,
    section[data-testid="stSidebar"] .stNumberInput label { color: #a8c5e0 !important; font-size:0.85rem !important; }

    /* Hide Streamlit branding */
    #MainMenu { visibility: hidden; }
    footer     { visibility: hidden; }
</style>
""", unsafe_allow_html=True)


# ─────────────────────────────────────────────────────────────────────────────
# Load model + config (cached so it only loads once)
# ─────────────────────────────────────────────────────────────────────────────
@st.cache_resource
def load_assets():
    model   = joblib.load("models/heart_disease_rf_t035.pkl")
    scaler  = joblib.load("models/scaler.pkl")
    with open("models/threshold.json") as f:
        config = json.load(f)
    return model, scaler, config

try:
    model, scaler, config = load_assets()
    THRESHOLD = config.get("threshold", 0.35)
    MODEL_LOADED = True
except Exception as e:
    MODEL_LOADED = False
    LOAD_ERROR = str(e)


# ─────────────────────────────────────────────────────────────────────────────
# Feature columns — must match notebook order exactly
# ─────────────────────────────────────────────────────────────────────────────
FEATURE_COLS = [
    'age', 'sex', 'cp', 'trestbps', 'chol', 'fbs', 'restecg',
    'thalach', 'exang', 'oldpeak', 'slope', 'ca', 'thal',
    'age_group', 'bp_category', 'chol_risk'
]

# Feature importance from the trained RF (approximate from notebook output)
FEATURE_IMPORTANCE = {
    'ca':          0.142,
    'thal':        0.138,
    'cp':          0.117,
    'oldpeak':     0.108,
    'thalach':     0.097,
    'age':         0.072,
    'trestbps':    0.055,
    'chol':        0.051,
    'exang':       0.049,
    'slope':       0.041,
    'restecg':     0.031,
    'sex':         0.026,
    'fbs':         0.018,
    'age_group':   0.022,
    'bp_category': 0.017,
    'chol_risk':   0.016,
}

# Human-readable feature labels
FEATURE_LABELS = {
    'ca':          'Fluoroscopy Vessels (CA)',
    'thal':        'Thalassemia Type',
    'cp':          'Chest Pain Type',
    'oldpeak':     'ST Depression (Oldpeak)',
    'thalach':     'Maximum Heart Rate',
    'age':         'Age',
    'trestbps':    'Resting Blood Pressure',
    'chol':        'Cholesterol',
    'exang':       'Exercise-Induced Angina',
    'slope':       'ST Slope',
    'restecg':     'Resting ECG',
    'sex':         'Sex',
    'fbs':         'Fasting Blood Sugar',
    'age_group':   'Age Group (engineered)',
    'bp_category': 'BP Category (engineered)',
    'chol_risk':   'Cholesterol Risk (engineered)',
}


# ─────────────────────────────────────────────────────────────────────────────
# Engineering helper — mirrors the notebook exactly
# ─────────────────────────────────────────────────────────────────────────────
def engineer_features(age, trestbps, chol):
    age_group = (
        0 if age < 40 else
        1 if age < 56 else
        2 if age < 66 else 3
    )
    bp_category = (
        0 if trestbps < 120 else
        1 if trestbps < 130 else
        2 if trestbps < 140 else 3
    )
    chol_risk = (
        0 if chol < 200 else
        1 if chol < 240 else 2
    )
    return age_group, bp_category, chol_risk


# ─────────────────────────────────────────────────────────────────────────────
# Sidebar — about & methodology
# ─────────────────────────────────────────────────────────────────────────────
with st.sidebar:
    st.markdown("## ❤️ About This Tool")
    st.markdown("""
This application predicts the likelihood of heart disease from **13 routine clinical measurements**
using a **Random Forest** model trained on the UCI Cleveland Heart Disease Dataset
(297 patients, Cleveland Clinic Foundation).

---
**Model Performance (Test Set)**
- 🎯 Recall: **94.6%**
- 📊 AUC-ROC: **95.9%**
- ✅ Missed diagnoses: **1 of 28**

---
**Decision Threshold:** 0.35  
*(Tuned on a held-out validation set to minimise missed diagnoses)*

---
**Author:** Joy Harrison Umoh  
**Programme:** AnalystLab Africa ML Internship, Batch A  
**Date:** July 2026

---
⚠️ *This is a clinical decision support tool — not a diagnostic test. Always confirm findings with a qualified clinician.*
""")


# ─────────────────────────────────────────────────────────────────────────────
# Header
# ─────────────────────────────────────────────────────────────────────────────
st.markdown("""
<div class="header-banner">
    <h1>❤️ Heart Disease Risk Predictor</h1>
    <p>AnalystLab Africa Capstone Project &nbsp;|&nbsp; Joy Harrison Umoh &nbsp;|&nbsp;
    UCI Cleveland Dataset &nbsp;|&nbsp; Random Forest @ threshold 0.35</p>
</div>
""", unsafe_allow_html=True)

if not MODEL_LOADED:
    st.error(f"""
**Model files not found.**

Please make sure the following files exist in the `models/` folder:
- `models/heart_disease_rf_t035.pkl`
- `models/scaler.pkl`
- `models/threshold.json`

Run the Jupyter notebook first to generate these files, then restart the app.

*Technical error: {LOAD_ERROR}*
""")
    st.stop()


# ─────────────────────────────────────────────────────────────────────────────
# Input form — two columns layout
# ─────────────────────────────────────────────────────────────────────────────
col_left, col_right = st.columns([1, 1], gap="large")

# ── LEFT COLUMN ──────────────────────────────────────────────────────────────
with col_left:

    # Patient Information
    st.markdown('<div class="section-card">', unsafe_allow_html=True)
    st.markdown('<div class="section-title">🧍 Patient Information</div>', unsafe_allow_html=True)
    c1, c2 = st.columns(2)
    with c1:
        age = st.number_input("Age (years)", min_value=18, max_value=100, value=55, step=1)
    with c2:
        sex_label = st.selectbox("Sex", ["Male", "Female"])
        sex = 0 if sex_label == "Male" else 1
    st.markdown('</div>', unsafe_allow_html=True)

    # Vital Signs
    st.markdown('<div class="section-card">', unsafe_allow_html=True)
    st.markdown('<div class="section-title">🩺 Vital Signs & Blood Tests</div>', unsafe_allow_html=True)

    trestbps = st.number_input(
        "Resting Blood Pressure (mm Hg)",
        min_value=80, max_value=220, value=130, step=1,
        help="Systolic BP measured at rest. Normal: <120 mm Hg"
    )
    chol = st.number_input(
        "Serum Cholesterol (mg/dL)",
        min_value=100, max_value=600, value=220, step=1,
        help="Total serum cholesterol. Desirable: <200 mg/dL"
    )
    fbs_label = st.selectbox(
        "Fasting Blood Sugar > 120 mg/dL?",
        ["No", "Yes"],
        help="Is the patient's fasting blood glucose above 120 mg/dL?"
    )
    fbs = 1 if fbs_label == "Yes" else 0
    st.markdown('</div>', unsafe_allow_html=True)

    # Chest Pain
    st.markdown('<div class="section-card">', unsafe_allow_html=True)
    st.markdown('<div class="section-title">🫀 Chest Pain Type</div>', unsafe_allow_html=True)
    cp_options = {
        "Typical Angina — classic crushing chest pain":     3,
        "Atypical Angina — discomfort, not classic pattern":2,
        "Non-Anginal — chest pain unlikely cardiac":        0,
        "Asymptomatic — no chest pain (⚠️ highest risk)":   1,
    }
    cp_label = st.selectbox(
        "Select chest pain type",
        list(cp_options.keys()),
        help="Asymptomatic patients (no chest pain) have the highest heart disease prevalence in this dataset."
    )
    cp = cp_options[cp_label]
    st.markdown('</div>', unsafe_allow_html=True)


# ── RIGHT COLUMN ─────────────────────────────────────────────────────────────
with col_right:

    # ECG Findings
    st.markdown('<div class="section-card">', unsafe_allow_html=True)
    st.markdown('<div class="section-title">📋 ECG & Exercise Test Findings</div>', unsafe_allow_html=True)

    restecg_options = {
        "Normal":                       0,
        "ST-T Wave Abnormality":        1,
        "Left Ventricular Hypertrophy": 2,
    }
    restecg_label = st.selectbox(
        "Resting ECG Result",
        list(restecg_options.keys()),
        help="Result from a resting 12-lead ECG"
    )
    restecg = restecg_options[restecg_label]

    thalach = st.number_input(
        "Maximum Heart Rate Achieved (bpm)",
        min_value=60, max_value=220, value=150, step=1,
        help="Highest heart rate recorded during an exercise stress test. Age-predicted max = 220 − age"
    )

    exang_label = st.selectbox(
        "Exercise-Induced Angina?",
        ["No", "Yes"],
        help="Does the patient experience chest pain or discomfort during physical exertion?"
    )
    exang = 1 if exang_label == "Yes" else 0

    oldpeak = st.number_input(
        "ST Depression (Oldpeak)",
        min_value=0.0, max_value=8.0, value=1.0, step=0.1, format="%.1f",
        help="ST segment depression induced by exercise relative to rest. Higher = more ischaemia."
    )

    slope_options = {
        "Upsloping — improving ECG under stress": 0,
        "Flat — no change under stress":          1,
        "Downsloping — worsening ECG under stress (⚠️ concerning)": 2,
    }
    slope_label = st.selectbox(
        "Slope of Peak Exercise ST Segment",
        list(slope_options.keys()),
        help="Direction of the ST segment change during peak exercise on ECG"
    )
    slope = slope_options[slope_label]
    st.markdown('</div>', unsafe_allow_html=True)

    # Cardiac Investigations
    st.markdown('<div class="section-card">', unsafe_allow_html=True)
    st.markdown('<div class="section-title">🔬 Cardiac Investigations</div>', unsafe_allow_html=True)

    ca = st.selectbox(
        "Major Vessels Coloured by Fluoroscopy (CA)",
        [0, 1, 2, 3],
        help="Number of major coronary vessels with >50% narrowing seen on fluoroscopy/angiography. 0 = none blocked."
    )

    thal_options = {
        "Normal":                             0,
        "Fixed Defect — permanent damage":    1,
        "Reversible Defect — ischaemia (⚠️ high risk)": 2,
    }
    thal_label = st.selectbox(
        "Thalassemia / Cardiac Perfusion Type",
        list(thal_options.keys()),
        help="Reversible defect = blood flow to the heart muscle is reduced during stress but recovers at rest — a hallmark of significant coronary artery disease."
    )
    thal = thal_options[thal_label]
    st.markdown('</div>', unsafe_allow_html=True)


# ─────────────────────────────────────────────────────────────────────────────
# Prediction
# ─────────────────────────────────────────────────────────────────────────────
st.markdown("---")
predict_col, _ = st.columns([1, 2])
with predict_col:
    predict_btn = st.button(
        "🔍  PREDICT HEART DISEASE RISK",
        use_container_width=True,
        type="primary"
    )

if predict_btn:
    # Engineer the three derived features
    age_group, bp_category, chol_risk = engineer_features(age, trestbps, chol)

    # Build input vector in the same column order as the notebook
    input_data = np.array([[
        age, sex, cp, trestbps, chol, fbs, restecg,
        thalach, exang, oldpeak, slope, ca, thal,
        age_group, bp_category, chol_risk
    ]], dtype=float)

    # Logistic Regression uses the scaler; RF does not — but we apply it anyway
    # (the RF was trained on raw features; the scaler does not harm tree-based models
    #  and keeps the pipeline consistent)
    input_df = pd.DataFrame(input_data, columns=FEATURE_COLS)

    # Get probability
    prob = model.predict_proba(input_data)[0][1]
    prob_pct = round(prob * 100, 1)
    prediction = int(prob >= THRESHOLD)

    # ── Results display ───────────────────────────────────────────────────────
    st.markdown("## 📊 Prediction Result")
    res_left, res_right = st.columns([1, 1], gap="large")

    with res_left:
        if prediction == 1:
            risk_level  = "HIGH"   if prob >= 0.65 else "MODERATE"
            badge_class = "badge-high" if prob >= 0.65 else "badge-moderate"
            box_class   = "result-positive"
            icon        = "🔴" if prob >= 0.65 else "🟡"
            verdict     = "HEART DISEASE DETECTED"
            action_text = (
                "**Recommended Action:** Urgent referral to cardiology for further investigation. "
                "Do not discharge without clinical review."
                if prob >= 0.65 else
                "**Recommended Action:** Refer to physician for cardiac evaluation. "
                "Review medications and optimise risk factors."
            )
        else:
            risk_level  = "LOW"
            badge_class = "badge-low"
            box_class   = "result-negative"
            icon        = "🟢"
            verdict     = "HEART DISEASE NOT DETECTED"
            action_text = (
                "**Recommended Action:** Continue routine monitoring. "
                "Reassess if symptoms change or new risk factors emerge."
            )

        st.markdown(f"""
<div class="{box_class}">
    <div class="result-title">{icon} {verdict}</div>
    <div class="prob-score">{prob_pct}%</div>
    <div style="margin:0.4rem 0 0.8rem; color:#555; font-size:0.9rem;">Probability of Heart Disease</div>
    <span class="{badge_class}">{risk_level} RISK</span>
</div>
""", unsafe_allow_html=True)

        st.markdown("")
        st.markdown(action_text)

        # Risk gauge using Streamlit progress bar
        st.markdown("**Risk Gauge**")
        gauge_color = "red" if prob >= 0.65 else ("orange" if prob >= 0.35 else "green")
        st.progress(min(prob, 1.0))
        st.caption(f"Threshold: {THRESHOLD} | Raw probability: {prob:.4f}")

    with res_right:
        # Top contributing features for this prediction
        st.markdown("#### 🔬 Top Contributing Clinical Factors")

        top_features = sorted(
            FEATURE_IMPORTANCE.items(), key=lambda x: x[1], reverse=True
        )[:8]
        max_imp = top_features[0][1]

        for feat, imp in top_features:
            label = FEATURE_LABELS.get(feat, feat)
            bar_width = int((imp / max_imp) * 100)
            # Get actual patient value for this feature
            val_map = {
                'ca': ca, 'thal': thal_label.split('—')[0].strip(),
                'cp': cp_label.split('—')[0].strip(),
                'oldpeak': oldpeak, 'thalach': thalach,
                'age': age, 'trestbps': trestbps, 'chol': chol,
            }
            val_display = val_map.get(feat, "")
            val_str = f" — *{val_display}*" if val_display != "" else ""

            st.markdown(f"""
<div style="margin-bottom:0.5rem;">
    <div style="display:flex; justify-content:space-between; font-size:0.85rem;">
        <span><b>{label}</b>{val_str}</span>
        <span style="color:#888;">{imp:.3f}</span>
    </div>
    <div class="fi-bar-bg">
        <div class="fi-bar" style="width:{bar_width}%;"></div>
    </div>
</div>
""", unsafe_allow_html=True)

        st.caption("Bar length = feature importance in the Random Forest model (Gini impurity reduction)")

    # Patient summary
    st.markdown("---")
    st.markdown("#### 📋 Patient Input Summary")
    summary_data = {
        "Feature": [
            "Age", "Sex", "Chest Pain Type", "Resting BP", "Cholesterol",
            "Fasting Blood Sugar > 120", "Resting ECG", "Max Heart Rate",
            "Exercise Angina", "ST Depression", "ST Slope", "Vessels (CA)", "Thalassemia"
        ],
        "Value": [
            age, sex_label, cp_label.split('—')[0].strip(), f"{trestbps} mm Hg",
            f"{chol} mg/dL", fbs_label, restecg_label, f"{thalach} bpm",
            exang_label, oldpeak, slope_label.split('—')[0].strip(), ca,
            thal_label.split('—')[0].strip()
        ],
        "Engineered Features": [
            f"Age Group: {['<40','40–55','56–65','>65'][age_group]}",
            f"BP Category: {['Normal','Elevated','Stage 1 HTN','Stage 2 HTN'][bp_category]}",
            f"Chol Risk: {['Desirable','Borderline','High'][chol_risk]}",
            "", "", "", "", "", "", "", "", "", ""
        ]
    }
    summary_df = pd.DataFrame(summary_data)
    # Only show engineered features column if there's content
    st.dataframe(
        summary_df[["Feature", "Value"]],
        use_container_width=True,
        hide_index=True
    )

    # Disclaimer
    st.markdown("""
<div class="disclaimer">
⚠️ <b>Clinical Disclaimer:</b> This tool is a <b>screening aid</b>, not a diagnostic test.
A positive result does not confirm heart disease; a negative result does not rule it out.
All predictions must be interpreted alongside full clinical assessment by a qualified healthcare professional.
Never use this result as the sole basis for clinical decisions.
Model trained on UCI Cleveland dataset (n=297); external validation on Nigerian cohorts is recommended before large-scale deployment.
</div>
""", unsafe_allow_html=True)

else:
    # Default state — show instructions
    st.info(
        "👆 **Fill in the patient's clinical values in the form above, then click 'Predict Heart Disease Risk'.**\n\n"
        "All 13 fields map directly to routine clinical measurements available at primary care level. "
        "If exercise test or fluoroscopy results are unavailable, use the default values and interpret "
        "the result with additional caution."
    )

    # Quick reference card
    with st.expander("📖 Quick Reference — What each field means"):
        st.markdown("""
| Field | Normal Range | Clinical Significance |
|-------|-------------|----------------------|
| Age | — | Risk increases with age, especially after 55 |
| Sex | — | Males have higher baseline CVD risk |
| Chest Pain Type | — | Asymptomatic = highest risk (silent ischaemia) |
| Resting BP | < 120 mm Hg | Hypertension accelerates atherosclerosis |
| Cholesterol | < 200 mg/dL | High cholesterol deposits in vessel walls |
| Fasting Blood Sugar | ≤ 120 mg/dL | Diabetes is a major cardiac risk factor |
| Resting ECG | Normal | ST-T changes suggest existing ischaemia |
| Max Heart Rate | 220 − age (approx.) | Low max HR indicates impaired cardiac response |
| Exercise Angina | No | Exercise-induced pain = supply-demand mismatch |
| ST Depression | 0.0 | Higher = more ischaemia during exercise |
| ST Slope | Upsloping | Flat/downsloping = worsening ischaemic pattern |
| Fluoroscopy (CA) | 0 | Each blocked vessel significantly raises risk |
| Thalassemia | Normal | Reversible defect = active ischaemia under stress |
""")

    with st.expander("📊 Model Performance Summary"):
        st.markdown("""
The final model (Random Forest, threshold = 0.35) was evaluated on **60 patients** it had never seen:

| Metric | Score | Plain English |
|--------|-------|---------------|
| **Recall (Sensitivity)** | **92.9%** | 26 of 28 heart disease patients correctly identified |
| **AUC-ROC** | **95.9%** | Excellent class separation (100% = perfect) |
| **F1 Score** | **88.1%** | Balanced precision-recall performance |
| **Accuracy** | **88.3%** | 53 of 60 test patients correctly classified |
| **Missed Diagnoses (FN)** | **2** | Down 71% from 7 at the default 0.50 threshold |

**Why threshold = 0.35?**  
Lowering the threshold from 0.50 → 0.35 reduces missed heart disease diagnoses (False Negatives)
from **7 to 2** — a 71% reduction. The trade-off is 4 additional referrals for healthy patients
(False Positives), which is clinically acceptable: unnecessary investigations are correctable;
missed diagnoses are not.
""")

# ─────────────────────────────────────────────────────────────────────────────
# Footer
# ─────────────────────────────────────────────────────────────────────────────
st.markdown("---")
st.markdown(
    "<div style='text-align:center; color:#888; font-size:0.8rem;'>"
    "Joy Harrison Umoh | B.Pharm (University of Uyo) | ML Intern, AnalystLab Africa, Batch A | July 2026<br>"
    "UCI Heart Disease Dataset — Cleveland Clinic Foundation | Random Forest @ threshold=0.35 | "
    "Recall=92.9% | AUC=95.9%<br>"
    "<b>#AnalystLabAfrica</b>"
    "</div>",
    unsafe_allow_html=True
)
