# ❤️ Heart Disease Prediction Using Machine Learning

> **Can a computer learn to spot heart disease from routine clinic measurements - before it becomes fatal?**  
> This project answers that question with a machine learning model built on real patient data from the Cleveland Clinic Foundation.



**Author:** Joy Harrison Umoh   
**Programme:** AnalystLab Africa - Machine Learning Internship  
**Duration:** May - July 2026  
**Dataset:** UCI Heart Disease Dataset - Cleveland Clinic Foundation (via Kaggle)  
**Final Model:** Random Forest Classifier | Recall = 92.9% | AUC-ROC = 95.9%



## 🌐 Live Application

>  **Try the app here:** [heart-disease-predictor.streamlit.app](https://heart-disease-predictor.streamlit.app)

The web application allows any healthcare worker - pharmacist, nurse, community health extension worker - to enter a patient's routine clinical measurements and receive an instant risk prediction. No coding knowledge required.


## 📋 Table of Contents

1. [Why This Project Matters](#-why-this-project-matters)
2. [What the Model Does](#-what-the-model-does)
3. [The Data - What Information Goes In](#-the-data--what-information-goes-in)
4. [How the Model Was Built](#-how-the-model-was-built)
5. [How to Use the Web App  Step-by-Step Guide](#-how-to-use-the-web-app--step-by-step-guide)
6. [Understanding Your Results](#-understanding-your-results)
7. [Model Performance  In Plain English](#-model-performance--in-plain-english)
8. [Key Clinical Findings](#-key-clinical-findings)
9. [For Developers - Running Locally](#-for-developers--running-locally)
11. [Project Files](#-project-files)
12. [Important Limitations](#-important-limitations)
13. [References](#-references)



## ❤️ Why This Project Matters

Cardiovascular disease (CVD) is the **number one cause of death worldwide**, responsible for approximately **17.9 million deaths every year** (World Health Organization, 2023). In Nigeria and across sub-Saharan Africa, the burden is growing fast - driven by rising rates of hypertension, diabetes, sedentary lifestyles, and critically, **limited access to early diagnosis**.

### The gap this project addresses

| The Problem | The Reality in Nigeria                                                        |
|-------------|-------------------------------------------------------------------------------|
| Heart disease often has **no obvious symptoms** until a fatal event | Many patients present late - sometimes in cardiac arrest                      |
| Definitive tests (echocardiography, stress tests) require specialist centres | Most Nigerians access care at primary health centres and community pharmacies |
| Cardiac risk scoring tools exist but are rarely used at community level | GPs and pharmacists see these patients daily but lack a quick risk tool       |

**As a licensed pharmacist working in Kano**, I interact daily with patients collecting antihypertensives, statins, and aspirin - all medications that manage cardiovascular risk. Yet there is no practical, validated tool I can use at the counter to flag a patient who may need urgent cardiology referral.

This project builds that tool.



## 🔍 What the Model Does

In simple terms: the model **reads 13 routine clinical measurements** and answers one question:

> *"Based on this patient's clinical profile, does this person likely have heart disease?"*

The output is:
- A **Yes/No prediction** (Heart Disease Detected / Not Detected)  
- A **risk probability score** from 0% to 100%  
- A **risk category**: 🟢 Low / 🟡 Moderate / 🔴 High  

The model was designed with one clinical priority above all others: **never miss a real heart disease patient**. In medicine, this property is called *sensitivity* or *recall* - and it is what separates a useful screening tool from a dangerous one.



## 📊 The Data — What Information Goes In

The model uses **13 clinical features** - all of which can be obtained through a routine clinic visit, basic blood tests, and a resting ECG. No specialist equipment is required.

| # | Feature | What It Measures | How It Is Recorded |
|---|---------|------------------|--------------------|
| 1 | **Age** | Patient's age in years | Number (e.g. 55) |
| 2 | **Sex** | Biological sex | Male / Female |
| 3 | **Chest Pain Type** | The nature of chest discomfort | 4 types (see below) |
| 4 | **Resting Blood Pressure** | BP at rest (mm Hg) | Number (e.g. 130) |
| 5 | **Cholesterol** | Total serum cholesterol (mg/dL) | Number (e.g. 220) |
| 6 | **Fasting Blood Sugar** | Whether blood sugar > 120 mg/dL | Yes / No |
| 7 | **Resting ECG** | Electrical activity of the heart at rest | 3 categories |
| 8 | **Maximum Heart Rate** | Highest heart rate achieved during exercise | Number (e.g. 150 bpm) |
| 9 | **Exercise-Induced Angina** | Chest pain brought on by exercise | Yes / No |
| 10 | **ST Depression (Oldpeak)** | ECG change seen during exercise | Number (e.g. 1.5) |
| 11 | **ST Slope** | Direction of ECG change during exercise | Upsloping / Flat / Downsloping |
| 12 | **Fluoroscopy Vessels (CA)** | Number of major blood vessels with blockage | 0 / 1 / 2 / 3 |
| 13 | **Thalassemia Type (Thal)** | Blood disorder type affecting oxygen delivery | Normal / Fixed defect / Reversible defect |

### Understanding Chest Pain Types

One of the most counter-intuitive findings in cardiology - and confirmed by this dataset - is that **"typical" chest pain is not the most dangerous type**:

| Chest Pain Type | Description | Risk Level |
|-----------------|-------------|------------|
| Typical Angina | Classic crushing chest pain, radiating to arm/jaw | Moderate |
| Atypical Angina | Chest discomfort that doesn't fit the classic pattern | Moderate |
| Non-Anginal | Chest pain unlikely to be cardiac in origin | Lower |
| **Asymptomatic** | **No chest pain at all** | **HIGHEST** |

> ⚠️ **The "silent heart attack" is a real phenomenon.** Asymptomatic chest pain type in this dataset is the strongest categorical predictor of heart disease - because these are the patients who walk in for a blood pressure check and leave without ever knowing their risk.

---

## 🏗️ How the Model Was Built

This section explains the methodology in plain language - no statistics degree required.

### Step 1: Getting the Data
The dataset contains **297 patient records** from the Cleveland Clinic Foundation - real patients who underwent cardiac evaluation. Each record includes the 13 features above plus a confirmed diagnosis.

- **No disease:** 160 patients (53.9%)  
- **Heart disease confirmed:** 137 patients (46.1%)

### Step 2: Cleaning the Data
A small number of records had missing information (9 rows out of 304). Rather than guess these values, those rows were removed - because in the most important features (thalassemia type, vessel blockage count), guessing could mislead the model.

### Step 3: Adding Clinical Knowledge
Three additional features were engineered from existing data, using recognised medical guidelines:

- **Age Group** - using cardiology risk bands (under 40 / 40–55 / 55–65 / over 65)  
- **Blood Pressure Category** - using the ACC/AHA 2017 guidelines (Normal / Elevated / Stage 1 HTN / Stage 2 HTN)  
- **Cholesterol Risk Tier** - using the NCEP ATP III guidelines (Desirable / Borderline / High)

### Step 4: Splitting the Data (The Most Important Design Decision)

The data was divided into **three separate groups**, not two:


| | 🟦 Training Set                      | 🟨 Validation Set                            | 🔒 Test Set                                                                  |
|---|--------------------------------------|----------------------------------------------|------------------------------------------------------------------------------|
| **Size** | 60% - 177 patients                   | 20% - 60 patients                            | 20% - 60 patients                                                            |
| **Purpose** | Model learns patterns from this data | Used to fine-tune how sensitive the model is | **Locked.** Opened only once at the very end to check real-world performance |

> **Why three groups?** If you tune your settings and test performance on the same data, you get an unfairly optimistic result - like practicing with the exam paper then claiming you scored well on the exam. The test set was only ever seen once, at the final stage.

### Step 5: Training Four Models

Four different algorithms were trained and compared:

| Model | What It Is (Plain English)                                                                                                       |
|-------|----------------------------------------------------------------------------------------------------------------------------------|
| **Logistic Regression** | Calculates the probability of disease using a weighted formula - like a scoring system used in medicine                          |
| **Decision Tree** | A series of yes/no questions arranged like a flowchart - e.g. "Is the heart rate below 140? → If yes, is ST depression above 2?" |
| **Random Forest** | Hundreds of decision trees voting together - like getting a second opinion from 200 doctors and going with the majority          |
| **XGBoost** | An advanced version of the same idea, where each tree learns from the mistakes of the previous ones                              |

### Step 6: Tuning the Sensitivity Threshold

By default, models predict "Heart Disease" when they are 50% or more confident. But in cardiac screening, 50% is too conservative - some real heart disease patients might score 40% and be incorrectly sent home.

The threshold was lowered to **35%** based on testing on the validation set:

```
At 50% threshold:  6 patients with real heart disease were missed
At 35% threshold:  only 1 patients with real heart disease was missed
                   - a 71% reduction in missed diagnoses 
```

This is the threshold used in the live application.

---

## 📱 How to Use the Web App - Step-by-Step Guide

> **Access the app:** [https://heartdiseasepredictionmodel-9yussjrrzysfmxscvrb2wa.streamlit.app](https://heart-disease-predictor.streamlit.app)

The app is designed to be used by healthcare workers at the point of care. Here is a complete walkthrough:

---

### Step 1 - Open the App

Open the link above in any web browser on your phone, tablet, or computer. No account or login is needed.

You will see a form divided into sections:

```
┌─────────────────────────────────────────────┐
│  💊 Heart Disease Risk Predictor            │
│  AnalystLab Africa Capstone — Joy H. Umoh  │
├─────────────────────────────────────────────┤
│  PATIENT INFORMATION                        │
│  ┌─────────────────┐  ┌──────────────────┐ │
│  │ Age             │  │ Sex              │ │
│  └─────────────────┘  └──────────────────┘ │
│                                             │
│  CLINICAL MEASUREMENTS                      │
│  ┌─────────────────────────────────────┐   │
│  │ Resting Blood Pressure (mm Hg)      │   │
│  └─────────────────────────────────────┘   │
│  ...                                        │
│  [  PREDICT RISK  ]                         │
└─────────────────────────────────────────────┘
```



### Step 2 - Enter Patient Information

Fill in each field using information from the patient's clinical record or current consultation:

#### 🧍 Basic Information

| Field | What to Enter | Example |
|-------|---------------|---------|
| Age | Patient's age in years | `58` |
| Sex | Select Male or Female | `Male` |

#### 🩺 Vital Signs & Blood Tests

| Field | What to Enter | Where to Get It | Example |
|-------|---------------|-----------------|---------|
| Resting Blood Pressure | Systolic BP in mm Hg | BP machine reading | `145` |
| Cholesterol | Total serum cholesterol in mg/dL | Latest blood test result | `230` |
| Fasting Blood Sugar | Is blood sugar > 120 mg/dL? | Blood glucose test | `Yes` |

#### 🫀 Cardiac-Specific Fields

| Field | What to Enter | Notes |
|-------|---------------|-------|
| Chest Pain Type | Select the most fitting description | Use "Asymptomatic" if no chest pain |
| Resting ECG | Result from resting ECG report | Written on ECG printout |
| Exercise-Induced Angina | Does chest pain occur during exercise? | From patient history |
| Maximum Heart Rate | Highest heart rate during exercise test | From treadmill/stress test if available; otherwise leave at default |
| ST Depression (Oldpeak) | Value from exercise ECG report | Written as a decimal (e.g. 1.5) |
| ST Slope | Direction noted on exercise ECG | From ECG report |
| Fluoroscopy Vessels (CA) | Number of vessels with >50% blockage | From angiography/cardiac catheterisation report |
| Thalassemia Type | Blood disorder type | From haematology report or known history |

> 💡 **If you don't have all values:** Some fields (like fluoroscopy and exercise ECG) may not be available at primary care level. Use the default values and note that the prediction confidence may be lower. The most important fields for screening are: Age, Sex, Chest Pain Type, Blood Pressure, Cholesterol, and Maximum Heart Rate.



### Step 3 - Click "Predict Risk"

After filling in the form, click the **Predict Risk** button at the bottom of the page.

The app processes the information in seconds.



### Step 4 - Read Your Results

You will see a results panel that looks like this:

```
┌─────────────────────────────────────────────────────┐
│                  PREDICTION RESULT                   │
│                                                     │
│   🔴  HEART DISEASE DETECTED                        │
│                                                     │
│   Risk Probability:  █████████░  87%                │
│                                                     │
│   Risk Category:  🔴 HIGH RISK                      │
│                                                     │
│ ─────────────────────────────────────────────────── │
│  Top Contributing Factors:                          │
│  • Reversible thalassemia defect detected           │
│  • 2 major vessels blocked (fluoroscopy)            │
│  • ST depression of 2.3 during exercise             │
│  • Maximum heart rate only 118 bpm (low for age)    │
│                                                     │
│ ─────────────────────────────────────────────────── │
│  ⚠️ This is a screening tool. Always confirm with    │
│  clinical assessment and specialist referral.        │
└─────────────────────────────────────────────────────┘
```



## 📖 Understanding Your Results

### What the Risk Levels Mean

| Result | Risk Probability | What It Means | Recommended Action |
|--------|-----------------|---------------|--------------------|
| 🟢 **Low Risk** | < 35% | The patient's profile does not suggest significant cardiac risk | Continue routine monitoring; reassess if symptoms change |
| 🟡 **Moderate Risk** | 35% – 65% | Some risk factors are present; clinical judgement needed | Refer to physician for further cardiac evaluation; optimise medications |
| 🔴 **High Risk** | > 65% | The patient's profile is consistent with heart disease | Urgent referral to cardiology; do not delay further investigation |

### What the Probability Score Means

The percentage score represents the model's **confidence that heart disease is present**. For example:

- **87%** → The model is highly confident this patient has heart disease
- **42%** → The model is uncertain; clinical judgement is essential
- **12%** → The model is reasonably confident this patient does NOT have heart disease

> ⚠️ **IMPORTANT:** A probability below 35% does not guarantee the absence of heart disease. This tool is a **screening aid**, not a diagnostic test. Always use it alongside clinical assessment.

### Why the App May Flag More Patients Than Expected

The model is deliberately set to **cast a wide net**. At the 35% probability threshold, it will flag some patients who turn out to be healthy - these are called false positives. This is intentional: it is far better to send a healthy patient for an ECG they didn't need than to send a heart disease patient home without investigation.

In medicine, this trade-off is called **sensitivity vs specificity**:
- **Sensitivity (what we optimise for):** Catching every real case, even if it means some extra referrals
- **Specificity:** Only flagging genuine cases, but risking missing some

For a cardiac screening tool, **sensitivity always comes first**.



## 📈 Model Performance 

The model was tested on **60 patients it had never seen before** (the locked test set). Here is what happened:

### The Most Important Number

```
Of 28 patients with confirmed heart disease in the test group:

  ✅ 27 patients were correctly identified as having heart disease
  ❌  1 patients were missed (false negatives)

That is a DETECTION RATE of 96.4%
```

### Comparing Default vs Tuned Threshold

```
                        Default (50%)    Tuned (35%)
                        ─────────────    ───────────
Patients correctly        
identified (Recall)         78.6%  →       96.4%

Missed diagnoses (FN)          6   →           1   (71% fewer)

Unnecessary referrals (FP)     2   →           5   (acceptable trade-off)
```

### Full Performance Summary

| Metric | Score     | Plain English                                                                  |
|--------|-----------|--------------------------------------------------------------------------------|
| **Recall (Sensitivity)** | **96.4%** | Detected 27 of 28 heart disease patients                                       |
| **AUC-ROC** | **95.9%** | Excellent ability to separate sick from healthy (100% = perfect, 50% = random) |
| **F1 Score** | **88.1%** | Balanced measure of precision and recall                                       |
| **Accuracy** | **88.3%** | Correct prediction on 53 of 60 test patients                                   |
| **False Negatives** | **1**     | Only 1 heart disease patients was missed                                       |

### Cross-Validation — Is This Reliable?

To confirm the model wasn't just lucky on one test set, it was evaluated on **five different random splits** of the data. The F1 score was consistent across all five (standard deviation of ±0.04), confirming the model generalises reliably.



## 🔬 Key Clinical Findings

Beyond the prediction itself, the model reveals which clinical features carry the most weight in diagnosing heart disease. These results are consistent with established cardiology literature.

### Top 5 Predictive Features (Most to Least Important)

| Rank | Feature                          | Why It Matters Clinically                                                                                                                                         |
|------|----------------------------------|-------------------------------------------------------------------------------------------------------------------------------------------------------------------|
| 🥇 1st | **CA - Fluoroscopy Vessels**     | Direct count of blocked coronary arteries. More blockages = more disease. This is the most definitive structural marker.                                          |
| 🥈 2nd | **Thal - Thalassemia Type**      | A "reversible defect" means blood flow to the heart muscle is reduced during stress but normalises at rest - the hallmark of significant coronary artery disease. |
| 🥉 3rd | **CP - Chest Pain Type**         | Counterintuitively, *asymptomatic* patients (no chest pain) are highest risk. Silent ischaemia is a well-documented and dangerous phenomenon.                     |
| 4th | **Oldpeak - ST Depression**      | The ECG signature of myocardial ischaemia during exercise. Greater depression = more severe oxygen deprivation to the heart muscle.                               |
| 5th | **Thalach - Maximum Heart Rate** | A healthy heart accelerates during exercise. A heart with narrowed arteries cannot. Low maximum heart rate for age is a clinical warning sign.                    |

> 💊 **Pharmacist's note:** Four of the top five features (CA, Thal, Oldpeak, Thalach) are from cardiac investigations, but the fifth (Chest Pain Type) and several others (BP, Cholesterol, Fasting Blood Sugar) are fully assessable at a community pharmacy. A pharmacy-based pre-screening using available features can still provide valuable risk stratification.



## 💻 For Developers - Running Locally

### Prerequisites

- Python 3.9 or above
- pip (Python package installer)
- Git

### Installation

```bash
# 1. Clone the repository
git clone https://github.com/yourusername/heart-disease-prediction.git
cd heart-disease-prediction

# 2. (Recommended) Create a virtual environment
python -m venv venv
source venv/bin/activate          # Mac/Linux
venv\Scripts\activate             # Windows

# 3. Install all required packages
pip install -r requirements.txt
```

### Running the Jupyter Notebook

```bash
# Start Jupyter
jupyter notebook

# Open: heart_disease_prediction.ipynb
```

The notebook runs end-to-end and will reproduce all outputs, visualisations, and model files.

### Running the Streamlit App Locally

```bash
# Make sure the models/ folder exists and contains the saved model
# (Run the notebook first if it doesn't)

streamlit run app.py
```

The app will open automatically in your browser at `http://localhost:8501`.

### Repository Structure

```
heart-disease-prediction/
│
├── heart_disease_prediction.ipynb   ← Complete ML notebook (run this first)
├── app.py                           ← Streamlit web application
├── Clevaland_heart_disease.csv      ← Source dataset (Cleveland Clinic)
├── requirements.txt                 ← Python dependencies
├── README.md                        ← This file
│
├── models/
│   ├── heart_disease_rf_t035.pkl    ← Saved Random Forest model
│   ├── scaler.pkl                   ← Feature scaler (StandardScaler)
│   └── threshold.json               ← Model configuration (threshold=0.35)
│
└── outputs/
    ├── 01_target_distribution.png
    ├── 02_feature_distributions.png
    ├── 03_categorical_features.png
    ├── 04_correlation_heatmap.png
    ├── 05_age_thalach_scatter.png
    ├── 10_threshold_tuning_fn.png
    ├── 11_model_comparison_t035.png
    ├── 12_fn_comparison_roc.png
    ├── 13_rf_confusion_comparison.png
    ├── 14_cross_validation.png
    └── 15_feature_importance_rf.png
```



## 📁 Project Files

| File                               | Description                                                                         |
|------------------------------------|-------------------------------------------------------------------------------------|
| `heart_disease_prediction.ipynb`   | Complete Jupyter notebook - data loading, EDA, preprocessing, modelling, evaluation |
| `app.py`                           | Streamlit web application for clinical deployment                                   |
| `Clevaland_heart_disease.csv`      | Raw dataset (Cleveland Clinic Foundation, via Kaggle)                               |
| `requirements.txt`                 | All Python dependencies with version constraints                                    |
| `models/heart_disease_rf_t035.pkl` | Trained Random Forest model (serialised with joblib)                                |
| `models/scaler.pkl`                | Fitted StandardScaler for feature normalisation                                     |
| `models/threshold.json`            | Model metadata including selected threshold                                         |
| `outputs/*.png`                    | All generated visualisations from the notebook                                      |



## ⚠️ Important Limitations

This tool is a **clinical decision support aid** - not a replacement for clinical judgment or specialist evaluation. Please read these limitations carefully before use.

1. **Training data origin:** The model was trained on data from the Cleveland Clinic Foundation in the United States. While the UCI Cleveland dataset is the most widely used benchmark for heart disease ML, it may not fully capture disease patterns in Nigerian or West African patients. **Validation on a local cohort is strongly recommended before large-scale deployment.**

2. **Missing feature values:** If fluoroscopy (CA) or thalassemia results are unavailable (as is common at primary care level), the model will still run but with reduced accuracy. Interpret results with additional caution.

3. **Not a diagnostic test:** A positive prediction does not confirm heart disease. It indicates elevated risk and the need for further investigation by a qualified clinician.

4. **Not a negative guarantee:** A negative prediction does not rule out heart disease. If a patient has strong clinical symptoms, refer regardless of the model's output.

5. **Threshold was set for screening:** The 35% threshold prioritises catching cases over avoiding false alarms. In settings where unnecessary referrals have high cost, the threshold may need to be adjusted with a clinician's input.

> 🩺 **Always refer to a physician or cardiologist for definitive diagnosis and management.**



## 📚 References

- World Health Organization (2023). *Cardiovascular diseases fact sheet.* WHO Global Health Observatory.
- Detrano, R., Janosi, A., Steinbrunn, W., et al. (1989). *International application of a new probability algorithm for the diagnosis of coronary artery disease.* American Journal of Cardiology, 64(5), 304–310.
- Grundy, S.M. et al. (2018). *AHA/ACC/AACVPR/AAPA/ABC/ACPM/ADA/AGS/APhA/ASPC/NLA/PCNA Guideline on the Management of Blood Cholesterol.* Circulation, 139(25).
- Whelton, P.K. et al. (2018). *2017 ACC/AHA/AAPA/ABC/ACPM High Blood Pressure Guidelines.* Journal of the American College of Cardiology, 71(19).
- Dataset Source. *Heart Disease Dataset.* [https://www.kaggle.com/datasets/redwankarimsony/heart-disease-data](https://archive.ics.uci.edu/dataset/45/heart+disease)



## 🙏 Acknowledgements

- **AnalystLab Africa** - for the structured Machine Learning internship programme and the capstone framework
- **Cleveland Clinic Foundation** - for making this landmark cardiac dataset publicly available



*This project was completed as the Week 8 Capstone for the AnalystLab Africa Machine Learning Internship.*  
*#AnalystLabAfrica*
