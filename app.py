"""
PhishGuard – Intelligent Phishing Website Detection System
==========================================================
Streamlit Web Application
-------------------------
A modern, cybersecurity-themed interactive dashboard for real-time
phishing URL inspection and machine learning model benchmarking.
"""

import os
import sys
import json
import time
from typing import Dict, Any, Optional

import streamlit as st
import pandas as pd
import numpy as np
import joblib

# Add src folder to system path for modular imports
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
SRC_DIR = os.path.join(BASE_DIR, "src")
if SRC_DIR not in sys.path:
    sys.path.append(SRC_DIR)

from feature_extraction import (
    FEATURE_COLUMNS,
    extract_features_single,
    is_valid_url
)

# ------------------------------------------------------------------------------
# App Configuration & Page Metadata
# ------------------------------------------------------------------------------
st.set_page_config(
    page_title="PhishGuard – Phishing URL Detection",
    page_icon="🛡️",
    layout="wide",
    initial_sidebar_state="expanded"
)

MODELS_DIR = os.path.join(BASE_DIR, "models")
RESULTS_DIR = os.path.join(BASE_DIR, "results")

MODEL_CONFIG = {
    "Decision Tree Classifier (Recommended)": {
        "file": "decision_tree_model.pkl",
        "badge": "Champion Model",
        "description": "Non-linear decision splits. Best threat recall (55.5%) and highest F1-Score (67.2%)."
    },
    "Logistic Regression": {
        "file": "logistic_model.pkl",
        "badge": "Linear Baseline",
        "description": "StandardScaler + Logistic Regression. 92.1% precision, conservative classifier."
    },
    "Support Vector Machine (SVM)": {
        "file": "svm_model.pkl",
        "badge": "Max-Margin Linear",
        "description": "StandardScaler + sigmoid-calibrated LinearSVC. 92.7% held-out precision; live scores are not verified probabilities."
    }
}


def set_current_page(page: str):
    st.session_state["current_page"] = page
    st.session_state["sidebar_nav"] = page


def launch_scanner():
    st.session_state["scanned_url"] = "https://www.google.com/"
    set_current_page("🔍 URL Scanner")


PAGES = [
    "🏠 Home Dashboard",
    "🔍 URL Scanner",
    "📊 Model Performance",
    "ℹ️ About & Architecture"
]

# Initialize navigation session state
if "current_page" not in st.session_state:
    st.session_state["current_page"] = "🔍 URL Scanner"

if "scanned_url" not in st.session_state:
    st.session_state["scanned_url"] = ""

# ------------------------------------------------------------------------------
# Custom Cyber-Security Dark Theme CSS
# ------------------------------------------------------------------------------
CYBER_CSS = """
<style>
    /* Dark Navy & Slate Theme */
    .stApp {
        background-color: #0b132b;
        color: #e2e8f0;
    }
    
    /* Header & Branding */
    .brand-title {
        font-family: 'Inter', system-ui, -apple-system, sans-serif;
        font-size: 2.4rem;
        font-weight: 800;
        background: linear-gradient(90deg, #38bdf8 0%, #818cf8 100%);
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
        margin-bottom: 0.2rem;
    }
    .brand-subtitle {
        font-size: 1.05rem;
        color: #94a3b8;
        margin-bottom: 1.5rem;
    }
    
    /* Modern Glassmorphic Cards */
    .cyber-card {
        background: #1c2541;
        border: 1px solid #334155;
        border-radius: 12px;
        padding: 1.25rem 1.5rem;
        margin-bottom: 1.2rem;
        box-shadow: 0 4px 14px rgba(0, 0, 0, 0.25);
    }
    .cyber-card-highlight {
        background: #1e293b;
        border-left: 4px solid #38bdf8;
        border-radius: 8px;
        padding: 1rem 1.25rem;
        margin-bottom: 1rem;
    }
    
    /* Threat Alert Badges */
    .badge-phish {
        background-color: rgba(239, 68, 68, 0.15);
        color: #ef4444;
        border: 1px solid #ef4444;
        padding: 0.4rem 1rem;
        border-radius: 9999px;
        font-weight: 700;
        display: inline-block;
        font-size: 0.95rem;
    }
    .badge-legit {
        background-color: rgba(16, 185, 129, 0.15);
        color: #10b981;
        border: 1px solid #10b981;
        padding: 0.4rem 1rem;
        border-radius: 9999px;
        font-weight: 700;
        display: inline-block;
        font-size: 0.95rem;
    }
    
    /* Feature Metrics Grid */
    .metric-value {
        font-size: 1.6rem;
        font-weight: 800;
        color: #38bdf8;
    }
    .metric-label {
        font-size: 0.85rem;
        color: #94a3b8;
        text-transform: uppercase;
        letter-spacing: 0.05em;
    }
    
    /* Disclaimer Note */
    .disclaimer-box {
        background-color: rgba(245, 158, 11, 0.08);
        border: 1px solid rgba(245, 158, 11, 0.3);
        border-radius: 8px;
        padding: 0.85rem 1rem;
        color: #f59e0b;
        font-size: 0.88rem;
        margin-top: 1rem;
    }
</style>
"""
st.markdown(CYBER_CSS, unsafe_allow_html=True)


# ------------------------------------------------------------------------------
# Cached Model & Data Loaders
# ------------------------------------------------------------------------------
@st.cache_resource
def load_trained_model(model_filename: str):
    """Loads and caches serialized model pipeline from disk."""
    model_path = os.path.join(MODELS_DIR, model_filename)
    if not os.path.exists(model_path):
        return None
    try:
        model = joblib.load(model_path)
        return model
    except Exception as e:
        st.error(f"Error loading model '{model_filename}': {e}")
        return None

@st.cache_data
def load_evaluation_metrics() -> Optional[pd.DataFrame]:
    """Loads the model comparison CSV generated in Stage 6."""
    csv_path = os.path.join(RESULTS_DIR, "model_comparison.csv")
    if os.path.exists(csv_path):
        return pd.read_csv(csv_path)
    return None

@st.cache_data
def load_feature_columns() -> list:
    """Loads ordered feature columns list from JSON."""
    json_path = os.path.join(MODELS_DIR, "feature_columns.json")
    if os.path.exists(json_path):
        with open(json_path, "r", encoding="utf-8") as f:
            return json.load(f)
    return FEATURE_COLUMNS


# ------------------------------------------------------------------------------
# Sidebar Navigation
# ------------------------------------------------------------------------------
with st.sidebar:
    st.markdown("## 🛡️ **PhishGuard**")
    st.markdown("`v1.0 – B.Tech AI & Data Science`")
    st.markdown("---")
    
    # Keep sidebar selection synchronized with session state
    nav_selection = st.radio(
        "Navigation",
        PAGES,
        index=PAGES.index(st.session_state["current_page"]),
        key="sidebar_nav"
    )
    st.session_state["current_page"] = nav_selection
    
    st.markdown("---")
    st.markdown("### ⚙️ **Active Model**")
    selected_model_name = st.selectbox(
        "Select Classification Engine:",
        list(MODEL_CONFIG.keys()),
        index=0
    )
    
    cfg = MODEL_CONFIG[selected_model_name]
    st.caption(f"**{cfg['badge']}**: {cfg['description']}")
    
    st.markdown("---")
    st.markdown(
        "<div style='font-size:0.8rem; color:#64748b; text-align:center;'>"
        "Designed for Offline Safe URL Inspection.<br>No network packets sent."
        "</div>",
        unsafe_allow_html=True
    )


# ------------------------------------------------------------------------------
# SECTION 1: HOME DASHBOARD
# ------------------------------------------------------------------------------
if st.session_state["current_page"] == "🏠 Home Dashboard":
    st.markdown('<div class="brand-title">🛡️ PhishGuard Intelligent Detection</div>', unsafe_allow_html=True)
    st.markdown(
        '<div class="brand-subtitle">'
        'Machine Learning-driven cybersecurity system for real-time lexical inspection of fraudulent web addresses.'
        '</div>',
        unsafe_allow_html=True
    )
    
    col1, col2, col3 = st.columns(3)
    with col1:
        st.markdown(
            '<div class="cyber-card">'
            '<div class="metric-value">507,194</div>'
            '            <div class="metric-label">URL Records in Dataset</div>'
            '<p style="font-size:0.88rem; color:#94a3b8; margin-top:0.5rem;">'
            'Labeled records in the local cleaned URL dataset.'
            '</p>'
            '</div>',
            unsafe_allow_html=True
        )
    with col2:
        st.markdown(
            '<div class="cyber-card">'
            '<div class="metric-value">12 Features</div>'
            '<div class="metric-label">Lexical & Structural Checks</div>'
            '<p style="font-size:0.88rem; color:#94a3b8; margin-top:0.5rem;">'
            'Purely client-side string extraction with zero network exposure.'
            '</p>'
            '</div>',
            unsafe_allow_html=True
        )
    with col3:
        st.markdown(
            '<div class="cyber-card">'
            '<div class="metric-value">87.78%</div>'
            '<div class="metric-label">Champion Accuracy</div>'
            '<p style="font-size:0.88rem; color:#94a3b8; margin-top:0.5rem;">'
            'Achieved on 101,439 held-out test records using Decision Trees.'
            '</p>'
            '</div>',
            unsafe_allow_html=True
        )
        
    st.markdown("### 🎯 What is Phishing?")
    st.markdown(
        """
        Phishing is a social engineering cyberattack where attackers craft fraudulent websites disguised 
        as legitimate organizations (such as PayPal, Google, Microsoft, or banking portals) to harvest 
        confidential user credentials, financial information, or session tokens.
        """
    )
    
    st.markdown("### ⚙️ How PhishGuard Protects You")
    st.markdown(
        """
        Traditional blacklists fail against zero-day phishing domains generated on-the-fly. 
        **PhishGuard** analyzes the **morphological and structural DNA** of the raw URL string without ever 
        connecting to the dangerous remote server:
        
        1. **Lexical Inspection:** Measures URL length, character ratios, digit density, and symbol frequencies.
        2. **Security Indicators:** Detects raw IP hosting, brand-impersonation tokens (`verify`, `secure`, `login`), and subdomains.
        3. **Machine Learning Classification:** Passes 12 extracted numerical attributes into saved classifiers to produce a prediction and model risk score.
        """
    )
    
    st.markdown("---")
    st.markdown("### 🚀 Quick Navigation")
    nav_c1, nav_c2 = st.columns(2)
    with nav_c1:
        st.button(
            "🔍 Launch URL Scanner Now",
            type="primary",
            use_container_width=True,
            on_click=launch_scanner
        )
    with nav_c2:
        st.button(
            "📊 View Model Performance Dashboard",
            use_container_width=True,
            on_click=set_current_page,
            args=("📊 Model Performance",)
        )


# ------------------------------------------------------------------------------
# SECTION 2 & 3: URL SCANNER & PREDICTION RESULTS
# ------------------------------------------------------------------------------
elif st.session_state["current_page"] == "🔍 URL Scanner":
    st.markdown('<div class="brand-title">🔍 Real-Time URL Security Scanner</div>', unsafe_allow_html=True)
    st.markdown(
        '<div class="brand-subtitle">'
        'Inspect any web link for social-engineering and credential-harvesting indicators.'
        '</div>',
        unsafe_allow_html=True
    )
    
    # Preset quick-test samples
    st.markdown("**Quick Test Samples:**")
    preset_cols = st.columns(4)
    with preset_cols[0]:
        if st.button("🟢 Safe: Google Search", use_container_width=True):
            st.session_state["scanned_url"] = "https://www.google.com/"
            st.rerun()
    with preset_cols[1]:
        if st.button("🔴 Phish: IP Address Host", use_container_width=True):
            st.session_state["scanned_url"] = "http://192.168.1.100/admin/secure/login.php"
            st.rerun()
    with preset_cols[2]:
        if st.button("🔴 Phish: Deep Subdomain", use_container_width=True):
            st.session_state["scanned_url"] = "https://portal.secure.banking.verify.paypal.com.account-update.xyz/login"
            st.rerun()
    with preset_cols[3]:
        if st.button("🔴 Phish: Open Redirect", use_container_width=True):
            st.session_state["scanned_url"] = "http://legit-looking.com//fake-login/index.php?url=http://evil.com"
            st.rerun()
            
    # URL Input Field (bound to session_state)
    url_input = st.text_input(
        "Enter website URL to analyze:",
        value=st.session_state.get("scanned_url", ""),
        placeholder="e.g., https://paypal.com-account-verification.info/signin",
        help="Paste any complete or bare website link. PhishGuard will NOT connect to the URL."
    )
    
    scan_clicked = st.button("🚀 Scan Website URL", type="primary", use_container_width=True)
    
    # Prediction Workflow
    if scan_clicked:
        if not is_valid_url(url_input):
            st.warning("⚠️ Enter a valid HTTP(S) URL or bare hostname before scanning.")
        else:
            cleaned_input = url_input.strip()
            
            # Step 1: Feature Extraction
            with st.spinner("Extracting lexical indicators and security attributes..."):
                time.sleep(0.15)  # Subtle feedback feel
                features_df = extract_features_single(cleaned_input)
                
                # Verify column alignment with training metadata
                expected_cols = load_feature_columns()
                features_df = features_df[expected_cols]
                
            # Step 2: Model Loading & Inference
            cfg = MODEL_CONFIG[selected_model_name]
            model = load_trained_model(cfg["file"])
            
            if model is None:
                st.error(f"❌ Model artifact `{cfg['file']}` could not be located in `models/` directory.")
            else:
                t_start = time.time()
                prediction = model.predict(features_df)[0]
                t_elapsed = (time.time() - t_start) * 1000  # ms
                
                # Surface model output as a score; calibration is not validated.
                phishing_score = None
                if hasattr(model, "predict_proba"):
                    proba = model.predict_proba(features_df)[0]
                    phishing_score = proba[1] * 100
                    
                st.markdown("---")
                st.markdown("### 📋 Prediction Outcome")
                
                # Display Result Card
                if prediction == 1:
                    st.markdown(
                        f"""
                        <div class="cyber-card" style="border-left: 6px solid #ef4444; background: rgba(239, 68, 68, 0.08);">
                            <div style="display:flex; justify-content:space-between; align-items:center;">
                                <span class="badge-phish">⚠️ POTENTIALLY PHISHING / HIGH RISK</span>
                                <span style="font-size:0.85rem; color:#94a3b8;">Inference: {t_elapsed:.2f} ms</span>
                            </div>
                            <h3 style="color:#ef4444; margin-top:0.75rem; margin-bottom:0.25rem;">
                                Deceptive or High-Risk Characteristics Detected
                            </h3>
                            <p style="color:#cbd5e1; font-size:0.95rem;">
                                Active Model: <strong>{selected_model_name}</strong><br>
                                This URL exhibits structural patterns strongly associated with fraudulent websites.
                                <strong>Do not enter credentials, financial information, or personal data.</strong>
                            </p>
                        </div>
                        """,
                        unsafe_allow_html=True
                    )
                else:
                    st.markdown(
                        f"""
                        <div class="cyber-card" style="border-left: 6px solid #10b981; background: rgba(16, 185, 129, 0.08);">
                            <div style="display:flex; justify-content:space-between; align-items:center;">
                                <span class="badge-legit">🛡️ NO OBVIOUS PHISHING INDICATORS DETECTED</span>
                                <span style="font-size:0.85rem; color:#94a3b8;">Inference: {t_elapsed:.2f} ms</span>
                            </div>
                            <h3 style="color:#10b981; margin-top:0.75rem; margin-bottom:0.25rem;">
                                URL Structure Appears Standard
                            </h3>
                            <p style="color:#cbd5e1; font-size:0.95rem;">
                                Active Model: <strong>{selected_model_name}</strong><br>
                                The submitted URL does not exhibit anomalous length, raw IP hosting, or known spoofing markers.
                            </p>
                        </div>
                        """,
                        unsafe_allow_html=True
                    )
                    
                # Phishing Risk Meter
                if phishing_score is not None:
                    st.markdown(f"**Model Risk Score: `{phishing_score:.1f}/100`**")
                    st.caption("Model-derived score; it is not a verified, calibrated probability.")
                    st.progress(min(max(int(phishing_score), 0), 100))
                    
                # Mandatory Security Disclaimer
                st.markdown(
                    """
                    <div class="disclaimer-box">
                        <strong>⚠️ Security Notice & Limitation:</strong> PhishGuard inspects static URL syntax only. 
                        A status of <em>"No Obvious Phishing Indicators Detected"</em> does <strong>not</strong> guarantee a website is 100% safe, 
                        as compromised legitimate domains or zero-day homographs may bypass lexical filters. 
                        Always verify SSL certificates and domain authenticity.
                    </div>
                    """,
                    unsafe_allow_html=True
                )
                
                # Extracted URL Characteristics Breakdown Grid
                st.markdown("### 🔬 Extracted URL Characteristics")
                feat_dict = features_df.iloc[0].to_dict()
                
                c1, c2, c3, c4 = st.columns(4)
                with c1:
                    st.metric("Total Length", f"{int(feat_dict['url_length'])} chars")
                    st.metric("Hyphen Count", int(feat_dict['num_hyphens']))
                    st.metric("Digits Count", int(feat_dict['num_digits']))
                with c2:
                    st.metric("Domain Length", f"{int(feat_dict['domain_length'])} chars")
                    st.metric("Dot Count", int(feat_dict['num_dots']))
                    st.metric("Subdomains", int(feat_dict['num_subdomains']))
                with c3:
                    st.metric("Explicit HTTPS?", "Yes (1)" if feat_dict['is_https'] == 1 else "No (0)")
                    st.metric("Suspicious Keywords", int(feat_dict['num_suspicious_keywords']))
                    st.metric("Special Delimiters", int(feat_dict['num_special_chars']))
                with c4:
                    st.metric("Raw IP Address?", "YES ⚠️" if feat_dict['has_ip_address'] == 1 else "No")
                    st.metric("Contains '@' Symbol?", "YES ⚠️" if feat_dict['has_at_symbol'] == 1 else "No")
                    st.metric("Redirection Marker?", "YES ⚠️" if feat_dict['has_redirection'] == 1 else "No")


# ------------------------------------------------------------------------------
# SECTION 4: MODEL PERFORMANCE DASHBOARD
# ------------------------------------------------------------------------------
elif st.session_state["current_page"] == "📊 Model Performance":
    st.markdown('<div class="brand-title">📊 Model Benchmark & Performance Dashboard</div>', unsafe_allow_html=True)
    st.markdown(
        '<div class="brand-subtitle">'
        'Empirical evaluation results across 101,439 held-out test URLs (20% stratified test split).'
        '</div>',
        unsafe_allow_html=True
    )
    
    df_eval = load_evaluation_metrics()
    
    if df_eval is not None:
        # Top KPI Highlights
        col1, col2, col3, col4 = st.columns(4)
        with col1:
            st.metric("Evaluated Test Set", "101,439 URLs", "Stratified Split")
        with col2:
            st.metric("Champion Model", "Decision Tree", "Depth = 12")
        with col3:
            st.metric("Best Phishing Recall", "55.49%", "+16.1% vs Linear")
        with col4:
            st.metric("Lowest Missed Phish (FN)", "10,174", "3,717 fewer than SVM")
            
        st.markdown("### 📋 Empirical Metrics Summary Table")
        
        display_df = df_eval.copy()
        pct_cols = ["Accuracy", "Precision (Phishing)", "Recall (Phishing)", "F1-Score (Phishing)", "Specificity (Legitimate)", "ROC-AUC"]
        for c in pct_cols:
            if c in display_df.columns:
                display_df[c] = (display_df[c] * 100).round(2).astype(str) + "%"
                
        int_cols = ["True Negatives (TN)", "False Positives (FP)", "False Negatives (FN)", "True Positives (TP)"]
        for c in int_cols:
            if c in display_df.columns:
                display_df[c] = display_df[c].astype(int).map("{:,}".format)
                
        st.dataframe(display_df, use_container_width=True)
        
        st.markdown("---")
        st.markdown("### 📈 Visual Benchmark Charts")
        
        tab1, tab2, tab3 = st.tabs(["Accuracy Comparison", "Precision / Recall / F1", "Confusion Matrices"])
        
        with tab1:
            acc_chart = os.path.join(RESULTS_DIR, "accuracy_comparison.png")
            if os.path.exists(acc_chart):
                st.image(acc_chart, caption="Figure 1: Test Accuracy Comparison across Classifiers", use_container_width=True)
            else:
                st.info("Accuracy chart not found.")
                
        with tab2:
            metric_chart = os.path.join(RESULTS_DIR, "metrics_comparison.png")
            if os.path.exists(metric_chart):
                st.image(metric_chart, caption="Figure 2: Phishing Precision, Recall, and F1-Score Trade-off", use_container_width=True)
            else:
                st.info("Metrics comparison chart not found.")
                
        with tab3:
            cm_chart = os.path.join(RESULTS_DIR, "confusion_matrices_combined.png")
            if os.path.exists(cm_chart):
                st.image(cm_chart, caption="Figure 3: Side-by-Side Confusion Matrices (True Negatives, False Alarms, Missed Threats, Caught Attacks)", use_container_width=True)
            else:
                st.info("Combined confusion matrix chart not found.")
    else:
        st.warning("No evaluation metrics file (`results/model_comparison.csv`) found.")


# ------------------------------------------------------------------------------
# SECTION 5: ABOUT & ARCHITECTURE
# ------------------------------------------------------------------------------
elif st.session_state["current_page"] == "ℹ️ About & Architecture":
    st.markdown('<div class="brand-title">ℹ️ About PhishGuard</div>', unsafe_allow_html=True)
    st.markdown(
        '<div class="brand-subtitle">'
        'Intelligent Phishing Website Detection System – Architecture & Academic Context'
        '</div>',
        unsafe_allow_html=True
    )
    
    st.markdown(
        """
        ### 🎓 Project Background
        Developed as an academic capstone project for **B.Tech Artificial Intelligence and Data Science**.
        
        **Project Objective:**
        To engineer an offline, high-speed cyber-defense filter capable of detecting malicious phishing URLs 
        at line rate without relying on delayed third-party threat feeds or making network connections to live attack servers.
        
        ---
        
        ### 🛠️ Technology Stack
        * **Language:** Python 3.10+
        * **User Interface:** Streamlit
        * **Data Science & ML:** Scikit-learn, Pandas, NumPy
        * **Model Serialization:** Joblib
        * **Visualizations:** Seaborn, Matplotlib
        * **Security Parsing:** `urllib.parse`, `ipaddress`
        
        ---
        
        ### 📂 Dataset Information
        * **Provenance:** Local CSV files do not include source or license metadata; confirm provenance before redistribution.
        * **Local dataset:** **507,194 labeled URL records** from `data/cleaned_urls.csv`; exact URL duplicates were not present in the inspected file.
        * **Class Balance:** 77.46% Legitimate (`392,896`), 22.54% Phishing (`114,298`).
        * **HTTPS coverage:** only 7 records are marked HTTPS, and all 7 are labeled phishing; no legitimate HTTPS records are present.
        
        ---
        
        ### 🧠 Algorithms Implemented
        1. **Logistic Regression:** Probabilistic linear baseline with L-BFGS solver and standard scaling.
        2. **Support Vector Machine (LinearSVC):** Max-margin classifier with sigmoid calibration; its live risk scores have not been verified as calibrated probabilities.
        3. **Decision Tree Classifier (Champion):** Non-linear hierarchical tree with `max_depth=12`. 
           Captures threshold interactions (e.g., combining IP presence, symbol counts, and path length) 
           yielding superior recall and minimum false negatives.
        
        ---
        
        ### ⚠️ System Limitations
        * **Zero-Day Homograph Attacks:** Unicode internationalized domain names (IDN) spoofing Latin characters require Punycode translation.
        * **Compromised Legitimate Hosts:** If a legitimate server (e.g. AWS S3, Google Drive) hosts a phishing form, lexical domain checks alone may not distinguish the malicious form without content inspection.
        """
    )
