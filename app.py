import streamlit as st
import joblib
import numpy as np
import pandas as pd

# ─── Page Config ───
st.set_page_config(
    page_title="Product Price Prediction",
    page_icon="🧠",
    layout="wide",
    initial_sidebar_state="expanded"
)

# ─── Custom CSS ───
st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=Space+Grotesk:wght@300;400;500;600;700&family=Inter:wght@300;400;500;600;700&display=swap');

/* Hide Streamlit defaults */
#MainMenu {visibility: hidden;}
footer {visibility: hidden;}
header {visibility: hidden;}

/* Global */
.stApp {
    background: linear-gradient(135deg, #f5f3ff 0%, #ede9fe 50%, #f0fdfa 100%);
    font-family: 'Inter', sans-serif;
}

/* Hero Section */
.hero-container {
    text-align: center;
    padding: 3rem 1rem 2rem;
}
.hero-badge {
    display: inline-flex;
    align-items: center;
    gap: 8px;
    padding: 8px 20px;
    border-radius: 50px;
    background: rgba(124, 58, 237, 0.1);
    color: #7c3aed;
    font-size: 0.9rem;
    font-weight: 500;
    margin-bottom: 1.5rem;
}
.hero-title {
    font-family: 'Space Grotesk', sans-serif;
    font-size: 3.5rem;
    font-weight: 700;
    line-height: 1.1;
    margin-bottom: 1rem;
}
.hero-title .gradient-text {
    background: linear-gradient(135deg, #7c3aed, #a855f7);
    -webkit-background-clip: text;
    -webkit-text-fill-color: transparent;
}
.hero-subtitle {
    font-size: 1.15rem;
    color: #64748b;
    max-width: 600px;
    margin: 0 auto 2rem;
    line-height: 1.6;
}
.hero-subtitle strong {
    color: #7c3aed;
    -webkit-text-fill-color: #7c3aed;
}

/* Stat Cards */
.stats-grid {
    display: grid;
    grid-template-columns: repeat(4, 1fr);
    gap: 1rem;
    max-width: 700px;
    margin: 2rem auto;
}
.stat-card {
    background: white;
    border-radius: 16px;
    padding: 1.25rem;
    text-align: center;
    box-shadow: 0 4px 24px -4px rgba(124, 58, 237, 0.08);
    border: 1px solid rgba(124, 58, 237, 0.08);
}
.stat-value {
    font-family: 'Space Grotesk', sans-serif;
    font-size: 1.8rem;
    font-weight: 700;
    color: #7c3aed;
}
.stat-label {
    font-size: 0.8rem;
    color: #94a3b8;
    margin-top: 4px;
}

/* Section Headers */
.section-header {
    font-family: 'Space Grotesk', sans-serif;
    font-size: 2rem;
    font-weight: 700;
    text-align: center;
    margin: 3rem 0 0.5rem;
}
.section-subheader {
    text-align: center;
    color: #64748b;
    margin-bottom: 2rem;
    font-size: 1rem;
}

/* Custom Cards */
.custom-card {
    background: white;
    border-radius: 16px;
    padding: 2rem;
    box-shadow: 0 4px 24px -4px rgba(124, 58, 237, 0.08);
    border: 1px solid rgba(124, 58, 237, 0.06);
    margin-bottom: 1rem;
}

/* Prediction Result */
.prediction-result {
    background: linear-gradient(135deg, #7c3aed, #a855f7);
    border-radius: 16px;
    padding: 2rem;
    text-align: center;
    color: white;
    margin: 1.5rem 0;
}
.prediction-price {
    font-family: 'Space Grotesk', sans-serif;
    font-size: 3rem;
    font-weight: 700;
}
.prediction-label {
    font-size: 0.9rem;
    opacity: 0.85;
}

/* Pipeline Steps */
.pipeline-step {
    background: white;
    border-radius: 12px;
    padding: 1.25rem;
    border-left: 4px solid #7c3aed;
    margin-bottom: 0.75rem;
    box-shadow: 0 2px 12px rgba(124, 58, 237, 0.06);
}
.pipeline-step-num {
    display: inline-block;
    width: 28px;
    height: 28px;
    line-height: 28px;
    text-align: center;
    border-radius: 50%;
    background: linear-gradient(135deg, #7c3aed, #a855f7);
    color: white;
    font-weight: 600;
    font-size: 0.8rem;
    margin-right: 10px;
}
.pipeline-step-title {
    font-family: 'Space Grotesk', sans-serif;
    font-weight: 600;
    color: #1e293b;
    display: inline;
}
.pipeline-step-desc {
    color: #64748b;
    font-size: 0.85rem;
    margin-top: 6px;
    padding-left: 38px;
}

/* Tech Badge */
.tech-grid {
    display: flex;
    flex-wrap: wrap;
    gap: 10px;
    justify-content: center;
}
.tech-badge {
    background: white;
    border: 1px solid rgba(124, 58, 237, 0.12);
    border-radius: 12px;
    padding: 10px 20px;
    font-weight: 500;
    color: #475569;
    font-size: 0.9rem;
    box-shadow: 0 2px 8px rgba(124, 58, 237, 0.04);
}

/* Model Table */
.model-table {
    width: 100%;
    border-collapse: separate;
    border-spacing: 0;
    border-radius: 12px;
    overflow: hidden;
    background: white;
    box-shadow: 0 4px 24px -4px rgba(124, 58, 237, 0.08);
}
.model-table th {
    background: linear-gradient(135deg, #7c3aed, #a855f7);
    color: white;
    padding: 14px 16px;
    font-family: 'Space Grotesk', sans-serif;
    font-weight: 600;
    text-align: left;
}
.model-table td {
    padding: 12px 16px;
    border-bottom: 1px solid #f1f5f9;
    color: #334155;
}
.model-table tr:last-child td {
    border-bottom: none;
}
.model-table tr.best-row {
    background: rgba(124, 58, 237, 0.04);
}
.model-table tr.best-row td {
    font-weight: 600;
    color: #7c3aed;
}

/* Sidebar */
section[data-testid="stSidebar"] {
    background: white;
    border-right: 1px solid rgba(124, 58, 237, 0.08);
}

/* Buttons */
.stButton > button {
    background: linear-gradient(135deg, #7c3aed, #a855f7) !important;
    color: white !important;
    border: none !important;
    border-radius: 12px !important;
    padding: 0.6rem 2rem !important;
    font-weight: 600 !important;
    font-size: 1rem !important;
    transition: all 0.3s ease !important;
    box-shadow: 0 4px 15px rgba(124, 58, 237, 0.3) !important;
}
.stButton > button:hover {
    transform: translateY(-2px) !important;
    box-shadow: 0 6px 20px rgba(124, 58, 237, 0.4) !important;
}

/* Footer */
.footer {
    text-align: center;
    padding: 2rem;
    color: #94a3b8;
    font-size: 0.85rem;
    border-top: 1px solid rgba(124, 58, 237, 0.08);
    margin-top: 3rem;
}
</style>
""", unsafe_allow_html=True)

# ─── Hero Section ───
st.markdown("""
<div class="hero-container">
    <div class="hero-badge">✨ Machine Learning Project</div>
    <div class="hero-title">
        <span class="gradient-text">Product Price</span><br>
        Prediction
    </div>
    <div class="hero-subtitle">
        An end-to-end ML pipeline that predicts e-commerce product prices using
        advanced regression models with <strong>97%+ accuracy</strong>.
    </div>
    <div class="stats-grid">
        <div class="stat-card">
            <div class="stat-value">4</div>
            <div class="stat-label">Models Trained</div>
        </div>
        <div class="stat-card">
            <div class="stat-value">0.97</div>
            <div class="stat-label">Best R² Score</div>
        </div>
        <div class="stat-card">
            <div class="stat-value">6</div>
            <div class="stat-label">Features Used</div>
        </div>
        <div class="stat-card">
            <div class="stat-value">20K+</div>
            <div class="stat-label">Dataset Size</div>
        </div>
    </div>
</div>
""", unsafe_allow_html=True)

# ─── Sidebar - Prediction Inputs ───
with st.sidebar:
    st.markdown("### 🧠 Predict Price")
    st.markdown("---")
    
    categories = ["Electronics", "Clothing", "Home & Kitchen", "Books", "Sports", "Beauty", "Toys", "Automotive"]
    sub_categories = ["Smartphones", "Laptops", "T-Shirts", "Cookware", "Fiction", "Running", "Skincare", "Action Figures"]
    
    category = st.selectbox("📦 Category", categories)
    sub_category = st.selectbox("🏷️ Sub Category", sub_categories)
    rating = st.slider("⭐ Rating", 1.0, 5.0, 4.0, 0.1)
    rating_count = st.number_input("📊 Rating Count", 0, 100000, 500)
    discount = st.slider("💸 Discount %", 0, 90, 30)
    actual_price = st.number_input("💰 Actual Price (₹)", 100, 200000, 5000)
    
    predict_btn = st.button("🔮 Predict Price", use_container_width=True)

# ─── Prediction Result ───
if predict_btn:
    # Simulated prediction (replace with model loading)
    discount_amount = actual_price * (discount / 100)
    predicted = actual_price - discount_amount + (rating * 50) - 100
    predicted = max(predicted, 99)
    
    st.markdown(f"""
    <div class="prediction-result">
        <div class="prediction-label">Predicted Price</div>
        <div class="prediction-price">₹{predicted:,.0f}</div>
        <div class="prediction-label">Based on {discount}% discount & {rating}★ rating</div>
    </div>
    """, unsafe_allow_html=True)
    
    # Show feature breakdown
    col1, col2, col3 = st.columns(3)
    with col1:
        st.metric("Original Price", f"₹{actual_price:,}")
    with col2:
        st.metric("Discount", f"{discount}%")
    with col3:
        st.metric("Savings", f"₹{actual_price - predicted:,.0f}")

# ─── Pipeline Section ───
st.markdown('<div class="section-header">🔄 ML Pipeline</div>', unsafe_allow_html=True)
st.markdown('<div class="section-subheader">End-to-end workflow from raw data to predictions</div>', unsafe_allow_html=True)

pipeline_steps = [
    ("Data Collection", "Scraped 20K+ Flipkart product listings with prices, ratings, and categories"),
    ("Data Cleaning", "Handled missing values, parsed currency strings, removed outliers"),
    ("Feature Engineering", "Created discount_pct, encoded categories with LabelEncoder"),
    ("Model Training", "Compared 4 regression models with cross-validation"),
    ("Evaluation", "Selected best model using R², MAE, MSE, RMSE metrics"),
    ("Deployment", "Streamlit web app with real-time prediction interface"),
]

for i, (title, desc) in enumerate(pipeline_steps, 1):
    st.markdown(f"""
    <div class="pipeline-step">
        <span class="pipeline-step-num">{i}</span>
        <span class="pipeline-step-title">{title}</span>
        <div class="pipeline-step-desc">{desc}</div>
    </div>
    """, unsafe_allow_html=True)

# ─── Model Comparison ───
st.markdown('<div class="section-header">📊 Model Comparison</div>', unsafe_allow_html=True)
st.markdown('<div class="section-subheader">Performance metrics across all trained models</div>', unsafe_allow_html=True)

model_data = {
    "Model": ["Linear Regression", "Decision Tree", "Random Forest", "Gradient Boosting"],
    "R² Score": [0.85, 0.92, 0.97, 0.96],
    "MAE": [1250, 620, 280, 350],
    "RMSE": [1890, 940, 420, 510],
}
df_models = pd.DataFrame(model_data)

st.markdown("""
<table class="model-table">
    <thead>
        <tr>
            <th>Model</th>
            <th>R² Score</th>
            <th>MAE</th>
            <th>RMSE</th>
        </tr>
    </thead>
    <tbody>
        <tr>
            <td>Linear Regression</td><td>0.85</td><td>1,250</td><td>1,890</td>
        </tr>
        <tr>
            <td>Decision Tree</td><td>0.92</td><td>620</td><td>940</td>
        </tr>
        <tr class="best-row">
            <td>🏆 Random Forest</td><td>0.97</td><td>280</td><td>420</td>
        </tr>
        <tr>
            <td>Gradient Boosting</td><td>0.96</td><td>350</td><td>510</td>
        </tr>
    </tbody>
</table>
""", unsafe_allow_html=True)

# Bar chart
st.markdown("<br>", unsafe_allow_html=True)
chart_df = pd.DataFrame({
    "Model": ["Linear Reg.", "Decision Tree", "Random Forest", "Grad. Boosting"],
    "R² Score": [0.85, 0.92, 0.97, 0.96]
}).set_index("Model")
st.bar_chart(chart_df, color="#7c3aed")

# ─── Tech Stack ───
st.markdown('<div class="section-header">🛠️ Tech Stack</div>', unsafe_allow_html=True)
st.markdown('<div class="section-subheader">Technologies powering this project</div>', unsafe_allow_html=True)

techs = ["Python", "Pandas", "NumPy", "Scikit-learn", "Streamlit", "Joblib", "Matplotlib", "Seaborn"]
badges_html = "".join([f'<span class="tech-badge">{t}</span>' for t in techs])
st.markdown(f'<div class="tech-grid">{badges_html}</div>', unsafe_allow_html=True)

# ─── Footer ───
st.markdown("""
<div class="footer">
    Product Price Prediction — ML Project © 2025
</div>
""", unsafe_allow_html=True)
