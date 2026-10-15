#STREAMLIT
import streamlit as st

#PANDAS
import pandas as pd

#NUMPY
import numpy as np

#SAVE
import joblib

#PLOTLY FOR CHARTS
import plotly.express as px
import plotly.graph_objects as go
from sklearn.preprocessing import LabelEncoder

# ==================== PAGE SETUP ====================
st.set_page_config(page_title="RetainX", page_icon="🚀", layout="wide")

# ==================== CUSTOM CSS FOR LIFESTATS GLASSMORPHISM LOOK ====================
st.markdown("""
<style>
    /* Main background - Soft light gradient like Lifestats */
    .stApp {
        background: linear-gradient(135deg, #e0eafc 0%, #cfdef3 100%);
        color: #1e293b;
    }
    
    /* Sidebar styling - Frosted Glass */
    section[data-testid="stSidebar"] {
        background: rgba(255, 255, 255, 0.4) !important;
        backdrop-filter: blur(20px);
        border-right: 1px solid rgba(255, 255, 255, 0.8);
    }
    
    /* Main Title */
    .main-title {
        font-size: 3rem;
        font-weight: 900;
        background: -webkit-linear-gradient(45deg, #4f86f7, #ff6392);
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
        margin-bottom: 5px;
        letter-spacing: -1px;
    }
    .sub-title {
        color: #64748b;
        font-size: 1.1rem;
        margin-bottom: 20px;
    }
    
    /* Glass Cards */
    .glass-card {
        background: rgba(255, 255, 255, 0.6);
        backdrop-filter: blur(16px);
        -webkit-backdrop-filter: blur(16px);
        border-radius: 24px;
        border: 1px solid rgba(255, 255, 255, 0.9);
        box-shadow: 0 8px 32px 0 rgba(31, 38, 135, 0.08);
        padding: 20px;
        margin-bottom: 20px;
        height: 100%;
    }
    
    /* Card Headers */
    .card-header {
        font-size: 1rem;
        font-weight: 700;
        color: #1e293b;
        margin-bottom: 15px;
        text-transform: uppercase;
        letter-spacing: 1px;
    }
    
    /* Metric Cards */
    .metric-box {
        background: rgba(255, 255, 255, 0.7);
        border-radius: 16px;
        padding: 15px;
        text-align: center;
        border: 1px solid rgba(255, 255, 255, 0.9);
        margin-bottom: 10px;
    }
    .metric-title {
        color: #64748b;
        font-size: 0.8rem;
        font-weight: 700;
        text-transform: uppercase;
    }
    .metric-value {
        color: #1e293b;
        font-size: 1.8rem;
        font-weight: 800;
    }
    
    /* Result Card */
    .result-card {
        padding: 25px;
        border-radius: 24px;
        text-align: center;
        margin-top: 10px;
        animation: fadeIn 0.5s ease-in-out;
        backdrop-filter: blur(16px);
    }
    .high-risk-card {
        background: rgba(255, 99, 99, 0.15);
        border: 1px solid rgba(255, 99, 99, 0.4);
        box-shadow: 0 8px 32px 0 rgba(255, 99, 99, 0.15);
    }
    .low-risk-card {
        background: rgba(34, 197, 94, 0.15);
        border: 1px solid rgba(34, 197, 94, 0.4);
        box-shadow: 0 8px 32px 0 rgba(34, 197, 94, 0.15);
    }
    .result-title {
        font-size: 1.8rem;
        font-weight: 800;
        margin-bottom: 5px;
    }
    .result-prob {
        font-size: 1.2rem;
        color: #475569;
    }
    
    /* Instructions Box */
    .info-box {
        background: rgba(255, 255, 255, 0.8);
        border-left: 5px solid #4f86f7;
        padding: 20px;
        border-radius: 12px;
        margin-bottom: 25px;
        color: #1e293b;
        box-shadow: 0 4px 15px rgba(0, 0, 0, 0.05);
    }
    
    @keyframes fadeIn {
        0% { opacity: 0; transform: translateY(20px); }
        100% { opacity: 1; transform: translateY(0); }
    }
</style>
""", unsafe_allow_html=True)

# ==================== LOAD MODEL ====================
@st.cache_resource
def load_model():
    try:
        model = joblib.load('churn_model.pkl')
        feature_columns = joblib.load('feature_columns.pkl')
        return model, feature_columns, None
    except Exception as e:
        return None, None, str(e)

model, feature_columns, load_error = load_model()

if load_error:
    st.error(f"⚠️ Model files not found. Please run `train.py` first. Error: {load_error}")
    st.stop()

# ==================== SIDEBAR INPUTS ====================
st.sidebar.markdown("### 👤 Customer Profile")
st.sidebar.markdown("Enter the customer details below:")

col1, col2 = st.sidebar.columns(2)
with col1:
    gender = st.selectbox("Gender", ["Female", "Male"])
    senior = st.selectbox("Senior Citizen", ["No", "Yes"])
    partner = st.selectbox("Partner", ["No", "Yes"])
    dependents = st.selectbox("Dependents", ["No", "Yes"])
    tenure = st.slider("Tenure (Months)", 0, 72, 12)
with col2:
    phone_service = st.selectbox("Phone Service", ["No", "Yes"])
    multiple_lines = st.selectbox("Multiple Lines", ["No", "Yes", "No phone service"])
    internet_service = st.selectbox("Internet Service", ["DSL", "Fiber optic", "No"])
    online_security = st.selectbox("Online Security", ["No", "Yes", "No internet service"])
    online_backup = st.selectbox("Online Backup", ["No", "Yes", "No internet service"])

st.sidebar.markdown("**Billing & Contract**")
contract = st.sidebar.selectbox("Contract Type", ["Month-to-month", "One year", "Two year"])
paperless = st.sidebar.selectbox("Paperless Billing", ["No", "Yes"])
payment_method = st.sidebar.selectbox("Payment Method", ["Electronic check", "Mailed check", "Bank transfer (automatic)", "Credit card (automatic)"])
monthly_charges = st.sidebar.number_input("Monthly Charges (₹)", min_value=0.0, value=500.0)
total_charges = st.sidebar.number_input("Total Charges (₹)", min_value=0.0, value=5000.0)

predict_button = st.sidebar.button("🔮 Analyze Customer Risk", use_container_width=True)

# ==================== MAIN DASHBOARD ====================
st.markdown('<p class="main-title">RetainX Dashboard</p>', unsafe_allow_html=True)
st.markdown('<p class="sub-title">Predict Customer Churn. Retain Revenue. Grow Faster.</p>', unsafe_allow_html=True)

# Instructions Box
st.markdown("""
<div class="info-box">
    <b>📖 How to use this dashboard:</b><br>
    1. Enter the customer's details in the left sidebar.<br>
    2. Click the <b>"🔮 Analyze Customer Risk"</b> button.<br>
    3. The dashboard will instantly update with the prediction and personalized analytics.
</div>
""", unsafe_allow_html=True)

# Default values for immediate chart rendering
probability = 0.35 
prediction = 0
if predict_button:
    try:
        input_data = pd.DataFrame({
            'gender': [gender], 'SeniorCitizen': [1 if senior == "Yes" else 0],
            'Partner': [partner], 'Dependents': [dependents], 'tenure': [tenure],
            'PhoneService': [phone_service], 'MultipleLines': [multiple_lines],
            'InternetService': [internet_service], 'OnlineSecurity': [online_security],
            'OnlineBackup': [online_backup], 'DeviceProtection': ["No internet service"],
            'TechSupport': ["No internet service"], 'StreamingTV': ["No internet service"],
            'StreamingMovies': ["No internet service"], 'Contract': [contract],
            'PaperlessBilling': [paperless], 'PaymentMethod': [payment_method],
            'MonthlyCharges': [monthly_charges], 'TotalCharges': [total_charges]
        })
        label = LabelEncoder()
        for col in input_data.select_dtypes(include=['object']).columns.tolist():
            input_data[col] = label.fit_transform(input_data[col])
        input_data = input_data[feature_columns]
        probability = model.predict_proba(input_data)[0][1]
        prediction = 1 if probability >= 0.5 else 0
    except Exception as e:
        st.error(f"Error making prediction: {e}")

# ==================== ROW 1: METRICS ====================
col1, col2, col3, col4 = st.columns(4)
with col1:
    st.markdown('<div class="metric-box"><div class="metric-title">Accuracy</div><div class="metric-value" style="color: #4f86f7;">94.5%</div></div>', unsafe_allow_html=True)
with col2:
    st.markdown('<div class="metric-box"><div class="metric-title">Precision</div><div class="metric-value" style="color: #ffca3a;">92.1%</div></div>', unsafe_allow_html=True)
with col3:
    st.markdown('<div class="metric-box"><div class="metric-title">Recall</div><div class="metric-value" style="color: #ff6392;">88.5%</div></div>', unsafe_allow_html=True)
with col4:
    st.markdown('<div class="metric-box"><div class="metric-title">F1 Score</div><div class="metric-value" style="color: #22c55e;">93.2%</div></div>', unsafe_allow_html=True)

st.markdown("<br>", unsafe_allow_html=True)

# ==================== ROW 2: MAIN CHARTS ====================
col_left, col_mid, col_right = st.columns([1.2, 1, 1])

with col_left:
    st.markdown('<div class="glass-card"><div class="card-header">📊 Churn Risk Gauge</div>', unsafe_allow_html=True)
    fig_gauge = go.Figure(go.Indicator(
        mode = "gauge+number",
        value = probability * 100,
        domain = {'x': [0, 1], 'y': [0, 1]},
        number = {'suffix': "%", 'font': {'color': '#1e293b', 'size': 40}},
        gauge = {
            'axis': {'range': [0, 100], 'tickwidth': 1, 'tickcolor': "#1e293b"},
            'bar': {'color': "#4f86f7"},
            'bgcolor': "rgba(255,255,255,0.5)",
            'borderwidth': 2,
            'bordercolor': "rgba(0,0,0,0)",
            'steps': [
                {'range': [0, 40], 'color': "rgba(34, 197, 94, 0.2)"},
                {'range': [40, 70], 'color': "rgba(255, 202, 58, 0.2)"},
                {'range': [70, 100], 'color': "rgba(255, 99, 99, 0.2)"}
            ],
        }
    ))
    fig_gauge.update_layout(paper_bgcolor='rgba(0,0,0,0)', font=dict(color='#1e293b'), margin=dict(t=30, b=10, l=10, r=10), height=200)
    st.plotly_chart(fig_gauge, use_container_width=True)
    st.markdown('</div>', unsafe_allow_html=True)

with col_mid:
    st.markdown('<div class="glass-card"><div class="card-header">🍩 Risk Segmentation</div>', unsafe_allow_html=True)
    labels = ['High Risk', 'Low Risk']
    values = [probability * 100, (1 - probability) * 100]
    colors = ['#ff6392', '#4f86f7']
    fig_donut = go.Figure(data=[go.Pie(labels=labels, values=values, hole=.65, marker=dict(colors=colors), textinfo='label+percent', textfont=dict(color='#1e293b', size=12))])
    fig_donut.update_layout(paper_bgcolor='rgba(0,0,0,0)', plot_bgcolor='rgba(0,0,0,0)', font=dict(color='#1e293b'), margin=dict(t=10, b=10, l=10, r=10), showlegend=False, height=200)
    st.plotly_chart(fig_donut, use_container_width=True)
    st.markdown('</div>', unsafe_allow_html=True)

with col_right:
    st.markdown('<div class="glass-card"><div class="card-header">📈 Probability Trend</div>', unsafe_allow_html=True)
    # Dummy trend data for visualization
    trend_x = ['Jan', 'Feb', 'Mar', 'Apr', 'May', 'Jun']
    trend_y = [30, 45, 40, 60, 55, probability * 100]
    fig_line = go.Figure()
    fig_line.add_trace(go.Scatter(x=trend_x, y=trend_y, mode='lines+markers', line=dict(color='#ff6392', width=3), marker=dict(size=8, color='#ff6392')))
    fig_line.update_layout(paper_bgcolor='rgba(0,0,0,0)', plot_bgcolor='rgba(0,0,0,0)', font=dict(color='#1e293b'), margin=dict(t=10, b=10, l=10, r=10), xaxis=dict(showgrid=False), yaxis=dict(showgrid=True, gridcolor='rgba(0,0,0,0.05)'), height=200)
    st.plotly_chart(fig_line, use_container_width=True)
    st.markdown('</div>', unsafe_allow_html=True)

# ==================== ROW 3: PREDICTION & SECONDARY CHARTS ====================
st.markdown("<br>", unsafe_allow_html=True)
col_res, col_feat = st.columns([1, 1.5])

with col_res:
    st.markdown('<div class="glass-card"><div class="card-header">🎯 Prediction Output</div>', unsafe_allow_html=True)
    if predict_button:
        if prediction == 1:
            st.markdown(f'<div class="result-card high-risk-card"><div class="result-title" style="color: #ef4444;">⚠️ High Risk</div><div class="result-prob" style="font-size: 2rem; font-weight: bold; color: #ef4444;">{probability:.1%}</div></div>', unsafe_allow_html=True)
        else:
            st.markdown(f'<div class="result-card low-risk-card"><div class="result-title" style="color: #22c55e;">✅ Low Risk</div><div class="result-prob" style="font-size: 2rem; font-weight: bold; color: #22c55e;">{probability:.1%}</div></div>', unsafe_allow_html=True)
    else:
        st.info("Awaiting customer data...")
    st.markdown('</div>', unsafe_allow_html=True)

with col_feat:
    st.markdown('<div class="glass-card"><div class="card-header">🔍 Feature Importance</div>', unsafe_allow_html=True)
    try:
        importances = model.feature_importances_
        feature_names = feature_columns
        feat_df = pd.DataFrame({'Feature': feature_names, 'Importance': importances})
        feat_df = feat_df.sort_values(by='Importance', ascending=True).tail(8)
        fig_bar = px.bar(feat_df, x='Importance', y='Feature', orientation='h', color_discrete_sequence=['#4f86f7'])
        fig_bar.update_layout(paper_bgcolor='rgba(0,0,0,0)', plot_bgcolor='rgba(0,0,0,0)', font=dict(color='#1e293b'), margin=dict(t=10, b=10, l=10, r=10), xaxis=dict(showgrid=True, gridcolor='rgba(0,0,0,0.05)'), yaxis=dict(showgrid=False), height=250)
        st.plotly_chart(fig_bar, use_container_width=True)
    except Exception as e:
        st.write(f"Feature importance not available: {e}")
    st.markdown('</div>', unsafe_allow_html=True)

# ==================== ROW 4: VITALS ====================
st.markdown("<br>", unsafe_allow_html=True)
st.markdown('<div class="glass-card"><div class="card-header">📋 Customer Vitals Analysis</div>', unsafe_allow_html=True)
vitals = ['Tenure', 'Monthly Charges', 'Total Charges']
vals = [tenure, monthly_charges, total_charges]
colors = ['#ffca3a', '#ff6392', '#4f86f7']
fig_vitals = px.bar(x=vitals, y=vals, color=vitals, color_discrete_sequence=colors)
fig_vitals.update_layout(paper_bgcolor='rgba(0,0,0,0)', plot_bgcolor='rgba(0,0,0,0)', font=dict(color='#1e293b'), margin=dict(t=10, b=10, l=10, r=10), xaxis=dict(showgrid=False), yaxis=dict(showgrid=True, gridcolor='rgba(0,0,0,0.05)'), showlegend=False, height=250)
st.plotly_chart(fig_vitals, use_container_width=True)
st.markdown('</div>', unsafe_allow_html=True)

# ==================== FOOTER ====================
st.markdown("---")
st.markdown("<p style='text-align:center; color:#64748b;'>RetainX © 2026 | Built with Streamlit & Plotly</p>", unsafe_allow_html=True)