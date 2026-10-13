#STREAMLIT
import streamlit as st

#PANDAS
import pandas as pd

#NUMPY
import numpy as np

#SAVE
import joblib
from sklearn.preprocessing import LabelEncoder

# ==================== PAGE SETUP ====================
st.set_page_config(page_title="RetainX", page_icon="🚀", layout="wide")

# ==================== CUSTOM CSS FOR PROFESSIONAL LOOK ====================
st.markdown("""
<style>
    /* Main background */
    .stApp {
        background: linear-gradient(135deg, #0f172a 0%, #1e293b 100%);
        color: #f8fafc;
    }
    
    /* Hide default streamlit header */
    header {visibility: hidden;}
    
    /* Title styling */
    .main-title {
        font-size: 3.5rem;
        font-weight: 900;
        background: -webkit-linear-gradient(45deg, #38bdf8, #818cf8, #c084fc);
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
        text-align: center;
        margin-bottom: 0px;
        letter-spacing: 2px;
    }
    
    .sub-title {
        text-align: center;
        color: #94a3b8;
        font-size: 1.1rem;
        margin-bottom: 30px;
        letter-spacing: 1px;
    }
    
    /* Input Card styling */
    div[data-testid="stVerticalBlock"] > div {
        background-color: rgba(30, 41, 59, 0.7);
        border-radius: 12px;
        padding: 10px;
        border: 1px solid rgba(255, 255, 255, 0.1);
    }
    
    /* Input labels */
    label {
        color: #cbd5e1 !important;
        font-weight: 500 !important;
    }
    
    /* Button styling */
    .stButton>button {
        width: 100%;
        background: linear-gradient(90deg, #3b82f6 0%, #8b5cf6 100%);
        color: white;
        font-size: 1.2rem;
        font-weight: bold;
        padding: 15px;
        border-radius: 10px;
        border: none;
        transition: transform 0.2s, box-shadow 0.2s;
        margin-top: 20px;
    }
    
    .stButton>button:hover {
        transform: scale(1.02);
        box-shadow: 0 0 20px rgba(139, 92, 246, 0.6);
        color: white;
    }
    
    /* Result Cards */
    .result-card {
        padding: 30px;
        border-radius: 15px;
        text-align: center;
        margin-top: 20px;
        animation: fadeIn 0.5s ease-in-out;
    }
    
    .churn-card {
        background: linear-gradient(135deg, rgba(239, 68, 68, 0.2), rgba(153, 27, 27, 0.4));
        border: 1px solid #ef4444;
        box-shadow: 0 0 30px rgba(239, 68, 68, 0.2);
    }
    
    .stay-card {
        background: linear-gradient(135deg, rgba(34, 197, 94, 0.2), rgba(20, 83, 45, 0.4));
        border: 1px solid #22c55e;
        box-shadow: 0 0 30px rgba(34, 197, 94, 0.2);
    }
    
    .result-title {
        font-size: 2rem;
        font-weight: bold;
        margin-bottom: 10px;
    }
    
    .result-prob {
        font-size: 1.5rem;
        color: #e2e8f0;
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
    model = joblib.load('churn_model.pkl')
    feature_columns = joblib.load('feature_columns.pkl')
    return model, feature_columns

model, feature_columns = load_model()

# ==================== HEADER ====================
st.markdown('<p class="main-title">🚀 RetainX</p>', unsafe_allow_html=True)
st.markdown('<p class="sub-title">Predict Customer Churn. Retain Revenue. Grow Faster.</p>', unsafe_allow_html=True)

# ==================== USER INPUTS ====================
st.markdown("### 👤 Customer Profile")

col1, col2, col3 = st.columns(3)

with col1:
    st.markdown("**Demographics**")
    gender = st.selectbox("Gender", ["Female", "Male"])
    senior = st.selectbox("Senior Citizen", ["No", "Yes"])
    partner = st.selectbox("Partner", ["No", "Yes"])
    dependents = st.selectbox("Dependents", ["No", "Yes"])
    tenure = st.slider("Tenure (Months)", 0, 72, 12)

with col2:
    st.markdown("**Services**")
    phone_service = st.selectbox("Phone Service", ["No", "Yes"])
    multiple_lines = st.selectbox("Multiple Lines", ["No", "Yes", "No phone service"])
    internet_service = st.selectbox("Internet Service", ["DSL", "Fiber optic", "No"])
    online_security = st.selectbox("Online Security", ["No", "Yes", "No internet service"])
    online_backup = st.selectbox("Online Backup", ["No", "Yes", "No internet service"])

with col3:
    st.markdown("**Billing & Contract**")
    contract = st.selectbox("Contract Type", ["Month-to-month", "One year", "Two year"])
    paperless = st.selectbox("Paperless Billing", ["No", "Yes"])
    payment_method = st.selectbox("Payment Method", ["Electronic check", "Mailed check", "Bank transfer (automatic)", "Credit card (automatic)"])
    # CHANGED TO INDIAN RUPEES
    monthly_charges = st.number_input("Monthly Charges (₹)", min_value=0.0, value=500.0)
    total_charges = st.number_input("Total Charges (₹)", min_value=0.0, value=5000.0)

# ==================== PREDICTION ====================
if st.button("🔮 Analyze Customer Risk"):

    # CONVERT SENIOR TO NUMBER
    senior_num = 1 if senior == "Yes" else 0

    # CREATE DATAFRAME FROM INPUTS
    input_data = pd.DataFrame({
        'gender': [gender],
        'SeniorCitizen': [senior_num],
        'Partner': [partner],
        'Dependents': [dependents],
        'tenure': [tenure],
        'PhoneService': [phone_service],
        'MultipleLines': [multiple_lines],
        'InternetService': [internet_service],
        'OnlineSecurity': [online_security],
        'OnlineBackup': [online_backup],
        'DeviceProtection': ["No internet service"],
        'TechSupport': ["No internet service"],
        'StreamingTV': ["No internet service"],
        'StreamingMovies': ["No internet service"],
        'Contract': [contract],
        'PaperlessBilling': [paperless],
        'PaymentMethod': [payment_method],
        'MonthlyCharges': [monthly_charges],
        'TotalCharges': [total_charges]
    })

    # ENCODE INPUTS THE SAME WAY AS TRAINING
    label = LabelEncoder()
    
    for col in input_data.select_dtypes(include=['object']).columns.tolist():
        input_data[col] = label.fit_transform(input_data[col])

    # MATCH COLUMN ORDER
    input_data = input_data[feature_columns]

    # PREDICT
    prediction = model.predict(input_data)[0]
    probability = model.predict_proba(input_data)[0][1]

    # SHOW RESULT
    st.markdown("---")
    
    if prediction == 1:
        st.markdown(f"""
        <div class="result-card churn-card">
            <div class="result-title" style="color: #f87171;">⚠️ High Risk of Churn</div>
            <div class="result-prob">This customer is likely to leave.</div>
            <div class="result-prob" style="font-size: 2rem; font-weight: bold; margin-top: 10px; color: #ef4444;">
                {probability:.1%} Probability
            </div>
        </div>
        """, unsafe_allow_html=True)
    else:
        st.markdown(f"""
        <div class="result-card stay-card">
            <div class="result-title" style="color: #4ade80;">✅ Customer Will Stay</div>
            <div class="result-prob">This customer is likely to remain loyal.</div>
            <div class="result-prob" style="font-size: 2rem; font-weight: bold; margin-top: 10px; color: #22c55e;">
                {probability:.1%} Probability
            </div>
        </div>
        """, unsafe_allow_html=True)