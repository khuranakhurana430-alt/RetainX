import streamlit as st
import pandas as pd
import numpy as np
import joblib
import plotly.express as px
import plotly.graph_objects as go

# 1. PAGE CONFIG (Sets the Browser Tab Icon and Title)
st.set_page_config(
    page_title="Churn Prediction App", 
    page_icon="📊", 
    layout="wide"
)

# Load artifacts
@st.cache_resource
def load_artifacts():
    model = joblib.load('churn_model.pkl')
    scaler = joblib.load('scaler.pkl')
    feature_columns = joblib.load('feature_columns.pkl')
    threshold = joblib.load('best_threshold.pkl')
    return model, scaler, feature_columns, threshold

model, scaler, feature_columns, threshold = load_artifacts()

# 2. SIDEBAR WITH LOGO
# Using a high-quality analytics icon from Flaticon
logo_url = "https://cdn-icons-png.flaticon.com/512/2103/2103633.png"
st.sidebar.image(logo_url, width=120)
st.sidebar.title("Customer Churn Analytics")
st.sidebar.markdown("---")
st.sidebar.info(
    "This app uses an XGBoost model trained on the IBM Telco Customer Churn dataset. "
    "It predicts the likelihood of a customer leaving the company based on their profile and services."
)
st.sidebar.markdown("---")
st.sidebar.write(f"**Model Accuracy:** ~80.5%")
st.sidebar.write(f"**Decision Threshold:** {threshold:.2f}")

# MAIN PAGE HEADER
st.title("📊 Customer Churn Prediction")
st.markdown("Fill in customer details below to predict churn probability and view insights.")

# Input Form
col1, col2 = st.columns(2)

with col1:
    st.subheader("Demographics")
    gender = st.selectbox("Gender", ["Female", "Male"])
    senior = st.selectbox("Senior Citizen", ["No", "Yes"])
    partner = st.selectbox("Partner", ["No", "Yes"])
    dependents = st.selectbox("Dependents", ["No", "Yes"])
    tenure = st.slider("Tenure (months)", 0, 72, 12)

with col2:
    st.subheader("Services & Billing")
    phone_service = st.selectbox("Phone Service", ["No", "Yes"])
    multiple_lines = st.selectbox("Multiple Lines", ["No", "Yes", "No phone service"])
    internet_service = st.selectbox("Internet Service", ["DSL", "Fiber optic", "No"])
    online_security = st.selectbox("Online Security", ["No", "Yes", "No internet service"])
    tech_support = st.selectbox("Tech Support", ["No", "Yes", "No internet service"])
    contract = st.selectbox("Contract", ["Month-to-month", "One year", "Two year"])
    payment_method = st.selectbox("Payment Method",
        ["Electronic check", "Mailed check", "Bank transfer (automatic)", "Credit card (automatic)"])
    monthly_charges = st.number_input("Monthly Charges ($)", 0.0, 200.0, 70.0)
    total_charges = st.number_input("Total Charges ($)", 0.0, 10000.0, 1000.0)

# Prediction Logic
if st.button("🔮 Predict Churn", type="primary", use_container_width=True):
    
    # 1. Prepare Input
    input_data = {
        'gender': gender, 'SeniorCitizen': 1 if senior == "Yes" else 0,
        'Partner': partner, 'Dependents': dependents, 'tenure': tenure,
        'PhoneService': phone_service, 'MultipleLines': multiple_lines,
        'InternetService': internet_service, 'OnlineSecurity': online_security,
        'TechSupport': tech_support, 'Contract': contract,
        'PaymentMethod': payment_method, 'MonthlyCharges': monthly_charges,
        'TotalCharges': total_charges,
    }

    input_df = pd.DataFrame([input_data])
    input_encoded = pd.get_dummies(input_df)
    
    for col in feature_columns:
        if col not in input_encoded.columns:
            input_encoded[col] = 0
            
    input_encoded = input_encoded[feature_columns]
    input_scaled = scaler.transform(input_encoded)

    # 2. Make Prediction
    proba = model.predict_proba(input_scaled)[0, 1]
    prediction = "CHURN" if proba >= threshold else "NO CHURN"

    st.divider()
    
    # --- RESULTS LAYOUT ---
    res_col1, res_col2 = st.columns([1, 1])

    with res_col1:
        st.subheader("Prediction Result")
        if proba >= threshold:
            st.error(f"### ⚠️ {prediction}")
            st.write("This customer is **likely to churn**.")
        else:
            st.success(f"### ✅ {prediction}")
            st.write("This customer is **likely to stay**.")
        
        # Plotly Gauge Chart
        fig_gauge = go.Figure(go.Indicator(
            mode = "gauge+number",
            value = proba * 100,
            number = {'suffix': "%"},
            domain = {'x': [0, 1], 'y': [0, 1]},
            title = {'text': "Churn Probability", 'font': {'size': 24}},
            gauge = {
                'axis': {'range': [None, 100], 'tickwidth': 1, 'tickcolor': "darkblue"},
                'bar': {'color': "rgba(0,0,0,0)"},
                'bgcolor': "white",
                'borderwidth': 2,
                'bordercolor': "gray",
                'steps': [
                    {'range': [0, threshold * 100], 'color': "lightgreen"},
                    {'range': [threshold * 100, 100], 'color': "salmon"}
                ],
                'threshold': {
                    'line': {'color': "red", 'width': 4},
                    'thickness': 0.75,
                    'value': proba * 100
                }
            }
        ))
        fig_gauge.update_layout(height=300, margin=dict(l=20, r=20, t=50, b=20))
        st.plotly_chart(fig_gauge, use_container_width=True)

    with res_col2:
        st.subheader("Top Reasons for Prediction")
        st.write("These are the features the model weighs most heavily.")
        
        # Extract feature importances
        try:
            base_xgb = model.calibrated_classifiers_[0].estimator
            importances = base_xgb.feature_importances_
            fi_df = pd.DataFrame({'Feature': feature_columns, 'Importance': importances})
            fi_df = fi_df.sort_values(by='Importance', ascending=True).tail(10)
            
            fi_df['Feature'] = fi_df['Feature'].str.replace('_', ' ').str.title()
            
            fig_fi = px.bar(
                fi_df, x='Importance', y='Feature', orientation='h',
                color='Importance', color_continuous_scale='Blues',
                title="Feature Importance (Top 10)"
            )
            fig_fi.update_layout(showlegend=False, height=350, margin=dict(l=20, r=20, t=40, b=20))
            st.plotly_chart(fig_fi, use_container_width=True)
        except Exception as e:
            st.warning("Could not load feature importance chart.")

    # --- CUSTOMER PROFILE COMPARISON ---
    st.divider()
    st.subheader("Customer vs Average Profile")
    st.write("How this customer's tenure and billing compare to the average customer.")
    
    avg_tenure = 32.37
    avg_monthly = 64.76
    avg_total = 2283.30

    compare_df = pd.DataFrame({
        'Metric': ['Tenure (Months)', 'Monthly Charges ($)', 'Total Charges ($)'],
        'This Customer': [tenure, monthly_charges, total_charges],
        'Dataset Average': [avg_tenure, avg_monthly, avg_total]
    })
    
    compare_melt = compare_df.melt(id_vars='Metric', var_name='Type', value_name='Value')
    
    fig_compare = px.bar(
        compare_melt, x='Metric', y='Value', color='Type', barmode='group',
        color_discrete_sequence=['#00CC96', '#AB63FA']
    )
    fig_compare.update_layout(height=300, margin=dict(l=20, r=20, t=20, b=20))
    st.plotly_chart(fig_compare, use_container_width=True)