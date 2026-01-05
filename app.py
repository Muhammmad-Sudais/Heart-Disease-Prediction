import streamlit as st
import pandas as pd
import numpy as np
import joblib

# Professional styling configuration
st.set_page_config(
    page_title="Heart Disease Prediction - Medical AI Assistant",
    page_icon="❤️",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS for professional styling
st.markdown("""
<style>
    .main-header {
        background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
        padding: 2rem;
        border-radius: 15px;
        margin-bottom: 2rem;
        text-align: center;
        color: white;
        box-shadow: 0 10px 30px rgba(0,0,0,0.1);
    }
    .metric-card {
        background: white;
        padding: 1.5rem;
        border-radius: 10px;
        box-shadow: 0 4px 15px rgba(0,0,0,0.1);
        margin: 1rem 0;
        border-left: 4px solid #667eea;
        transition: all 0.3s ease;
        cursor: pointer;
    }
    .metric-card:hover {
        transform: translateY(-5px);
        box-shadow: 0 8px 25px rgba(102, 126, 234, 0.2);
        border-left: 4px solid #764ba2;
    }
    .success-result {
        background: linear-gradient(135deg, #84fab0 0%, #8fd3f4 100%);
        padding: 2rem;
        border-radius: 15px;
        text-align: center;
        color: white;
        box-shadow: 0 10px 30px rgba(0,0,0,0.1);
        transition: all 0.3s ease;
    }
    .success-result:hover {
        transform: scale(1.02);
        box-shadow: 0 15px 40px rgba(132, 250, 176, 0.3);
    }
    .warning-result {
        background: linear-gradient(135deg, #fa709a 0%, #fee140 100%);
        padding: 2rem;
        border-radius: 15px;
        text-align: center;
        color: white;
        box-shadow: 0 10px 30px rgba(0,0,0,0.1);
        transition: all 0.3s ease;
    }
    .warning-result:hover {
        transform: scale(1.02);
        box-shadow: 0 15px 40px rgba(250, 112, 154, 0.3);
    }
    .info-section {
        background: #f8f9fa;
        padding: 1.5rem;
        border-radius: 10px;
        margin: 1rem 0;
        border-left: 4px solid #17a2b8;
        transition: all 0.3s ease;
    }
    .info-section:hover {
        background: #e9ecef;
        border-left: 4px solid #138496;
        transform: translateX(5px);
    }
    .stButton > button {
        background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
        color: white;
        border: none;
        padding: 0.5rem 2rem;
        font-weight: bold;
        border-radius: 25px;
        transition: all 0.3s ease;
    }
    .stButton > button:hover {
        transform: translateY(-2px);
        box-shadow: 0 5px 15px rgba(102, 126, 234, 0.4);
    }
    .footer {
        text-align: center;
        padding: 2rem;
        background: #f8f9fa;
        border-radius: 10px;
        margin-top: 2rem;
        color: #6c757d;
        transition: all 0.3s ease;
    }
    .footer:hover {
        background: #e9ecef;
        transform: translateY(-2px);
    }
    /* Input field hover effects */
    .stTextInput > div > div > input,
    .stNumberInput > div > div > input,
    .stSelectbox > div > div > select {
        transition: all 0.3s ease;
    }
    .stTextInput > div > div > input:focus,
    .stNumberInput > div > div > input:focus,
    .stSelectbox > div > div > select:focus {
        border-color: #667eea !important;
        box-shadow: 0 0 0 3px rgba(102, 126, 234, 0.1) !important;
    }
    /* Header hover effect */
    .main-header:hover {
        transform: translateY(-3px);
        box-shadow: 0 15px 40px rgba(0,0,0,0.15);
    }
</style>
""", unsafe_allow_html=True)

# Load the trained model
try:
    model = joblib.load('heart_disease_model.pkl')
except FileNotFoundError:
    st.error("Model file not found. Please run train_model.py first.")
    st.stop()

# Professional header
st.markdown("""
<div class="main-header">
    <h1>❤️ Heart Disease Prediction System</h1>
    <p>Advanced AI-Powered Medical Risk Assessment Tool</p>
    <p style="font-size: 0.9rem; opacity: 0.9;">Enter your health parameters below for instant risk evaluation</p>
</div>
""", unsafe_allow_html=True)

# Information section
col_info1, col_info2, col_info3 = st.columns(3)
with col_info1:
    st.markdown("""
    <div class="metric-card">
        <h4>🔬 Medical Grade</h4>
        <p>Clinically validated algorithm with high accuracy</p>
    </div>
    """, unsafe_allow_html=True)

with col_info2:
    st.markdown("""
    <div class="metric-card">
        <h4>⚡ Instant Results</h4>
        <p>Get your risk assessment in seconds</p>
    </div>
    """, unsafe_allow_html=True)

with col_info3:
    st.markdown("""
    <div class="metric-card">
        <h4>🔒 Privacy First</h4>
        <p>Your data stays on your device</p>
    </div>
    """, unsafe_allow_html=True)

# Input form
with st.form("prediction_form"):
    col1, col2 = st.columns(2)
    
    with col1:
        age = st.number_input("Age", min_value=1, max_value=120, value=50)
        sex = st.selectbox("Sex", options=[0, 1], format_func=lambda x: "Male" if x == 1 else "Female")
        cp = st.selectbox("Chest Pain Type", options=[1, 2, 3, 4], help="1: typical angina, 2: atypical angina, 3: non-anginal pain, 4: asymptomatic")
        trestbps = st.number_input("Resting Blood Pressure (mm Hg)", min_value=50, max_value=250, value=120)
        chol = st.number_input("Cholesterol (mg/dl)", min_value=100, max_value=600, value=200)
        fbs = st.selectbox("Fasting Blood Sugar > 120 mg/dl", options=[0, 1], format_func=lambda x: "True" if x == 1 else "False")
        restecg = st.selectbox("Resting ECG Results", options=[0, 1, 2], help="0: normal, 1: ST-T wave abnormality, 2: left ventricular hypertrophy")

    with col2:
        thalach = st.number_input("Max Heart Rate Achieved", min_value=50, max_value=250, value=150)
        exang = st.selectbox("Exercise Induced Angina", options=[0, 1], format_func=lambda x: "Yes" if x == 1 else "No")
        oldpeak = st.number_input("ST Depression Induced by Exercise", min_value=0.0, max_value=10.0, value=1.0)
        slope = st.selectbox("Slope of Peak Exercise ST Segment", options=[1, 2, 3], help="1: upsloping, 2: flat, 3: downsloping")
        ca = st.selectbox("Number of Major Vessels (0-3)", options=[0, 1, 2, 3])
        thal = st.selectbox("Thalassemia", options=[3, 6, 7], help="3: normal, 6: fixed defect, 7: reversable defect")

    # Enhanced submit button
    submit_button = st.form_submit_button("🔍 Analyze Risk", help="Click to analyze your heart disease risk")

if submit_button:
    # Create a dataframe for prediction
    input_data = pd.DataFrame({
        'age': [age],
        'sex': [sex],
        'cp': [cp],
        'trestbps': [trestbps],
        'chol': [chol],
        'fbs': [fbs],
        'restecg': [restecg],
        'thalach': [thalach],
        'exang': [exang],
        'oldpeak': [oldpeak],
        'slope': [slope],
        'ca': [ca],
        'thal': [thal]
    })
    
    # Make prediction
    prediction = model.predict(input_data)[0]
    probability = model.predict_proba(input_data)[0][1]
    
    # Professional result display
    st.markdown("---")
    
    if prediction == 1:
        st.markdown("""
        <div class="warning-result">
            <h2>⚠️ High Risk Detected</h2>
            <h3>Risk Probability: {:.1%}</h3>
            <p><strong>Recommendation:</strong> Please consult a cardiologist immediately for comprehensive evaluation.</p>
            <p>📞 Emergency: Call your local emergency services if experiencing chest pain, shortness of breath, or other severe symptoms.</p>
        </div>
        """.
        format(probability), unsafe_allow_html=True)
        
        # Risk factors section
        st.subheader("📋 Risk Factors Analysis")
        risk_factors = []
        if age > 65:
            risk_factors.append("Age above 65")
        if sex == 1:
            risk_factors.append("Male gender")
        if trestbps > 140:
            risk_factors.append("High blood pressure")
        if chol > 240:
            risk_factors.append("High cholesterol")
        if fbs == 1:
            risk_factors.append("Elevated fasting blood sugar")
            
        if risk_factors:
            st.warning("Identified risk factors: " + ", ".join(risk_factors))
    else:
        st.markdown("""
        <div class="success-result">
            <h2>✅ Low Risk Detected</h2>
            <h3>Risk Probability: {:.1%}</h3>
            <p><strong>Great news!</strong> Your current parameters indicate low risk for heart disease.</p>
            <p>💪 Continue maintaining a healthy lifestyle with regular exercise and balanced nutrition.</p>
        </div>
        """.
        format(probability), unsafe_allow_html=True)
        
        # Prevention tips
        st.subheader("🌟 Prevention Tips")
        st.markdown("""
        <div class="info-section">
            <h4>Maintain Your Heart Health:</h4>
            <ul>
                <li>🏃‍♂️ Exercise regularly (30 minutes, 5 days a week)</li>
                <li>🥗 Eat a balanced diet rich in fruits and vegetables</li>
                <li>🚭 Avoid smoking and limit alcohol consumption</li>
                <li>😴 Get 7-8 hours of quality sleep</li>
                <li>🧘‍♂️ Manage stress through meditation or yoga</li>
                <li>📊 Schedule regular health check-ups</li>
            </ul>
        </div>
        """, unsafe_allow_html=True)
        
        # Professional footer
        st.markdown("""
        <div class="footer">
            <p><strong>Medical Disclaimer:</strong> This tool is for informational purposes only and should not replace professional medical advice.</p>
            <p>Always consult with qualified healthcare providers for medical diagnosis and treatment.</p>
            <p style="font-size: 0.8rem; margin-top: 1rem;"> 2024 Heart Disease Prediction System | Powered by Machine Learning</p>
        </div>
        """, unsafe_allow_html=True)
