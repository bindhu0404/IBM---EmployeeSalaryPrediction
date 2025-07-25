import streamlit as st
import pandas as pd
import numpy as np
import joblib
from PIL import Image

# Load saved model and assets
model = joblib.load("model1/model.pkl")
encoder = joblib.load("model1/encoder.pkl")
columns = joblib.load("model1/columns.pkl")

st.set_page_config(page_title="Employee Income Classifier", layout="centered")
st.title("💼 Employee Income Classifier")
st.markdown("This app predicts whether an employee's income is **>50K or <=50K** based on their profile.")

# Styling for pointer hover
st.markdown("""
    <style>
    .stSelectbox:hover, .stSlider:hover, .stNumberInput:hover {
        cursor: pointer !important;
    }
    </style>
""", unsafe_allow_html=True)

# Sidebar inputs
st.sidebar.header("Enter Employee Details")

# Input form
workclass = st.sidebar.selectbox("Workclass", ["Private", "Self-emp-not-inc", "Self-emp-inc", "Federal-gov", "Local-gov", "State-gov", "Without-pay", "Never-worked"])
education = st.sidebar.selectbox("Education", ["Bachelors", "HS-grad", "11th", "Masters", "9th", "Some-college", "Assoc-acdm", "Assoc-voc", "7th-8th", "Doctorate", "5th-6th", "10th", "1st-4th", "Preschool", "12th"])
marital_status = st.sidebar.selectbox("Marital Status", ["Married-civ-spouse", "Divorced", "Never-married", "Separated", "Widowed", "Married-spouse-absent", "Married-AF-spouse"])
occupation = st.sidebar.selectbox("Occupation", ["Tech-support", "Craft-repair", "Other-service", "Sales", "Exec-managerial", "Prof-specialty", "Handlers-cleaners", "Machine-op-inspct", "Adm-clerical", "Farming-fishing", "Transport-moving", "Priv-house-serv", "Protective-serv", "Armed-Forces"])
relationship = st.sidebar.selectbox("Relationship", ["Wife", "Own-child", "Husband", "Not-in-family", "Other-relative", "Unmarried"])
race = st.sidebar.selectbox("Race", ["White", "Asian-Pac-Islander", "Amer-Indian-Eskimo", "Other", "Black"])
sex = st.sidebar.selectbox("Sex", ["Female", "Male"])
native_country = st.sidebar.selectbox("Native Country", ["United-States", "India", "Mexico", "Philippines", "Germany", "Canada", "England", "Cuba", "Iran", "China", "France", "Puerto-Rico", "Jamaica", "Vietnam", "Japan", "Italy", "Greece", "Columbia", "Thailand", "Ecuador", "Poland", "Honduras", "Ireland", "Hungary", "Scotland", "Guatemala", "Nicaragua", "Trinadad&Tobago", "Laos", "Taiwan", "Haiti", "Hong", "South", "Yugoslavia", "El-Salvador", "Dominican-Republic", "Portugal", "Outlying-US(Guam-USVI-etc)", "Cambodia", "Holand-Netherlands", "Peru"])

age = st.sidebar.slider("Age", 17, 90, 30)
fnlwgt = st.sidebar.number_input("Fnlwgt", 10000, 1000000, 200000)
education_num = st.sidebar.slider("Education Number", 1, 16, 10)
hours_per_week = st.sidebar.slider("Hours per Week", 1, 99, 40)

# Create input DataFrame
user_data = pd.DataFrame({
    'age': [age],
    'workclass': [workclass],
    'fnlwgt': [fnlwgt],
    'education': [education],
    'educational-num': [education_num],
    'marital-status': [marital_status],
    'occupation': [occupation],
    'relationship': [relationship],
    'race': [race],
    'gender': [sex],
    'hours-per-week': [hours_per_week],
    'native-country': [native_country],
    'capital-gain': [0],  # default dummy values
    'capital-loss': [0]
})

# Prediction logic
def predict_income(data):
    cat_features = data.select_dtypes(include='object').columns
    try:
        for col in cat_features:
            if col in encoder:
                le = encoder[col]
                val = data[col].iloc[0]
                if val in le.classes_:
                    data[col] = le.transform([val])
                else:
                    st.error(f"Unknown value '{val}' for {col}. Please select a valid option.")
                    return None, None
    except Exception as e:
        st.error(f"Encoding error: {e}")
        return None, None

    try:
        data = data[columns]
        pred = model.predict(data)[0]
        prob = model.predict_proba(data)[0][pred]
        return pred, prob
    except Exception as e:
        st.error(f"Prediction error: {e}")
        return None, None

# Prediction Trigger
if st.button("Predict Income"):
    pred, prob = predict_income(user_data)
    if pred is not None:
        income_label = ">50K" if pred == 1 else "<=50K"
        st.success(f"**Prediction:** {income_label}")
        st.info(f"**Confidence:** {prob:.2%}")
        st.progress(min(max(int(prob * 100), 1), 100))

# Static charts
st.markdown("---")
st.subheader("📊 Top Influential Features")
st.image("model1/feature_importance.png", caption="Top features from Random Forest model", use_container_width=True)

st.markdown("---")
st.subheader("📉 Actual vs Predicted")
st.image("model1/actual_vs_predicted.png", caption="Model's predictions vs actual income values", use_container_width=True)
