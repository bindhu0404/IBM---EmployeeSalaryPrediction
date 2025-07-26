import streamlit as st
import pandas as pd
import numpy as np
import pickle

# Load model, encoders, and columns
with open("model1/model.pkl", "rb") as f:
    model = pickle.load(f)

with open("model1/encoder.pkl", "rb") as f:
    encoders = pickle.load(f)

with open("model1/columns.pkl", "rb") as f:
    model_columns = pickle.load(f)

st.set_page_config(page_title="Income Prediction", layout="centered")

st.title("🧠 Income Classification App")
st.write("Enter your details below to predict whether your income is >50K or <=50K.")

# Sample input fields (based on UCI Adult dataset)
def user_input():
    age = st.number_input("Age", min_value=18, max_value=100, value=30)
    workclass = st.selectbox("Workclass", [
        'Private', 'Self-emp-not-inc', 'Self-emp-inc', 'Federal-gov',
        'Local-gov', 'State-gov', 'Without-pay', 'Never-worked'
    ])
    education = st.selectbox("Education", [
        'Bachelors', 'Some-college', '11th', 'HS-grad', 'Prof-school',
        'Assoc-acdm', 'Assoc-voc', '9th', '7th-8th', '12th', 'Masters',
        '1st-4th', '10th', 'Doctorate', '5th-6th', 'Preschool'
    ])
    educational_num = st.number_input("Educational Number", min_value=1, max_value=20, value=10)
    marital_status = st.selectbox("Marital Status", [
        'Married-civ-spouse', 'Divorced', 'Never-married',
        'Separated', 'Widowed', 'Married-spouse-absent', 'Married-AF-spouse'
    ])
    occupation = st.selectbox("Occupation", [
        'Tech-support', 'Craft-repair', 'Other-service', 'Sales',
        'Exec-managerial', 'Prof-specialty', 'Handlers-cleaners',
        'Machine-op-inspct', 'Adm-clerical', 'Farming-fishing',
        'Transport-moving', 'Priv-house-serv', 'Protective-serv', 'Armed-Forces'
    ])
    relationship = st.selectbox("Relationship", [
        'Wife', 'Own-child', 'Husband', 'Not-in-family',
        'Other-relative', 'Unmarried'
    ])
    race = st.selectbox("Race", [
        'White', 'Asian-Pac-Islander', 'Amer-Indian-Eskimo', 'Other', 'Black'
    ])
    gender = st.selectbox("Gender", ['Male', 'Female'])
    hours_per_week = st.slider("Hours per Week", min_value=1, max_value=100, value=40)
    native_country = st.selectbox("Native Country", [
        'United-States', 'Cambodia', 'England', 'Puerto-Rico', 'Canada',
        'Germany', 'Outlying-US(Guam-USVI-etc)', 'India', 'Japan', 'Greece',
        'South', 'China', 'Cuba', 'Iran', 'Honduras', 'Philippines', 'Italy',
        'Poland', 'Jamaica', 'Vietnam', 'Mexico', 'Portugal', 'Ireland',
        'France', 'Dominican-Republic', 'Laos', 'Ecuador', 'Taiwan',
        'Haiti', 'Columbia', 'Hungary', 'Guatemala', 'Nicaragua', 'Scotland',
        'Thailand', 'Yugoslavia', 'El-Salvador', 'Trinadad&Tobago', 'Peru',
        'Hong', 'Holand-Netherlands'
    ])

    data = {
        'age': age,
        'workclass': workclass,
        'education': education,
        'educational-num': educational_num,
        'marital-status': marital_status,
        'occupation': occupation,
        'relationship': relationship,
        'race': race,
        'gender': gender,
        'hours-per-week': hours_per_week,
        'native-country': native_country
    }

    return pd.DataFrame([data])

# Preprocess user input to match model input
def preprocess_input(input_df):
    df = input_df.copy()

    # Apply encoders
    for col in df.columns:
        if col in encoders:
            le = encoders[col]
            try:
                df[col] = le.transform(df[col])
            except ValueError:
                st.error(f"Unknown category in column '{col}': '{df[col].values[0]}'")
                st.stop()

    # Add missing columns (if any)
    for col in model_columns:
        if col not in df.columns:
            df[col] = 0

    # Reorder columns
    df = df[model_columns]

    return df

# Run prediction
user_df = user_input()

if st.button("Predict Income"):
    processed = preprocess_input(user_df)
    prediction = model.predict(processed)[0]
    prediction_proba = model.predict_proba(processed)[0][prediction]

    if prediction == 1:
        st.success(f"✅ Predicted Income: >50K (Confidence: {prediction_proba:.2%})")
    else:
        st.warning(f"❌ Predicted Income: <=50K (Confidence: {prediction_proba:.2%})")

    # Debug
    with st.expander("See input data"):
        st.write(user_df)

    with st.expander("See model-ready data"):
        st.write(processed)
