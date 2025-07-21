# import streamlit as st
# import pandas as pd
# import numpy as np
# import joblib
# import matplotlib.pyplot as plt
# import seaborn as sns

# # Load model, encoder, and scaler
# model = joblib.load("model/model.pkl")
# encoders = joblib.load("model/encoder.pkl")
# scaler = joblib.load("model/scaler.pkl")

# # Load dataset headers for reference
# df = pd.read_csv("data/adult 3.csv")
# df = df.replace('?', np.nan).dropna()
# X = df.drop('income', axis=1)

# # Set page config
# st.set_page_config(page_title="Employee Salary Predictor", layout="wide", initial_sidebar_state="expanded")

# # Apply custom dark style
# with open("style.css") as f:
#     st.markdown(f"<style>{f.read()}</style>", unsafe_allow_html=True)

# st.title("💼 Employee Salary Prediction")
# st.markdown("Predict whether a person earns >50K or <=50K based on their profile.")

# # Sidebar user inputs
# st.sidebar.header("🔧 Enter Employee Details")

# user_input = {}

# for col in X.columns:
#     if X[col].dtype == 'object':
#         options = df[col].unique().tolist()
#         user_input[col] = st.sidebar.selectbox(col.title(), options)
#     else:
#         min_val = int(df[col].min())
#         max_val = int(df[col].max())
#         mean_val = int(df[col].mean())
#         user_input[col] = st.sidebar.slider(col.title(), min_val, max_val, mean_val)

# # Transform input to dataframe
# input_df = pd.DataFrame([user_input])

# # Encode categorical
# for col in input_df.select_dtypes(include='object').columns:
#     input_df[col] = encoders[col].transform(input_df[col])

# # Scale numerical
# input_df[input_df.select_dtypes(include=['int64', 'float64']).columns] = scaler.transform(input_df.select_dtypes(include=['int64', 'float64']))

# # Predict
# if st.sidebar.button("Predict Salary"):
#     prediction = model.predict(input_df)[0]
#     proba = model.predict_proba(input_df)[0][prediction]
#     label = ">50K" if prediction == 1 else "<=50K"

#     st.subheader("📊 Prediction Result")
#     st.success(f"**Predicted Salary: {label}**")
#     st.progress(proba)
#     st.write(f"Prediction Confidence: {proba * 100:.2f}%")

#     # Show Feature Importance
#     # st.subheader("📌 Feature Importance")
#     # st.image("model/feature_importance.png")

#     # Show Accuracy Progression Graph
#     st.subheader("📈 Model Accuracy Curve")
#     st.image("model/accuracy_plot.png")

# # Footer
# st.markdown("---")
# st.markdown("Made with ❤️ using Streamlit | Random Forest Classifier")
import streamlit as st
import pandas as pd
import numpy as np
import pickle
from PIL import Image

# Load model components
model = pickle.load(open("model1/model.pkl", "rb"))
encoder = pickle.load(open("model1/encoder.pkl", "rb"))
columns = pickle.load(open("model1/columns.pkl", "rb"))

# Set page config
st.set_page_config(page_title="Employee Income Classifier", layout="centered")
st.title("💼 Employee Income Classifier")
st.markdown("This app predicts whether an employee's income is **>50K or <=50K** based on their profile.")

# Custom CSS to show pointer cursor on hover
st.markdown(
    """
    <style>
    .stSelectbox:hover, .stSlider:hover, .stNumberInput:hover {
        cursor: pointer !important;
    }
    </style>
    """,
    unsafe_allow_html=True
)

# Sidebar input
st.sidebar.header("Enter Employee Details")

# Define form inputs
workclass = st.sidebar.selectbox("Workclass", ["Private", "Self-emp-not-inc", "Self-emp-inc", "Federal-gov", "Local-gov", "State-gov", "Without-pay", "Never-worked"])
education = st.sidebar.selectbox("Education", ["Bachelors", "HS-grad", "11th", "Masters", "9th", "Some-college", "Assoc-acdm", "Assoc-voc", "7th-8th", "Doctorate", "5th-6th", "10th", "1st-4th", "Preschool", "12th"])
marital_status = st.sidebar.selectbox("Marital Status", ["Married-civ-spouse", "Divorced", "Never-married", "Separated", "Widowed", "Married-spouse-absent", "Married-AF-spouse"])
occupation = st.sidebar.selectbox("Occupation", ["Tech-support", "Craft-repair", "Other-service", "Sales", "Exec-managerial", "Prof-specialty", "Handlers-cleaners", "Machine-op-inspct", "Adm-clerical", "Farming-fishing", "Transport-moving", "Priv-house-serv", "Protective-serv", "Armed-Forces"])
relationship = st.sidebar.selectbox("Relationship", ["Wife", "Own-child", "Husband", "Not-in-family", "Other-relative", "Unmarried"])
race = st.sidebar.selectbox("Race", ["White", "Asian-Pac-Islander", "Amer-Indian-Eskimo", "Other", "Black"])
sex = st.sidebar.selectbox("Sex", ["Female", "Male"])
native_country = st.sidebar.selectbox("Native Country", ["United-States", "Cambodia", "England", "Puerto-Rico", "Canada", "Germany", "Outlying-US(Guam-USVI-etc)", "India", "Japan", "Greece", "South", "China", "Cuba", "Iran", "Honduras", "Philippines", "Italy", "Poland", "Jamaica", "Vietnam", "Mexico", "Portugal", "Ireland", "France", "Dominican-Republic", "Laos", "Ecuador", "Taiwan", "Haiti", "Columbia", "Hungary", "Guatemala", "Nicaragua", "Scotland", "Thailand", "Yugoslavia", "El-Salvador", "Trinadad&Tobago", "Peru", "Hong", "Holand-Netherlands"])

age = st.sidebar.slider("Age", 17, 90, 30)
fnlwgt = st.sidebar.number_input("Fnlwgt", 10000, 1000000, 200000)
education_num = st.sidebar.slider("Education Number", 1, 16, 10)
hours_per_week = st.sidebar.slider("Hours per Week", 1, 99, 40)

# Create input DataFrame with exact column names
user_data = pd.DataFrame({
    'age': [age],
    'workclass': [workclass],
    'fnlwgt': [fnlwgt],
    'education': [education],
    'educational-num': [education_num],  # fixed name to match model
    'marital-status': [marital_status],
    'occupation': [occupation],
    'relationship': [relationship],
    'race': [race],
    'gender': [sex],  # fixed name to match model
    'hours-per-week': [hours_per_week],
    'native-country': [native_country],
    'capital-gain': [0],  # dummy value
    'capital-loss': [0]   # dummy value
})

# Prediction function
def predict_income(data):
    cat_features = data.select_dtypes(include='object').columns
    try:
        for col in cat_features:
            if col in encoder:
                le = encoder[col]
                data[col] = le.transform(data[col])
    except Exception as e:
        st.error(f"Encoding error: {e}")
        return None, None

    try:
        data = data[columns]  # Ensure correct column order
        pred = model.predict(data)[0]
        prob = model.predict_proba(data)[0][pred]
        return pred, prob
    except Exception as e:
        st.error(f"Prediction error: {e}")
        return None, None

# Make prediction
if st.button("Predict Income"):
    pred, prob = predict_income(user_data)
    if pred is not None:
        income_label = ">50K" if pred == 1 else "<=50K"
        st.success(f"**Prediction:** {income_label}")
        st.info(f"**Confidence:** {prob:.2%}")
        st.progress(min(max(int(prob * 100), 1), 100))

# Feature importance section (static chart)
st.markdown("---")
st.subheader("📊 Top Influential Features")
st.image("model1/feature_importance.png", caption="Most influential features from the model", use_column_width=True)

# Actual vs Predicted (Static)
st.markdown("---")
st.subheader("📉 Actual vs Predicted (Demo)")
st.image("model1/actual_vs_predicted.png", caption="This plot displays the model's predictions vs actual income classes.")
