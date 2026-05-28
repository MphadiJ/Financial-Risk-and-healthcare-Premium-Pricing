import streamlit as st
import pandas as pd
import sys
import os

# Path fix FIRST, before any custom imports
PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
SRC_PATH = os.path.join(PROJECT_ROOT, "src")
sys.path.insert(0, SRC_PATH)

#  Now import
from inference.Prediction import Predictor

st.set_page_config(page_title="Medical Insurance Charges Predictor", layout="wide")
st.title("Annual Medical Insurance Charges Predictor")
st.write("Enter your information below manually or upload a CSV file to predict insurance charges using a trained Random Forest model.")

# Model path points to the repo's models folder, not your local PC
MODEL_PATH = os.path.join(PROJECT_ROOT, "models", "artifacts.pkl")

predictor = Predictor(
    artifacts_path=MODEL_PATH,
    debug=False
)

# Manual Input Section
st.header("Manual Input")
age = st.number_input("Age", min_value=0, max_value=100, value=30)
bmi = st.number_input("BMI", min_value=10.0, max_value=60.0, value=25.0, step=0.1)
children = st.number_input("Number of Children", min_value=0, max_value=10, value=0)
sex = st.selectbox("Sex", options=["male", "female"])
smoker = st.selectbox("Smoker", options=["yes", "no"])
region = st.selectbox("Region", options=["northwest", "northeast", "southwest", "southeast"])

manual_input_df = pd.DataFrame([{
    "age": age, "bmi": bmi, "children": children,
    "sex": sex, "smoker": smoker, "region": region
}])

# CSV Upload Section
st.header("Upload CSV")
uploaded_file = st.file_uploader("Upload a CSV with the same columns as training data", type=["csv"])

if uploaded_file:
    csv_df = pd.read_csv(uploaded_file)
    st.write("Preview of uploaded data:")
    st.dataframe(csv_df.head())
    input_df = csv_df
else:
    input_df = manual_input_df

if st.button("Generate Predictions"):
    with st.spinner("Predicting..."):
        EXPECTED_COLS = [
            'age', 'sex', 'bmi', 'children', 'smoker',
            'region_northeast', 'region_northwest', 'region_southeast', 'region_southwest'
        ]
        processed = predictor.preprocessor.transform(input_df)
        processed = processed[EXPECTED_COLS]
        predictions = predictor.model.predict(processed)
        result = input_df.copy()
        result["Predicted Charges"] = predictions
    st.success("Predictions Generated!")
    st.dataframe(result)
