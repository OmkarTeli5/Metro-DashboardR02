import streamlit as st
import pandas as pd
import joblib
import numpy as np

# Load model and preprocessing objects
model = joblib.load("model.pkl")
preprocessor = joblib.load("preprocessor.pkl")
features = joblib.load("features.pkl")

# Set Streamlit page config
st.set_page_config(page_title="Metro Station Civil Cost Estimator", layout="wide")
st.title("🚉 Metro Station Civil Cost Estimator")
st.markdown("Estimate metro station civil construction cost using 10 key design parameters.")

# Station type presets
station_type = st.sidebar.selectbox("Select Station Type:", ["Regular", "Terminal", "Interchange", "Custom"])

autofill_presets = {
    "Regular": [1, 1, 3, 100, 150, 20, 15, 5, 2, 1],
    "Terminal": [2, 2, 4, 110, 180, 25, 18, 6, 2, 2],
    "Interchange": [3, 3, 5, 120, 200, 30, 20, 7, 3, 3],
    "Custom": [None] * len(features)
}

# Manual input UI
user_inputs = {}
st.sidebar.header("🧾 Input Parameters")
for i, feature in enumerate(features):
    default_val = autofill_presets[station_type][i]
    if default_val is not None:
        user_inputs[feature] = st.sidebar.number_input(label=feature, value=float(default_val), step=1.0)
    else:
        user_inputs[feature] = st.sidebar.number_input(label=feature, step=1.0)

# Prediction logic
def predict_cost(input_dict):
    df = pd.DataFrame([input_dict])
    df = df[features]  # Ensure correct column order
    df_transformed = preprocessor.transform(df)
    prediction = model.predict(df_transformed)[0]
    return round(prediction, 2)

if st.button("💰 Predict Civil Cost"):
    try:
        cost = predict_cost(user_inputs)
        st.success(f"Estimated Civil Cost: ₹{cost} Cr")
    except Exception as e:
        st.error(f"Prediction failed: {str(e)}")

# Excel batch upload section
st.markdown("### 📂 Batch Prediction via Excel")
uploaded_file = st.file_uploader("Upload Excel file with station parameters", type=["xlsx"])

if uploaded_file is not None:
    try:
        df_excel = pd.read_excel(uploaded_file)
        missing_cols = [col for col in features if col not in df_excel.columns]
        if missing_cols:
            st.error(f"Missing required columns: {missing_cols}")
        else:
            df_excel = df_excel[features]
            df_transformed = preprocessor.transform(df_excel)
            preds = model.predict(df_transformed)
            df_excel["Predicted Civil Cost (Cr)"] = np.round(preds, 2)
            st.success("Batch prediction successful!")
            st.dataframe(df_excel)
            csv = df_excel.to_csv(index=False).encode("utf-8")
            st.download_button("📥 Download Predictions CSV", csv, "predictions.csv", "text/csv")
    except Exception as e:
        st.error(f"Error processing file: {e}")
