import streamlit as st
import pandas as pd
import joblib
import numpy as np
import io

# Load files
model = joblib.load("model.pkl")
preprocessor = joblib.load("preprocessor.pkl")
feature_columns = joblib.load("features.pkl")

# Streamlit UI
st.set_page_config(page_title="Metro Civil Cost Estimator", layout="wide")
st.title("🚇 Metro Station Civil Cost Estimator")
st.markdown("Estimate civil cost using 10 key design and location parameters.")

# Template download
def generate_template(columns):
    df_template = pd.DataFrame(columns=columns)
    buffer = io.BytesIO()
    df_template.to_excel(buffer, index=False)
    buffer.seek(0)
    return buffer

if st.sidebar.button("📥 Download Excel Template"):
    st.download_button(
        label="Download Template",
        data=generate_template(feature_columns),
        file_name="metro_input_template.xlsx",
        mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet"
    )

# Autofill presets
station_type = st.selectbox("Select Station Type:", ["Regular", "Terminal", "Interchange", "Custom"])

presets = {
    "Regular": ["delhi", "underground", 3, 125, 210, 32, 18, 5, 3, "typ-a"],
    "Terminal": ["mumbai", "elevated", 2, 145, 180, 30, 16, 4, 2, "typ-b"],
    "Interchange": ["chennai", "underground", 4, 160, 240, 36, 20, 6, 4, "typ-c"],
    "Custom": [None] * len(feature_columns)
}

# User input form
st.sidebar.header("🧮 Input Parameters")
inputs = {}
for i, feature in enumerate(feature_columns):
    default_val = presets[station_type][i]
    inputs[feature] = st.sidebar.text_input(label=feature, value=str(default_val) if default_val else "")

# Prediction function
def predict(inputs):
    df_input = pd.DataFrame([inputs])
    df_input = df_input.astype({col: "float" if col not in ["city", "metro_type", "station_typology"] else "object" for col in df_input.columns})
    X = preprocessor.transform(df_input)
    return round(model.predict(X)[0], 2)

if st.button("💰 Predict Civil Cost"):
    try:
        cost = predict(inputs)
        st.success(f"Predicted Civil Cost: ₹{cost} Cr")
    except Exception as e:
        st.error(f"Prediction failed: {str(e)}")

# Excel upload for batch prediction
st.markdown("### 📂 Upload Excel for Batch Prediction")
uploaded_file = st.file_uploader("Upload .xlsx file with all 10 columns", type=["xlsx"])

if uploaded_file is not None:
    try:
        df = pd.read_excel(uploaded_file)
        missing = [col for col in feature_columns if col not in df.columns]
        extra = [col for col in df.columns if col not in feature_columns]

        if missing:
            st.error(f"Missing required columns: {missing}")
            st.stop()
        if extra:
            st.warning(f"Extra unused columns: {extra}")

        df = df.astype({col: "float" if col not in ["city", "metro_type", "station_typology"] else "object" for col in df.columns})
        X = preprocessor.transform(df)
        preds = model.predict(X)
        df["Predicted Cost (Cr)"] = np.round(preds, 2)
        st.success("Batch prediction successful!")
        st.dataframe(df)
        csv = df.to_csv(index=False).encode("utf-8")
        st.download_button("📥 Download Result CSV", csv, "predictions.csv", "text/csv")
    except Exception as e:
        st.error(f"Error: {e}")
