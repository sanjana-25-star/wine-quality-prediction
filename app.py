import streamlit as st
import numpy as np
from sklearn.ensemble import RandomForestClassifier

st.title("🍷 Wine Quality Prediction (Random Forest)")

st.write("Enter wine properties to predict quality")

# Inputs (real wine features)
fixed_acidity = st.number_input("Fixed Acidity", min_value=0.0)
volatile_acidity = st.number_input("Volatile Acidity", min_value=0.0)
citric_acid = st.number_input("Citric Acid", min_value=0.0)
residual_sugar = st.number_input("Residual Sugar", min_value=0.0)
chlorides = st.number_input("Chlorides", min_value=0.0)
free_sulfur_dioxide = st.number_input("Free Sulfur Dioxide", min_value=0.0)
total_sulfur_dioxide = st.number_input("Total Sulfur Dioxide", min_value=0.0)
density = st.number_input("Density", min_value=0.0)
pH = st.number_input("pH", min_value=0.0)
sulphates = st.number_input("Sulphates", min_value=0.0)
alcohol = st.number_input("Alcohol", min_value=0.0)

# Dummy dataset (same structure)
X = np.array([
    [7.4,0.7,0,1.9,0.076,11,34,0.9978,3.51,0.56,9.4],
    [7.8,0.88,0,2.6,0.098,25,67,0.9968,3.20,0.68,9.8],
    [7.8,0.76,0.04,2.3,0.092,15,54,0.9970,3.26,0.65,9.8],
    [11.2,0.28,0.56,1.9,0.075,17,60,0.9980,3.16,0.58,9.8],
    [7.4,0.70,0.00,1.9,0.076,11,34,0.9978,3.51,0.56,9.4],
    [7.9,0.60,0.06,1.6,0.069,15,59,0.9964,3.30,0.46,9.4]
])

y = np.array([0,0,0,1,0,1])  # 0 = Bad, 1 = Good

# Train model
model = RandomForestClassifier()
model.fit(X, y)

# Prediction
if st.button("Predict Quality"):
    input_data = np.array([[fixed_acidity, volatile_acidity, citric_acid, residual_sugar,
                            chlorides, free_sulfur_dioxide, total_sulfur_dioxide,
                            density, pH, sulphates, alcohol]])

    prediction = model.predict(input_data)

    if prediction[0] == 1:
        st.success("Good Quality Wine 🍷✅")
    else:
        st.error("Bad Quality Wine ❌")

# Sidebar
st.sidebar.header("About")
st.sidebar.write("Random Forest model predicting wine quality.")
