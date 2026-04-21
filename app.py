import streamlit as st
import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
import joblib
import os

from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score

st.set_page_config(page_title="Wine Quality Predictor", layout="wide")

st.title("🍷 Wine Quality Prediction App")

# Sidebar
st.sidebar.header("Options")
option = st.sidebar.selectbox(
    "Choose Action",
    ["View Dataset", "Train Model", "Visualize Data", "Predict Quality"]
)

# Load dataset
@st.cache_data
def load_data():
    df = pd.read_csv("winequality-red.csv")
    return df

df = load_data()

# ------------------ VIEW DATA ------------------
if option == "View Dataset":
    st.subheader("Dataset Preview")
    st.dataframe(df.head(20))

    st.write("Shape:", df.shape)

# ------------------ TRAIN MODEL ------------------
elif option == "Train Model":
    st.subheader("Train Machine Learning Model")

    # Convert to binary classification
    df['quality'] = df['quality'].apply(lambda x: 1 if x >= 7 else 0)

    X = df.drop('quality', axis=1)
    y = df['quality']

    test_size = st.slider("Test Size", 0.1, 0.5, 0.2)

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=test_size, random_state=42
    )

    n_estimators = st.slider("Number of Trees", 50, 300, 100)

    if st.button("Train Model"):
        model = RandomForestClassifier(
            n_estimators=n_estimators,
            random_state=42
        )
        model.fit(X_train, y_train)

        y_pred = model.predict(X_test)
        acc = accuracy_score(y_test, y_pred)

        st.success(f"Model trained successfully! Accuracy: {acc:.2f}")

        # Save model
        joblib.dump(model, "wine_model.pkl")
        st.info("Model saved as wine_model.pkl")

# ------------------ VISUALIZATION ------------------
elif option == "Visualize Data":
    st.subheader("Data Visualization")

    if st.checkbox("Show Correlation Heatmap"):
        fig, ax = plt.subplots()
        sns.heatmap(df.corr(), annot=True, cmap="coolwarm", ax=ax)
        st.pyplot(fig)

    if st.checkbox("Show Feature Distribution"):
        column = st.selectbox("Select Feature", df.columns)
        fig, ax = plt.subplots()
        sns.histplot(df[column], kde=True, ax=ax)
        st.pyplot(fig)

# ------------------ PREDICTION ------------------
elif option == "Predict Quality":
    st.subheader("Predict Wine Quality")

    if not os.path.exists("wine_model.pkl"):
        st.warning("⚠️ Train the model first!")
    else:
        model = joblib.load("wine_model.pkl")

        st.write("Enter Wine Features:")

        features = []
        for col in df.columns[:-1]:
            val = st.slider(col, float(df[col].min()), float(df[col].max()))
            features.append(val)

        if st.button("Predict"):
            result = model.predict([features])[0]

            if result == 1:
                st.success("🍷 Good Quality Wine!")
            else:
                st.error("❌ Bad Quality Wine")

        # Feature Importance
        st.subheader("Feature Importance")
        importance = model.feature_importances_
        fig, ax = plt.subplots()
        ax.barh(df.columns[:-1], importance)
        st.pyplot(fig)
