import streamlit as st
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression, LogisticRegression
from sklearn.ensemble import RandomForestRegressor, RandomForestClassifier
from sklearn.tree import DecisionTreeRegressor, DecisionTreeClassifier
from sklearn.svm import SVR, SVC
from sklearn.metrics import accuracy_score, mean_squared_error, r2_score, confusion_matrix
import matplotlib.pyplot as plt
import seaborn as sns
import joblib

# ---------------------------------------------
# PAGE CONFIG
# ---------------------------------------------
st.set_page_config(page_title="Streamlit", layout="wide")

# ---------------------------------------------
# SIDEBAR
# ---------------------------------------------
st.sidebar.title("Model Type")

model_type = st.sidebar.selectbox(
    "Model Type",
    ["Regression", "Classification"]
)

if model_type == "Regression":
    algorithm = st.sidebar.selectbox(
        "Algorithms",
        ["Linear Regression", "Random Forest Regressor", "Support Vector Regressor", "Decision Tree Regressor"]
    )
else:
    algorithm = st.sidebar.selectbox(
        "Algorithms",
        ["Logistic Regression", "Random Forest Classifier", "Support Vector Classifier", "Decision Tree Classifier"]
    )

# ---------------------------------------------
# MAIN PAGE HEADING
# ---------------------------------------------
st.title("Machine Learning Models")
st.subheader(f"{algorithm}")

# ---------------------------------------------
# Radio Buttons
# ---------------------------------------------
col1, col2 = st.columns(2)

with col1:
    fit_intercept = st.radio("Fit Intercept", [True, False])

with col2:
    copy_x = st.radio("Copy X", [True, False])

# ---------------------------------------------
# File Upload
# ---------------------------------------------
uploaded_file = st.file_uploader("Choose the File", type=["csv"])

if uploaded_file:

    # Read full data
    df = pd.read_csv(uploaded_file)

    st.write("📄 Full Uploaded Data")
    st.dataframe(df)   # FULL DATA DISPLAY

    st.write(f"Total Rows: {df.shape[0]}, Total Columns: {df.shape[1]}")

    # ------------------------------------------------------
    # MULTIPLE FEATURE SELECTION
    # ------------------------------------------------------
    feature_cols = st.multiselect("Select Features", df.columns)

    # ---------------------------------------------
    # Select Label
    # ---------------------------------------------
    label_col = st.selectbox("Select Label", df.columns)

    # ---------------------------------------------
    # Submit Button
    # ---------------------------------------------
    if st.button("Submit"):

        # Validations
        if len(feature_cols) == 0:
            st.error("Please select at least one feature!")
        elif label_col in feature_cols:
            st.error("Label column cannot be selected as a feature!")
        else:

            # Convert selected features into X matrix
            X = df[feature_cols]
            y = df[label_col]

            # Train/Test Split
            X_train, X_test, y_train, y_test = train_test_split(
                X, y, test_size=0.2, random_state=42
            )

            # ---------------------------------------------
            # MODEL SELECTION
            # ---------------------------------------------
            if algorithm == "Linear Regression":
                model = LinearRegression(fit_intercept=fit_intercept, copy_X=copy_x)

            elif algorithm == "Random Forest Regressor":
                model = RandomForestRegressor()

            elif algorithm == "Support Vector Regressor":
                model = SVR()

            elif algorithm == "Decision Tree Regressor":
                model = DecisionTreeRegressor()

            elif algorithm == "Logistic Regression":
                model = LogisticRegression()

            elif algorithm == "Random Forest Classifier":
                model = RandomForestClassifier()

            elif algorithm == "Support Vector Classifier":
                model = SVC()

            elif algorithm == "Decision Tree Classifier":
                model = DecisionTreeClassifier()

            # ---------------------------------------------
            # Train Model
            # ---------------------------------------------
            model.fit(X_train, y_train)
            y_pred = model.predict(X_test)

            st.success("Model Training Completed Successfully!")

            # ---------------------------------------------
            # RESULTS + GRAPH
            # ---------------------------------------------
            st.subheader("Results")

            # ---------------------------------------------
            # REGRESSION
            # ---------------------------------------------
            if model_type == "Regression":

                mse = mean_squared_error(y_test, y_pred)
                r2 = r2_score(y_test, y_pred)

                st.write(f"**MSE:** {mse}")
                st.write(f"**R² Score:** {r2}")

                # --- Actual vs Predicted Plot ---
                st.write("Actual vs Predicted Plot")

                fig, ax = plt.subplots()
                ax.scatter(y_test, y_pred)
                ax.set_xlabel("Actual Values")
                ax.set_ylabel("Predicted Values")
                ax.set_title("Actual vs Predicted")
                st.pyplot(fig)

            # ---------------------------------------------
            # CLASSIFICATION
            # ---------------------------------------------
            else:

                acc = accuracy_score(y_test, y_pred)
                st.write(f"**Accuracy:** {acc}")

                # --- Confusion Matrix ---
                st.write("Confusion Matrix")

                cm = confusion_matrix(y_test, y_pred)

                fig, ax = plt.subplots()
                sns.heatmap(cm, annot=True, cmap="Blues", fmt="d", ax=ax)
                ax.set_xlabel("Predicted")
                ax.set_ylabel("Actual")
                st.pyplot(fig)

            # ---------------------------------------------
            # Save Model
            # ---------------------------------------------
            joblib.dump(model, "trained_model.pkl")

            st.download_button(
                label="Download Model",
                data=open("trained_model.pkl", "rb").read(),
                file_name="trained_model.pkl",
                mime="application/octet-stream"
            )
