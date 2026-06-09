#=========================================
# TITANIC SURVIVAL ANALYSIS - STREAMLIT APP
# =========================================

import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import accuracy_score

from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier

# -----------------------------------------
# PAGE SETTINGS
# -----------------------------------------

st.set_page_config(
    page_title="Titanic Survival Analysis",
    layout="wide"
)

st.title("🚢 Titanic Survival Analysis Project")

st.markdown("---")

# -----------------------------------------
# LOAD DATASET
# -----------------------------------------

df = pd.read_csv("train.csv")

# -----------------------------------------
# DATA CLEANING
# -----------------------------------------

df["Age"] = df["Age"].fillna(df["Age"].median())

df["Embarked"] = df["Embarked"].fillna(
    df["Embarked"].mode()[0]
)

df["Fare"] = df["Fare"].fillna(
    df["Fare"].median()
)

if "Cabin" in df.columns:
    df.drop("Cabin", axis=1, inplace=True)

# -----------------------------------------
# FEATURE ENGINEERING
# -----------------------------------------

df["FamilySize"] = df["SibSp"] + df["Parch"] + 1

df["IsAlone"] = 0

df.loc[df["FamilySize"] == 1, "IsAlone"] = 1

df["Title"] = df["Name"].str.extract(
    r' ([A-Za-z]+)\.',
    expand=False
)

# -----------------------------------------
# SIDEBAR
# -----------------------------------------

menu = st.sidebar.selectbox(
    "Choose Section",
    [
        "Dataset",
        "Visualizations",
        "Heatmap",
        "Machine Learning",
        "Feature Importance"
    ]
)

# =========================================
# DATASET SECTION
# =========================================

if menu == "Dataset":

    st.subheader("📄 Dataset Preview")

    st.dataframe(df.head(10))

    st.subheader("📊 Dataset Shape")

    st.write("Rows:", df.shape[0])
    st.write("Columns:", df.shape[1])

    st.subheader("📌 Missing Values")

    st.write(df.isnull().sum())

    st.subheader("📈 Statistical Summary")

    st.dataframe(df.describe())

# =========================================
# VISUALIZATIONS
# =========================================

elif menu == "Visualizations":

    st.subheader("📊 Titanic Visualizations")

    # Survival Count
    fig1, ax1 = plt.subplots(figsize=(8, 5))

    sns.countplot(
        x="Survived",
        data=df,
        palette="Set2",
        ax=ax1
    )

    ax1.set_title("Survival Count")

    st.pyplot(fig1)

    # Gender Survival
    fig2, ax2 = plt.subplots(figsize=(8, 5))

    sns.countplot(
        x="Sex",
        hue="Survived",
        data=df,
        palette="coolwarm",
        ax=ax2
    )

    ax2.set_title("Survival by Gender")

    st.pyplot(fig2)

    # Passenger Class
    fig3, ax3 = plt.subplots(figsize=(8, 5))

    sns.countplot(
        x="Pclass",
        hue="Survived",
        data=df,
        palette="viridis",
        ax=ax3
    )

    ax3.set_title("Survival by Passenger Class")

    st.pyplot(fig3)

    # Age Distribution
    fig4, ax4 = plt.subplots(figsize=(8, 5))

    sns.histplot(
        df["Age"],
        bins=30,
        kde=True,
        ax=ax4
    )

    ax4.set_title("Age Distribution")

    st.pyplot(fig4)

    # Fare Distribution
    fig5, ax5 = plt.subplots(figsize=(8, 5))

    sns.histplot(
        df["Fare"],
        bins=30,
        kde=True,
        ax=ax5
    )

    ax5.set_title("Fare Distribution")

    st.pyplot(fig5)

# =========================================
# HEATMAP
# =========================================

elif menu == "Heatmap":

    st.subheader("🔥 Correlation Heatmap")

    numeric_df = df.select_dtypes(include=np.number)

    fig6, ax6 = plt.subplots(figsize=(12, 8))

    sns.heatmap(
        numeric_df.corr(),
        annot=True,
        cmap="coolwarm",
        ax=ax6
    )

    st.pyplot(fig6)

# =========================================
# MACHINE LEARNING
# =========================================

elif menu == "Machine Learning":

    # ENCODING
    label = LabelEncoder()

    df["Sex"] = label.fit_transform(df["Sex"])
    df["Embarked"] = label.fit_transform(df["Embarked"])
    df["Title"] = label.fit_transform(df["Title"])

    # DROP COLUMNS
    df.drop(
        ["PassengerId", "Name", "Ticket"],
        axis=1,
        inplace=True
    )

    # FEATURES & TARGET
    X = df.drop("Survived", axis=1)

    y = df["Survived"]

    # SPLIT
    X_train, X_test, y_train, y_test = train_test_split(
        X,
        y,
        test_size=0.2,
        random_state=42
    )

    st.subheader("🤖 Machine Learning Models")

    # Logistic Regression
    lr_model = LogisticRegression(max_iter=1000)

    lr_model.fit(X_train, y_train)

    lr_pred = lr_model.predict(X_test)

    lr_acc = accuracy_score(y_test, lr_pred)

    # Decision Tree
    dt_model = DecisionTreeClassifier(
        max_depth=5,
        random_state=42
    )

    dt_model.fit(X_train, y_train)

    dt_pred = dt_model.predict(X_test)

    dt_acc = accuracy_score(y_test, dt_pred)

    # Random Forest
    rf_model = RandomForestClassifier(
        n_estimators=100,
        random_state=42
    )

    rf_model.fit(X_train, y_train)

    rf_pred = rf_model.predict(X_test)

    rf_acc = accuracy_score(y_test, rf_pred)

    st.success(f"Logistic Regression Accuracy: {lr_acc:.2f}")

    st.success(f"Decision Tree Accuracy: {dt_acc:.2f}")

    st.success(f"Random Forest Accuracy: {rf_acc:.2f}")

    # Accuracy Chart
    model_df = pd.DataFrame({
        "Model": [
            "Logistic Regression",
            "Decision Tree",
            "Random Forest"
        ],
        "Accuracy": [
            lr_acc,
            dt_acc,
            rf_acc
        ]
    })

    fig7, ax7 = plt.subplots(figsize=(8, 5))

    sns.barplot(
        x="Model",
        y="Accuracy",
        data=model_df,
        palette="Set1",
        ax=ax7
    )

    ax7.set_title("Model Accuracy Comparison")

    st.pyplot(fig7)

# =========================================
# FEATURE IMPORTANCE
# =========================================

elif menu == "Feature Importance":

    label = LabelEncoder()

    df["Sex"] = label.fit_transform(df["Sex"])
    df["Embarked"] = label.fit_transform(df["Embarked"])
    df["Title"] = label.fit_transform(df["Title"])

    df.drop(
        ["PassengerId", "Name", "Ticket"],
        axis=1,
        inplace=True
    )

    X = df.drop("Survived", axis=1)

    y = df["Survived"]

    rf_model = RandomForestClassifier(
        n_estimators=100,
        random_state=42
    )

    rf_model.fit(X, y)

    importance = pd.DataFrame({
        "Feature": X.columns,
        "Importance": rf_model.feature_importances_
    })

    importance = importance.sort_values(
        by="Importance",
        ascending=False
    )

    st.subheader("⭐ Feature Importance")

    fig8, ax8 = plt.subplots(figsize=(10, 6))

    sns.barplot(
        x="Importance",
        y="Feature",
        data=importance,
        palette="viridis",
        ax=ax8
    )

    ax8.set_title("Feature Importance")

    st.pyplot(fig8)

# =========================================
# FOOTER
# =========================================

st.markdown("---")

st.success("Titanic Survival Analysis Project Completed Successfully!")