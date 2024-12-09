# Install Streamlit di terminal jika belum diinstal:
# pip install streamlit

import streamlit as st
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeClassifier, export_text
from sklearn.ensemble import RandomForestClassifier
from sklearn.svm import SVC
from sklearn.metrics import classification_report, confusion_matrix

# ====================== Fungsi Pendukung ======================

# Transformasi data
def transform_data(df):
    df['lama studi'] = df['lama studi'].apply(lambda x: "Tepat Waktu" if x <= 4 else "Tidak Tepat")
    df['ipk'] = df['ipk'].apply(lambda x: "Rendah" if x < 3 else ("Sedang" if 3 <= x <= 3.5 else "Tinggi"))
    ketereratan_mapping = {1: "Sangat Erat", 2: "Erat", 3: "Cukup", 4: "Kurang", 5: "Tidak"}
    df['ketereratan'] = df['ketereratan'].map(ketereratan_mapping)
    return df

# Data preparation
def prepare_data(df, features, target):
    X = pd.get_dummies(df[features])  # Encoding categorical variables
    y = df[target]
    return train_test_split(X, y, test_size=0.3, random_state=42)

# Model training and evaluation
def evaluate_model(model, X_train, X_test, y_train, y_test):
    model.fit(X_train, y_train)
    y_pred = model.predict(X_test)
    st.write("### Classification Report")
    st.text(classification_report(y_test, y_pred))
    st.write("### Confusion Matrix")
    st.write(confusion_matrix(y_test, y_pred))

# ====================== Streamlit App ======================

st.title("Pengolahan Dataset & Analisis Algoritma Data Mining")
st.write("Unggah dataset Excel dan lakukan analisis menggunakan algoritma Decision Tree, Random Forest, dan SVM.")

# Upload dataset
uploaded_file = st.file_uploader("Unggah File Excel", type=["xls", "xlsx"])
if uploaded_file:
    # Read and process Excel
    df = pd.read_excel(uploaded_file)
    st.write("### Data Awal")
    st.dataframe(df.head())

    # Transform and display data
    df = transform_data(df)
    st.write("### Data Setelah Transformasi")
    st.dataframe(df.head())

    # Select target and features
    target = st.selectbox("Pilih Kolom Target (Ketereratan)", df.columns)
    features = st.multiselect("Pilih Kolom Fitur", [col for col in df.columns if col != target])

    if target and features:
        # Prepare data
        X_train, X_test, y_train, y_test = prepare_data(df, features, target)

        # Select algorithm
        algorithm = st.selectbox("Pilih Algoritma", ["Decision Tree", "Random Forest", "SVM"])

        if algorithm == "Decision Tree":
            st.write("### Analisis Decision Tree")
            model = DecisionTreeClassifier(random_state=42)
            evaluate_model(model, X_train, X_test, y_train, y_test)
            st.text(export_text(model, feature_names=X_train.columns.tolist()))

        elif algorithm == "Random Forest":
            st.write("### Analisis Random Forest")
            model = RandomForestClassifier(random_state=42, n_estimators=100)
            evaluate_model(model, X_train, X_test, y_train, y_test)
            st.write("### Fitur Penting")
            feature_importances = model.feature_importances_
            st.write({name: importance for name, importance in zip(X_train.columns, feature_importances)})

        elif algorithm == "SVM":
            kernel = st.selectbox("Pilih Kernel untuk SVM", ["linear", "rbf"])
            st.write(f"### Analisis SVM ({kernel} kernel)")
            model = SVC(kernel=kernel, random_state=42)
            evaluate_model(model, X_train, X_test, y_train, y_test)

# Jalankan aplikasi ini dengan `streamlit run script_name.py`
