import streamlit as st
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.metrics import accuracy_score, classification_report, roc_auc_score
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.svm import SVC
from imblearn.over_sampling import SMOTE
from sklearn.metrics import roc_curve, auc
from sklearn.preprocessing import label_binarize
import matplotlib.pyplot as plt

# Fungsi untuk mengunggah dan memproses dataset PKPA dan BAAK
def process_pkpa(file_name):
    excel_file = pd.ExcelFile(file_name)
    if "Rekap" in excel_file.sheet_names:
        df = pd.read_excel(file_name, sheet_name="Rekap", usecols="C,D,E,T,AC", skiprows=1)
        df.columns = ["Kode Progdi", "nim", "nama", "pekerjaan", "ketereratan"]
        df = df.dropna()
        df = df[~df['ketereratan'].astype(str).str.strip().isin(["0", "4", "5"])]  # Filter kategori yang tidak diperlukan
        df = df[~df['Kode Progdi'].isin(['01', '02', '03'])]  # Hapus kode prodi tertentu
        df = df[df['nim'].astype(str).str.isdigit() & (df['nim'].astype(str).str.len() == 9)]
        return df[["nim", "ketereratan"]]
    return pd.DataFrame()

def process_baak(file_name):
    excel_file = pd.ExcelFile(file_name)
    processed_sheets = []
    for sheet in excel_file.sheet_names:
        df = pd.read_excel(file_name, sheet_name=sheet, usecols="B,C,D,E", skiprows=1)
        df.columns = ["nim", "nama", "lama studi", "ipk"]
        df = df.dropna()
        df = df[~df['nim'].apply(lambda x: str(x)[4:6] in ['01', '02', '03'])]  # Hapus nim dengan kode tertentu
        df = df[df["lama studi"] >= 3]  # Pastikan lama studi >= 3 tahun
        processed_sheets.append(df)
    return pd.concat(processed_sheets, ignore_index=True) if processed_sheets else pd.DataFrame()

# Mengunggah dataset melalui Streamlit
uploaded_pkpa = st.file_uploader("Unggah Dataset PKPA", type=["xlsx"], accept_multiple_files=True)
uploaded_baak = st.file_uploader("Unggah Dataset BAAK", type=["xlsx"], accept_multiple_files=True)

if uploaded_pkpa and uploaded_baak:
    df_pkpa = pd.concat([process_pkpa(f) for f in uploaded_pkpa], ignore_index=True)
    df_baak = pd.concat([process_baak(f) for f in uploaded_baak], ignore_index=True)

    # Gabungkan dataset
    df_merged = df_pkpa.merge(df_baak, on="nim", how="left")

    # Proses kategori dan data untuk analisis
    df_merged["lama studi kategori"] = df_merged["lama studi"].apply(lambda x: 1 if x <= 4 else (2 if 4.1 <= x <= 4.5 else 3))
    df_merged["ipk kategori"] = df_merged["ipk"].apply(lambda x: 3 if x < 3 else (2 if 3 <= x <= 3.5 else 1))
    dataset = df_merged.drop(columns=["nim"])

    # Pilih fitur dan filter sampel
    data_uji = dataset[['lama studi kategori', 'ipk kategori', 'ketereratan']].copy()
    data_uji.rename(columns={'lama studi kategori': 'lama studi', 'ipk kategori': 'ipk'}, inplace=True)
    data_uji_filtered = data_uji.groupby('ketereratan', group_keys=False).apply(lambda x: x.sample(min(len(x), 100))).reset_index(drop=True)

    # Label encoding
    le_ipk = LabelEncoder()
    le_studi = LabelEncoder()
    le_ketereratan = LabelEncoder()
    data_uji_filtered['ipk'] = le_ipk.fit_transform(data_uji_filtered['ipk'].astype(str))
    data_uji_filtered['lama studi'] = le_studi.fit_transform(data_uji_filtered['lama studi'].astype(str))
    data_uji_filtered['ketereratan'] = le_ketereratan.fit_transform(data_uji_filtered['ketereratan'].astype(str))

    # Pembagian data
    X = data_uji_filtered[['ipk', 'lama studi']]
    y = data_uji_filtered['ketereratan']
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    # SMOTE untuk Imbalanced Data
    smote = SMOTE(random_state=42)
    X_resampled, y_resampled = smote.fit_resample(X_train, y_train)

    # Normalisasi
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_resampled)
    X_test_scaled = scaler.transform(X_test)

    # Pilihan menu algoritma
    algorithm = st.selectbox("Pilih Algoritma", ("Decision Tree", "Random Forest", "SVM"))

    # Model Decision Tree
    if algorithm == "Decision Tree":
        model = DecisionTreeClassifier(random_state=42)
        model.fit(X_train_scaled, y_resampled)
        y_pred = model.predict(X_test_scaled)
        accuracy = accuracy_score(y_test, y_pred)
        st.write(f"Akurasi Model Decision Tree: {accuracy}")
        st.write("Laporan Klasifikasi:\n", classification_report(y_test, y_pred, target_names=le_ketereratan.classes_))

    # Model Random Forest
    elif algorithm == "Random Forest":
        model = RandomForestClassifier(random_state=42)
        model.fit(X_train_scaled, y_resampled)
        y_pred = model.predict(X_test_scaled)
        accuracy = accuracy_score(y_test, y_pred)
        st.write(f"Akurasi Model Random Forest: {accuracy}")
        st.write("Laporan Klasifikasi:\n", classification_report(y_test, y_pred, target_names=le_ketereratan.classes_))

    # Model SVM
    elif algorithm == "SVM":
        model = SVC(kernel='rbf', probability=True, random_state=42)
        model.fit(X_train_scaled, y_resampled)
        y_pred = model.predict(X_test_scaled)
        accuracy = accuracy_score(y_test, y_pred)
        y_proba = model.predict_proba(X_test_scaled)
        roc_auc = roc_auc_score(y_test, y_proba, multi_class='ovr')
        st.write(f"Akurasi Model SVM: {accuracy}")
        st.write(f"ROC AUC Score: {roc_auc}")
        st.write("Laporan Klasifikasi:\n", classification_report(y_test, y_pred, target_names=le_ketereratan.classes_))

        # Plot ROC Curve
        y_test_binarized = label_binarize(y_test, classes=np.unique(y))
        fpr, tpr, roc_auc = {}, {}, {}
        n_classes = len(le_ketereratan.classes_)
        for i in range(n_classes):
            fpr[i], tpr[i], _ = roc_curve(y_test_binarized[:, i], y_proba[:, i])
            roc_auc[i] = auc(fpr[i], tpr[i])
        
        plt.figure(figsize=(8, 6))
        colors = ['blue', 'red', 'green', 'orange', 'purple']
        for i, color in zip(range(n_classes), colors):
            plt.plot(fpr[i], tpr[i], color=color, lw=2,
                     label=f'Kelas {le_ketereratan.classes_[i]} (AUC = {roc_auc[i]:.2f})')
        plt.plot([0, 1], [0, 1], 'k--', lw=2)
        plt.xlabel('False Positive Rate')
        plt.ylabel('True Positive Rate')
        plt.title('Kurva ROC-AUC untuk Model SVM')
        plt.legend(loc='lower right')
        plt.grid()
        st.pyplot()
