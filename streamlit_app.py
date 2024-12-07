import pandas as pd
import streamlit as st

# Fungsi untuk memproses dataset PKPA
def process_pkpa(file):
    try:
        # Membaca sheet 'Rekap'
        excel_file = pd.ExcelFile(file)
        if "Rekap" in excel_file.sheet_names:
            df = pd.read_excel(file, sheet_name="Rekap", usecols="C,D,E,T,AC", skiprows=1)
            df.columns = ["Kode Progdi", "nim", "nama", "pekerjaan", "ketereratan"]

            # Menghapus baris dengan missing values atau jika kolom D tidak berisi 9 karakter
            df = df.dropna()
            df = df[df['nim'].apply(lambda x: len(str(x)) == 9)]  # Memastikan nim memiliki 9 karakter

            # Menghapus baris yang berisi 01, 02, 03 pada kolom C
            df = df[~df['Kode Progdi'].isin(['01', '02', '03'])]
            
            return df
        else:
            st.error(f"Sheet 'Rekap' tidak ditemukan dalam file {file.name}")
            return None
    except Exception as e:
        st.error(f"Error saat memproses {file.name}: {e}")
        return None

# Fungsi untuk memproses dataset BAAK
def process_baak(file):
    try:
        excel_file = pd.ExcelFile(file)
        processed_data_baak = []
        for sheet in excel_file.sheet_names:
            df = pd.read_excel(file, sheet_name=sheet, usecols="B,C,E,F", skiprows=1)
            df.columns = ["nim", "nama", "ipk", "lama studi"]
            df = df.dropna()
            df = df[~df['nim'].apply(lambda x: str(x)[4:6] in ['01', '02', '03'])]
            processed_data_baak.append(df)
        return pd.concat(processed_data_baak, ignore_index=True)
    except Exception as e:
        st.error(f"Error saat memproses {file.name}: {e}")
        return None

# Fungsi untuk menggabungkan kedua dataset berdasarkan kolom 'nim'
def join_data(df_pkpa, df_baak):
    if df_pkpa is not None and df_baak is not None:
        # Melakukan join berdasarkan kolom 'nim'
        joined_df = pd.merge(df_pkpa, df_baak, on="nim", how="inner")

        # Memilih kolom yang diinginkan: ipk, lama studi, ketereratan
        filtered_df = joined_df[["ipk", "lama studi", "ketereratan"]]

        # Menghapus baris di mana kolom 'ketereratan' bernilai 0
        filtered_df = filtered_df[filtered_df["ketereratan"] != 0]

        return filtered_df
    else:
        st.error("Tidak ada data yang cukup untuk dilakukan join.")
        return None

# Streamlit app layout
st.title("Data Preprocessing for PKPA and BAAK")

st.sidebar.header("Upload Files")

# File Upload
uploaded_pkpa = st.sidebar.file_uploader("Upload PKPA File", type=["xlsx", "xls"])
uploaded_baak = st.sidebar.file_uploader("Upload BAAK File", type=["xlsx", "xls"])

# Process and Display Data
if uploaded_pkpa and uploaded_baak:
    st.write("Memproses Data PKPA...")
    df_pkpa = process_pkpa(uploaded_pkpa)

    st.write("Memproses Data BAAK...")
    df_baak = process_baak(uploaded_baak)

    st.write("Melakukan Join Data PKPA dan BAAK berdasarkan kolom 'nim'...")
    result_df = join_data(df_pkpa, df_baak)

    if result_df is not None:
        st.write("Hasil Join:")
        st.dataframe(result_df)  # Menampilkan hasil join

        # Simpan hasil join
        st.download_button(
            label="Unduh Hasil Join sebagai CSV",
            data=result_df.to_csv(index=False),
            file_name="joined_data.csv",
            mime="text/csv"
        )
        st.download_button(
            label="Unduh Hasil Join sebagai XLSX",
            data=result_df.to_excel(index=False, engine='xlsxwriter'),
            file_name="joined_data.xlsx",
            mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet"
        )
else:
    st.info("Unggah file PKPA dan BAAK untuk memulai pemrosesan.")
