import streamlit as st
import pandas as pd
import joblib
import os

# Konfigurasi Path yang aman untuk sistem Linux di Cloud
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
MODEL_PATH = os.path.join(BASE_DIR, 'model', 'best_model_student_dropout.joblib')
SCALER_PATH = os.path.join(BASE_DIR, 'model', 'scaler_model.joblib')

# Fungsi Load Assets dengan Cache agar hemat RAM
@st.cache_resource
def load_assets():
    if not os.path.exists(MODEL_PATH) or not os.path.exists(SCALER_PATH):
        return None, None
    try:
        model = joblib.load(MODEL_PATH)
        scaler = joblib.load(SCALER_PATH)
        return model, scaler
    except:
        return None, None

st.set_page_config(page_title="Prediksi Dropout", layout="centered")

def main():
    st.title("🎓 Student Dropout Predictor")
    st.info("Jaya Jaya Institut - Sistem Analisis Prediksi")

    # Muat Model
    model, scaler = load_assets()
    if model is None:
        st.error("❌ Model gagal dimuat. Pastikan folder 'model' berisi file .joblib sudah di-upload ke GitHub.")
        return

    # Upload File
    uploaded_file = st.file_uploader("Unggah file data siswa (.csv, .xlsx)", type=["csv", "xlsx"])

    if uploaded_file:
        try:
            # Baca data
            df = pd.read_csv(uploaded_file) if uploaded_file.name.endswith('.csv') else pd.read_excel(uploaded_file)
            
            # Kolom wajib
            features = [
                'Tuition_fees_up_to_date', 'Scholarship_holder', 'Age_at_enrollment',
                'Curricular_units_1st_sem_approved', 'Curricular_units_2nd_sem_approved',
                'Curricular_units_2nd_sem_grade', 'Debtor', 'Gender', 'Academic_Trend', 'GDP'
            ]

            if all(col in df.columns for col in features):
                if st.button("🚀 Jalankan Analisis"):
                    with st.spinner('Sedang menghitung...'):
                        # Preprocessing & Prediksi
                        input_data = df[features].fillna(0)
                        data_scaled = scaler.transform(input_data)
                        preds = model.predict(data_scaled)

                        # Mapping Label
                        label_map = {0: 'Dropout', 1: 'Enrolled', 2: 'Graduate'}
                        df['Hasil_Prediksi'] = [label_map.get(p, 'Unknown') for p in preds]

                        # Tampilkan Ringkasan
                        st.success("✅ Analisis Selesai!")
                        col1, col2 = st.columns(2)
                        col1.metric("Total Siswa", len(df))
                        col2.metric("Potensi Dropout", len(df[df['Hasil_Prediksi'] == 'Dropout']))

                        # Preview Data (Dibatasi 100 baris pertama agar ringan)
                        st.write("### Preview Hasil (100 Data Pertama)")
                        st.dataframe(df[['Hasil_Prediksi'] + features].head(100), use_container_width=True)

                        # Tombol Download
                        csv = df.to_csv(index=False).encode('utf-8')
                        st.download_button("📥 Download Hasil Lengkap (.csv)", data=csv, file_name="hasil_prediksi.csv", mime="text/csv")
            else:
                st.warning("⚠️ File tidak memiliki kolom yang sesuai. Pastikan data sesuai format.")
        
        except Exception as e:
            st.error(f"Terjadi kesalahan: {e}")

if __name__ == "__main__":
    main()
