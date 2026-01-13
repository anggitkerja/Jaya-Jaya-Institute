import streamlit as st
import pandas as pd
import joblib
import os

# Set lokasi file model dan scaler
MODEL_PATH = './model/best_model_student_dropout.joblib'
SCALER_PATH = './model/scaler_model.joblib'

@st.cache_resource
def load_assets():
    model = joblib.load(MODEL_PATH)
    scaler = joblib.load(SCALER_PATH)
    return model, scaler

st.set_page_config(page_title="Prediksi Dropout Siswa", layout="wide")

st.markdown("""
    <style>
    .main { background-color: #f5f7f9; }
    .title-text { padding-top: 10px; }
    </style>
    """, unsafe_allow_html=True)

_, col_mid, _ = st.columns([1, 2, 1])

with col_mid:
    header_col1, header_col2 = st.columns([1, 6])
    
    with header_col1:
        st.image("https://img.icons8.com/fluency/96/graduation-cap.png", width=70)
        
    with header_col2:
        st.markdown('<div class="title-text"><h1>Student Dropout App</h1></div>', unsafe_allow_html=True)
    
    st.info("Sistem Prediksi Kelulusan Siswa - Jaya Jaya Institut")
    
    try:
        best_model, scaler = load_assets()
    except Exception as e:
        st.error(f"Gagal memuat model: {e}")
        st.stop()

    st.subheader("📤 Unggah Data Siswa")
    uploaded_file = st.file_uploader(
        "Pilih file dengan format .csv, .xlsx, atau .xls", 
        type=["csv", "xlsx", "xls"]
    )

    if uploaded_file:
        file_extension = os.path.splitext(uploaded_file.name)[1].lower()
        raw_filename = os.path.splitext(uploaded_file.name)[0]
        
        try:
            if file_extension == '.csv':
                df = pd.read_csv(uploaded_file)
            else:
                df = pd.read_excel(uploaded_file)
            
            features = [
                'Tuition_fees_up_to_date', 'Scholarship_holder', 'Age_at_enrollment',
                'Curricular_units_1st_sem_approved', 'Curricular_units_2nd_sem_approved',
                'Curricular_units_2nd_sem_grade', 'Debtor', 'Gender', 'Academic_Trend', 'GDP'
            ]

            if all(col in df.columns for col in features):
                if st.button("🚀 Mulai Analisis Prediksi"):
                    with st.spinner('Menganalisis data...'):
                        data_scaled = scaler.transform(df[features].fillna(0))
                        predictions = best_model.predict(data_scaled)
                        
                        status_labels = {0: 'Dropout', 1: 'Enrolled', 2: 'Graduate'}
                        df['Status_Prediction'] = [status_labels.get(p) for p in predictions]
                        
                        df.index = range(1, len(df) + 1)
                        df.index.name = 'No'

                        st.success("✅ Analisis Selesai!")
                        
                        c1, col_space, c2 = st.columns([1, 0.5, 1])
                        c1.metric("Total Data", len(df))
                        c2.metric("Potensi Dropout", len(df[df['Status_Prediction'] == 'Dropout']))

                        st.write("### Detail Hasil Prediksi")
                        display_cols = ['Status_Prediction'] + [c for c in df.columns if c != 'Status_Prediction']
                        st.dataframe(df[display_cols], use_container_width=True)
                        
                        csv_output = df.to_csv(index=True).encode('utf-8')
                        st.download_button(
                            label="📥 Download Hasil Prediksi (.csv)",
                            data=csv_output,
                            file_name=f"prediksi_{raw_filename}.csv",
                            mime="text/csv"
                        )
            else:
                st.error("⚠️ Format kolom tidak sesuai! Pastikan file memiliki kolom fitur yang diperlukan.")
                
        except Exception as e:
            st.error(f"Terjadi kesalahan saat membaca file: {e}")

    st.markdown("---")
    st.caption("© 2026 Jaya Jaya Institut - Business Intelligence Unit")