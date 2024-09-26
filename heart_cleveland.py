import pickle
import numpy as np
import streamlit as st

# Judul Web
st.title('PREDIKSI PENYAKIT JANTUNG DENGAN DATA MINING')

# Membaca Model
penyakitJantung_model = pickle.load(open('svc_model.sav', 'rb'))

# Membuat Kolom Input
col1, col2, col3 = st.columns(3)

with col1:
    age = st.number_input('Input Nilai age')
with col2:
    sex = st.number_input('Input Nilai sex')
with col3:
    cp = st.number_input('Input Nilai cp')
with col1:
    trestbps = st.number_input('Input Nilai trestbps')
with col2:
    chol = st.number_input('Input Nilai chol')
with col3:
    fbs = st.number_input('Input Nilai fbs')
with col1:
    restecg = st.number_input('Input Nilai restecg')
with col2:
    thalach = st.number_input('Input Nilai thalach')
with col3:
    exang = st.number_input('Input Nilai exang')
with col1:
    oldpeak = st.number_input('Input Nilai oldpeak')
with col2:
    slope = st.number_input('Input Nilai slope')
with col3:
    ca = st.number_input('Input Nilai ca')
with col1:
    thal = st.number_input('Input Nilai thal')

# Prediksi Berdasarkan Input
if st.button('Prediksi'):
    # Menyiapkan data input
    input_data = (age, sex, cp, trestbps, chol, fbs, restecg, thalach, exang, oldpeak, slope, ca, thal)
    input_data_as_numpy_array = np.array(input_data).reshape(1, -1)

    # Normalisasi input data (jika diperlukan)
    # scaler = ... # load scaler yang sesuai
    # input_data_scaled = scaler.transform(input_data_as_numpy_array)

    # Prediksi
    prediction = penyakitJantung_model.predict(input_data_as_numpy_array)

    if prediction[0] == 0:
        st.success('Pasien Tidak Terkena Penyakit Jantung')
    else:
        st.error('Pasien Terkena Penyakit Jantung')
