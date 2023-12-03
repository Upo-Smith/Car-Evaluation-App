import streamlit as st
import numpy as np
import pickle
import pandas as pd

st.write("""# Car Evaluation""")

tab1, tab2, tab3 = st.tabs(["Data", "Preprocessing Data", "Implementasi"])

with tab1:
    def input():
        
        buying_map = {'murah': 0, 'medium': 1, 'mahal': 2, 'sangat mahal': 3}
        maint_map = {'murah': 0, 'medium': 1, 'mahal': 2, 'sangat mahal': 3}
        doors_map = {2: 0, 3: 1, 4: 2, '5-lebih': 3}
        persons_map = {2: 0, 4: 1, 'lebih': 2}
        lug_boot_map = {'kecil': 0, 'medium': 1, 'besar': 2}
        safety_map = {'rendah': 0, 'medium': 1, 'tinggi': 2}

        buying = st.select_slider('Harga beli mobil', options=['murah','medium','mahal','sangat mahal'])
        maint = st.select_slider('Harga perbaikan (maintenance) mobil', options=['murah','medium','mahal','sangat mahal'])
        doors = st.select_slider('Jumlah pintu mobil', options=[2,3,4,'5-lebih'])
        persons = st.select_slider('Jumlah orang yang bisa dimuat dalam mobil', options=[2,4,'lebih'])
        lug_boot = st.select_slider('Ukuran bagasi mobil', options=['kecil','medium','besar'])
        safety = st.select_slider('Tingkat keselamatan mobil', options=['rendah', 'medium', 'tinggi'])
        
        buying_numeric = buying_map[buying]
        maint_numeric = maint_map[maint]
        doors_numeric = doors_map[doors]
        persons_numeric = persons_map[persons]
        lug_boot_numeric = lug_boot_map[lug_boot]
        safety_numeric = safety_map[safety]

        data = {
        'buying': buying_numeric,
        'maint': maint_numeric,
        'doors': doors_numeric,
        'persons': persons_numeric,
        'lug_boot': lug_boot_numeric,
        'safety': safety_numeric}

        fitur = pd.DataFrame(data, index=[0])
        return fitur

    data = input()

    st.write(data)

with tab2:
    mm = pickle.load(open('scaler.pkl', 'rb'))

    data_sc = mm.transform(data)
    
    st.write('Hasil Preprocessing menggunakan Min-Max Scaler dari data tersebut adalah')
    st.write(data_sc)

with tab3:
    treeClf = pickle.load(open('treeClf.pkl', 'rb'))

    st.write("Menggunakan Model Decision Tree (99%), maka hasil prediksi dari data tersebut:")
    
    type = np.array(['Tidak dapat diterima', 'Dapat diterima', 'Bagus', 'Sangat Bagus'])

    
    predict1 = treeClf.predict(data_sc)
    st.write(type[predict1])
