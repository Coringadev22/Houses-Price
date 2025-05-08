import streamlit as st
import pandas as pd
import numpy as np
import joblib  
from sklearn.preprocessing import StandardScaler

# Carregando modelo treinado
reg_rl = joblib.load('modelo_regressao.pkl')  

st.write("App carregado com sucesso.")
st.title('Previsão de Preço de Casas 🏠')

# Inputs do usuário (exemplo com poucos campos)
lot_area = st.number_input('Área do Lote (LotArea)', value=8450)
year_built = st.number_input('Ano de Construção (YearBuilt)', value=2003)
overall_qual = st.slider('Qualidade Geral (OverallQual)', 1, 10, 5)
gr_liv_area = st.number_input('Área habitável acima do solo (GrLivArea)', value=1710)

# Botão de previsão
if st.button('Prever Preço'):
    # Monta os dados no mesmo formato que o modelo espera
    entrada = pd.DataFrame([{
        'LotArea': lot_area,
        'YearBuilt': year_built,
        'OverallQual': overall_qual,
        'GrLivArea': gr_liv_area
    }])
    
    # Previsão
    preco_previsto = reg_rl.predict(entrada)
    st.success(f'Preço estimado da casa: ${preco_previsto[0]:,.2f}')
