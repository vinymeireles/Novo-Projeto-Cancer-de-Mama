import numpy as np
import streamlit as st
import pandas as pd
import plotly.express as px
from plotly.subplots import make_subplots
from plotly import graph_objs as go
import matplotlib.pyplot as plt

#Apps
st.set_page_config(page_title="App Previsão de Câncer de Mama", page_icon= ":bar_chart:")
st.header("🎗Analytics Câncer de Mama📊", divider='violet')

st.sidebar.image("img/outubro_rosa.png", width=300)

# Style
with open('style.css')as f:
    st.markdown(f"<style>{f.read()}</style>", unsafe_allow_html = True)

#Texto na página
st.markdown("""<h6 style='text-align: justify;'> Essa aplicação demonstra uma análise exploratória dos dados de previsão de sobrevivência de pacientes com câncer de mama num conjunto de dados disponível no [www.kaggle.com]. Essa aplicação utilizando Machine Learning - IA tem o intuito de demonstrar graficamente as correlações e previsões de pacientes que possam a sobreviver ao tratamento da doença. </h6""", unsafe_allow_html=True) 

st.markdown("""<h6 style='text-align: justify;'> A utilização de Machine Learning para a detecção do câncer de mama vem crescendo cada vez mais e contribuindo para diagnósticos mais rápidos e precisos. De acordo com a Sociedade Brasileira de Mastologia, uma em cada 12 mulheres terá um tumor nas mamas até os 90 anos. Infelizmente, o câncer de mama é a principal causa de morte entre as mulheres, de todos os diferentes tipos de câncer. 
Uma das principais características do câncer de mama é que quanto mais precoce for o seu diagnóstico, maiores são as chances de tratamento. Entretanto, uma pesquisa realizada revelou que mais de 3,8 milhões de mulheres na faixa de 50 a 69 anos nunca haviam feito autoexame ou mamografia. A fim de aumentar a conscientização a respeito da prevenção e diagnóstico precoce, há, todo ano, a campanha Outubro Rosa, que visa alertar principalmente as mulheres sobre esta causa. </h6""", unsafe_allow_html=True)
st.markdown("""<h6 style='text-align: justify;'> 🌍Fonte: [https://sigmoidal.ai/machine-learning-para-a-deteccao-de-cancer-de-mama]. </h6""", unsafe_allow_html=True) 
col1, col2 = st.columns(2)
with col1:
    st.image("img/rosa.jpg", width= 330)
with col2:    
    st.markdown("""<h6 style='text-align: justify;'>Contribuindo com essa conscientização e com a campanha Outubro Rosa, preparei uma aplicação em Data Science onde construí um modelo de Machine Learning capaz de detectar e prever numeros sobrevivência de paciente com câncer de mama. Com o uso de dados obtidos em sites governamentais na área da saúde é possuir utilizar a Inteligência Artificial e prever casos em pacientes que podem ou não sobreviver a essa doença maligna.</h6""", unsafe_allow_html=True)

st.subheader("📋 Informações sobre o câncer de mama:", divider='violet')
st.markdown("""<h6 style='text-align: justify;'> O câncer de mama é o tipo que mais acomete mulheres em todo o mundo, tanto em países em desenvolvimento quanto em países desenvolvidos. Cerca de 2,3 milhões de casos novos foram estimados para o ano de 2020 em todo o mundo, o que representa cerca de 24,5% de todos os tipos de neoplasias diagnosticadas nas mulheres. As taxas de incidência variam entre as diferentes regiões do planeta, com as maiores taxas nos países desenvolvidos. Para o Brasil, foram estimados 66.280 casos novos de câncer de mama em 2021, com um risco estimado de 61,61 casos a cada 100 mil mulheres.
O câncer de mama também ocupa a primeira posição em mortalidade por câncer entre as mulheres no Brasil, com taxa de mortalidade ajustada por idade, pela população mundial, para 2019, de 14,23/100 mil. As maiores taxas de incidência e de mortalidade estão nas regiões Sul e Sudeste do Brasil. Os principais sinais e sintomas suspeitos de câncer de mama são: caroço (nódulo), geralmente endurecido, fixo e indolor; pele da mama avermelhada ou parecida com casca de laranja, alterações no bico do peito (mamilo) e saída espontânea de líquido de um dos mamilos. Também podem aparecer pequenos nódulos no pescoço ou na região embaixo dos braços (axilas).</h6""", unsafe_allow_html=True)

st.markdown("""<h4 style='text-align: justify;'>  Fatores de risco: </h4""", unsafe_allow_html=True)
st.markdown("""<h6 style='text-align: justify;'> Não há uma causa única para o câncer de mama. Diversos fatores estão relacionados ao desenvolvimento da doença entre as mulheres, como: envelhecimento, determinantes relacionados à vida reprodutiva da mulher, histórico familiar de câncer de mama, consumo de álcool, excesso de peso, atividade física insuficente e exposição à radiação ionizante. Os principais fatores são:</h6""", unsafe_allow_html=True)
st.markdown("""<h5 style='text-align: justify;'> ❎ Os principais fatores são:</h5""", unsafe_allow_html=True)
st.markdown("")
st.markdown("""<h5 style='text-align: justify;'> 1️⃣ Comportamentais/Ambientais: </h5""", unsafe_allow_html=True)
st.markdown("""<h6 style='text-align: justify;'> 🟣Obesidade e sobrepeso, após a menopausa. </h6""", unsafe_allow_html=True)
st.markdown("""<h6 style='text-align: justify;'> 🟣Atividade física insuficiente (menos de 150 minutos de atividade física moderada por semana). </h6""", unsafe_allow_html=True)
st.markdown("""<h6 style='text-align: justify;'> 🟣Consumo de bebida alcoólica. </h6""", unsafe_allow_html=True)
st.markdown("""<h6 style='text-align: justify;'> 🟣Exposição frequente a radiações ionizantes (Raios-X, tomografia computadorizada, mamografia). </h6""", unsafe_allow_html=True)
st.markdown("""<h6 style='text-align: justify;'> 🟣História de tratamento prévio com radioterapia no tórax. </h6""", unsafe_allow_html=True)
st.markdown("")

st.markdown("""<h5 style='text-align: justify;'> 2️⃣ Aspectos da vida reprodutiva/hormonais: </h5""", unsafe_allow_html=True)
st.markdown("""<h6 style='text-align: justify;'> 🟣Primeira menstruação (menarca) antes de 12 anos. </h6""", unsafe_allow_html=True)
st.markdown("""<h6 style='text-align: justify;'> 🟣Não ter filhos. </h6""", unsafe_allow_html=True)
st.markdown("""<h6 style='text-align: justify;'> 🟣Primeira gravidez após os 30 anos. </h6""", unsafe_allow_html=True)
st.markdown("""<h6 style='text-align: justify;'> 🟣Parar de menstruar (menopausa) após os 55 anos. </h6""", unsafe_allow_html=True)
st.markdown("""<h6 style='text-align: justify;'> 🟣Uso de contraceptivos hormonais (estrogênio-progesterona). </h6""", unsafe_allow_html=True)
st.markdown("""<h6 style='text-align: justify;'> 🟣Ter feito terapia de reposição hormonal (estrogênio-progesterona), principalmente por mais de 5 anos. </h6""", unsafe_allow_html=True)
st.markdown("")

st.markdown("""<h5 style='text-align: justify;'> 3️⃣ Hereditários/Genéticos: </h5""", unsafe_allow_html=True)
st.markdown("""<h6 style='text-align: justify;'> 🟣Histórico familiar de câncer de ovário; de câncer de mama em mulheres, principalmente antes dos 50 anos; e caso de câncer de mama em homem. </h6""", unsafe_allow_html=True)
st.markdown("""<h6 style='text-align: justify;'> 🟣Alteração genética, especialmente nos genes BRCA1 e BRCA2. </h6""", unsafe_allow_html=True)
st.markdown("")
st.markdown("""<h6 style='text-align: justify;'> A mulher que possui esses fatores genéticos tem risco elevado para câncer de mama. </h6""", unsafe_allow_html=True)
st.markdown("")
st.markdown("")
st.markdown("""<h6 style='text-align: justify;'> 🌍 Fonte: [https://www.gov.br/inca/pt-br/assuntos/campanhas/2022/outubro-rosa] </h6""", unsafe_allow_html=True)

st.toast("Página atualizada!", icon='✅')

#### Logo sidebar######

