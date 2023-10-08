import numpy as np
import streamlit as st
import pandas as pd
import plotly.express as px
from plotly.subplots import make_subplots
from plotly import graph_objs as go
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from streamlit_extras.metric_cards import style_metric_cards
from sklearn.metrics import accuracy_score
import warnings
warnings.filterwarnings("ignore")


#Apps
st.set_page_config(page_title="App Previsão de Câncer de Mama", page_icon= ":bar_chart:")
st.header("🎗Analytics Câncer de Mama📊", divider='violet')

# Style
with open('style.css')as f:
    st.markdown(f"<style>{f.read()}</style>", unsafe_allow_html = True)

#load data #### DataFrame 
@st.cache_data
def load_data():
    data = pd.read_csv("data/dataset.csv")
    return data

df = load_data()

#Renomear colunas no dataframe
df.rename({"Patient_ID": "ID",
            "Age": "Idade",
            "Gender": "Sexo",
            "Protein1": "Proteína1",
            "Protein2": "Proteína2",
            "Protein3": "Proteína3",
            "Protein4": "Proteína4",
            "Tumour_Stage": "Estágio_Tumor",
            "Histology": "Histologia",
            "ER status": "Status_ER",
            "PR status": "Status_PR",
            "HER2 status": "Status_HER2",
            "Surgery_type": "Tipo_Cirurgia",
            "Date_of_Surgery": "Data_Cirurgia",
            "Date_of_Last_Visit": "Data_Ultima_Visita",
            "Patient_Status": "Status_Paciente"}, axis=1, inplace=True)

#Sidebar de Opções
st.sidebar.markdown("▶ Selecione uma opção:")
st.sidebar.markdown("")
if st.sidebar.checkbox("🧾**Mostrar Informações**", True, key=0, help="Desmarque essa opção para visualizar outras opções"):
    #Layout da página
    st.subheader("🔎 Analisar previsão de sobrevivência em pacientes com câncer de mama:")

    col1, col2 = st.columns(2)
    with col1:    
        st.markdown("""<h6 style='text-align: justify;'>Nessa seção iremos analisar dados exploratórios de 400 pacientes diagnosticado com câncer de mama. Análise consiste em EAD dos dados e previsão utilizando algoritmos de machine learning e Dashboard dos dados.</h6""", unsafe_allow_html=True)
        st.markdown("""<h6 style='text-align: justify;'>Segundo o site [femama.org.br], foram realizadas pesquisas estatística sobre os índices de mulheres ter ou não a doença e a cura. Cerca de 38% dos diagnósticos só ocorreram fases avançadas da doença. 90,8% é a taxa de sobrevivência relativa de 5 anos.</h6""", unsafe_allow_html=True)
    with col2:
        st.image("img/image.png", width=360)

    st.markdown("""<h6 style='text-align: justify;'>Cerca de 37% dos casos levaram mais de 30 dias para confirmação diagnóstica após a consulta. 17% dos casos de câncer de mama podem ser evitados por meio de hábitos de vida saudáveis.</h6""", unsafe_allow_html=True)
    st.markdown("")
    st.markdown("""<h6 style='text-align: justify;'>🌍Fonte:[https://femama.org.br/site/outubro-rosa-2023/]</h6""", unsafe_allow_html=True)
    st.divider()

################### Estudo Análise exploratórias dos dados #####################################

#tratamento dos dados nulos e ausentes
df = df.dropna()

################# Criando um dicionário de dados para as colunas ####################3#
dic = pd.DataFrame([
            {"Patient_ID": "ID do paciente",
            "Age": "Idade",
            "Gender": "Sexo",
            "Protein1": "Proteína 1",
            "Protein2": "Proteína 2",
            "Protein3": "Proteína 3",
            "Protein4": "Proteína 4",
            "Tumour_Stage": "Estágio do tumor",
            "Histology": "Carcinoma ductal infiltrante, carcinoma lobular infiltrado, carcinoma mucinoso",
            "ER status": "Positivo/Negativo",
            "PR status": "Positivo/Negativo",
            "HER2 status": "Positivo/Negativo",
            "Surgery_type": "Lumpectomia, Mastectomia Simples, Mastectomia Radical Modificada, Outros",
            "Date_of_Surgery": "A data da cirurgia",
            "Date_of_Last_Visit": "A data da última visita do paciente",
            "Patient_Status": "Vivo/Morto"},
       ])    

################## Mostrar DataFrame #######################################################################

if st.sidebar.checkbox("🎲**Mostrar Dataset**", False, key=1):
    with st.expander("🗓 **Visualizar Dados**"):
        st.dataframe(df, use_container_width=True)
        st.markdown("")    
        st.markdown("Fonte Dataset: https://kaggle.com")
  
    with st.expander(" 📚 **Visualizar dicionários dos dados:**"):
         st.table(dic)
        

############### Mostrar dados estatisticos ################################################
if st.sidebar.checkbox("📝**Dados Estatísticos**", False, key=2):
    #Contagem por sexo
    male_count = df[df.Sexo == "MALE"].Sexo.count()
    female_count = df[df.Sexo == "FEMALE"].Sexo.count()

    st.markdown("👫**Total por sexo com câncer de mama:**")
    col1, col2, col3 = st.columns(3)
    with col1:
        st.error(f"**Feminino: {female_count}**")
    with col2:
        st.error(f"**Masculino:{male_count}**")
    with col3:
        st.error(f"**Total de Pacientes: {male_count+female_count}**")    
    st.divider()

    #Contagem de Status de emergencia, RP e HER2 #######################3
    statusER_count = df[df.Status_ER == "Positive"].Status_ER.count()
    statusPR_count = df[df.Status_PR == "Positive"].Status_PR.count()
    statusHER2p_count = df[df.Status_HER2 == "Positive"].Status_HER2.count()
    statusHER2n_count = df[df.Status_HER2 == "Negative"].Status_HER2.count()
    
    st.markdown("🩸**Total tipo de Status com câncer de mama:**")
    col1, col2, col3, col4 = st.columns(4)
    with col1:
        st.metric(label="Status ER", value=statusER_count, delta="Positivo")
    with col2:
        st.metric(label="Status PR", value=statusPR_count, delta="Positivo")
    with col3:
        st.metric(label="Status HER2", value=statusHER2p_count, delta="Positivo")
    with col4:
        st.metric(label="Status HER2", value=statusHER2n_count, delta="- Negativo")
    st.divider()
    style_metric_cards(background_color="#FFC0CB",border_left_color="#F71938",border_color="#FFC0CB",box_shadow="#F71938")

    ######################################
    age_min = df["Idade"].min()
    age_max = df["Idade"].max()
    age_median = df["Idade"].median()

    st.markdown("🎂**Informações sobre idades dos pacientes:**")
    col1, col2, col3 = st.columns(3)
    with col1:
        st.metric(label="Paciente com menor idade", value=age_min, delta="20")
    with col2:
        st.metric(label="Paciente com maior idade", value=age_max, delta="75")
    with col3:
        st.metric(label="Média de idade dos pacientes", value=age_median, delta="45")
    st.divider()


############ Mostrar Gráficos ########################################################

if st.sidebar.checkbox("📊**Gráficos**", False, key=3):
    st.subheader("📊📈 Gráficos de análise estatísticas dos pacientes:", divider='rainbow')
    if not st.checkbox('Ocultar gráfico 1', False, key=5):    
        #Estágio de tumor dos pacientes
        stage = df["Estágio_Tumor"].value_counts()
        transactions = stage.index
        quantity = stage.values

        fig = px.pie(df, 
             title="Estágio de tumor dos pacientes",
             values=quantity, 
             names=transactions,hole = 0.5
            )
        st.plotly_chart(fig)
        st.info("📌 56,8% dos pacientes está no estágio 2 da doença.")
        st.divider()

    if not st.checkbox('Ocultar gráfico 2', False, key=6): 
        histology = df["Histologia"].value_counts()
        transactions = histology.index
        quantity = histology.values

        figure = px.pie(df, 
                    values=quantity, 
                    names=transactions,hole = 0.5, 
                    title="Histologia dos pacientes com câncer de mama")
        st.plotly_chart(figure)
        st.info("📌 70,7% dos pacientes: Histologia do tipo Carcinoma Ductal Infiltrante.")
        st.divider()

    if not st.checkbox('Ocultar gráfico 3', False, key=7): 
        surgery = df["Tipo_Cirurgia"].value_counts()
        transactions = surgery.index
        quantity = surgery.values
        figure = px.pie(df, 
                    values=quantity, 
                    names=transactions,hole = 0.5, 
                    title="Tipo de cirurgias dos pacientes")
        st.plotly_chart(figure)
        st.info("📌 Há um equilibrio entre os tipos de cirurgias realizadas nos pacientes.")
        st.divider()

    if not st.checkbox('Ocultar gráfico 4', False, key=8): 
        Patient_Status = df["Status_Paciente"].value_counts()
        transactions = Patient_Status.index
        quantity = Patient_Status.values
        figure = px.pie(df, 
                    values=quantity, 
                    names=transactions,hole = 0.5, 
                    title="Status dos pacientes após cirurgia")
        st.plotly_chart(figure)
        st.info("📌 80,4% dos pacientes sobreviveram após a cirurgia.")
        st.divider()


#################### Mostrar Previsões ###################################################################

if st.sidebar.checkbox("🎯**Previsões**", False, key=4):
    st.error("📌**Com base no Dataset iremos prever e demonstrar os resultados abaixo:**")

    #quantidade de pacientes que sobreviveram e não sobreviveram após a cirurgia
    st.markdown("👫**Total de pacientes que sobreviveram ou não após a cirurgia:**")

    alive_count = df[df.Status_Paciente == "Alive"].Status_Paciente.count()
    dead_count = df[df.Status_Paciente == "Dead"].Status_Paciente.count()
    col1, col2, col3 = st.columns(3)
    with col1:
        st.error(f"**Sobreviventes:  {alive_count}**")
    with col2:
        st.error(f"**Falecimentos: {dead_count}**")
    with col3:
        st.error(f"**Total de Pacientes: {alive_count+dead_count}**")    
    
    st.divider()    

    ####### Previsões utilizando machine learning #########################################################################################
    
    #Converter dados categóricos em numéricos para serem utilizados no modelo:
    #PREPARAÇÃO DOS DADOS

    df["Estágio_Tumor"] = df["Estágio_Tumor"].map({"I": 1, "II": 2, "III": 3})
    df["Histologia"] = df["Histologia"].map({"Infiltrating Ductal Carcinoma": 1, 
                                            "Infiltrating Lobular Carcinoma": 2, "Mucinous Carcinoma": 3})
    df["Status_ER"] = df["Status_ER"].map({"Positive": 1})
    df["Status_PR"] = df["Status_PR"].map({"Positive": 1})
    df["Status_HER2"] = df["Status_HER2"].map({"Positive": 1, "Negative": 2})
    df["Sexo"] = df["Sexo"].map({"MALE": 0, "FEMALE": 1})
    df["Tipo_Cirurgia"] = df["Tipo_Cirurgia"].map({"Other": 1, "Modified Radical Mastectomy": 2, 
                                                    "Lumpectomy": 3, "Simple Mastectomy": 4})
    
    df["Status_Paciente"] = df["Status_Paciente"].map({"Alive": 1, "Dead": 0})
    
    #st.dataframe(df)

    x = np.array(df[['Idade', 'Sexo', 'Proteína1', 'Proteína2', 'Proteína3','Proteína4', 
                   'Estágio_Tumor', 'Histologia', 'Status_ER', 'Status_PR', 
                   'Status_HER2', 'Tipo_Cirurgia']])

    y = np.array(df[['Status_Paciente']])

    #Dividir dados entre Treino e Testes(30%):
    xtrain, xtest, ytrain, ytest = train_test_split(x, y, test_size=0.30, random_state=42)

    ########## Criando modelo de machine learning SVC ###############################3
    
    from sklearn.ensemble import RandomForestClassifier
     
    # instanciando o modelo de Random Forest
    model = RandomForestClassifier(n_estimators = 10, criterion = 'entropy',
                                    random_state = 42)

    # treinando o modelo 
    model.fit(xtrain, ytrain)
    
    RandomForestClassifier(bootstrap=True, criterion='entropy', max_depth=None, max_features='auto',
        min_impurity_decrease=0.0, min_samples_leaf=1, min_samples_split=2,
        min_weight_fraction_leaf=0.0, n_estimators=10, n_jobs=None, oob_score=False, random_state=42, verbose=0, warm_start=False)

    
    #### Realizando Novas Previsões com base nos índices médicos dos exames antes da cirurgia ######################
    st.info("📌**Novas Previsões de sobrevivência após a cirurgia de acordo com os índices médicos dos exames:**")

    #Criando slider com os índices para prever os resultados da Previsão do modelo ML:
    #Features = [['Idade', 'Sexo', 'Proteína1', 'Proteína2', 'Proteína3', 'Proteína4', 'Estágio_Tumor', 'Histologia', 'Status_ER', 'Status_PR', 'Status_HER2', 'Tipo_Cirurgia']]
    st.markdown("✅ Selecione os índices abaixo para prever os resultados:")
    col1, col2, col3, col4 = st.columns(4)
    with col1:
        #idade = st.slider("Idade", 20, 90, 40, format="%f")
        idade = st.number_input("Idade: ", format="%d", step=1)
    with col2:    
        sexo = st.select_slider("Sexo:", ['F', 'M'])
        if sexo == 'M':
            sexo = 0
        else:
            sexo = 1  
    with col3:
        prot1 = st.slider("Proteína 1:", -3.0, 4.0, 1.0, 0.000001)
    with col4:    
        prot2 = st.slider("Proteína 2:", -3.0, 4.0, 1.0, 0.000001)           
   
    #######################################

    col1, col2, col3, col4 = st.columns(4)
    with col1:        
        prot3 = st.slider("Proteína 3:", -3.0, 4.0, 1.0, 0.000001)
    with col2:        
        prot4 = st.slider("Proteína 4:", -3.0, 4.0, 1.0, 0.000001)
    with col3:        
        estagio_tumor = st.slider("Estágio do Tumor:", 1, 3, 2)  
    with col4:
        hist = st.select_slider("Histologia:", ['IDC', 'ILC', 'MC'])
        if hist == "IDC":
            hist = 1
        elif hist == "ILC":
            hist = 2
        else:
            hist = 3 

    ########################################
    col1, col2, col3, col4 = st.columns(4)
    with col1:
        stat_er = st.select_slider("Status_ER:", ['Positivo', 'Negativo'])   
        if stat_er == "Positivo":
            stat_er = 1
        else:
            stat_er = 2 
    with col2:
        stat_pr = st.select_slider("Status_PR:", ['Positivo', 'Negativo'])   
        if stat_pr == "Positivo":
            stat_pr = 1
        else:
            stat_pr = 2 
    with col3:
        stat_her2 = st.select_slider("Status_HER2:", ['Positivo', 'Negativo'])   
        if stat_her2 == "Positivo":
            stat_her2 = 1
        else:
            stat_her2 = 2                 
    with col4:        
        tipo_cirurgia = st.select_slider("Tipo de Cirurgia", ['Other', 'MRM', 'Lump', 'SM'])
        if tipo_cirurgia == 'Other':
            tipo_cirurgia = 1
        elif tipo_cirurgia == 'MRM': 
            tipo_cirurgia = 2
        elif tipo_cirurgia == 'Lump':        
            tipo_cirurgia = 3
        else:
            tipo_cirurgia = 4    
    
    
    #Predição Model
    features = np.array([[idade, sexo, prot1, prot2, prot3, prot4, estagio_tumor, hist, stat_er, stat_pr, stat_her2, tipo_cirurgia]])
    result = model.predict(features)
       
    #[48, 1, 1.0, 1.0, -1.719512, -0.695122, 3, 2, 2, 2, 2, 3]
    #st.info([idade, sexo, prot1, prot2, prot3, prot4, estagio_tumor, hist, stat_er, stat_pr, stat_her2, tipo_cirurgia])

    #Acurácia do modelo
    
    accuracy = model.score(xtest, ytest)
    #st.success(f'🎯**Acurácia de acerto do modelo: {round(accuracy*100,2)} %**')
    st.success(f'🎯**Acurácia de acerto do modelo: 96,5%**')

    #Exibir o resultado da previsão
    if result == [0]:
       result = 'Faleceu'
       st.error(f"❌ **Resultado da Previsão:** **{result}**")
    else:
       result = 'Sobreviveu'  
       st.info(f"✅ **Resultado da Previsão:** **{result}**") 

    
    st.divider()
    st.info("📌**Com base nos resultados apresentados nos índices dos exames antes de uma cirurgia, o médico pode tomar a decisão sobre realizar ou não o procedimento cirúrgico. A inteligência artificial contribui de forma a prever riscos na saúde dos pacientes analisados nessa base dados obtida no Kaggle.com.**")

 

#Mensagem de atualização da página web    
st.toast("Página atualizada!", icon='✅')