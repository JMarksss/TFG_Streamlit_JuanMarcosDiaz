import pandas as pd
import streamlit as st
import functions as f
pd.options.display.float_format = '{:,.3f}'.format 

st.title('_Dataset_')
st.markdown('Para poner en práctica estos algoritmos, utilizaremos una determinada base de datos. El _dataset_ lo obtenemos del famoso repositorío de aprendizaje automático [Kaggle](https://www.kaggle.com/datasets/rashikrahmanpritom/heart-attack-analysis-prediction-dataset). Elegimos este _dataset_ ya que por su composición lo creemos adecuado para poder practicar todos los algoritmos mencionados en la memoria. Esta pensado originalmente para clasificar pacientes con mayor o menor riesgo de cardiopatía, aunque la presencia de variables numéricas también nos permitirá realizar análisis de regresión.')
st.image('images/2_Dataset/card.png')
st.divider()
st.markdown('## Análisis Descriptivo')
st.markdown('El _dataset_ cuenta con 303 filas (observaciones) y 14 columnas (variables) y ningún valor nulo. En el repositorio de [Kaggle](https://www.kaggle.com/datasets/rashikrahmanpritom/heart-attack-analysis-prediction-dataset/discussion/234843?sort=votes), podemos encontrar más información sobre las variables. Para facilitar su comprensión, hemos decididido modificar el nombre original de las variables y sus valores. Podemos observar el _dataframe_ entero en el siguiente desplegable.')

datos = pd.read_csv('data/heart.csv')
cat = f.clean_data(datos)

with st.expander("Dataframe Completo"):
    st.write(cat)

st.markdown('### **Análisis descriptipo de las variables numéricas**')

st.write(cat.describe())
st.markdown('- La edad de los pacientes está comprendida entre los 29 y los 77 años. La media y la mediana están próximas, en torno a los 54. Concluimos que los pacientes no son jóvenes.')
st.markdown('- La tensión arterial varía, si tenemos en cuenta como valores normales los menores a 120, deducimos que 3 de cada 4 pacientes tienen una presión arterial alarmante. Presencia de anomalías con valores muy altos.')
st.markdown('- Respecto al colesterol obtenemos información similar, teniendo en cuenta como normales los valores comprendidos entre 125 y 200, más del 75% de pacientes presentan valores superiores. Presencia de anomalías con valores muy altos.')
st.markdown('- Los valores normales para frecuencia cardiaca máxima varían con la edad, por lo que esta variable por sí sola no aporta demasiada información.')
st.markdown('- Al menos la mitad de los pacientes no muestran ningún vaso mayor coloreado por la fluoroscopia')

st.markdown('###  **Distribución de las variables categóricas**')
option = st.selectbox(
    '¿Qué variable deseas estudiar?',
    ('Sexo', 'Dolor torácico', 'Glucemia', 'Electrocardiograma', 'Angina por ejercicio', 'Pendiente ST', 'Talasemia', 'Cardiopatía'))
if option == 'Sexo':
     st.image('images/2_Dataset/pie/SexoPie.png')
     st.markdown('Más de dos tercios son hombres.')
elif option == 'Dolor torácico':
     st.image('images/2_Dataset/pie/DolorPie.png')
     st.markdown('Casi la mitad de los pacientes no presentan dolor torácico, destaca la mayor presencia de angina atípica frente a la típica.')
elif option == 'Glucemia':
    st.image('images/2_Dataset/pie/GlucemiaPie.png')
    st.markdown('La inmensa mayoría de pacientes presenta valores mayores 120 mg/dl para la glucemia.')
elif option == 'Electrocardiograma':
    st.image('images/2_Dataset/pie/ElectroPie.png')
    st.markdown('Más de la mitad de los electrocardiogramas son normales y solo un 1.3% presenta anomalía de onda.')
elif option == 'Angina por ejercicio':
    st.image('images/2_Dataset/pie/AnginaPie.png')
    st.markdown('Más de dos tercios de los pacientes presentan angina provocada por el ejercico.')
elif option == 'Pendiente ST':
    st.image('images/2_Dataset/pie/PendientePie.png')
elif option == 'Talasemia':
    st.image('images/2_Dataset/pie/TalasemiaPie.png')
    st.markdown('La mayoría de los pacientes presenta valores normales en la talasemia, entre los que presentan defectos (~45%), la mayoría son reversibles y sólo hay un pequeño porcentaje de pruebas son nulas.')
else:
    st.image('images/2_Dataset/pie/CardiopatiaPie.png')
    st.markdown('Más de la mitad de los pacientes tiene una probabilidad altade sufrir cadiopatía.')

st.markdown('### **Relación entre las variables numéricas y la variable salida**')
option2 = st.selectbox(
    '¿Qué variable deseas estudiar?',
    ('Edad', 'Tensión arterial', 'Nivel de colesterol', 'Frecuencia cardíaca máxima', 'Depresión ST', 'Vasos mayores coloreados'))
if option2 == 'Edad':
    st.image('images/2_Dataset/boxplot/EdadBox.png')
    st.markdown('Sorprendentemente en esta muestra los pacientes mayores tienen menor probabilidad de sufrir una cardiopatía. Esto se puede deber a que la muestra esté recogida en un hospital, en el que los pacientes más jóvenes acuden con más síntomas, mientras que para los mayores se puede tratar de una revisión.')
elif option2 == 'Tensión arterial':
    st.image('images/2_Dataset/boxplot/TensiónBox.png')
    st.markdown('No existen grandes diferencias en los niveles de tensión para los dos grupos de pacientes. Se observan incluso valores ligeramente más altos para los pacientes con menos riesgo.')
elif option2 == 'Nivel de colesterol':
    st.image('images/2_Dataset/boxplot/ColesterolBox.png')
    st.markdown('No existen grandes diferencias en los niveles de colesterol para los dos grupos de pacientes. Se observan incluso valores ligeramente más altos para los pacientes con menos riesgo.')
elif option2 == 'Frecuencia cardíaca máxima':
    st.image('images/2_Dataset/boxplot/FrecBox.png')
    st.markdown('La frecuencia máxima si que parece mayor para los pacientes con mayor riesgo.')
elif option2 == 'Depresión ST':
    st.image('images/2_Dataset/boxplot/DepresionBox.png')
else:
    st.image('images/2_Dataset/boxplot/VasosBox.png')

st.markdown('### **Relación entre las variables categóricas y la variable salida**')
option3 = st.selectbox(
    '¿Qué variable deseas estudiar?',
    ('Sexo', 'Dolor torácico', 'Glucemia', 'Electrocardiograma', 'Angina por ejercicio', 'Pendiente ST', 'Talasemia'))
if option3 == 'Sexo':
    st.write(pd.crosstab(cat['Sexo'], cat['Cardiopatía'], normalize=0))
    st.markdown('Las mujeres tienen más riesgo de cardiopatía que los hombres.')
elif option3 == 'Dolor torácico':
    st.write(pd.crosstab(cat['Dolor'], cat['Cardiopatía'], normalize=0))
    st.markdown('Los pacientes que no presentan dolor torácico tienen menor riesgo de sufrir una cardiopatía frente al resto.')
elif option3 == 'Glucemia':
    st.write(pd.crosstab(cat['Dolor'], cat['Cardiopatía'], normalize=0))
    st.markdown('Los niveles de glucemia no parecen afectar al riesgo de cardiopatía.')   
elif option3 == 'Electrocardiograma':
    st.write(pd.crosstab(cat['Electro'], cat['Cardiopatía'], normalize=0))
    st.markdown('Muy sorprendentemente, los pacientes con un electrocardiograma normal presentan más riesgo de sufrir una cardiopatía frente a los que presentan otros resultados en esta prueba.')   
elif option3 == 'Angina por ejercicio':
    st.write(pd.crosstab(cat['Angina_ej'], cat['Cardiopatía'], normalize=0))
    st.markdown('Los pacientes con angina derivada del ejercicio físico también presentan mator riesgo')   
elif option3 == 'Pendiente ST':
    st.write(pd.crosstab(cat['Pendiente_ST'], cat['Cardiopatía'], normalize=0))  
else:
    st.write(pd.crosstab(cat['Talasemia'], cat['Cardiopatía'], normalize=0))
    st.markdown('Sorprende que los pacientes que presentan una talasemia normal tengan mayor riesgo que aquellos que presentan defectos.')   

st.divider()
st.markdown('### Conclusiones')
st.markdown('Los resultados obtenidos en este primer análisis exploratorio son muy sorprendentes. Muchos de los indicadores que habitualmente se consideran como factores de riesgo cardiacos, parecen no influir o influir negativamente en el desarrollo de esta enfermedad. Podemos deducir que, o bien se trata de una muestra pequeña que no recoge suficientes observaciones o simplemente estamos trabajando con un dataset artificial que no se ajusta a la realidad.')
st.divider()
st.image('images/usal.png')
st.caption('Trabajo de Fin de Grado en Estadística')
st.caption('Aplicación web interactiva para el análisis de datos multivariantes mediante técnicas de aprendizaje automático')
st.caption('Juan Marcos Díaz')
