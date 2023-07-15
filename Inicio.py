import streamlit as st

st.sidebar.write('# Índice')
st.title('Aprendizaje Automático')
st.caption('Juan Marcos Díaz')
st.image('./images/1_Inicio/ML.jpg')
st.markdown('En 1959, Artur Samuel definió el **Aprendizaje Automático** o **_Machine Learning_** cómo el área de estudio que confiere a los ordenadores la capacidad de aprender una tarea específica sin ser explícitamente programados para ella. En esta web pondremos en práctica los distintos algoritmos estudiados en la memoria, agrupados en dos grandes categorías:')
st.markdown(' - **Aprendizaje Supervisado**: algoritmos que funcionan y se entrenan con conjuntos de datos etiquetados, es decir, conjuntos de observaciones en las que se conoce la salida esperada y cuyo objetivo es predecir esta variable salida para nuevos conjuntos de datos. Dependiendo de si la variable etiqueta es categórica o numérica trabajaremos con algoritmos de clasificación o regresión respectivamente.')
st.markdown(' - **Aprendizaje No Supervisado**: son algoritmos que trabajan con conjuntos de datos no etiquetados (no existe una variable salida) y cuya misión es encontrar estructuras y patrones ocultos en dichos datos. Dentro de este tipo de algoritmos encontramos el PCA y el _clustering_.')
with open('./data/TFG_JuanMarcosDiaz_signed.pdf', "rb") as file:
    btn = st.download_button(
            label="Descarga la memoria en formato pdf",
            data=file,
            file_name="TFG_JuanMarcosDiaz_signed.pdf",
    )
st.divider()
st.markdown('## Cuestionario para decidir el tipo de análisis')
st.markdown('Para determinar qué tipo de algoritmo de los estudiados en la memoria debemos utilizar, en función del tipo de datos y nuestro objetivo, podemos utilizar el siguiente cuestionario:')
c1 = st.selectbox(
    '1) ¿Cómo son nuestros datos?',
    ('Existe una variable etiqueta o salida que queremos predecir', 'No existe una variable salida'))
if c1 == 'Existe una variable etiqueta o salida que queremos predecir':
    c2 = st.selectbox(
    '2) ¿Cómo es la variable salida?',
    ('Numérica', 'Categórica'))
    if c2 == 'Numérica':
        c3 = st.selectbox(
            '3) ¿Cuantas variables independientes quieres emplear en tu modelo?',
            ('Una', 'Más de una'))
        if c3 == 'Una':
            st.markdown('Debes realizar un análisis de **Regresión Lineal Simple**')
        elif c3 == 'Más de una':
            st.markdown('Debes realizar un análisis de **Regresión Lineal Múltiple**')
    elif c2 == 'Categórica':
         c4 = st.selectbox(
            '3) ¿Tus datos son linealmente separables?',
            ('Sí', 'No'))
         if c4 == 'Sí':
            st.markdown('Debes realizar un análisis de **Clasificación con Regresión Logística o SVM con Kernel Lineal**')
         elif c4 == 'No':
            st.markdown('Debes realizar un análisis de **Clasificación con SVM con Kernel No Lineal (RBF, Polinómico o Sigmoide)**')
elif c1 == 'No existe una variable salida':
    c5 = st.selectbox(
        '2) ¿Cual es tu objetivo?',
        ('Buscar similitudes entre las observaciones y agrupar los datos', 'Reducir el tamaño de los datos para facilitar su interpretación o usarlos en otros análisis'))
    if c5 == 'Buscar similitudes entre las observaciones y agrupar los datos':
        c6 = st.selectbox(
            '3) ¿Sabes cuántos grupos quieres formar?',
            ('Sí', 'No'))
        if c6 == 'Sí':
            st.markdown('Debes realizar un análisis de **Clustering Particional (K-Means, K-Medoids)**')
        elif c6 == 'No':
            st.markdown('Debes realizar un análisis de **Clustering Aglomerativo o Basado en Densidad (DBSCAN)**')
    elif c5 == 'Reducir el tamaño de los datos para facilitar su interpretación o usarlos en otros análisis':
        st.markdown('Debes realizar un análisis de **Reducción de la Dimensionalidad con PCA**')
st.divider()
st.image('./images/usal.png')
st.caption('Trabajo de Fin de Grado en Estadística')
st.caption('Aplicación web interactiva para el análisis de datos multivariantes mediante técnicas de aprendizaje automático')
st.caption('Juan Marcos Díaz')