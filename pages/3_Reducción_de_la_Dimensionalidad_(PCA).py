import pandas as pd
import streamlit as st
import functions as f
pd.options.display.float_format = '{:,.3f}'.format 
st.set_option('deprecation.showPyplotGlobalUse', False)

st.markdown('# Reducción de la Dimensionalidad (PCA)')
st.markdown('El Análisis de Componentes principales (PCA, por sus siglas en inglés), es una técnica estadística empleada en análisis multivariante qué se utiliza para reducir el número de variables transformando las variables originales mediante combinaciones lineales de estas en un nuevo conjunto de variables denominadas "componentes principales".')
st.markdown('La idea principal de este algoritmo es buscar la dirección en la que los datos presentan la mayor varianza entendiendo que a mayor varianza, más información. Una vez encontrada esta dirección, se deberán proyectar los datos originales sobre esta, obteniendo así la primera "componente". De manera análoga, se busca la segunda dirección de mayor varianza y se proyecta sobre ella, de tal forma que se obtienen tantas componentes principales como variables originales, con la particularidad de que cada componente explica un porcentaje de variabilidad (y por tanto de información), mayor que la siguiente componente.')
st.image('images/4_PCA/pca.jpg')
st.markdown('Es tarea del investigador decidir con cuantas componentes se queda, existiendo varios métodos que le guiarán a la hora de tomar esta decisión evaluando cuanta información se pierde.')
st.markdown(' - **Criterio de varianza explicado**: se establece un umbral mínimo de varianza explicada (en torno al 70% o el 80%) y se seleccionan suficientes componentes principales para alcanzar ese umbral, teniendo en cuenta que las primeras componentes explican la mayor proporción de la varianza.')
st.markdown(' - **Criterio del codo**: se representa gráficamente el porcentaje de varianza explicada (eje Y) por cada componente principal ordenada (eje X). El número de componentes a elegir nos lo indicará el punto de inflexión (o "codo") de la gráfica, indicando que seleccionar más gráficas no supondría un aumento considerable de la varianza explicada. Este gráfico se denomina _Scree Plot_.')
st.divider()
st.markdown('## PCA Interactivo')
var = st.slider('Selecciona el porcentaje de varianza mínimo explicado', 0, 100, 1)
esc = st.radio(
    "Selecciona el tipo de escalado",
    ('Estándar', 'Min-Max', 'Robusto'))

datos = pd.read_csv('data/heart.csv')
cat = f.clean_data(datos)
Y = datos['output'] 
X = cat.drop(['Cardiopatía'], axis=1)
X_st = f.standard(X)
X_mm = f.minmax(X)
X_r = f.robust(X)
Xr = pd.get_dummies(X, drop_first=True)

if esc == 'Estándar':
    X_s = X_st
elif esc == 'Min-Max':
    X_s = X_mm
else:
    X_s = X_r
if st.button('Compruébalo!'):
    pca, X_pca = f.pca(X_s, var/100)
    st.write('Número de componentes:', X_pca.shape[1])
    st.write('Porcentaje de varianza explicada por cada componente (entre 0 y 1):')
    st.write(pca.explained_variance_ratio_.reshape(1,-1))
    st.pyplot(f.plot_pca(pca))
st.divider()
st.markdown('## Conclusiones')
st.markdown('Necesitamos 5 variables, 6 en el caso del escalado robusto, para conservar al menos el 70% de la varianza. Parece que el esclado Min-Max es el que mejores resultados obtiene y, atendiendo al criterio del codo parece que 3 es el número correcto de componentes para este tipo de escalado. Con este número de componentes estaríamos conservando en torno al 60% de la varianza.')
st.divider()
st.image('images/usal.png')
st.caption('Trabajo de Fin de Grado en Estadística')
st.caption('Aplicación web interactiva para el análisis de datos multivariantes mediante técnicas de aprendizaje automático')
st.caption('Juan Marcos Díaz')