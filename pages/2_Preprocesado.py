import pandas as pd
import streamlit as st
import functions as f
pd.options.display.float_format = '{:,.3f}'.format 

st.markdown('# Preprocesado')
st.markdown('Los algoritmos de aprendizaje automático aprenden y trabajan con datos y, aunque existen algunos que no lo necesitan, para poder aplicar correctamente la mayoría de ellos primero debemos transformar los datos en lugar de trabajar con los datos en crudos directamente. Esta parte del tratamiento de los datos se conoce como preprocesado y es de vital importancia. Algunas de las principales técnicas son:')
st.markdown('## Codificación')
st.markdown('Consiste en transformar datos no numéricos (texto, imágenes…) en datos numéricos, aunque en nuestro caso vamos a considerar sólo las técnicas que transforman variables categóricas a numéricas:')
st.markdown('- **Codificación Ordinal**: útil para variables categóricas en las que exista un orden natural. Por ejemplo, tres categorías: Menor de 30 años, 30-60 años y Más de 60 años, les podemos asignar las variables 0, 1 y 2 respectivamente sin perder información.')
st.markdown('- **Codificación _One-Hot_**: consiste en crear variables _dummies_ para cada una (suponemos $n$ categorías) o para $n-1$ categorías de la variable. Gráficammente')
st.image('images/3_Preproces/onehot.jpg')

st.markdown('## Discretización')
st.markdown('Entendemos la discretización como el proceso que convierte datos continuos (con un rango infinito de posibles valores) en datos discretos, es decir, transformamos una variable continua y dividimos sus valores en categorías o intervalos discretos, donde cada valor dentro de un intervalo se considera equivalente.')
st.image('images/3_Preproces/disc.jpg')

st.markdown('## Escalado')
st.markdown('El objetivo de esta técnica es ajustar los valores de las variables de modo que se encuentren en una escala uniforme y comparable. Cuando los datos de entrada tienen diferentes magnitudes o unidades, algunos algoritmos pueden dar más peso a las variables con valores más grandes, lo que puede afectar negativamente el rendimiento del modelo, por lo que el escalado es muy importante, ya que puede mejorar la precisión y la eficacia de estos modelos. Algunos de los métodos de escalado más populares son:')
st.markdown('- **Estandarización**: es uno de los métodos más utilizados y útiles, supone que los datos son más o menos normales. Consiste en restar a cada observación la media de la variable 𝜇 y dividir por la desviación típica 𝜎 de tal modo que las nuevas variables tendrán 𝜇 = 0 y 𝜎 = 1.')
st.latex(r'''\mu_i := \frac{1}{n}\sum_{j=1}^nx_i^{(j)}\qquad \sigma_i := \sqrt{\frac{1}{n}\sum_{j=1}^n(x_i^{(j)}-\mu_i)^2}''')
st.latex(r'''x_i := \frac{x_i-\mu_i}{\sigma_i}''')
st.markdown('- **Escalado Min-Max**: : es útil cuando los valores extremos son importantes en nuestro estudio. Todos los valores quedarán comprendidos en el intervalo [0,1]. La principal limitación de esta técnica es que es muy sensible a los outliers o valores extremos.')
st.latex(r'''max=\max_{0<j<n} x_i^{(j)}\qquad min=\min_{0<j<n} x_i^{(j)}''')
st.latex(r'''x_i := \frac{x_i-min}{max-min}''')
st.markdown('- **Escalado Robusto**: : técnica similar al estandarizado, pero ignorando los outliers. Consiste en restar la mediana 𝑄2 y escalar entre el primer y el tercer cuartil [𝑄1, 𝑄3].')
st.latex(r'''x_i := \frac{x_i-Q_2}{Q_3-Q_1}''')
st.image('images/3_Preproces/esc.jpg')
st.divider()
st.markdown('En el caso particular de nuestro conjunto de datos, hemos aplicado la codificación _One-hot_ y los distintos tipos de escalado para futuros análisis. Además hemos separado la variable respuesta Cardiopatía $Y$, del resto $X$.')

datos = pd.read_csv('data/heart.csv')
cat = f.clean_data(datos)
Y = datos['output'] 
X = cat.drop(['Cardiopatía'], axis=1)
X_st = f.standard(X)
X_mm = f.minmax(X)
X_r = f.robust(X)
Xr = pd.get_dummies(X, drop_first=True)

with st.expander('Distintos _datasets_ escalados con codificación _One-Hot_'):
    pre_option = st.selectbox(
    '¿Qué transformación deseas observar?',
    ('Escalado estándar', 'Escalado Min-Max', 'Escalado robusto', 'Codificación sin escalar'))
    if pre_option == 'Escalado estándar':
        st.write(X_st)
    elif pre_option == 'Escalado Min-Max':
        st.write(X_mm)
    elif pre_option == 'Escalado robusto':
        st.write(X_r)
    else:
        st.write(Xr)

st.divider()
st.image('images/usal.png')
st.caption('Trabajo de Fin de Grado en Estadística')
st.caption('Aplicación web interactiva para el análisis de datos multivariantes mediante técnicas de aprendizaje automático')
st.caption('Juan Marcos Díaz')
