import pandas as pd
import numpy as np
import streamlit as st
import functions as f
from sklearn.linear_model import LinearRegression
r = 12 #Aleatoriedad
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
pd.options.display.float_format = '{:,.3f}'.format 

datos = pd.read_csv('data/heart.csv')
cat = f.clean_data(datos)
Y = datos['output'] 
X = cat.drop(['Cardiopatía'], axis=1)
Xr = pd.get_dummies(X, drop_first=True)

st.markdown('# Regresión')
st.markdown('Los modelos de regresión buscan predecir un valor numérico en función del resto de variables. Dependiendo del número de variables explicativas o regresoras del modelo trabajaremos con modelos de regresión simple (una sola variable explicativa) o regresión múltiple (más de una variable explicativa). Aunque existen distintos algoritmos de regresión, a en este trabajo nos centraremos en describir la regresión lineal, el modelo más simple que representa la relación de las variables en forma de línea recta.')
st.divider()
st.markdown('### Métricas:')
st.markdown('Las métricas para evaluar los modelos de regresión se centran en comprobar si las predicciones de un modelo de regresión ($\hat{Y}$) se ajustan y se aproximan a los valores reales de la variable dependiente $Y$. Para evaluar los modelos dividimos el conjunto total en dos subconjuntos: entrenamiento y test, de tal forma que entrenamos los modelos sobre el primer subconjunto y evaluamos su desempeño sobre el segundo.')
with st.expander("**:star: Despliega para conocer algunas de estas métricas :exclamation:**"):
    st.markdown('- **Coeficiente de determinación** $R^2$: indica cuánta varianza de la variable dependiente puede ser explicada por el modelo de regresión lineal. Puede tomar valores entre 0 y 1. De tal forma que, un valor igual a 0 indica que el modelo no explica ninguna variación en la variable dependiente, mientras que un valor de 1 indicaría que el modelo explica toda la variación en la variable dependiente.')
    st.latex(r'''R^2=1-\frac{SC_{Res}}{SC_{Tot}}=1-\frac{\sum_{i=1}^{n}\left(y_i-\widehat{y_i}\right)^2}{\sum_{i=1}^{n}\left(y_i-\bar{y}\right)^2}''')
    st.markdown('- **Error cuadrático medio** ($MSE$): es una de las métricas más utilizadas en regresión lineal y mide la diferencia entre los valores observados y los predichos, por tanto, cuanto menor sea su valor, mejor se ajustará nuestro modelo a los datos.')
    st.latex(r'''MSE=\frac{SC_{Res}}{n}=\frac{\sum_{i=1}^{n}\left(y_i-\widehat{y_i}\right)^2}{n}''')
    st.markdown('- **Error absoluto medio** ($MAE$): similar al $MSE$, esta métrica tiene en cuenta las diferencias en valor absoluto entre valores reales y predichos, en lugar de los cuadrados de tal forma que es menos sensible a los valores atípicos.')
    st.latex(r'''MAE=\frac{\sum_{i=1}^{n}\left|y_i-\widehat{y_i}\right|}{n}''')

st.divider()
st.markdown('## Regresión Lineal Simple')
st.markdown('Consideremos dos variables cuantitativas $X$ e $Y$. El modelo muestral que explica el comportamiento de $Y$ frente a $X$ es:')
st.latex(r'''Y_i=\widehat{\beta_0}+\widehat{\beta_1}X_i+\widehat{u_i}''')
st.markdown('Donde $i=1,\ldots,n$ representa el número de observaciones; $\widehat{β_0}, \widehat{β_1}$ los coeficientes del modelo y $\widehat{u_i}$ el residuo o término aleatorio. El objetivo es construir un modelo de predicción.')
st.latex(r'''\widehat{Y_i} =\widehat{\beta_0}+\widehat{\beta_1}X_i''')
with st.expander("**:star: Despliega para saber más :exclamation:**"):
    st.markdown('Para estimar el valor de los coeficientes, buscaremos minimizar la suma de los cuadrados de los residuos que se corresponden con la diferencia entre el valor real ($Y$) y el esperado ($\widehat{Y}$). Con esta técnica, que se conoce como Mínimos Cuadrados Ordinarios:.')
    st.latex(r'''\min{G}=min\sum_{i=1}^{n}\widehat{{u_i}^2}=\min{\sum_{i=1}^{n}\left(Y_i-\widehat{Y_i}\right)^2}=\min{\sum_{i=1}^{n}\left(Y_i-\widehat{\beta_0}-\widehat{\beta_1}X_i\right)^2}''')
    st.markdown('Derivando respecto de los coeficientes e igualando a 0 obtenemos las **Ecuaciones Normales** de las que podemos finalmente deducir el valor de los coeficientes:')
    st.latex(r'''\widehat{\beta_0}=\overline{Y}-\widehat{\beta_1}\overline{X}''')
    st.latex(r'''\widehat{\beta_1}=\frac{S_{XY}}{S_X^2}''')
    st.markdown('Con $\overline{X},\overline{Y}$ las medias de $X,Y$ respectiamente, $S_{XY}$ la covarianza entre las dos variables y $S_X^2$ la varianza de la variable independiente.')

st.markdown('### Regresión Lineal Simple Interactiva')
st.markdown('En esta sección, vamos a evaluar distintos ajustes entre las variables numéricas de nuestro dataset sin escalar que nos resultan más interesantes.')
with st.expander("**:star: Despliega para aplicar los distintos modelos :exclamation:**"):
    dep = st.selectbox(
        'Selecciona la variable dependiente $Y$',
        ('Edad', 'Tensión', 'Colesterol', 'Frecuencia_Max'))
    indep = st.selectbox(
        'Selecciona la variable independiente $X$',
        ('Edad', 'Tensión', 'Colesterol', 'Frecuencia_Max'))
    test = st.slider(
        '¿Qué porcentaje de los datos quieres que represente el subconjunto de test?',
        min_value=5, max_value=40, step=1)
    slr = LinearRegression()
    if st.button('Haz Click!'):
        y = Xr[dep]
        x = Xr[indep]
        X_train, X_test, y_train, y_test = train_test_split(x, y, random_state = r, test_size=test/100)
        slr.fit(np.array(X_train).reshape(-1, 1), y_train)
        y_pred = slr.predict(np.array(X_test).reshape(-1, 1))
        st.write('**Coeficiente ($β_1$)**')
        st.write(slr.coef_)
        st.write('**Intercepto ($β_0$)**')
        st.write(round(slr.intercept_,3)) 
        st.write('**Métricas**')
        st.write('$MSE$:', round(mean_squared_error(y_test, y_pred),3))
        st.write('$MAE$:', round(mean_absolute_error(y_test, y_pred),3))
        st.write('$R^2$:', round(r2_score(y_test, y_pred),3))
        fig = plt.figure(figsize=(10, 5))
        sns.regplot(x=X_train,y=y_train)
        st.pyplot(fig)

    st.markdown('#### Evaluación')
    st.markdown('Obviando los ajustes de cada variable sobre si misma (que evidentemente son perfectos), los resultados no son nada buenos. Esto se puede deber a que las variables numéricas de este _dataset_ no están pensadas para realizar este tipo de análisis y se ven muy influenciadas por los _outliers_. Además, podemos observar que, en general y como cabría esperar, a medida que aumenta el tamaño del test los resultados empeoran. Sorprende que en algunos casos se obtienen resultados de $R^2$ negativos, lo que en teoría es imposible. Esto se debe a la construcción de la métrica en Scikit-Learn y reflejaría que el ajuste es muy malo y en este caso, podemos sustituir los valores negativos por 0.')

st.divider()
st.markdown('## Regresión Lineal Múltiple')
st.markdown('En este caso, nuestros modelos contarán con más de una variable independiente. Generalizando los resultados del caso simple para $k$ variables independientes, la forma matricial del modelo muestral sería:')
st.latex(r'''Y = X\beta+u''')
st.latex(r'''Y=\left(\begin{matrix}Y_1\\\vdots\\Y_n\\\end{matrix}\right),\>X=\left(\begin{matrix}1&x_{1,1}&\ldots&x_{1,k}\\\vdots&\vdots&\ddots&\vdots\\1&x_{n,1}&\ldots&x_{n,k}\\\end{matrix}\right),\>\beta=\left(\begin{matrix}\beta_0\\\vdots\\\beta_k\\\end{matrix}\right),\>u=\left(\begin{matrix}u_1\\\vdots\\u_n\\\end{matrix}\right)''')
st.markdown('Aplicando mínimos cuadrados obtenemos el valor de los coeficientes:')
st.latex(r'''\beta=\left(X^TX\right)^{-1}X^TY''')

st.markdown('### Regresión Lineal Múltiple Interactiva')
st.markdown('Estudiemos las regresiones de las variables numéricas Edad, Tensión, Colesterol y Frecuencia_Max tomando como variables independientes todas las demás.')
with st.expander("**:star: Despliega para aplicar los distintos modelos :exclamation:**"):
    dep2 = st.selectbox(
        'Selecciona la variable dependiente $Y$',
        ('Edad', 'Tensión', 'Colesterol', 'Frecuencia_Max'), key=2)
    test2 = st.slider(
        '¿Qué porcentaje de los datos quieres que represente el subconjunto de test?',
        min_value=5, max_value=40, step=1, key=3)
    mlr = LinearRegression()
    if st.button('Compruébalo!'):
        ym = Xr[dep2]
        xm = Xr.drop([dep2],axis=1)
        Xm_train, Xm_test, ym_train, ym_test = train_test_split(xm, ym, random_state = r, test_size=test2/100)
        mlr.fit(Xm_train, ym_train)
        ym_pred = mlr.predict(Xm_test) 
        st.write('**Coeficientes ($β_i, i=1,\dots,n$)**')
        st.write(mlr.coef_.reshape(1,-1))
        st.write('**Intercepto ($β_0$)**')
        st.write(round(mlr.intercept_,3))
        st.write('**Métricas**')
        st.write('$MSE$:', round(mean_squared_error(ym_test, ym_pred),3))
        st.write('$MAE$:', round(mean_absolute_error(ym_test, ym_pred),3))
        st.write('$R^2$:', round(r2_score(ym_test, ym_pred),3))
    st.markdown('#### Evaluación')
    st.markdown('En los modelos múltiples, aunque son algo mejores que en el caso simple, los resultados siguen sin ser buenos. Esto se debe principalmente a que las variables numéricas de este _dataset_ no están pensadas para realizar este tipo de análisis y se ven muy influenciadas por los _outliers_. Además, podemos observar que, en general y como cabría esperar, a medida que aumenta el tamaño del test los resultados empeoran. El corficiente de determinación $R^2$ aumenta respecto al caso simple ya que tenemos más variables explicativas.')
st.divider()  
st.image('images/usal.png')
st.caption('Trabajo de Fin de Grado en Estadística')
st.caption('Aplicación web interactiva para el análisis de datos multivariantes mediante técnicas de aprendizaje automático')
st.caption('Juan Marcos Díaz')
        

    

