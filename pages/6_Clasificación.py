import pandas as pd
import streamlit as st
import functions as f
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, balanced_accuracy_score, f1_score, precision_score, recall_score
r = 23 #Aleatoriedad
pd.options.display.float_format = '{:,.3f}'.format 
st.set_option('deprecation.showPyplotGlobalUse', False)

datos = pd.read_csv('data/heart.csv')
cat = f.clean_data(datos)
Y = datos['output'] 
X = cat.drop(['Cardiopatía'], axis=1)
X_st = f.standard(X)
X_mm = f.minmax(X)
X_r = f.robust(X)

st.markdown('# Clasificación')
st.markdown('Los algoritmos de clasificación se definen como aquellos algoritmos de aprendizaje supervisado  que se utilizan para predecir una clase o categoría. Los modelos que empleen estos algoritmos serán entrenados con un conjunto de datos etiquetados en los que la variable respuesta $Y$ será de tipo cualitativo. Si esta variable respuesta es dicotómica, es decir que sólo acepta dos categorías, trabajaremos con algoritmos de clasificación binaria como será nuestro caso. Por otro lado, si la variable acepta más de dos posibles respuestas, la clasificación será múltiple.')
st.image('images/7_Clasif/clas.png')
st.divider()
st.markdown('### Evaluación y Métricas:')
st.markdown('Para evaluar los modelos podemos aplicar principalmente dos enfoques distintos:')
st.markdown('- **División Train-Test**: dividir el conjunto total en dos subconjuntos: entrenamiento y test, de tal forma que entrenamos los modelos sobre el primer subconjunto y evaluamos su desempeño sobre el segundo.')
st.markdown('- **Validación cruzada**: dividir el conjunto de datos en $k$ subconjuntos del mismo tamaño, repitiendo el proceso de entrenamiento y test $k$ veces, utilizando cada uno de los subconjuntos como conjunto de evaluación y el resto como conjunto de entrenamiento y finalmente calcular la media de los resultados obtenidos en cada iteración para obtener una estimación final del modelo. Pudes comprobarlo gráficamente haciendo _click_ en la siguiente _checkbox_:')
vc= st.checkbox('Gráfico de Validación-Cruzada')
if vc:
    st.image('images/7_Clasif/valc.jpg')
st.markdown('En cuanto a las métricas, todas  parten de una idea común: buscan recompensar los aciertos, es decir, las coincidencias entre la categoría real y la categoría predicha o, en otras palabras, la igualdad de valores entre $Y$ e $\hat{Y}$.')
with st.expander("**:star: Despliega para conocer algunas de estas métricas :exclamation:**"):
    st.markdown('Para problemas de clasificación binarios tenemos:')
    st.markdown('- **Matriz de confusión** (_Confusion Matrix_): tabla cruzada que recoge el número de observaciones clasificadas correctamente para cada clase.')
    st.image('images/7_Clasif/matc.jpg')
    st.markdown('- **Exactitud** (_Accuracy_): proporción de observaciones correctamente clasificadas.')
    st.latex(r'''Accuracy=\frac{VP+VN}{VP+VN+FN+FP}=\frac{VP+VN}{N}''')
    st.markdown('- **Precisión** (_Precision_): se utiliza para evaluar la capacidad del modelo para no clasificar como positivas aquellas observaciones que no lo son.')
    st.latex(r'''Precision=\frac{VP}{VP+FP}''')
    st.markdown('- **Sensibilidad** (_Recall_): mide la capacidad del modelo para detectar correctamente las observaciones positivas.')
    st.latex(r'''Recall=\frac{VP}{VP+FN}''')
    st.markdown('- **Exactitud Balanceada** (_Balanced Accuracy_): similar a la exactitud estándar, es especialmente útil cuando trabajamos con datasets desbalanceados, es decir, en los que una categoría domina sobre el resto.')
    st.latex(r'''Balanced_Accuracy=\frac{\frac{VP}{VP+FN}+\frac{VN}{VN+FP}}{2}''')
    st.markdown('- **F1-Score**: media armónica entre la sensibilidad y la precisión del modelo.')
    st.latex(r'''F1-Score=\frac{2}{precision^{-1}+recall^{-1}}=2\cdot\frac{precision\cdot recall}{precision+recall}''')
st.divider()
st.markdown('La clasificación es el área dentro del aprendizaje automático en la que se han desarrollado (y se siguen desarrollando) más algoritmos. En esta sección, describiremos tan sólo algunos de los más populares.')
st.markdown('## Regresión Logística')
st.markdown('La regresión logística es una extensión de la regresión lineal para el caso de variables dependientes cualitativas binarias, aunque puede ser también utilizada en problemas de clasificación múltiples.')
with st.expander("**:star: Despliega para saber más :exclamation:**"):
    st.markdown('A partir de los conceptos básicos de la regresión lineal, se utiliza la **función sigmoide**. Esta función, define una curva que se ajusta mejor a este tipo de problemas y que puede utilizarse para predecir probabilidades, es decir, salidas en el intervalo [0,1]') 
    col1, col2 = st.columns(2)
    with col1:
        st.latex(r'''sigm\left(x\right)=\frac{1}{1+e^{-x}}''')
    with col2:
        st.image('images/7_Clasif/sigm.jpg')
    st.markdown('Suponiendo un modelo simple, $y=β_0+βx$, siendo $y$ la variable respuesta o dependiente y $x$ la variable independiente, la fórmula para la regresión logística se corresponderá con la probabilidad de que la variable salida $\hat{Y}$ tome el valor 1 para cada observación $i$.')
    st.latex(r'''P_i=P\left(\widehat{Y_i}=1\right)=\frac{1}{1+e^{-L_i}}=\frac{1}{1+e^{-\left(\beta_0+\beta_1x_i\right)}}''')
    st.markdown('De tal forma que si $P_i>0.5$, entonces $\widehat{Y_i} = 1$; mientras que para valores de $P_i<0.5$, tendremos que $\widehat{Y_i} = 0$. Por convenio, para valores de $P_i=0.5$ le asignaremos también el grupo 1.')
st.divider()
st.markdown('## Máquinas de Soporte Vectorial (SVM)')
st.markdown('Son uno de los mejores y más utilizados algoritmos de clasificación, aunque también pueden ser utilizados en regresión. Su principal objetivo es encontrar el hiperplano que mejor separe las muestras de las diferentes clases en el espacio, es decir, busca el hiperplano que proporcione la máxima separación.')
with st.expander("**:star: Despliega para saber más :exclamation:**"):
    st.markdown('Para un problema de clasificación binario, buscamos separar las observaciones $x_i \in R^d$ con sus respectivas etiquetas (en este caso para facilitar los cálculos, -1 o 1) por un hiperplano $H$ tal que:')
    st.latex(r'''H\ =\ \{x\in R^d\>|\>\ a^T\cdot x-b=0\}''')
    st.markdown('Con $a \in R^{d}$ y $b\in\ R$ sin especificar y de tal forma que se divida el espacio de observaciones en dos partes para separar las dos clases: $H_+$, y $H_-$. El objetivo es encontar el hiperplano $H$ que maximice la separación $D$ entre las dos observaciones más cercanas de cada clase')
    st.image('images/7_Clasif/svm.jpg')
    st.markdown('Finalmente el proceso se reduce a un problema de optimización de la siguiente fórmula:')
    st.latex(r'''\max_{\delta\geq0}{\sum_{i=1}^{n}\delta_i}-\frac{1}{2}\sum_{i,j=1}^{n}\delta_i\cdot\delta_j\cdot y_i\cdot y_j\cdot x_i^T\cdot x_j,\>\text{tal que}\>\sum_{i=1}^{n}\delta_i\cdot y_i=0''')
    st.latex(r'''\text{Con}\>\delta=\left(\delta_1,\ldots,\delta_n\right)^T\in R^n\>\text{multiplicadores de Lagrange.}''')
    st.markdown('Las SVM permiten utilizar _**Kernels**_ para afrontar problemas no linealmente separables. Un kernel es una función matemática que se utiliza para transformar el espacio de características original en un espacio de características de mayor dimensionalidad, buscando que en este nuevo espacio las clases si sean linealmente separables')
    st.markdown('Dada una función $\phi:\>R^d \longrightarrow R^r$ con $r>d$, de mapeo de características que convierta los vectores $x_i,x_j$ de la ecuación dual en linealmente separables. Debemos pues encontrar una función Kernel $K:\>R^d\times R^d \longrightarrow R$ tal que:')
    st.latex(r'''\phi\left(x_i\right)^T\cdot\phi\left(x_i\right)=K\left(x_i,x_j\right)''')
    st.markdown('Algunos de los Kernels más utilizados son:')
    st.markdown('- **Lineal**: no realiza transformaciones y se utiliza cuando los datos ya son linealmente separables.')
    st.latex(r'''K\left(x_i,x_j\right)=x_i\cdot x_j''')
    st.markdown('- **Polinómico**: se utiliza una función polinómica para mapear los datos a un espacio de mayor dimensión. El grado $d$ se puede ajustar.')
    st.latex(r'''K\left(x_i,x_j\right)=\left(x_i\cdot x_j+c\right)^d''')
    st.markdown('- **RBF**: transforma los datos a un espacio de dimensionalidad infinita utilizando funciones de base radial. El parámetro delta ($\delta$) controla el alcance de la influencia de cada ejemplo de entrenamiento.')
    st.latex(r'''K\left(x_i,x_j\right)=exp(-δ*\|xi-xj\|^2)''')
    st.markdown('- **Sigmoide**: aplica la función tangente hiperbólica ($tanh$) con rasgos de función sigmoide a los vectores de características.')
    st.latex(r'''K\left(x_i,x_j\right)=tanh\left(x_i\cdot x_j+c\right)''')
st.divider()
st.markdown('## Clasificación Interactiva')
st.markdown('La tarea original de este dataset es clasificar a los pacientes según el riesgo que tengan de padecer una cardiopatía en dos grupos, mayor o menor riesgo. Pongamos en práctica los distintos algoritmos estudiados en esta sección y evaluémoslos con las distintas métricas aprendidas eligiendo entre validación cruzada o división train-test. Para obtener mejores resultados, trabajaremos con los datos escalados:')
with st.expander("**:star: Despliega para aplicar los distintos modelos :exclamation:**"):
    esc = st.radio(
    "Selecciona el tipo de escalado",
    ('Estándar', 'Min-Max', 'Robusto'))
    if esc == 'Estándar':
        X_s = X_st
    elif esc == 'Min-Max':
        X_s = X_mm
    else:
        X_s = X_r
    eval = st.radio(
    "Selecciona el tipo de evaluación",
    ('Validación cruzada', 'División train-test'))
    if eval == 'Validación cruzada':
        k = st.slider(
            '¿Cúantos subconjuntos $k$ quieres?',
            min_value=2, max_value=10, step=1)

        clas_met = f.cross_validate_metrics(X_s, Y, k)
        st.write(clas_met)
        st.write('**Gráficamente**')
        met1 = st.selectbox('Selecciona la métrica:',
                        ('Accuracy','F1','Recall','Precision','Balanced_Accuracy'))
        fig1 = plt.figure(figsize=(10, 5))
        sns.barplot(x= clas_met.index , y= clas_met[met1])
        st.pyplot(fig1)
    else:
        test = st.slider(
        '¿Qué porcentaje de los datos quieres que represente el subconjunto de test?',
        min_value=5, max_value=40, step=1)
        X_train, X_test, y_train, y_test = train_test_split(X_s, Y, random_state = r, test_size=test/100)
        algm = st.selectbox('Elige el modelo que deseas evaluar',
                        ('Regresión Logística', 'SVM con kernel Lineal', 'SVM con kernel Polinómico','SVM con kernel RBF', 'SVM con kernel Sigmoide'))
        
        y_pred = f.clas_pred_mod(X_train, y_train, X_test, algm)
        st.write('**Matriz de Confusión**')
        st.pyplot(f.matriz_confusion(y_test, y_pred))
        st.write('**Métricas**')
        st.write('Exactitud (_Accuracy_): ', round(accuracy_score(y_test, y_pred), 3))
        st.write('Exactitud Balanceada (_Balanced Accuracy_): ', 
                 round(balanced_accuracy_score(y_test, y_pred), 3))
        st.write('F1-Score: ', round(f1_score(y_test, y_pred), 3))
        st.write('Precisión (_Precision_): ', round(precision_score(y_test, y_pred), 3))
        st.write('Sensibilidad (_Recall_): ', round(recall_score(y_test, y_pred), 3))

    st.markdown('### Evaluación')
    st.markdown('Conseguimos muy buenos resultados con prácticamente todas las combinaciones. Esto se debe principalmente a que este _dataset_ estaba pensado originalmente para esta función. Además, podemos observar como en general, los resultados empeoran al aumentar el tamaño del test o reducir el número $k$ de subgrupos en la validación cruzada. En cuanto a los algoritmos, obtenemos resultados muy similares.')
st.divider()
st.image('images/usal.png')
st.caption('Trabajo de Fin de Grado en Estadística')
st.caption('Aplicación web interactiva para el análisis de datos multivariantes mediante técnicas de aprendizaje automático')
st.caption('Juan Marcos Díaz')
      

    
    

