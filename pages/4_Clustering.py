import pandas as pd
import streamlit as st
import functions as f
import joblib
from sklearn.cluster import DBSCAN
from sklearn.metrics import silhouette_score, calinski_harabasz_score, davies_bouldin_score
pd.options.display.float_format = '{:,.3f}'.format 

datos = pd.read_csv('data/heart.csv')
cat = f.clean_data(datos)
Y = datos['output'] 
X = cat.drop(['Cardiopatía'], axis=1)
X_st = f.standard(X)
pca, X_pca = f.pca(X_st, 0.7)

st.markdown('# _Clustering_')
st.markdown('Consiste en agrupar un conjunto de datos en subconjuntos, conglomerados o "_clusters_" en función de la similitud entre ellos, de tal forma que los individuos pertenecientes a cada _cluster_ sean más similares entre ellos mismos que frente a los de los otros _clusters_. Así pues, el objetivo principal del _clustering_ es identificar patrones o estructuras en los datos, que pueden ser útiles para la toma de decisiones o para descubrir información valiosa.')
st.image('images/5_Clustering/clus.png')
st.markdown('En general, el _clustering_ funciona mejor cuando trabajamos con _datasets_ con un número reducido de variables. Además, al trabajar con distancias, para su correcta implementación es necesario que las variables estén en un rango similar, es decir, que estén escaladas. Por estos motivos, y sin pérdida de generalidad, para la implementación y evaluación de estos modelos en esta sección, vamos a utilizar el _dataset_ con escalado estandar y tras aplicarle PCA que conserve al menos el 70% de la varanza (5 componentes)')
st.markdown('Para evaluar la similitud se emplean distintas medidas de distancia dependiendo de los datos y también del algoritmo empleado. Aunque existen infinidad de medidas, podemos ver algunas de las más utilizadas en el siguiente desplegable:')
with st.expander("**:star: Despliega para conocer las distintas distancias :exclamation:**"):
    st.markdown('- **Distancia de Manhattan**: para dos puntos $A$ y $B$, con coordenadas $X_i$ e $Y_i$ respectivamente donde $i=1,\dots,n$ indica la dimensión, la distancia es:')
    st.latex(r'''d(A,B) = \sum_{i=1}^n|X_i-Y_i|''')
    st.markdown('- **Distancia Euclídea**')
    st.latex(r'''d(A,B) = \sqrt{\sum_{i=1}^n(X_i-Y_i)^2}''')
    st.markdown('- **Distancia de Chebyshev**')
    st.latex(r'''d(A,B) = \max_i|X_i-Y_i|''')
    st.markdown('- **Distancia de Mahalanobis**: sea $S$ la matriz de covarianza entre $A$ y $B$')
    st.latex(r'''d(A,B) = \sqrt{(X_i-Y_i)S^{-1}(X_i-Y_i)^T}''')
    st.image('images/5_Clustering/dist.jpg')

st.divider()
st.markdown('### Métricas')
st.markdown('Para casi todos los algoritmos de _clustering_, debemos utilizar distintas técnicas y métricas para determinar cuál es el número de clusters óptimo en cada caso, así como para evaluar la calidad de las agrupaciones. Es conveniente utilizar varias métricas a la hora de evaluar la calidad de los _clusters_ ya que no existe una sola métrica universalmente aceptada para esta tarea y, dependiendo de cada caso concreto, priorizaremos distintas características de los clusters.')
with st.expander("**:star: Despliega para conocer algunas de estas métricas :exclamation:**"):
    st.markdown('- **Coeficiente de Silhouette**: mide la similitud de cada observación con su propio _cluster_ en comparación con el resto de _clusters_. Este valor varía entre -1 y 1, de tal forma que un valor cercano a 1 indica que el punto está asignado correctamente, un valor de 0 indicaría indiferencia y un valor negativo indicaría que la observación estaría mejor agrupada en otro _cluster_. Donde $d_{int}(x)$ será la distancia media entre el punto $x$ y el resto de los puntos de su cluster, mientras que $d_{ext}(x)$ es la distancia media entre el punto $x$ y los puntos del _cluster_ más cercano al que no pertenezca.')
    st.latex(r'''s\left(x\right)=\frac{d_{ext}\left(x\right)-d_{int}\left(x\right)}{\max_x{\left(d_{int}\left(x\right)-d_{ext}\left(x\right)\right)}}''')
    st.markdown('- **Inercia (SSE)**: mide la suma de las distancias al cuadrado entre cada punto y el punto central o centroide de su _cluster_ correspondiente. Un valor bajo de esta métrica indicará que los puntos están más cerca de su centroide, es decir, los _clusters_ serán más compactos. Elegida una medida de distancia ($d$), siendo $k$ el número de _clusters_ y $x$ es un unto perteneciente al _cluster_ $C_i$ cuyo punto central es $m_i$ con $i=1,\cdots,k$ tenemos:')
    st.latex(r'''SSE=\sum_{i=1}^{k}\sum_{x\in C_i}{d^2\left(m_i,x\right)}''')
    st.markdown('- **Separación (SSB)**: nos indicará como de lejos o cuan separados están unos _clusters_ de otros. Dados una distancia ($d$), $k$ el número de _clusters_, $n_i$ el número de elementos pertenecientes al _cluster_ $i$, $m_i$ el centroide de dicho _cluster_ y $\overline{x}$ la media del conjunto total de datos.')
    st.latex(r'''SSB=\sum_{i=1}^{k}n_i \cdot d^2(m_i,\bar{x})''')
    st.markdown('- **Coeficiente de Calinski-Harabasz (CH)**: esta métrica es una fusión de las dos anteriores. Mide la dispersión intra _cluster_ y la dispersión entre los distintos _clusters_, de tal forma que un valor alto de este coeficiente indica una mejor calidad de _clustering_. Con $n$ el número total de observaciones y $k$ el de _clusters_ tenemos:')
    st.latex(r'''CH=\frac{SSB/\left(k-1\right)}{SSE/\left(n-k\right)}''')
    st.markdown('- **Índice de Davies-Bouldin (DBI)**: evalúa la compacidad y separación de los _clusters_, de tal forma que un valor bajo de esta métrica indicaría calidad en el agrupamiento. Para $k$ _clusters_ y con $\overline{d_i}, \overline{d_j}$ la distancia media entre cada punto del _cluster_ $i$ o $j$ respectivamente a su centroide; $m_i$ y $m_j$ dichos centroides y $d(m_i,m_j)$ la distancia entre ambos tenemos:')
    st.latex(r'''DBI=\frac{1}{k}\sum_{i=1,i \neq j}^{k}\max{\left(\frac{\bar{d_i}+\bar{d_j}}{d\left(m_i,m_j\right)}\right)}''')

st.divider()

st.markdown('Al existir muchas formas distintas de agrupar los datos en función de muchas medidas diferentes, cuando hablamos de _clustering_ no hablamos de un único algoritmo, sino de una técnica de agrupamiento de objetos que engloba numerosos algoritmos. A continuación, estudiemos algunas de estas técnicas.')

st.markdown('## _Clustering_ Jerárquico')
st.markdown('Este tipo de agrupa las observaciones o individuos en una estructura jerárquica de _clusters_, de manera que crea _clusters_ sucesivos utilizando otros previamente establecidos. Dentro de este tipo, destacan los algoritmos **Aglomerativos**. Estos modelos comienzan considerando cada individuo como un _cluster_ y los van fusionando entre ellos formando así _clusters_ cada vez más grandes. Una vez elegida la medida de distancia (por ejemplo, siendo $d$ la distancia euclídea), comenzamos uniendo los dos clusters (en este caso individuos) más cercanos en función de alguno de los siguientes criterios:')
with st.expander("**:star: Despliega para conocer los distintos criterios :exclamation:**"):
    st.markdown('- **Distancia máxima (_Complete Linkage_)**: dados dos clusters $A, B$ con $x_i\in A (i=1,\dots,n)$ y $y_j\in B (j=1,\dots,m)$ los elementos o individuos pertenecientes a los clusters y $d$ la medida de distancia escogida. Se define como se define la distancia máxima entre los clusters $A$ y $B$ como la distancia entre las observaciones más alejadas $𝑥_𝑖 ∈ 𝐴$ e $𝑦_𝑗 ∈ 𝐵$:')
    st.latex(r'''\max_{i,j} {d(x_i,y_j), x_i\in A, y_j\in B}''')
    st.markdown('- **Distancia minima (_Single Linkage_)**: se define la distancia mínima entre los clusters $A$ y $B$ como la distancia más pequeña para cada par $𝑥_𝑖 ∈ 𝐴$ e $𝑦_𝑗 ∈ 𝐵$:')
    st.latex(r'''\min_{i,j} {d(x_i,y_j), x_i\in A, y_j\in B}''')
    st.markdown('- **Distancia media (_Average Linkage_)**: se define la distancia media entre los clusters $A$ y $B$ como el promedio de todas las distancias entre cada par $𝑥_𝑖 ∈ 𝐴$ e $𝑦_𝑗 ∈ B$')
    st.latex(r'''\frac{1}{n\cdot m}\sum_{i=1}^n\sum_{j=1}^m d(x_i, y_j)''')
    st.markdown('- **Distancia de Ward**:  tiene en cuenta la varianza dentro de cada grupo y busca minimizar el incremento de esta al combinar dos grupos en uno solo. Comienza considerando las distancias entre clusters como el cuadrado de la distancia euclídea y, en cada iteración se busca el par de clusters con cuya unión se minimice esta varianza.')

st.markdown('### _Clustering_ Jerárquico Aglomerativo Interactivo')
st.markdown('Este tipo de _clustering_ tiene la ventaja de que no necesitamos saber previamente el número de grupos que queremos formar, a traves de un dendograma podemos observar cúal sería el número óptimo para cada distancia y criterio. Además podemos evaluar la calidad de los distintos modelos con las gráficas vistas anteriormente.')

with st.expander("**:star: Despliega para aplicar los distintos modelos :exclamation:**"):
    st.markdown('#### Dendogramas')
    cr = st.selectbox(
        "Selecciona el criterio (Ward solo admite la distancia Euclídea)",
        ('Distancia Máxima', 'Distancia Mínima', 'Distancia Media', 'Distancia De Ward'))
    if cr == 'Distancia De Ward':
        st.image('images/5_Clustering/dendogram/ward.png')
        st.markdown('Parece una buena agrupación con 2 _clusters_ bien diferenciados y balanceados.')
    else:
        dist = st.selectbox(
            "Selecciona la distancia",
            ('Distancia Euclídea', 'Distancia de Manhattan', 'Distancia de Chebyshov', 'Distancia de Mahalanobis'))
        if (cr == 'Distancia Máxima') and (dist == 'Distancia Euclídea'):
            st.image('images/5_Clustering/dendogram/maxeu.png')
            st.markdown('Distinguimos 5 _clusters_ distintos bastante bien diferenciados.')
        elif (cr == 'Distancia Máxima') and (dist == 'Distancia de Manhattan'):
            st.image('images/5_Clustering/dendogram/maxmh.png')
            st.markdown('Distinguimos 5 _clusters_ distintos, pero desbalanceados.')
        elif (cr == 'Distancia Máxima') and (dist == 'Distancia de Chebyshov'):
            st.image('images/5_Clustering/dendogram/maxch.png')
            st.markdown('Distinguimos 6 _clusters_ .')
        elif (cr == 'Distancia Máxima') and (dist == 'Distancia de Mahalanobis'):
            st.image('images/5_Clustering/dendogram/maxma.png')
            st.markdown('Distinguimos 5 _clusters_, n grupo está compuesto sólo por un individuo (_outlier_).')
        elif (cr == 'Distancia Mínima') and (dist == 'Distancia Euclídea'):
            st.image('images/5_Clustering/dendogram/mineu.png')   
            st.markdown('Distinguimos 2 _clusters_ muy desbalanceados por la presencia de _outliers_.')
        elif (cr == 'Distancia Mínima') and (dist == 'Distancia de Manhattan'):
            st.image('images/5_Clustering/dendogram/minmh.png')   
            st.markdown('Distinguimos 2 _clusters_ muy desbalanceados por la presencia de _outliers_.')
        elif (cr == 'Distancia Mínima') and (dist == 'Distancia de Chebyshov'):
            st.image('images/5_Clustering/dendogram/minch.png')   
            st.markdown('Distinguimos 2 _clusters_ muy desbalanceados por la presencia de _outliers_.')
        elif (cr == 'Distancia Mínima') and (dist == 'Distancia de Mahalanobis'):
            st.image('images/5_Clustering/dendogram/minma.png')   
            st.markdown('Distinguimos 2 _clusters_ muy desbalanceados por la presencia de _outliers_.')
        elif (cr == 'Distancia Media') and (dist == 'Distancia Euclídea'):
            st.image('images/5_Clustering/dendogram/aveu.png')   
            st.markdown('Distinguimos 3 _clusters_ muy desbalanceados por la presencia de _outliers_.')
        elif (cr == 'Distancia Media') and (dist == 'Distancia de Manhattan'):
            st.image('images/5_Clustering/dendogram/avmh.png')   
            st.markdown('Distinguimos 2 _clusters_ muy desbalanceados por la presencia de _outliers_.')
        elif (cr == 'Distancia Media') and (dist == 'Distancia de Chebyshov'):
            st.image('images/5_Clustering/dendogram/avch.png')   
            st.markdown('Distinguimos 5 _clusters_ muy desbalanceados por la presencia de _outliers_.')
        elif (cr == 'Distancia Media') and (dist == 'Distancia de Mahalanobis'):
            st.image('images/5_Clustering/dendogram/avma.png') 
            st.write('Distinguimos varios _clusters_ pero muy descompensados, uno destaca y el resto son muy pequeños.')
        elif (cr == 'Distancia Media') and (dist == 'Distancia Euclídea'):
            st.image('images/5_Clustering/dendogram/avma.png')   
            st.markdown('Distinguimos alrededor de 9 _clusters_ muy desbalanceados por la presencia de _outliers_.')

    st.markdown('#### Evaluación')
    agg_metrics = joblib.load(open('data/agg_metrics.pkl','rb')) #Cargamos la tabla de métricas rápidamente
    met = st.selectbox(
        "Selecciona la métrica",
        ('Silhouette', 'Calinski-Harabasz', 'Davies-Bouldin'))
    if met == 'Silhouette':
        st.image('images/5_Clustering/metrics/aggsil.png')
    elif met == 'Calinski-Harabasz':
        st.image('images/5_Clustering/metrics/aggcal.png')
    else:
        st.image('images/5_Clustering/metrics/aggdb.png')
    
    dm = st.checkbox('_Dataframe_ completo de métricas')
    if dm:
        st.write(agg_metrics)
    
    st.markdown('Atendiendo a las métricas, observamos que los modelos que emplearon el criterio de distancia mínima, obtienen muy buenos valores en las métricas de Silhouette y Davies-Bouldin, pero no obtendrían buenos resultados en la de Calinski-Harabasz. Observando sus dendogramas, está claro que estos modelos están muy afectadospor la presencia de uno o varios _outliers_ que se diferencian enormemente del resto de valores.')
    st.markdown('Por otra parte el **modelo de Ward** es el que mejor valor obtiene en la métrica de Calinski-Harabasz, representando esta el ratio entre inercia y separación, además obtiene valores relativamente buenos de Silhouette. Observando el dendograma concluimos que esta división es la que mejor se ajusta a los datos y que por tanto, **2 es el número correcto de conglomerados**.')
st.divider()

st.markdown('## _Clustering_ Particional')
st.markdown('Estos algoritmos se basan en la especificación de un número inicial de grupos o _clusters_, y en la reasignación iterativa de observaciones, individuos o puntos entre los grupos hasta la convergencia. Además, estos algoritmos suelen determinar todos los grupos a la vez. Destacan principalmente dos algoritmos de este tipo:')
with st.expander('**:star: Despliega para conocer estos algoritmos :exclamation:**'):
    col1, col2 = st.columns(2)
    with col1:
        st.markdown("### K-Means")
        st.markdown('La idea general del algoritmo es asignar cada punto u observación al _cluster_ con el centro, denominado centroide, más cercano. Este **centroide** es la media de todos los puntos pertenecientes al _cluster_, es decir, sus coordenadas se corresponden la media aritmética para cada dimensión por separado sobre todos los puntos del _cluster_.')
        st.image('images/5_Clustering/kmeans.jpg')
    with col2:
        st.markdown("### K-Medoids")
        st.markdown('En lugar de seleccionar los centroides (medias aritméticas), el K-Medoids selecciona los medoides como centros de los clusters. Un **medoide** es una observación de un _cluster_ que minimiza la suma de las distancias al resto de observaciones al mismo _cluster_. Es decir, un medoide es el objeto que se encuentra más cerca de todos los demás objetos en el cluster, siendo así el objeto más representativo de dicho _cluster_.')
   
st.markdown('### _Clustering_ Particional Interactivo')
st.markdown('Aunque gracias al _clustering_ jerárquico ya sabemos que el número óptimo de conglomerados es 2, pongamos en práctica estos algoritmos y evaluemos su desempeño para distintos valores de $k$.')

with st.expander('**:star: Despliega para aplicar los distintos modelos :exclamation:**'):
    alg = st.radio(
    "Selecciona el algoritmo",('K-Means', 'K-Medoids'))
    k = st.slider('Selecciona el número de _clusters_ $k$', min_value=2, max_value=10, step=1)
    if st.button('Compruébalo!'):
        km = f.km(X_pca, alg, k)
        st.write('**Métricas**')
        st.write('Silhouette:', round(silhouette_score(X_pca, km.labels_), 3))
        st.write('Calinski-Harabasz:', round(calinski_harabasz_score(X_pca, km.labels_), 3))
        st.write('Davies-Bouldin', round(davies_bouldin_score(X_pca, km.labels_), 3))

    st.markdown('#### Evaluación')
    st.markdown('Atendiendo a las métricas, de nuevo deducimos que **2 es el número adecuado de _clusters_** para ammbos algoritmos. Además, podemos observar que los resultados son muy parecidos, pero ligeramente **mejores para el K-Means**. Si comparamos estos modelos con el de Ward también obtenemos mejores resultados.')
st.divider()
st.markdown('## _Clustering_ Basado en Densidad (DBSCAN)')
st.markdown('En este algoritmo, las regiones con una alta densidad de puntos indican la existencia de _clusters_, mientras que las regiones con una baja densidad de puntos indicarían _clusters_ de ruido. Así pues, la idea clave es que, para cada punto de un _cluster_, la vecindad de un radio dado tiene que contener al menos un número mínimo de puntos.')
with st.expander('**:star: Despliega para saber más:exclamation:**'):
    st.markdown('Este algoritmo necesita de al menos dos hiper-parámetros fijados de antemano:')
    st.markdown('- **Radio** ($r$): longitud que delimita el área, región o vecindario para cada punto.')
    st.markdown('- **Puntos mínimos** ($MinP$): número mínimo de puntos u observaciones que deben existir en una región o vecindario para que sea considerada un _cluster_.')
    st.image('images/5_Clustering/dbscan.jpg')

st.markdown('### DBSCAN Interactivo')
st.markdown('Póngamos en práctica estos modelos y evaluemos sus resultados de manera interactiva seleccionando el tipo de distancia, así como los hiper-parámetros $r$ y $MinP$')
with st.expander('**:star: Despliega para aplicar los distintos modelos :exclamation:**'):
    rad = st.slider('Radio $r$', min_value=0.1, max_value=2.0, step=0.1)
    MinP = st.slider('Puntos mínimos $MinP$', min_value=2, max_value=5, step=1)
    metr = st.selectbox('Distancia',
        ('euclidean', 'manhattan', 'chebyshev'))
    if st.button('Adelante!'):
        dbs = DBSCAN(eps=rad, min_samples=MinP, metric=metr).fit(X_pca)
        st.write('**Métricas**')
        st.write('Silhouette:', round(silhouette_score(X_pca, dbs.labels_), 3))
        st.write('Calinski-Harabasz:', round(calinski_harabasz_score(X_pca, dbs.labels_), 3))
        st.write('Davies-Bouldin', round(davies_bouldin_score(X_pca, dbs.labels_), 3))
    
    st.markdown('#### Evaluación')
    st.markdown('Atendiendo a las métricas, concluimos que en general no estamos consiguiendo buenas agrupaciones con este método ya que resulta **difícil determinar el valor de los hiper-parámetros**. Además, algunas combinaciones dan errores.')
st.divider()
st.image('images/usal.png')
st.caption('Trabajo de Fin de Grado en Estadística')
st.caption('Aplicación web interactiva para el análisis de datos multivariantes mediante técnicas de aprendizaje automático')
st.caption('Juan Marcos Díaz')


