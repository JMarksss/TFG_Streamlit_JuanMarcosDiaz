import pandas as pd
import streamlit as st
import numpy as np
import functions as f
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.preprocessing import LabelEncoder
import scipy.cluster.hierarchy as shc
from sklearn.cluster import AgglomerativeClustering, DBSCAN
from sklearn.metrics import silhouette_score, calinski_harabasz_score, davies_bouldin_score, mean_squared_error, mean_absolute_error, r2_score, confusion_matrix, accuracy_score, balanced_accuracy_score, f1_score, precision_score, recall_score
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import train_test_split
r = 23 #Aleatoriedad
pd.options.display.float_format = '{:,.3f}'.format 
st.set_option('deprecation.showPyplotGlobalUse', False)

st.markdown('# Analiza tus propios datos')
st.markdown('Sube tu conjunto de datos en formato .csv y completa tu propio análisis')
file = st.file_uploader("Selecciona tu archivo (Máximo 200MB)", type='csv')
if file is not None:
    dt = pd.read_csv(file)
    st.write('**_Dataframe_ completo**')
    st.write(dt)
    st.write('## Análisis descriptivo')
    with st.expander('Despliega para observar el análisis'):
        st.write(dt.describe(include='all'))
    st.divider()
    st.write('## Preprocesado')
    col1 = dt.columns
    st.write('En primer lugar debes estudiar si existe alguna variable a eliminar. Puedes saltarte esta pregunta pero es recomendable que borres aquellas columnas que sólo sirvan como identificador (Nobre, Id...), que tengan muchos valores perdidos o variables categóricas con muchas respuestas distintas')
    with st.expander('_Dataframe_ de valores perdidos'):
        nulos = pd.concat([dt.isna().sum(), dt.isna().mean()*100 ], axis=1)
        nulos.columns = ['Total nulos', 'Porcentaje %']
        st.write(nulos)
    var_el = st.multiselect(
    '¿Consideras alguna o algunas variables irelevantes y deseas eliminarlas?',
    col1)
    if var_el is None:
        ndt = dt
    else:
        ndt = dt.drop(var_el,axis=1)
    col2 = ndt.columns
    p1 = st.selectbox(
    '¿Existe una variable etiqueta o salida $Y$ que queramos predecir?',
    ('No', 'Sí'))
    if p1 == 'Sí':
        p2 = st.selectbox(
            'Elija la variable',(col2))
        p3 = st.selectbox(
            '¿Esta en formato numérico?',
            ('Sí', 'No, codifícala'))
        if p3 == 'Sí':
            Y = ndt[p2].fillna(ndt[p2].median())
        else:
            Y = ndt[p2].fillna(ndt[p2].mode())
            le = LabelEncoder()
            Y = le.fit_transform(ndt[p2])
        X = ndt.drop([p2],axis=1)
    else:
        X = ndt
        Y = 'No existe'
    st.write('Ahora aplicaremos una codificación _One-Hot_ para las variables categóricas y el tipo de escalado que elijas para las variables numéricas. Además, los valores perdidos se sustituirán por la mediana. Si siguieran existiendo nulos significaria')
    numeric_cols = X.select_dtypes(include=['float64', 'int']).columns.to_list()
    cat_cols = X.select_dtypes(include=['object', 'category']).columns.to_list()
    X[numeric_cols] = X[numeric_cols].fillna(X.median())
    X[cat_cols] = X[cat_cols].fillna(X.mode())
    esc = st.radio(
        "Selecciona el tipo de escalado",
        ('Estándar', 'Min-Max', 'Robusto', 'Ninguno'))
    if esc == 'Estándar':
        Xs = f.standard(X)
    elif esc == 'Min-Max':
        Xs = f.minmax(X)
    elif esc == 'Robusto':
        Xs = f.robust(X)
    else:
        Xs = pd.get_dummies(X, drop_first=True)

    with st.expander('Datos preprocesados $X$ y variable dependiente $Y$'):
        st.write(Xs)
        st.write('Variable dependiente $Y$:',Y)

    st.divider()
    st.markdown('## Análisis')
    anl = st.radio(
        "Selecciona el tipo de anáslisis",
        ('Reducción dimensionalidad (PCA)', 'Clustering', 'Regresión', 'Clasificación'))
    if anl == 'Reducción dimensionalidad (PCA)':
        st.markdown('### Reducción dimensionalidad (PCA)')
        var1 = st.slider('Selecciona el porcentaje de varianza mínimo explicado', 0, 100, 1)
        if st.button('Compruébalo!'):
            pca, X_pca = f.pca(Xs, var1/100)
            st.write('Número de componentes:', X_pca.shape[1])
            st.write('Porcentaje de varianza explicada por cada componente (entre 0 y 1):')
            st.write(pca.explained_variance_ratio_.reshape(1,-1))
            st.pyplot(f.plot_pca(pca))
            st.write('**Matriz transformada**')
            st.write(X_pca)
    elif anl == 'Clustering':
        st.markdown('### _Clustering_')
        st.write('Antes de continuar con tu análisis es recomendable realizar una reducción de la dimensionalidad con PCA.')
        clus_p1 = st.radio(
            "¿Quieres reducir la dimensión de tu conjunto de datos antes del _clustering_?",
            ('Sí', 'No'))
        if clus_p1 == 'Sí':
            var2 = st.slider('Selecciona el porcentaje de varianza mínimo explicado ', 0, 100, 1)
            pca, X_clus = f.pca(Xs, var2/100)
            st.write('Número de componentes:', X_clus.shape[1])
        else:
            X_clus = Xs
        clus_p2 = st.selectbox(
            'Elige el tipo de _clustering_',
            ('Patricional(K-Means, K-medoids)', 'Basado en Densidad (DBSCAN)', 'Jerárquico-Aglomerativo', ))
        if clus_p2 == 'Jerárquico-Aglomerativo':
            st.write('Ten paciencia, puede llevar algo de tiempo...')
            st.write('**Dendogramas** ')
            with st.expander('Despliega para estudiar los distintos dendogramas'):
                st.write('Pueden tardar un poco...')
                cr = st.selectbox(
                    "Selecciona el criterio (Ward solo admite la distancia Euclídea)",
                    ('Distancia Máxima', 'Distancia Mínima', 'Distancia Media', 'Distancia De Ward'))
                if cr == 'Distancia De Ward':
                    with st.spinner('Cargando...'):
                        fig = plt.figure()
                        shc.dendrogram(shc.linkage(X_clus, method='ward'))
                        st.pyplot(fig)
                else:
                    dist1 = st.selectbox(
                        "Selecciona la distancia",
                        ('Distancia Euclídea', 'Distancia de Manhattan', 'Distancia de Chebyshov', 'Distancia de Mahalanobis'))
                    if (cr=='Distancia Máxima') and (dist1=='Distancia Euclídea'):
                        with st.spinner('Cargando...'):
                            fig = plt.figure()
                            shc.dendrogram(shc.linkage(X_clus, method='complete', metric='euclidean'))
                            st.pyplot(fig)
                    elif (cr=='Distancia Máxima') and (dist1=='Distancia de Manhattan'):
                        with st.spinner('Cargando...'):
                            fig = plt.figure()
                            shc.dendrogram(shc.linkage(X_clus, method='complete', metric='cityblock'))
                            st.pyplot(fig)
                    elif (cr=='Distancia Máxima') and (dist1=='Distancia de Chebyshov'):
                        with st.spinner('Cargando...'):
                            fig = plt.figure()
                            shc.dendrogram(shc.linkage(X_clus, method='complete', metric='chebyshev'))
                            st.pyplot(fig)
                    elif (cr=='Distancia Máxima') and (dist1=='Distancia de Mahalanobis'):
                        with st.spinner('Cargando...'):
                            fig = plt.figure()
                            shc.dendrogram(shc.linkage(X_clus, method='complete', metric='mahalanobis'))
                            st.pyplot(fig)
                    elif (cr=='Distancia Mínima') and (dist1=='Distancia Euclídea'):
                        with st.spinner('Cargando...'):
                            fig = plt.figure()
                            shc.dendrogram(shc.linkage(X_clus, method='single', metric='euclidean'))
                            st.pyplot(fig)
                    elif (cr=='Distancia Mínima') and (dist1=='Distancia de Manhattan'):
                        with st.spinner('Cargando...'):
                            fig = plt.figure()
                            shc.dendrogram(shc.linkage(X_clus, method='single', metric='cityblock'))
                            st.pyplot(fig)
                    elif (cr=='Distancia Mínima') and (dist1=='Distancia de Chebyshov'):
                        with st.spinner('Cargando...'):
                            fig = plt.figure()
                            shc.dendrogram(shc.linkage(X_clus, method='single', metric='chebyshev'))
                            st.pyplot(fig)
                    elif (cr=='Distancia Mínima') and (dist1=='Distancia de Mahalanobis'):
                        with st.spinner('Cargando...'):
                            fig = plt.figure()
                            shc.dendrogram(shc.linkage(X_clus, method='single', metric='mahalanobis'))
                            st.pyplot(fig)
                    elif (cr=='Distancia Media') and (dist1=='Distancia Euclídea'):
                        with st.spinner('Cargando...'):
                            fig = plt.figure()
                            shc.dendrogram(shc.linkage(X_clus, method='average', metric='euclidean'))
                            st.pyplot(fig)
                    elif (cr=='Distancia Media') and (dist1=='Distancia de Manhattan'):
                        with st.spinner('Cargando...'):
                            fig = plt.figure()
                            shc.dendrogram(shc.linkage(X_clus, method='average', metric='cityblock'))
                            st.pyplot(fig)
                    elif (cr=='Distancia Media') and (dist1=='Distancia de Chebyshov'):
                        with st.spinner('Cargando...'):
                            fig = plt.figure()
                            shc.dendrogram(shc.linkage(X_clus, method='average', metric='chebyshev'))
                            st.pyplot(fig)
                    elif (cr=='Distancia Media') and (dist1=='Distancia de Mahalanobis'):
                        with st.spinner('Cargando...'):
                            fig = plt.figure()
                            shc.dendrogram(shc.linkage(X_clus, method='average', metric='mahalanobis'))
                            st.pyplot(fig)
            
            st.write('**Modelo**')
            st.write('En función de los resultados observados en los dendogramas, define el modelo que quieres evaluar')

            k1 = st.slider('Selecciona el número de _clusters_ $k$', min_value=2, max_value=10, step=1)
            cr2 = st.selectbox(
                "Selecciona el criterio (Ward solo admite la distancia Euclídea) ",
                ('Distancia Máxima', 'Distancia Mínima', 'Distancia Media', 'Distancia De Ward'))
            if cr2 == 'Distancia De Ward':
                aff='euclidean'
                link='ward'
            else:
                dist2 = st.selectbox(
                    "Selecciona la distancia ",
                    ('Distancia Euclídea', 'Distancia de Manhattan', 'Distancia de Chebyshov', 'Distancia de Mahalanobis'))
                if (cr2=='Distancia Máxima') and (dist2=='Distancia Euclídea'):
                    aff='euclidean'
                    link='complete'
                elif (cr2=='Distancia Máxima') and (dist2=='Distancia de Manhattan'):
                    aff='manhattan'
                    link='complete'
                elif (cr2=='Distancia Máxima') and (dist2=='Distancia de Chebyshov'):
                    aff='chebyshev'
                    link='complete'
                elif (cr2=='Distancia Máxima') and (dist2=='Distancia de Mahalanobis'):
                    aff='mahalanobis'
                    link='complete'
                elif (cr2=='Distancia Mínima') and (dist2=='Distancia Euclídea'):
                    aff='euclidean'
                    link='single'
                elif (cr2=='Distancia Mínima') and (dist2=='Distancia de Manhattan'):
                    aff='manhattan'
                    link='single'
                elif (cr2=='Distancia Mínima') and (dist2=='Distancia de Chebyshov'):
                    aff='chebyshev'
                    link='single'
                elif (cr2=='Distancia Mínima') and (dist2=='Distancia de Mahalanobis'):
                    aff='mahalanobis'
                    link='single'
                elif (cr2=='Distancia Media') and (dist2=='Distancia Euclídea'):
                    aff='euclidean'
                    link='average'
                elif (cr2=='Distancia Media') and (dist2=='Distancia de Manhattan'):
                    aff='manhattan'
                    link='average'
                elif (cr2=='Distancia Media') and (dist2=='Distancia de Chebyshov'):
                    aff='chebyshev'
                    link='average'
                elif (cr2=='Distancia Media') and (dist2=='Distancia de Mahalanobis'):
                    aff='mahalanobis'
                    link='single'

            agg_mod = AgglomerativeClustering(n_clusters=k1, affinity=aff, linkage=link)
            agg_mod.fit(X_clus)    
            st.write('**Métricas**')
            st.write('Silhouette:', round(silhouette_score(X_clus, agg_mod.labels_), 3))
            st.write('Calinski-Harabasz:', round(calinski_harabasz_score(X_clus, agg_mod.labels_), 3))
            st.write('Davies-Bouldin', round(davies_bouldin_score(X_clus, agg_mod.labels_), 3))

        elif clus_p2 == 'Patricional(K-Means, K-medoids)':
            st.write('**Modelo**')
            algor = st.radio(
                "Selecciona el algoritmo",('K-Means', 'K-Medoids'))
            k_part = st.slider('Selecciona el número de _clusters_ $k$', min_value=2, max_value=10, step=1)
            if st.button('Compruébalo!'):
                km = f.km(X_clus, algor, k_part)
                st.write('**Métricas**')
                st.write('Silhouette:', round(silhouette_score(X_clus, km.labels_), 3))
                st.write('Calinski-Harabasz:', round(calinski_harabasz_score(X_clus, km.labels_), 3))
                st.write('Davies-Bouldin', round(davies_bouldin_score(X_clus, km.labels_), 3))
        else:
            st.write('**Modelo**')
            rad = st.slider('Radio $r$', min_value=0.1, max_value=2.0, step=0.1)
            MinP = st.slider('Puntos mínimos $MinP$', min_value=2, max_value=5, step=1)
            metr_db = st.selectbox('Distancia',
                ('euclidean', 'manhattan', 'chebyshev'))
            if st.button('Adelante!'):
                dbs = DBSCAN(eps=rad, min_samples=MinP, metric=metr_db).fit(X_clus)
                st.write('**Métricas**')
                st.write('Silhouette:', round(silhouette_score(X_clus, dbs.labels_), 3))
                st.write('Calinski-Harabasz:', round(calinski_harabasz_score(X_clus, dbs.labels_), 3))
                st.write('Davies-Bouldin', round(davies_bouldin_score(X_clus, dbs.labels_), 3))


    elif anl == 'Regresión':
        st.markdown('## Regresión')
        st.markdown('Entendemos que quieres estudiar la variable dependiente $Y$ seleccionada en el preprocesado. Si no es así, modifica esta información en la sección **Preprocesado**, recuerda que para este análisis debes estudiar los datos sin escalar.')
        if esc == 'Ninguno':
            var_reg = st.multiselect(
                    'Selecciona las variables independientes para tu modelo',
                    Xs.columns)
            if len(var_reg) != 0:
                test_reg = st.slider(
                        '¿Qué porcentaje de los datos quieres que represente el subconjunto de test?',
                        min_value=5, max_value=40, step=1)
                lr = LinearRegression()
                y_reg = Y
                x_reg = Xs[var_reg]
                X_train, X_test, y_train, y_test = train_test_split(x_reg, y_reg, random_state=r, test_size=test_reg/100)
            
                if len(var_reg) == 1:
                        lr.fit(np.array(X_train).reshape(-1, 1), y_train)
                        st.write('**Coeficiente ($β_1$)**')
                        st.write(lr.coef_)
                        st.write('**Intercepto ($β_0$)**')
                        st.write(round(lr.intercept_,3))
                        y_pred = lr.predict(np.array(X_test).reshape(-1, 1)) 
                        st.write('**Métricas**')
                        st.write('$MSE$:', round(mean_squared_error(y_test, y_pred),3))
                        st.write('$MAE$:', round(mean_absolute_error(y_test, y_pred),3))
                        st.write('$R^2$:', round(r2_score(y_test, y_pred),3))
                else:
                        lr.fit(X_train, y_train)
                        st.write('**Coeficientes ($β_i, i=1,\dots,n$)**')
                        st.write(lr.coef_.reshape(1,-1))
                        st.write('**Intercepto ($β_0$)**')
                        st.write(lr.intercept_)
                        y_pred = lr.predict(X_test) 
                        st.write('**Métricas**')
                        st.write('$MSE$:', round(mean_squared_error(y_test, y_pred),3))
                        st.write('$MAE$:', round(mean_absolute_error(y_test, y_pred),3))
                        st.write('$R^2$:', round(r2_score(y_test, y_pred),3))
            else:
                st.write('No has seleccionado ninguna variable independiente!')
        else:
            st.write('En la sección de preprocesado, los datos sin escalar.')
    else:
            eval = st.radio(
                "Selecciona el tipo de evaluación",
                ('Validación cruzada', 'División train-test'))
            if eval == 'Validación cruzada':
                k_clas = st.slider(
                    '¿Cúantos subconjuntos $k$ quieres?',
                    min_value=2, max_value=10, step=1)

                clas_met = f.cross_validate_metrics(Xs, Y, k_clas)
                st.write(clas_met)
                st.write('**Gráficamente**')
                met1 = st.selectbox('Selecciona la métrica:',
                            ('Accuracy','F1','Recall','Precision','Balanced_Accuracy'))
                fig_clas = plt.figure(figsize=(10, 5))
                sns.barplot(x= clas_met.index , y= clas_met[met1])
                st.pyplot(fig_clas)
            else:
                test_clas = st.slider(
                    '¿Qué porcentaje de los datos quieres que represente el subconjunto de test? ',
                    min_value=5, max_value=40, step=1)
                X_train, X_test, y_train, y_test = train_test_split(Xs, Y, random_state = r, test_size=test_clas/100)
                algorr = st.selectbox('Elige el modelo que deseas evaluar',
                        ('Regresión Logística', 'SVM con kernel Lineal', 'SVM con kernel Polinómico','SVM con kernel RBF', 'SVM con kernel Sigmoide'))
        
                y_pred = f.clas_pred_mod(X_train, y_train, X_test, algorr)
                st.write('**Matriz de Confusión**')
                if len(Y.value_counts()) == 2:
                    st.pyplot(f.matriz_confusion(y_test, y_pred))
                elif len(Y.value_counts()) > 2:
                    m_con = plt.figure(figsize=())
                    sns.heatmap(confusion_matrix(y_test, y_pred), annot=True, fmt='', cmap='Blues')
                    st.pyplot(m_con)
                st.write('**Métricas**')
                st.write('Exactitud (_Accuracy_): ', round(accuracy_score(y_test, y_pred), 3))
                st.write('Exactitud Balanceada (_Balanced Accuracy_): ', 
                     round(balanced_accuracy_score(y_test, y_pred), 3))
                st.write('F1-Score: ', round(f1_score(y_test, y_pred), 3))
                st.write('Precisión (_Precision_): ', round(precision_score(y_test, y_pred), 3))
                st.write('Sensibilidad (_Recall_): ', round(recall_score(y_test, y_pred), 3))



st.divider()
st.image('images/usal.png')
st.caption('Trabajo de Fin de Grado en Estadística')
st.caption('Aplicación web interactiva para el análisis de datos multivariantes mediante técnicas de aprendizaje automático')
st.caption('Juan Marcos Díaz')