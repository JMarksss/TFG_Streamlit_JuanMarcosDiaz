import pandas as pd
import numpy as np
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import OneHotEncoder
from sklearn.preprocessing import StandardScaler, MinMaxScaler, RobustScaler
from sklearn.decomposition import PCA
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.cluster import KMeans
from sklearn_extra.cluster import KMedoids
from sklearn.svm import SVC, LinearSVC
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import cross_validate
from sklearn.metrics import confusion_matrix
from sklearn.svm import SVC, LinearSVC
from sklearn.linear_model import LogisticRegression
r = 23

def clean_data(datos):
    cat = datos.copy()

    cat['sex'] = cat['sex'].astype(str).replace({'0': 'Mujer', '1': 'Hombre'})
    cat['cp'] = cat['cp'].astype(str).replace({'0': 'Asintomático', '1': 'Angina_típica', '2': 'Angina_atípica', 
                                           '3': 'Dolor_no_anginoso'})
    cat['fbs'] = cat['fbs'].astype(str).replace({'0': '>120 mg/dl', '1': '<120 mg/dl'})
    cat['restecg'] = cat['restecg'].astype(str).replace({'0': 'Hipertrofia', '1': 'Normal', '2': 'Anomalía_de_onda'})
    cat['exng'] = cat['exng'].astype(str).replace({'0': 'Sí', '1': 'No'})
    cat['slp'] = cat['slp'].astype(str).replace({'0': 'Descendente', '1': 'Plano', '2': 'Ascendente'})
    cat['thall'] = cat['thall'].astype(str).replace({'0': 'Nulo', '1': 'Defecto_fijo', '2': 'Normal', '3': 'Defecto_reversible'})
    cat['output'] = cat['output'].astype(str).replace({'0': 'Menor_prob', '1': 'Mayor_prob'})

    cat.columns = ['Edad', 'Sexo', 'Dolor', 'Tensión', 'Colesterol', 'Glucemia', 'Electro', 'Frecuencia_Max', 'Angina_ej',
               'Depresión_ST', 'Pendiente_ST', 'Vasos_col', 'Talasemia', 'Cardiopatía']
    return cat

def standard(X):
    numeric_cols = X.select_dtypes(include=['float64', 'int']).columns.to_list()
    cat_cols = X.select_dtypes(include=['object', 'category']).columns.to_list()
    s = ColumnTransformer(
        [('standard', StandardScaler(), numeric_cols),
         ('onehot', OneHotEncoder(handle_unknown='ignore', drop='first'), cat_cols)],
         remainder='passthrough')
    return s.fit_transform(X)

def minmax(X):
    numeric_cols = X.select_dtypes(include=['float64', 'int']).columns.to_list()
    cat_cols = X.select_dtypes(include=['object', 'category']).columns.to_list()
    s = ColumnTransformer(
        [('standard', MinMaxScaler(), numeric_cols),
         ('onehot', OneHotEncoder(handle_unknown='ignore', drop='first'), cat_cols)],
         remainder='passthrough')
    return s.fit_transform(X)

def robust(X):
    numeric_cols = X.select_dtypes(include=['float64', 'int']).columns.to_list()
    cat_cols = X.select_dtypes(include=['object', 'category']).columns.to_list()
    s = ColumnTransformer(
        [('standard', RobustScaler(), numeric_cols),
         ('onehot', OneHotEncoder(handle_unknown='ignore', drop='first'), cat_cols)],
         remainder='passthrough')
    return s.fit_transform(X)

def pca(X, var):
    pca = PCA(n_components=var)
    X_pca = pca.fit_transform(X)
    return pca, X_pca

def plot_pca(pca):
    PC_values = np.arange(pca.n_components_) + 1
    plt.plot(PC_values, pca.explained_variance_ratio_, 'o-', linewidth=2, color='blue')
    plt.title('Scree Plot - Criterio del Codo')
    plt.xlabel('Componente Principal')
    plt.ylabel('Varianza Explicada')
    plt.show()

def km(X, algm, k):
    if algm == 'K-Means':
        km = KMeans(n_clusters=k, max_iter=50, random_state=23)
        km.fit(X)
    elif algm == 'K-Medoids':
        km = KMedoids(n_clusters=k, random_state=23)
        km.fit(X)
    else:
        km = 'Error'
    return km




def cross_validate_metrics(X, y, k):
    Clas_Modelos = {
        'Reg_Log': LogisticRegression(random_state=r),
        'SVM_rbf': SVC(random_state=r),
        'SVM_poly': SVC(kernel='poly', random_state=r),
        'SVM_sigmoid': SVC(kernel='sigmoid', random_state=r),
        'SVM_linear': LinearSVC(random_state=r),
    } 
    clas_metrics = pd.DataFrame(columns=['Modelo', 'Accuracy', 'F1', 'Recall', 'Precision', 'Balanced_Accuracy'])
    
    for nom, mod in Clas_Modelos.items():
        results = cross_validate(mod, X, y, cv=k, scoring=(
            ['accuracy', 'f1', 'recall', 'precision', 'balanced_accuracy']))
        clas_metrics = clas_metrics.append({
            'Modelo': nom,
            'Accuracy': results['test_accuracy'].mean(),
            'F1': results['test_f1'].mean(),
            'Recall': results['test_recall'].mean(),
            'Precision':results['test_precision'].mean(),
            'Balanced_Accuracy':results['test_balanced_accuracy'].mean(),
        }, ignore_index=True)

    clas_metrics = clas_metrics.round(3).sort_values("F1", ascending=False)
    clas_metrics.index =clas_metrics['Modelo']
    clas_metrics.drop(['Modelo'], axis=1, inplace=True)
    return clas_metrics

def matriz_confusion (a, b):
    cf_matrix = confusion_matrix(a, b)
    group_names = ['True Neg','False Pos','False Neg','True Pos']
    group_counts = ['{0:0.0f}'.format(value) for value in
                cf_matrix.flatten()]
    labels = [f'{v1}\n{v2}' for v1, v2 in
          zip(group_names,group_counts)]
    labels = np.asarray(labels).reshape(2,2)
    
    sns.heatmap(cf_matrix, annot=labels, fmt='', cmap='Blues')
    plt.show()

def clas_pred_mod(X_train,y_train,X_test, algm):
    if algm == 'Regresión Logística':
        mod = LogisticRegression(random_state=r)
    elif algm == 'SVM con kernel Lineal':
        mod = LinearSVC(random_state=r)
    elif algm == 'SVM con kernel Polinómico':
        mod = SVC(kernel='poly', random_state=r)
    elif algm == 'SVM con kernel RBF':
        mod = SVC(random_state=r)
    else:
        mod = SVC(kernel='sigmoid', random_state=r)
    
    mod.fit(X_train,y_train)
    y_pred = mod.predict(X_test)
    return y_pred

