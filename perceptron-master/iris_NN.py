#!/usr/bin/env python
# coding: utf-8

# # Ejemplo de entrenamiento de una red neuronal MLP
# 
# Este es un ejemplo para el curso de Inteligencia Artificial 2019-II, entrenando un modelo de red neuronal MLP para el conjunto de datos iris.cvs.
# 
# El conjunto de datos Iris contiene datos sobre tres tipos de flores Iris. Este es un conjunto de datos multi-variables construidos por Edgar Anderson para cuantificar la variación morfológica de tres especies de flores de iris.
# 
# El conjunto de datos contiene tres clases de flores que son: Iris Setosa, Iris Versicolour, e Iris Virginica. Cada clase cuenta con 50 ejemplos registrados, para un total de 150 ejemplos en el conjunto de datos. 
# 
# Los atributos, variables independientes, o características registrados para cada ejemplo son:
# 
# 1. sepal length in cm 
# 2. sepal width in cm 
# 3. petal length in cm 
# 4. petal width in cm
# 
# El conjunto de datos está disponible en: https://archive.ics.uci.edu/ml/datasets/Iris/

# In[1]:


import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
import numpy as np
import warnings

from sklearn.model_selection import train_test_split
from sklearn.neural_network import MLPClassifier
from sklearn.neural_network import MLPRegressor
from sklearn import metrics
from sklearn.metrics import classification_report
from sklearn.metrics import accuracy_score
from sklearn.metrics import mean_squared_error
from sklearn.exceptions import ConvergenceWarning
from sklearn.model_selection import GridSearchCV


# Cargando el conjunto de datos de un archivo extensión .cvs, y mostrando información del archivo.

# In[2]:


#Cargando datos
#No se le olvide actualizar este path a la ubicación del archivo de datos
iris = pd.read_csv("training.csv")
test=pd.read_csv("testing.csv")
#Informacion de los datos
print(iris.info())


# Visualizando la distribución de las clases a través de un histograma.

# In[3]:


#Histograma del atributo clase
ax=plt.subplots(1,1,figsize=(10,8))
sns.countplot('class',data=iris)
plt.title("numero de datos")
plt.show()


# Visualizando los histogramas de cada atributo.

# In[4]:


#Histograma de atributos predictores

iris.hist(edgecolor='black', linewidth=1.2)
fig=plt.gcf()
#fig.set_size_inches(12,12)
plt.show()


# Diagrama de cajas de los atributos o variables independientes.

# In[5]:


#boxplot de las variables numericas
#iris = iris.drop('Id',axis=1)
box_data = iris #variable representing the data array
box_target = iris["class"] #variable representing the labels array
sns.boxplot(data = box_data,width=0.5,fliersize=6)
#sns.set(rc={'figure.figsize':(2,15)})
plt.show()


# Observando la correlación entre variables permite descubrir posibles dependencias entre las variables independientes.

# In[6]:


#observando correlacion entre variables


print (iris)
X = iris.iloc[:, 0:6]
f, ax = plt.subplots(figsize=(10, 8))
corr = X.corr()
print(corr)
sns.heatmap(corr, mask=np.zeros_like(corr, dtype=np.bool), 
          cmap=sns.diverging_palette(220, 10, as_cmap=True),square=True, ax=ax, linewidths=.5)
plt.show() 


# En la matriz de correlación se observa un alto coeficiente para las variables petalWidth y PetalLength. Podemos mirar el comportamiento de las dos variables utilizando regresión lineal.

# In[7]:


#observando relaciones entre los datos
#sns.regplot(x='PetalLengthCm', y='PetalWidthCm', data=iris);
#sns.set(rc={'figure.figsize':(2,5)})
#plt.show()


# Una vez observado y analizado las variables del conjunto de datos vamos a hacer una primera prueba preliminar para observar cómo se comportaría el modelo de red neuronal. La configuración de este primer modelo se indica a través de los parámetros de MPLClassifier

# In[8]:


#Separando los datos en conjuntos de entrenaimiento y prueba
X = iris.iloc[:,1:6].values
y = iris.iloc[:, 0:1].values
 
x_test = test.iloc[:,1:6].values
y_test  = test.iloc[:, 0:1].values



#print (X)
#print (y)


#Como esta es una primera prueba prelimintar coloco esta instrucción para que nos me saque un warning
#debido a que el modelo no+o alcanza a converger
#warnings.filterwarnings("ignore", category=ConvergenceWarning, module="sklearn")

#Entrenando un modelo de red neuronal MLP para clasificación
#MLPClassifier permite configurar las capas ocultas del modelo, la instrucción de abajo indica que el modelo tendrá
#dos capas ocultas cada una con 3 neuronas. Algo como esto hidden_layer_sizes = (3,3,2) indicarían tres capas ocultas con
#3,3 y 2 neuronas respectivamente
model =  MLPClassifier(hidden_layer_sizes = (8,3,3), alpha=0.00001, max_iter=100000) 
model.fit(X,y.ravel()) #Training the model


# Una vez entrenado el modelo, debemos evaluarlo sobre el conjunto de datos reservado para prueba, y utilizar algunas métricas para observar que tan bien quedo entrenado el modelo. En esta primera prueba utilizamos como métricas el porcentaje de precisión del modelo y la matriz de confusión.

# In[9]:


#Test the model
predictions = model.predict(x_test)
print(accuracy_score(y_test, predictions))
print("Confusion matrix:\n%s" % metrics.confusion_matrix(y_test, predictions))


# Ahora vamos a ajustar los parámetros del modelo utilizando GridSearch

# In[13]:


param_grid = [{'hidden_layer_sizes' : [(3,3),(3,2),(4,2),(5,2),(2,2,2)], 'max_iter':[100,1000,1000,100000]}, 
              {'alpha':  [0.0001,0.01,0.000001,0.0001]}]







model = MLPClassifier()
grid_search = GridSearchCV(model, param_grid, cv=5, scoring='accuracy', iid=False)
grid_search.fit(X, y.ravel())


# In[15]:


print(grid_search.best_params_)

R_ind=grid_search.best_estimator_
new_predictions_R= R_ind.predict(x_test)




# Ahora vamos a graficar los resultados obtenidos. En la gráfica se podrá observar un plot de los datos original, de la aproximación obtenida con el primer modelo sin ajustar parámetros, y del modelo con los mejores parámetros encontrados por GridSearchCV.

'''X = np.arange(1, len(y_test)+1)
plt.figure()
plt.plot(X, y_test.ravel(), 'k', label='Datos Originales')
plt.plot(X, predictions, 'r', label='Primera Aproximación')
plt.plot(X, new_predictions_R, 'g', label='Segunda Aproximación')
plt.xlabel('data')
plt.ylabel('target')
plt.title('Original data Vs Predictions')
plt.legend()
plt.show()
'''


