import numpy as np #Librería numérica
import pandas as pd
import matplotlib.pyplot as plt # Para crear gráficos con matplotlib
#%matplotlib inline # Si quieres hacer estos gráficos dentro de un jupyter notebook

from sklearn.linear_model import LinearRegression #Regresión Lineal con scikit-learn
#Defining MAPE function
def mape(y_test, pred):
    y_test, pred = np.array(y_test), np.array(pred)
    mape = np.mean(np.abs((y_test - pred) / y_test))
    return mape
myfile = pd.read_excel("data1.xlsx")


##y = f(x) calculamos y a partir de la función que hemos generado

# hacemos un gráfico de los datos que hemos generado
x =  myfile['x'].values.reshape(-1,1)
y =  myfile['y'].values.reshape(-1,1)
plt.scatter(x,y,label='data', color='blue')
plt.title('Datos')
plt.show()

#aplicando regresion 

# Importamos la clase de Regresión Lineal de scikit-learn
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_absolute_error 

regresion_lineal = LinearRegression() # creamos una instancia de LinearRegression

# instruimos a la regresión lineal que aprenda de los datos (x,y)
regresion_lineal.fit(x.reshape(-1,1), y) 

# vemos los parámetros que ha estimado la regresión lineal
#print('w = ' + str(regresion_lineal.coef_) + ', b = ' + str(regresion_lineal.intercept_))

# resultado: w = [0.09183522], b = 1.2858792525736682

# vamos a predicir y = regresion_lineal(5)
nuevo_x = np.array([5]) 
prediccion = regresion_lineal.predict(nuevo_x.reshape(-1,1))
#print("Prediction: ",prediccion)

# importamos el cálculo del error cuadrático medio (MSE)
from sklearn.metrics import mean_squared_error
# Predecimos los valores y para los datos usados en el entrenamiento
prediccion_entrenamiento = regresion_lineal.predict(x.reshape(-1,1))
# Calculamos el Error Cuadrático Medio (MSE = Mean Squared Error)
mse = mean_squared_error(y_true = y, y_pred = prediccion_entrenamiento)
# La raíz cuadrada del MSE es el RMSE
#rmse = np.sqrt(mse)
print('Error Cuadrático Medio (MSE) = ' + str(mse))
#print('Raíz del Error Cuadrático Medio (RMSE) = ' + str(rmse))

# calculamos el coeficiente de determinación R2
#r2 = regresion_lineal.score(x.reshape(-1,1), y)
#print('Coeficiente de Determinación R2 = ' + str(r2))
# Using MAPE error metrics to check for the error rate and accuracy level
LR_MAPE= mape(y,prediccion)
print("Error porcentual absoluto medio: ",LR_MAPE)
mae = mean_absolute_error(y, prediccion_entrenamiento)
print("Error absoluto medio", mae )
