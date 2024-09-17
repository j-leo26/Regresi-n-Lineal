# Importamos las bibliotecas necesarias
import numpy as np  # Biblioteca para trabajar con arreglos y operaciones matemáticas
from sklearn.model_selection import train_test_split  # Función para dividir los datos en entrenamiento y prueba
from sklearn.linear_model import LinearRegression  # Modelo de regresión lineal
import matplotlib.pyplot as plt  # Biblioteca para graficar

# Datos de patrimonio del mes actual y mes anterior
# Creamos un arreglo de NumPy con los datos del patrimonio actual de varios meses
patrimonio_actual = np.array([
    3672.12, 3406.86, 3406.86, 3577.11, 3006.40, 3006.40, 3202.47, 3202.47,
    3238.12, 3774.78, 3234.08, 2377.50, 3210.34, 4666.79, 7446.90, 3213.96,
    29457.72, 4235.06, 3590.43, 3741.92, 3741.92, 3125.75, 2932.08, 2951.21,
    3087.61, 3087.61, 2975.50, 2975.50, 3044.81, 2881.89, 2547.95, 3083.34,
    1738.59, 3366.05, 3081.81, 3817.47, 3572.25, 3053.52, 5708.93, 3841.24,
    3598.50, 4301.24, 2994.00, 3580.53, 3652.23, 2457.50, 3025.51, 3573.03,
    2859.89
])

# Creamos un arreglo de NumPy con los datos del patrimonio del mes anterior correspondiente a los mismos meses
patrimonio_anterior = np.array([
    3210.43, 3209.28, 3209.28, 3207.39, 3206.17, 3206.17, 3202.47, 3202.47,
    3196.25, 3194.22, 3192.21, 3184.32, 3183.44, 3179.20, 3178.62, 3175.49,
    3174.68, 3167.54, 3163.62, 3130.26, 3130.26, 3125.75, 3122.60, 3113.62,
    3102.04, 3102.04, 3100.49, 3100.49, 3080.36, 3076.63, 3075.83, 3064.46,
    3052.01, 3044.07, 3040.83, 3034.25, 3026.84, 3013.81, 3012.17, 3001.36,
    2998.75, 2994.40, 2994.00, 2992.94, 2982.68, 2961.23, 2951.83, 2947.71,
    2938.19
])

# Convertimos las entradas a la forma correcta
x = patrimonio_anterior.reshape(-1, 1)  # Ajustamos el array para que tenga una sola columna (1 característica)
y = patrimonio_actual  # El patrimonio actual es la variable dependiente

# Dividimos los datos en conjuntos de entrenamiento y prueba
x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.2, random_state=42)  
# El 80% de los datos se utiliza para entrenamiento y el 20% para pruebas. random_state asegura la reproducibilidad

# Creamos un modelo de regresión lineal
model = LinearRegression()  # Instanciamos el modelo de regresión lineal
model.fit(x_train, y_train)  # Entrenamos el modelo usando los datos de entrenamiento

# Hacemos predicciones sobre los datos de prueba
y_pred = model.predict(x_test)  # Predecimos el patrimonio actual en base al patrimonio anterior (datos de prueba)

# Evaluamos el modelo
r2 = model.score(x_test, y_test)  # Calculamos el coeficiente R², que mide qué tan bien el modelo explica la variabilidad
print("R2 Score:", r2)  # Imprimimos el valor de R²

# Obtenemos los coeficientes del modelo
coeffient = model.coef_[0]  # El coeficiente de regresión (pendiente de la línea)
print("Coefficient:", coeffient)  # Imprimimos el coeficiente

intercept = model.intercept_  # El término de intersección o bias del modelo
print("Intercept:", intercept)  # Imprimimos la intersección

# Graficamos los resultados
plt.scatter(x_test, y_test, color='blue')  # Gráfico de dispersión de los datos de prueba (patrimonio anterior vs actual)
plt.plot(x_test, y_pred, color='red')  # Línea de regresión que muestra las predicciones
plt.xlabel('Patrimonio Mes Anterior')  # Etiqueta del eje x
plt.ylabel('Patrimonio Mes Actual')  # Etiqueta del eje y
plt.title('Modelo de Regresión Lineal para Predecir el Patrimonio')  # Título del gráfico
plt.show()  # Mostramos el gráfico

# Predicción del patrimonio para el próximo mes
patrimonio_mes_actual = np.array([[patrimonio_actual[-1]]])  # Usamos el último valor del patrimonio actual para predecir
prediccion_mes_siguiente = model.predict(patrimonio_mes_actual)  # Predecimos el patrimonio para el próximo mes
print(f"Predicción del patrimonio para el próximo mes: {prediccion_mes_siguiente[0]}")  # Mostramos la predicción
