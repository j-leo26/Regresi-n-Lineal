# Regresi-n-Lineal
Mateo Salamanca, Steven Anaya, Camilo Ramirez y Jose Chavarro


# Modelo de Regresión Lineal para Predecir Patrimonio

Este proyecto utiliza un modelo de regresión lineal simple para predecir el patrimonio de un mes basado en los datos de patrimonio del mes anterior. El modelo se entrena y evalúa utilizando un conjunto de datos histórico, y se realizan predicciones para el próximo mes.

## Descripción

El código realiza las siguientes operaciones:

1. **Importación de bibliotecas**:
   - `numpy` para manipulación de arreglos numéricos.
   - `sklearn` para la implementación del modelo de regresión lineal y división de los datos en entrenamiento y prueba.
   - `matplotlib` para la visualización de los resultados.
   
2. **Conjunto de datos**:
   - Dos conjuntos de datos: uno representa el patrimonio actual de varios meses y el otro representa el patrimonio del mes anterior.
   
3. **Transformación de los datos**:
   - Los datos del patrimonio anterior se reestructuran para ser utilizados como característica de entrada para el modelo.
   
4. **División del conjunto de datos**:
   - El conjunto de datos se divide en un conjunto de entrenamiento (80%) y un conjunto de prueba (20%) para entrenar y evaluar el modelo.
   
5. **Entrenamiento del modelo**:
   - Se utiliza el algoritmo de regresión lineal de `sklearn` para entrenar el modelo con los datos de entrenamiento.
   
6. **Predicción y evaluación**:
   - El modelo predice el patrimonio actual usando los datos de prueba.
   - Se evalúa el desempeño del modelo utilizando el coeficiente de determinación (R²).
   
7. **Visualización**:
   - Se genera un gráfico que muestra los datos de prueba reales y la línea de regresión que el modelo ha calculado.

8. **Predicción del próximo mes**:
   - Se utiliza el modelo entrenado para predecir el patrimonio del siguiente mes basándose en el último valor de patrimonio actual.

## Requisitos

Para ejecutar este proyecto, necesitas las siguientes dependencias:

- Python 3.x
- numpy
- scikit-learn
- matplotlib

Puedes instalarlas utilizando pip:
```bash
pip install numpy scikit-learn matplotlib
