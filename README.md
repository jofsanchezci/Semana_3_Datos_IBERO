
# Algoritmo SGD (Stochastic Gradient Descent)

## ¿Qué problema resuelve SGD?

El algoritmo **SGD** busca minimizar una función objetivo (función de costo), que comúnmente se utiliza en problemas de aprendizaje supervisado:

\[ J(\theta) = \frac{1}{n} \sum_{i=1}^{n} L(y^{(i)}, f(x^{(i)}; \theta)) \]

- **Objetivo**: Encontrar parámetros óptimos \(\theta\) que minimicen la función de pérdida.

---

## Idea intuitiva

SGD actualiza los parámetros utilizando solo una muestra aleatoria por iteración, generando actualizaciones rápidas, aunque ruidosas.

---

## Funcionamiento matemático paso a paso

**1. Inicialización** de parámetros \(\theta\):

\[ \theta \leftarrow \text{valor inicial aleatorio} \]

**2. Cálculo del gradiente** usando un solo ejemplo \((x^{(i)}, y^{(i)})\):

\[ \nabla_\theta J(\theta; x^{(i)}, y^{(i)}) \]

**3. Actualización** de parámetros:

\[ \theta \leftarrow \theta - \eta \nabla_\theta J(\theta; x^{(i)}, y^{(i)}) \]

**4. Repetir el proceso** hasta la convergencia.

---

## Ejemplo: Regresión lineal simple

- **Modelo**:
\[ f(x;\theta) = \theta_0 + \theta_1 x \]

- **Función de costo (MSE)**:
\[ J(\theta) = \frac{1}{2}(y - (\theta_0 + \theta_1 x))^2 \]

- **Actualización de parámetros**:

\[
\theta_0 \leftarrow \theta_0 + \eta(y - (\theta_0 + \theta_1 x)) \\
\theta_1 \leftarrow \theta_1 + \eta(y - (\theta_0 + \theta_1 x)) \cdot x
\]

---

## Hiperparámetros importantes

- **Tasa de aprendizaje (\(\eta\))**: determina el tamaño del paso.
- **Épocas**: número total de iteraciones sobre el conjunto de datos.
- **Batch size** (para Mini-batch SGD).

---

## Retos del SGD

- **Ruido**: debido al uso de una muestra única en cada paso.
- **Elección de tasa de aprendizaje**: crucial para lograr convergencia adecuada.

---

## Variantes y mejoras

- **SGD con Momentum**: añade impulso al aprendizaje.
- **Adaptive Learning Rates**: Adam, RMSProp, Adagrad (adaptan la tasa de aprendizaje).

---

## Pseudocódigo general

```pseudo
Inicializar θ aleatoriamente
Para cada época:
    Para cada ejemplo (x(i), y(i)):
        grad = calcular_gradiente(J(θ, x(i), y(i)))
        θ = θ - η * grad
```

---

## Implementación en Python (sin librerías)

```python
import numpy as np

# datos ficticios
X = np.array([1, 2, 3, 4])
Y = np.array([5, 7, 9, 11])

# inicialización
theta0, theta1 = 0, 0
eta = 0.01  # tasa de aprendizaje
epochs = 100

for epoch in range(epochs):
    for x, y in zip(X, Y):
        y_pred = theta0 + theta1 * x
        error = y - y_pred

        # actualización
        theta0 += eta * error
        theta1 += eta * error * x

print(f"θ0: {theta0}, θ1: {theta1}")
```

---

## Conclusión

SGD es un algoritmo efectivo para optimizar funciones objetivo en grandes conjuntos de datos, especialmente cuando se requiere rapidez y escalabilidad.
