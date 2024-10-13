import numpy as np
import matplotlib.pyplot as plt
import random  # Importamos el módulo random de Python
from scipy.misc import derivative

# Convierte un arreglo binario en un número decimal
def bin2dec(x):
    return int(''.join(map(lambda x: str(int(x)), x)), 2)

# Función objetivo o fitness que estamos evaluando. En este caso: f(x) = -x * sin(sqrt(abs(x)))
def funcion_objetivo(x):
    return -x * np.sin(np.sqrt(np.abs(x)))

# Genera un rango de valores de -512 a 512 para graficar la función objetivo
x1 = np.linspace(-512, 512, 1000)
f1 = funcion_objetivo(x1)

# Grafica la función objetivo para ver cómo se comporta
plt.figure()
plt.plot(x1, f1)
plt.title("Función objetivo")
plt.xlabel("x")
plt.ylabel("f(x)")
plt.grid()
plt.show()



# Evalúa cada individuo en la población
# La población es una lista de individuos donde cada individuo es un arreglo binario con su fitness.
def evaluar_poblacion(poblacion):
    for i in range(len(poblacion)):
        # Convierte el individuo de binario a decimal, ignorando el primer bit (que es el signo)
        ind_dec = bin2dec(poblacion[i][0][1:])
        # Si el primer bit es 1, el número es negativo
        if poblacion[i][0][0] == 1:
            ind_dec = ind_dec * -1
        # Calcula el fitness del individuo y lo guarda (invertido para minimizar)
        poblacion[i][1] = -1 * funcion_objetivo(ind_dec)

# Selección por competencia: selecciona los mejores K individuos de la población
def seleccion_por_competencia(poblacion, k):
    seleccionados = []
    tam_poblacion = len(poblacion)

    # Continua seleccionando hasta llenar el número de individuos de la población original
    while len(seleccionados) < tam_poblacion:
        # Selecciona aleatoriamente k competidores de la población
        indices_competidores = np.random.choice(len(poblacion), k, replace=False)
        competidores = [poblacion[i] for i in indices_competidores]

        # Ordena los competidores por su fitness (el mejor al principio)
        competidores_ordenados = sorted(competidores, key=lambda x: x[1])

        # Selecciona los dos mejores individuos
        seleccionados.append(competidores_ordenados[0])
        seleccionados.append(competidores_ordenados[1])

    # Devuelve la lista de seleccionados, limitada al tamaño original de la población
    return seleccionados[:tam_poblacion]

# Selección por ventanas: divide la población en ventanas y selecciona un individuo de cada ventana
def seleccion_por_ventanas(poblacion, num_ventanas):
    # Ordenar la población según su fitness de menor a mayor (para minimizar)
    poblacion_ordenada = sorted(poblacion, key=lambda x: x[1])

    seleccionados = []
    tam_poblacion = len(poblacion)

    # Define el tamaño de la primera ventana
    tam_ventana_inicial = int(tam_poblacion / num_ventanas)
    
    
    ventana_actual = tam_ventana_inicial
  

    # Repite para cada ventana
    for i in range(num_ventanas):
        # Define los límites de la ventana actual en la población ordenada
        ventana = poblacion_ordenada[0:ventana_actual]

        # Selecciona aleatoriamente un individuo de esta ventana
        seleccionado = random.choice(ventana)
        seleccionados.append(seleccionado)
        
        #Reduce el tamaño de la ventana en un 10% para la próxima iteración
        ventana_actual = max(1, int(ventana_actual * 0.9))  # Asegura que el tamaño sea al menos 1 y entero

    return seleccionados

# Función de cruce (crossover) que genera dos hijos por cada par de padres
def cruce(padre1, padre2):
    # Elige un punto de corte aleatorio (ignora el bit de signo)
    punto_corte = np.random.randint(1, len(padre1[0]) - 1)
    # Hijo 1: parte inicial de padre1 + parte final de padre2
    hijo1 = np.concatenate((padre1[0][:punto_corte], padre2[0][punto_corte:]))
    # Hijo 2: parte inicial de padre2 + parte final de padre1
    hijo2 = np.concatenate((padre2[0][:punto_corte], padre1[0][punto_corte:]))
    return [hijo1, []], [hijo2, []]  # Devuelve los dos hijos con espacio para su fitness

# Función de mutación para aplicar mutaciones a los individuos
def mutacion(individuo, prob_mutacion):
    # Recorre todos los bits del individuo
    for i in range(len(individuo[0])):
        # Aplica mutación con la probabilidad dada
        if np.random.rand() < prob_mutacion:
            # Cambia el bit (0 a 1 o 1 a 0)
            individuo[0][i] = 1 - individuo[0][i]
    return individuo

# Parámetros del algoritmo genético
cant_bits = 10          # Número de bits que representa a cada individuo
tam_poblacion = 100    # Tamaño de la población
prob_mutacion = 0.001   # Probabilidad de mutación
generaciones = 1000     # Número de generaciones

# Inicialización de la población con individuos aleatorios
poblacion = []
for i in range(tam_poblacion):
    individuo = np.random.randint(0, 2, cant_bits)  # Genera un individuo aleatorio
    poblacion.append([individuo, []])  # Guarda el individuo y un espacio para su fitness

# Evalúa la población inicial
evaluar_poblacion(poblacion)

# Bucle de generaciones
for generacion in range(generaciones):
    
    print("Generacion: ", generacion)
    mejor_individuo = min(poblacion, key=lambda x: x[1])
    print("Mejor individuo:", mejor_individuo[0]) 
    print("Valor decimal:", bin2dec(mejor_individuo[0][1:])) 
    print("Fitness:", mejor_individuo[1])
    

    #Seleccion de padres utilizando competencia con k =3
    seleccionados = seleccion_por_competencia(poblacion, 3)
    
    # Selección de padres utilizando selección por ventanas
    #seleccionados = seleccion_por_ventanas(poblacion, 10)

    nueva_poblacion = []
    # Realiza el cruce de los padres seleccionados
    for i in range(0, len(seleccionados), 2):
        padre1 = seleccionados[i]
        padre2 = seleccionados[i + 1]

        # Cruza a los padres para obtener dos hijos
        hijo1, hijo2 = cruce(padre1, padre2)

        # Aplica mutación a los hijos
        hijo1 = mutacion(hijo1, prob_mutacion)
        hijo2 = mutacion(hijo2, prob_mutacion)

        # Añade los hijos a la nueva población
        nueva_poblacion.append(hijo1)
        nueva_poblacion.append(hijo2)

    # Reemplaza la población anterior con la nueva
    poblacion = nueva_poblacion

    # Evalúa la nueva población
    evaluar_poblacion(poblacion)

# Al final, obtenemos el mejor individuo
mejor_individuo = min(poblacion, key=lambda x: x[1])
print("Mejor individuo FINAL: ", mejor_individuo[0]) 
print("Valor decimal FINAL: ", bin2dec(mejor_individuo[0][1:])) 
print("Fitness FINAL: ", mejor_individuo[1])

#-------------------------------------------------------------

# Definimos la función de gradiente descendente
def gradiente_descendente(x_inicial, tasa_aprendizaje, max_iteraciones):
    x_actual = x_inicial
    for iteracion in range(max_iteraciones):
        # Calcula el gradiente en la posición actual
        gradiente = derivative(funcion_objetivo, x_actual, dx=1e-6)

        # Actualiza la posición
        x_nuevo = x_actual - tasa_aprendizaje * gradiente
        
        # Opcional: imprime la iteración y el valor actual
        print(f"Iteración {iteracion + 1}: x = {x_actual}, f(x) = {funcion_objetivo(x_actual)}")

        # Actualiza la posición actual
        x_actual = x_nuevo
    
    return x_actual, funcion_objetivo(x_actual)

# Parámetros del algoritmo
x_inicial = 1.0         # Valor inicial
tasa_aprendizaje = 0.01 # Tasa de aprendizaje
max_iteraciones = 1000    # Número máximo de iteraciones

# Llamamos a la función de gradiente descendente
resultado = gradiente_descendente(x_inicial, tasa_aprendizaje, max_iteraciones)

# Mostramos el resultado final
print("Resultado final:")
print("x mínimo:", resultado[0])
print("Valor mínimo de la función:", resultado[1])



