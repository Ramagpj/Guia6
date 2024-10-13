import numpy as np
import matplotlib.pyplot as plt

# Convierte un arreglo binario en un número decimal
def bin2dec(x):
    return int(''.join(map(lambda x: str(int(x)), x)), 2)

# Función objetivo o fitness que estamos evaluando, Ejercicio1-A =  -x*sin(sqrt(abs(x)))
def funcion_objetivo(x):
    return -x * np.sin(np.sqrt(np.abs(x)))

# Genera un rango de valores de -512 a 512 para graficar la función
x1 = np.linspace(-512, 512, 1000)
f1 = funcion_objetivo(x1)

# Grafica la función para ver cómo se comporta
plt.figure()
plt.plot(x1, f1)
plt.title("Función objetivo")
plt.xlabel("x")
plt.ylabel("f(x)")
plt.grid()
plt.show()

# Evalúa cada individuo en la población
def evaluar_poblacion(poblacion):
    for i in range(len(poblacion)):
        # Convierte el individuo de binario a decimal (ignorando el primer bit)
        ind_dec = bin2dec(poblacion[i][0][1:])
        # Si el primer bit es 1, el número es negativo
        if poblacion[i][0][0] == 1:
            ind_dec = ind_dec * -1
        # Asigna el valor de la función fitness (invertido para minimizar)
        poblacion[i][1] = -1 * funcion_objetivo(ind_dec)

# Selección por competencia: selecciona los mejores K individuos de la población
def seleccion_por_competencia(poblacion, k):
    seleccionados = []
    tam_poblacion = len(poblacion)
    
    #hasta que seleccione la misma cantidad del tamaño de la poblacion
    while len(seleccionados) < tam_poblacion:
        # Selecciona k competidores de los índices de la población
        indices_competidores = np.random.choice(len(poblacion), k, replace=False)
        competidores=[]
        #Selecciona desde poblacion segundo los indices de indices_competidores
        for i in indices_competidores:
            competidores.append(poblacion[i])
        
        # Ordena los competidores por su fitness
        competidores_ordenados = sorted(competidores, key=lambda x: x[1])
        
        # Selecciona los dos mejores
        seleccionados.append(competidores_ordenados[0])
        seleccionados.append(competidores_ordenados[1])
    
    return seleccionados[:tam_poblacion]  # Cortamos si seleccionamos más de tam_poblacion

# Función de cruce para generar dos hijos por cada par de padres
def cruce(padre1, padre2):
    # Elegir un punto de corte aleatorio, ignora el bit de signo
    punto_corte = np.random.randint(1, len(padre1[0]) - 1)  
    # Hijo 1: combina la parte inicial de padre1 con la parte final de padre2
    hijo1 = np.concatenate((padre1[0][:punto_corte], padre2[0][punto_corte:]))  
    # Hijo 2: combina la parte inicial de padre2 con la parte final de padre1
    hijo2 = np.concatenate((padre2[0][:punto_corte], padre1[0][punto_corte:]))  
    return [hijo1, []], [hijo2, []]  # Retornar los hijos con espacio para fitness

# Función de mutación
def mutacion(individuo, prob_mutacion):
    for i in range(len(individuo[0])):
        # Si ocurre la mutación (con probabilidad prob_mutacion)
        #esto da un valor de probabilidad es decir desde 0, hasta 1
        if np.random.rand() < prob_mutacion:  
            individuo[0][i] = 1 - individuo[0][i]  # Cambia el bit
    return individuo

# Parámetros del algoritmo genético
cant_bits = 10  # Número de bits que representa a cada individuo
tam_poblacion = 100  # Tamaño de la población
prob_mutacion = 0.001  # Probabilidad de mutación
generaciones = 1000  # Número de generaciones

# Inicialización de la población con individuos aleatorios
poblacion = []
for i in range(tam_poblacion):
    individuo = np.random.randint(0, 2, cant_bits)  # Genera un individuo aleatorio
    poblacion.append([individuo, []])  # Guarda el individuo y un espacio para su fitness

# Evalúa la población inicial
evaluar_poblacion(poblacion)

# Ciclo de generaciones
for generacion in range(generaciones):
    # Selección de padres con selección por competencia
    seleccionados = seleccion_por_competencia(poblacion, k=3)  # Selecciona 250 pares (500 padres en total)

    nueva_poblacion = []
    #Itera de a dos en seleccionados, asi tengo los dos padres
    for i in range(0, len(seleccionados), 2):
        padre1 = seleccionados[i]
        padre2 = seleccionados[i + 1]

        # Aplicación de crossover para generar dos hijos
        hijo1, hijo2 = cruce(padre1, padre2)

        # Aplicación de mutación
        hijo1 = mutacion(hijo1, prob_mutacion)
        hijo2 = mutacion(hijo2, prob_mutacion)

        # Añadir los hijos a la nueva población
        nueva_poblacion.append(hijo1)
        nueva_poblacion.append(hijo2)

    # Reemplaza la población anterior con la nueva
    poblacion = nueva_poblacion

    # Evalúa la nueva población
    evaluar_poblacion(poblacion)

# Al final, podrías tenes el mínimo global
mejor_individuo = min(poblacion, key=lambda x: x[1])
print("Mejor individuo:", mejor_individuo[0]) 
print("Valor decimal:", bin2dec(mejor_individuo[0][1:])) 
print("Fitness:", mejor_individuo[1]) 
