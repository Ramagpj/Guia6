import numpy as np
import random  
import matplotlib.pyplot as plt

#Pasar binario a decimal
#def bin2dec(x):
#    return int(''.join(map(lambda x: str(int(x)), x)), 2)

# Función objetivo o fitness que estamos evaluando. 
def funcion_objetivo(x):
    return np.sum(x) #Ni idea, sumar todo?

#---------------------FUNCION DE EVALUACION, FITNESS PARA CADA INDIVIDUO--------
# Evalúa cada individuo en la población, donde cada individuo es un arreglo binario, es decir le carga su correspondiente fitness al individuo
def evaluar_poblacion(poblacion):
    for i in range(len(poblacion)):
        poblacion[i][1] = funcion_objetivo(poblacion[i][0])


#---------------------SELECCION POR COMPETENCIA----------------------------------
# Selección por competencia: selecciona los mejores K individuos de la población
def seleccion_por_competencia(poblacion, k):
    seleccionados = []
    competidores=[]
    tam_poblacion = len(poblacion)

    # Continua seleccionando hasta llenar el número de individuos de la población original
    while len(seleccionados) < tam_poblacion:
        # Selecciona aleatoriamente k competidores de la población
        indices_competidores = np.random.choice(len(poblacion), k, replace=False)
        for i in indices_competidores:
            competidores.append(poblacion[i])

        # Ordena los competidores por su fitness (el mejor al principio)
        competidores_ordenados = sorted(competidores, key=lambda x: x[1])

        # Selecciona los dos mejores individuos
        seleccionados.append(competidores_ordenados[0])
     

    # Devuelve la lista de seleccionados
    return seleccionados



#---------------------SELECCION POR VENTANA----------------------------------
# Selección por ventanas: divide la población en ventanas y selecciona un individuo de cada ventana
def seleccion_por_ventanas(poblacion, num_ventanas):
    # Ordenar la población según su fitness de menor a mayor (para minimizar)
    poblacion_ordenada = sorted(poblacion, key=lambda x: x[1])

    seleccionados = []
    tam_poblacion = len(poblacion)

    # Define el tamaño de la primera ventana
    tam_ventana_inicial = tam_poblacion
    
    
    while len(seleccionados) < tam_poblacion:
        ventana_actual = tam_ventana_inicial    
        # Repite para cada ventana
        for i in range(num_ventanas):
            # Define los límites de la ventana actual en la población ordenada
            ventana = poblacion_ordenada[0:ventana_actual]
    
            # Selecciona aleatoriamente un individuo de esta ventana
            seleccionado = random.choice(ventana)
            seleccionados.append(seleccionado)
            
            #Reduce el tamaño de la ventana en un 10% para la próxima iteración, vas haciendo mas chica las ventanas, siempre el con mas fitness va a estar siempre
            ventana_actual = max(1, int(ventana_actual * 0.9))  # Asegura que el tamaño sea al menos 1 y entero

    return seleccionados


#---------------------CRUCE----------------------------------
# Función de cruce que genera dos hijos por cada par de padres
def cruce(padre1, padre2):
    # Elige un punto de corte aleatorio (ignora el bit de signo)
    punto_corte = np.random.randint(1, len(padre1[0])-1)  
    # Hijo 1: parte inicial de padre1  + parte final de padre2
    hijo1 = np.concatenate((padre1[0][0:punto_corte], padre2[0][punto_corte:]))
    
    # Hijo 2: parte inicial de padre2 + parte final de padre1
    hijo2 = np.concatenate((padre2[0][0:punto_corte], padre1[0][punto_corte:]))

    return [hijo1, []], [hijo2, []]


#---------------------MUTACION----------------------------------
# Función de mutación para aplicar mutaciones a los individuos
def mutacion(individuo, prob_mutacion):
    # Recorre todos los bits del individuo
    for i in range(len(individuo[0])):
        # Aplica mutación con la probabilidad dada
        if np.random.rand() < prob_mutacion:
            individuo[0][i] = 1 - individuo[0][i]
    return individuo




#---------------------INICIALIZACION DEL ALGORITMO



# Parámetros del algoritmo genético
tam_poblacion = 100  # Tamaño de la población
cant_caracteristicas = 7129  # Cantidad de características
rango_min = -26775  # Valor mínimo de los decimales (Hice min en el exel)
rango_max = 71369   # Valor máximo de los decimales (Hice max en el exel)
generaciones = 10     # Número de generaciones
prob_mutacion = 0.001   # Probabilidad de mutación

poblacion = []

for i in range(tam_poblacion):
    # Genera un individuo aleatorio con valores decimales en cada característica
    individuo = np.random.uniform(rango_min, rango_max, cant_caracteristicas)
    poblacion.append([individuo, []])  # Guarda el individuo y un espacio para su fitness



#Agregar a la poblacion para cada individuo su funcion de fitnes
evaluar_poblacion(poblacion)

mejores_fitness=[]

generacion=0

# Ciclo de generaciones
while generacion < generaciones :
    
    generacion=generacion+1
    
    print("Generacion: ", generacion)
    mejor_individuo = min(poblacion, key=lambda x: x[1])
    print("Mejor individuo:", mejor_individuo[0]) 
    print("Fitness:", mejor_individuo[1])
    

    #Seleccion de padres utilizando competencia con k =3
    seleccionados = seleccion_por_competencia(poblacion, 3)
    
    # Selección de padres utilizando selección por ventanas
    #seleccionados = seleccion_por_ventanas(poblacion, 10)

    nueva_poblacion = []
    # Realiza el cruce de los padres seleccionados, voy pasando de a dos, porque tengo dos padres
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

    # Evalúa la nueva población, es decir le carga su funcion de fitness a cada individuo
    evaluar_poblacion(poblacion)
    
    mejores_fitness.append(mejor_individuo[1])
    
    

# Al final, tenemos el mejor individuo
print("---------------------------------------")
mejor_individuo = min(poblacion, key=lambda x: x[1])
print("Mejor individuo FINAL: ", mejor_individuo[0]) 
print("Fitness FINAL: ", mejor_individuo[1])




plt.figure()
plt.plot(mejores_fitness)
plt.title("Evolucion del fitness")


