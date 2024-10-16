import numpy as np
import matplotlib.pyplot as plt
import random 


#Pasar binario a decimal
def bin2dec(x):
    return int(''.join(map(lambda x: str(int(x)), x)), 2)

# Función objetivo o fitness que estamos evaluando. En este caso: f(x) = -x * sin(sqrt(abs(x)))
def funcion_objetivo(x):
    return -x * np.sin(np.sqrt(np.abs(x)))

#---------------------FUNCION DE EVALUACION, FITNESS PARA CADA INDIVIDUO--------
# Evalúa cada individuo en la población, donde cada individuo es un arreglo binario, es decir le carga su correspondiente fitness al individuo
def evaluar_poblacion(poblacion):
    for i in range(len(poblacion)):
        # Convierte el individuo de binario a decimal, ignorando el primer bit (que es el del signo)
        ind_dec = bin2dec(poblacion[i][0][1:])
        # Si el primer bit es 1, el número es negativo
        if poblacion[i][0][0] == 1:
            #Se multiplica con -1 para hacerlo negativo
            ind_dec = ind_dec * -1
        # Calcula el fitness del individuo y lo guarda (invertido para minimizar)
        poblacion[i][1] =  funcion_objetivo(ind_dec)


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

        # Selecciona al mejor individuo de esos K individuos seleccionados
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
 
    #Se hace una ventana igual al tamaño de la poblacion
    tam_ventana = int(tam_poblacion)
    #Se recorre segun la cantidad de ventanas
    for i in range(num_ventanas):
        #Para cada ventana se toma un cantidad de individuos random del 10% de la poblacion total
        for j in range (int(tam_poblacion*0.1)):
          seleccionados.append(random.choice(poblacion_ordenada[0:tam_ventana]))
          
         
     
        #Al tamaño de la ventana la reduzco un 10 porciento
        tam_ventana = int(tam_ventana - tam_poblacion*0.1) 
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
cant_bits = 10          # Número de bits que representa a cada individuo
tam_poblacion = 1000    # Tamaño de la población
prob_mutacion = 0.001   # Probabilidad de mutación
generaciones = 100     # Número de generaciones
generacion=0

# Inicialización de la población con individuos aleatorios
poblacion = []
for i in range(tam_poblacion):
    individuo = np.random.randint(0, 2, cant_bits)  # Genera un individuo aleatorio
    poblacion.append([individuo, []])  # Guarda el individuo y un espacio para su fitness

#Agregar a la poblacion para cada individuo su funcion de fitnes
evaluar_poblacion(poblacion)


mejores_fitness=[]

mejor_fitness = -np.inf
paciencia_contador=0


# Bucle de generaciones
while generacion < generaciones and paciencia_contador<10:
    generacion=generacion+1
    print("Generacion: ", generacion)
    mejor_individuo = min(poblacion, key=lambda x: x[1])
    print("Mejor individuo:", mejor_individuo[0]) 
    print("Valor decimal:", bin2dec(mejor_individuo[0][1:])) 
    print("Fitness:", mejor_individuo[1])
    

    #Seleccion de padres utilizando competencia con k =3
    #seleccionados = seleccion_por_competencia(poblacion, 3)
    
    # Selección de padres utilizando selección por ventanas
    seleccionados = seleccion_por_ventanas(poblacion, 10)

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
    
    
    if mejor_individuo[1] > mejor_fitness:
        mejor_fitness = mejor_individuo[1]
        paciencia_contador = 0  # Reinicia el contador de paciencia si hay mejora
    else:
        paciencia_contador += 1  # Incrementa el contador si no hay mejora

    
    
    

# Al final, tenemos el mejor individuo
print("---------------------------------------")
mejor_individuo = min(poblacion, key=lambda x: x[1])
print("Mejor individuo FINAL: ", mejor_individuo[0]) 
print("Valor decimal FINAL: ", bin2dec(mejor_individuo[0][1:])) 
print("Fitness FINAL: ", mejor_individuo[1])

#---------------------CALCULAR CON EL GRADIENTE----------------------------------------


#La derivada segun google es asi
def derivada_funcion_objetivo(x):
    if x == 0:
        return 0  # La derivada en x=0 puede ser considerada 0
    else:
        valor = -np.sin(np.sqrt(np.abs(x))) - (x * np.cos(np.sqrt(np.abs(x)))) / (2 * np.sqrt(np.abs(x))) *  np.sign(x)
        return valor


# Definimos la función de gradiente descendente
def gradiente_descendente(x_inicial, tasa_aprendizaje, max_iteraciones):
    x_actual = x_inicial
    for iteracion in range(max_iteraciones):
        # Calcula el gradiente en la posición actual
        gradiente = derivada_funcion_objetivo(x_actual)

        # Actualiza la posición
        x_nuevo = x_actual - tasa_aprendizaje * gradiente
    
        # Actualiza la posición actual
        x_actual = x_nuevo
    
    return x_actual, funcion_objetivo(x_actual)

# Parámetros del algoritmo
#Depende donde inicies x_inicial es en el minimo local que cae
x_inicial = 100         # Valor inicial
tasa_aprendizaje = 0.01 # Tasa de aprendizaje
max_iteraciones = 1000    # Número máximo de iteraciones

#Se llama a la funcion gradidente, igual siempre cae en minimos locales y no llega al global
resultado = gradiente_descendente(x_inicial, tasa_aprendizaje, max_iteraciones)

print("---------------------------------------")
# Mostramos el resultado final
print("Resultado final del gradiente:")
print("x mínimo:", resultado[0])
print("Valor mínimo de la función:", resultado[1])
print("---------------------------------------")



#------------------------GRAFICAR LA FUNCION------------------------------------------------------
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



plt.figure()
plt.plot(mejores_fitness)
plt.title("Evolucion del fitness")
