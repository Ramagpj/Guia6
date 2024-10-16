import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from scipy.optimize import minimize
import random 

# Convierte un arreglo binario en un número decimal
def bin2dec(x):
    return int(''.join(map(lambda x: str(int(x)), x)), 2)

# Función objetivo que estamos evaluando
def funcion_objetivo(x, y):
    return (x**2 + y**2)**0.25 * (np.sin(50 * (x**2 + y**2)**0.1)**2) + 1



#---------------------FUNCION DE EVALUACION, FITNESS PARA CADA INDIVIDUO--------
# Evalúa cada individuo en la población
def evaluar_poblacion(poblacion):
    for i in range(len(poblacion)):
        # Convierte el individuo de binario a decimal (ignorando el primer bit)
        ind_decX = bin2dec(poblacion[i][0][1:])
        ind_decY = bin2dec(poblacion[i][1][1:])
        
        # Si el primer bit es 1, el número es negativo
        if poblacion[i][0][0] == 1:
            ind_decX = -ind_decX
            
        if poblacion[i][1][0] == 1:
            ind_decY = -ind_decY
            
        # Asigna el valor de la función fitness 
        poblacion[i][2] = funcion_objetivo(ind_decX, ind_decY)


#---------------------SELECCION POR COMPETENCIA----------------------------------
# Selección por competencia: selecciona los mejores K individuos de la población
def seleccion_por_competencia(poblacion, k):
    seleccionados = []
    tam_poblacion = len(poblacion)
    competidores=[]
    while len(seleccionados) < tam_poblacion:
        # Selecciona k competidores de los índices de la población
    
        indices_competidores = np.random.choice(len(poblacion), k, replace=False)
        for i in indices_competidores:
            competidores.append(poblacion[i])
        
        
        # Ordena los competidores por su fitness
        competidores_ordenados = sorted(competidores, key=lambda x: x[2])
        
        # Selecciona los dos mejores
        seleccionados.append(competidores_ordenados[0])
    
    return seleccionados #Retorna los seleccionados


#---------------------SELECCION POR VENTANA----------------------------------
# Selección por ventanas: divide la población en ventanas y selecciona un individuo de cada ventana
def seleccion_por_ventanas(poblacion, num_ventanas):
    # Ordenar la población según su fitness de menor a mayor (para minimizar)
    poblacion_ordenada = sorted(poblacion, key=lambda x: x[2])

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


#---------------------CRUCE---------------------------------------------------
# Función de cruce para generar dos hijos por cada par de padres
def cruce(padre1, padre2):
    # Elegir un punto de corte aleatorio para cada individuo, ignora el bit de signo
    punto_corte_x = np.random.randint(1, len(padre1[0]) - 1)  
    punto_corte_y = np.random.randint(1, len(padre1[1]) - 1)

    # Hijo 1: combina la parte inicial de padre1 con la parte final de padre2
    hijo1 = [
        np.concatenate((padre1[0][0:punto_corte_x], padre2[0][punto_corte_x:])),  # Hijo X
        np.concatenate((padre1[1][0:punto_corte_y], padre2[1][punto_corte_y:])),  # Hijo Y
        []  # Espacio para fitness
    ]
    
    # Hijo 2: combina la parte inicial de padre2 con la parte final de padre1
    hijo2 = [
        np.concatenate((padre2[0][0:punto_corte_x], padre1[0][punto_corte_x:])),  # Hijo X
        np.concatenate((padre2[1][0:punto_corte_y], padre1[1][punto_corte_y:])),  # Hijo Y
        []  # Espacio para fitness
    ]

    return hijo1, hijo2  # Retornar los hijos con espacio para fitness

#---------------------MUTACION----------------------------------
def mutacion(individuo, prob_mutacion):
    for i in range(len(individuo[0])):
        # Si ocurre la mutación (con probabilidad prob_mutacion)
        if np.random.rand() < prob_mutacion:  
            individuo[0][i] = 1 - individuo[0][i]  # Cambia el bit en x
            individuo[1][i] = 1 - individuo[1][i]  # Cambia el bit en y
    return individuo


#---------------------INICIALIZACION DEL ALGORITMO----------------------------
# Parámetros del algoritmo genético
cant_bits = 8  # Número de bits que representa a cada individuo
tam_poblacion = 100  # Tamaño de la población
prob_mutacion = 0.001  # Probabilidad de mutación
generaciones = 1000  # Número de generaciones

# Inicialización de la población con individuos aleatorios
poblacion = []
for i in range(tam_poblacion):
    individuoX = np.random.randint(0, 2, cant_bits)  # Genera un individuo aleatorio en x
    individuoY = np.random.randint(0, 2, cant_bits)  # Genera un individuo aleatorio en y
    poblacion.append([individuoX, individuoY, []])  # Guarda el individuo y un espacio para su fitness

# Evalúa la población inicial
evaluar_poblacion(poblacion)


mejores_fitness=[]



generacion=0

# Ciclo de generaciones
while generacion < generaciones :
    
    generacion=generacion+1
    
    print("Generacion: ", generacion)
    mejor_individuo = min(poblacion, key=lambda x: x[2])
    print("Mejor individuo:", mejor_individuo[0]) 
    print("Valor decimal x:", bin2dec(mejor_individuo[0][1:])) 
    print("Valor decimal y:", bin2dec(mejor_individuo[1][1:])) 
    print("Fitness:", mejor_individuo[2])
    
    
    # Selección de padres con selección por competencia
    #seleccionados = seleccion_por_competencia(poblacion, k=10)
    
    
    
    # Selección de padres utilizando selección por ventanas
    seleccionados = seleccion_por_ventanas(poblacion, 10)

    nueva_poblacion = []
    # Itera de a dos en seleccionados, así tengo los dos padres
    for i in range(0, len(seleccionados), 2):
        padre1 = seleccionados[i]
        padre2 = seleccionados[i + 1]

        # Aplicación de cruce para generar dos hijos
        hijo1, hijo2 = cruce(padre1, padre2)

        # Aplicación de mutación
        hijo1 = mutacion(hijo1, prob_mutacion)
        hijo2 = mutacion(hijo2, prob_mutacion)

        # Añadir los hijos a la nueva población
        nueva_poblacion.append(hijo1)
        nueva_poblacion.append(hijo2)

    # Reemplaza la población anterior con la nueva
    poblacion = nueva_poblacion

    # Evalúa la nueva población, es decir le agrego su fitness a cada nuevo individuo
    evaluar_poblacion(poblacion)
    mejores_fitness.append(mejor_individuo[2])
    
    

    
    

# Se obtiene el mejor individuo
print("---------------------------------------")
mejor_individuo = min(poblacion, key=lambda x: x[2])
print("Mejor individuo:", mejor_individuo[0]) 
print("Valor decimal X:", bin2dec(mejor_individuo[0][1:])) 
print("Valor decimal Y:", bin2dec(mejor_individuo[1][1:])) 
print("Fitness:", mejor_individuo[2]) 





#------------------GRADIENTE DESCENDENTE----------------------------------

#Derivada de la funcion objetivo segun CHATGPT
def derivada_funcion_objetivo(x, y):
    r = x**2 + y**2
    
    # Manejo del caso donde r es cero
    if r == 0:
        return np.array([0.0, 0.0])  # Gradiente en (0,0) es 0 para ambos x e y

    # Derivada parcial con respecto a x
    dfx = (0.5 * x / r**0.75) * (np.sin(50 * r**0.1)**2) + \
          (r**0.25) * (2 * np.sin(50 * r**0.1) * np.cos(50 * r**0.1) * 50 * (0.2 * x / r**0.9))
    
    # Derivada parcial con respecto a y
    dfy = (0.5 * y / r**0.75) * (np.sin(50 * r**0.1)**2) + \
          (r**0.25) * (2 * np.sin(50 * r**0.1) * np.cos(50 * r**0.1) * 50 * (0.2 * y / r**0.9))
    
    return np.array([dfx, dfy])


# Definimos la función de gradiente descendente
def gradiente_descendente(x_inicial,y_inicial ,tasa_aprendizaje, max_iteraciones):
    x_actual = x_inicial
    y_actual = y_inicial
    for iteracion in range(max_iteraciones):
        # Calcula el gradiente en la posición actual
        gradiente = derivada_funcion_objetivo(x_actual,y_actual)

        # Actualiza la posición
        x_nuevo = x_actual - tasa_aprendizaje * gradiente[0]
        y_nuevo = y_actual - tasa_aprendizaje * gradiente[1]

        # Actualiza la posición actual
        x_actual = x_nuevo
        y_actual=  y_nuevo
    
    return x_actual,y_actual, funcion_objetivo(x_actual,y_actual)


#

# Parámetros del algoritmo, siempre cae en un minimo local segun el x_inicial e y_inicial que ponga
x_inicial = 100         # Valor inicial
y_inicial = 100         # Valor inicial
tasa_aprendizaje = 0.01 # Tasa de aprendizaje
max_iteraciones = 1000    # Número máximo de iteraciones

#Se llama a la funcion gradidente, igual siempre cae en minimos locales y no llega al global
resultado = gradiente_descendente(x_inicial,y_inicial,tasa_aprendizaje, max_iteraciones)

# Mostramos el resultado final
print("---------------------------------------")
print("Resultado final con el gradiente:")
print("x mínimo:", resultado[0])
print("y mínimo:", resultado[1])
print("Valor mínimo de la función:", resultado[2])
print("---------------------------------------")


#------------------------GRAFICAR LA FUNCION------------------------------------------------------

# Genera un rango de valores para graficar la función en 3D
x = np.linspace(-100, 100, 100)
y = np.linspace(-100, 100, 100)
X, Y = np.meshgrid(x, y)
Z = funcion_objetivo(X, Y)  # Calcula la función objetivo en la cuadrícula

# Crea el gráfico 3D
fig = plt.figure()
ax = fig.add_subplot(111, projection='3d')
ax.plot_surface(X, Y, Z, cmap='viridis')

# Etiquetas y título
ax.set_title('Gráfico 3D de la Función Objetivo')
ax.set_xlabel('X')
ax.set_ylabel('Y')
ax.set_zlabel('f(X, Y)')

plt.show()


plt.figure()
plt.plot(mejores_fitness)
plt.title("Evolucion del fitness")
