import numpy as np
from sklearn.metrics import accuracy_score
from sklearn.svm import SVC
import matplotlib.pyplot as plt
from sklearn.tree import DecisionTreeClassifier

train_data = np.loadtxt("leukemia_train.csv", delimiter=',')
test_data = np.loadtxt("leukemia_test.csv", delimiter=',')

caract_totales = 7129

X_trn_orig = train_data[:,0:caract_totales]
y_trn_orig = train_data[:,-1]
X_tst_orig = test_data[:,0:caract_totales]
y_tst_orig = test_data[:,-1]


# =============================================================================
# el genoma sera una cadena de la cantidad de bits igual a la cantidad de caracteristicas
# si hay un 0, no toma esa caracteristica
# si hay un 1, toma esa caracteristica
# el individuo sera un arreglo que en la posicion 0 tendra el genoma y en la posicion 1 el fitness
# =============================================================================
def fitness(individuo):
    #Configuracion del clasificador
    model = SVC(kernel='linear')
    #model = DecisionTreeClassifier(random_state=42)

    X_trn = []
    y_trn = y_trn_orig
    X_tst = []
    y_tst = y_tst_orig
    
    for gen in range(len(individuo)):
        if individuo[gen] == 1:
            X_trn.append(X_trn_orig[:,gen]) #agrego las caracteristicas correspondientes a X_trn y X_tst
            X_tst.append(X_tst_orig[:,gen])
    X_trn = np.array(X_trn).T
    X_tst = np.array(X_tst).T
    
    #entreno el modelo
    model.fit(X_trn, y_trn)
    
    # Prediccion en la prueba
    y_tst_pred = model.predict(X_tst)
    
    
    # == OPCION 1 DE FITNESS: solo tiene en cuenta el accuracy ==
    # accuracy = accuracy_score(y_tst, y_tst_pred)
    # fitness = accuracy
    
    # == OPCION 2 DE FITNESS: ademas del accuracy, cuanto menos caracteristicas seleccione mejor
    accuracy = accuracy_score(y_tst, y_tst_pred)
    coef_cant = 0.001
    coef_accuracy = 1
    fitness = coef_accuracy*accuracy - coef_cant*np.sum(individuo == 1)
    
    return fitness


def evaluar_poblacion(poblacion):
    for n_individuo in range(len(poblacion)):
        # if n_individuo % 50 == 0:
        #     print("Eval. ind:", n_individuo)
        individuo = poblacion[n_individuo][0]
        poblacion[n_individuo][1] = fitness(individuo)
        
    
#---------------------SELECCION POR COMPETENCIA----------------------------------
# Selección por competencia: selecciona los mejores K individuos de la población
def seleccion_por_competencia(poblacion, k):
    seleccionados = []
    tam_poblacion = len(poblacion)

    # Continua seleccionando hasta llenar el número de individuos de la población original
    while len(seleccionados) < tam_poblacion:
        competidores=[]
        # Selecciona aleatoriamente k competidores de la población
        indices_competidores = np.random.choice(len(poblacion), k, replace=False)
        for i in indices_competidores:
            competidores.append(poblacion[i])

        # Selecciona el mejor individuo basandose en el max fitness
        seleccionados.append(max(competidores,key=lambda x: x[1]))
    return seleccionados

#---------------------CRUCE----------------------------------
# Función de cruce que genera dos hijos por cada par de padres
def cruce(padre1, padre2):
    # Elige un punto de corte aleatorio
    punto_corte = np.random.randint(0, len(padre1[0])-1)   #si quiero ignorar el bit de signo pongo 1
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




# = FUNCION AUXILIAR =
# = Para ir monitoreando el accuracy en el metodo de fitness 2 =
def evaluar_accuracy(individuo):
    model = SVC(kernel='linear')
    X_trn = []
    y_trn = y_trn_orig
    X_tst = []
    y_tst = y_tst_orig
    
    for gen in range(len(individuo)):
        if individuo[gen] == 1:
            X_trn.append(X_trn_orig[:,gen]) #agrego las caracteristicas correspondientes a X_trn y X_tst
            X_tst.append(X_tst_orig[:,gen])
    X_trn = np.array(X_trn).T
    X_tst = np.array(X_tst).T
    
    #entreno el modelo
    model.fit(X_trn, y_trn)
    
    # Prediccion en la prueba
    y_tst_pred = model.predict(X_tst)
    
    accuracy = accuracy_score(y_tst, y_tst_pred)
    return accuracy
# =================




# =============================================================================
#                           INICIO DEL ALGORITMO
# =============================================================================
# Parametros
tam_poblacion = 30
prob_mutacion = 0.001   # Probabilidad de mutación
paciencia = 300
max_gen = 5000
# ----------

# Inicialización de la población con individuos aleatorios
poblacion = []
for i in range(tam_poblacion):
    individuo = np.random.randint(0, 2, caract_totales)  # Genera un individuo aleatorio
    poblacion.append([individuo, []])  # Guarda el individuo y un espacio para su fitness


paciencia_contador = 0
mejor_fitness = -np.inf
generacion = 0
mejores_fitness = []
mejores_accuracy = []

while paciencia_contador < paciencia and generacion < max_gen:
    
    print("\n==== GENERACION ",generacion," ===")
    evaluar_poblacion(poblacion)
    mejor_individuo = max(poblacion, key=lambda x: x[1])
    print("Mejor individuo:", mejor_individuo[0])
    print("Cantidad de caracteristicas:", np.sum(mejor_individuo[0] == 1))
    accuracy = evaluar_accuracy(mejor_individuo[0])
    print("Accuracy:", accuracy)
    mejores_accuracy.append(accuracy)
    print("Fitness:", mejor_individuo[1])
    mejores_fitness.append(mejor_individuo[1])
    
    print("Contador de paciencia:", paciencia_contador)
    if mejor_individuo[1] > mejor_fitness:
        mejor_fitness = mejor_individuo[1]
        paciencia_contador = 0  # Reinicia el contador de paciencia si hay mejora
    else:
        paciencia_contador += 1  # Incrementa el contador si no hay mejora
    
    #Seleccion de padres utilizando competencia con k =3
    seleccionados = seleccion_por_competencia(poblacion, 3)
    
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
    generacion += 1

plt.figure()
plt.plot(mejores_fitness)
plt.title("Evolucion del fitness")

plt.figure()
plt.plot(mejores_accuracy)
plt.title("Evolucion del accuracy")

