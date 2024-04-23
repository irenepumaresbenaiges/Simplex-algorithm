import numpy as np
from llegir import llegir_dades

def afegir_variables_artificials(A):
    """
    Afegeix variables artificials al problema. Retorna la matriu de les restriccions amb les noves variables afegides.
    """
    matriu_identitat = np.eye(num_restriccions)
    matriu_ampliada = np.hstack((A, matriu_identitat))
    return matriu_ampliada


def inicialitzacio_fase1(A, A_ampliada, b):
    """
    Inicialitza les variables necessàries per la primera fase del mètode de simplex.
    """
    num_vars_A = A.shape[1]
    num_vars_A_ampliada = A_ampliada.shape[1]
    
    basiques = [variable for variable in range(num_vars_A + 1, num_vars_A_ampliada + 1)]
    basiques_original = basiques.copy()
    no_basiques = [variable for variable in range(1, num_vars_A + 1)]

    c = np.zeros(num_vars_A_ampliada)   # Definim la nova funció objectiu del problema, que consta únicament de les variables artificials
    c[num_vars_A:] = 1 

    B = np.eye(num_restriccions)
    inversa_B = B.copy()

    A_n = A.copy()

    C_b = c[num_variables:].copy()
    C_n = c[:num_variables].copy()

    X_b = np.dot(inversa_B, b)
    X_n = np.zeros((num_variables,1))

    z = calcul_z(C_b, X_b, C_n, X_n)

    return basiques, basiques_original, no_basiques, c, B, inversa_B, A_n, C_b, C_n, X_b, X_n, z


def inicialitzacio_fase2(basiques, basiques_original, no_basiques, funcio_objectiu, b, inversa_B):
    """
    Inicialitza les variables necessàries per la segona fase del mètode de simplex.
    """
    for var in basiques_original:
        if var in no_basiques:
            no_basiques.remove(var)  

    C_b = funcio_objectiu[np.array(basiques)-1]
    C_n = funcio_objectiu[np.array(no_basiques)-1]

    X_b = np.dot(inversa_B, b)
    X_n = np.zeros((len(no_basiques),1)) 

    A_n = A[:,np.array(no_basiques)-1]

    z = calcul_z(C_b, X_b, C_n, X_n)

    return C_b, C_n, X_b, X_n, A_n, z


def calcul_z(C_b, X_b, C_n, X_n):
    """
    Calcula el valor de la funció objectiu.
    """
    producte_basiques = np.dot(C_b, X_b)
    producte_no_basiques = np.dot(C_n, X_n)
    z = np.sum(producte_basiques) + np.sum(producte_no_basiques)
    return z


def costos_reduits(C_n, C_b, inversa_B, A_n):
    """
    Calcula els vector de cosos reduïts de les variables no bàsiques.
    """
    producte = np.dot(np.dot(C_b, inversa_B), A_n)
    r = np.subtract(C_n, producte)
    return r


def es_optim(r): 
    """
    Verifica si la solució actual és òptima, retornant True si tots els elements dels costos reduïts són més grans o iguals a 0.
    """
    return all(element >= 0 for element in r)


def calcul_variable_entrada(r, no_basiques):
    """
    Selecciona la variable no bàsica a introduir a la base en la pròxima iteració.
    """
    negatius = []
    for i in range(r.shape[0]):
        if r[i] < 0:
            negatius.append(i)
    
    valors_negatius = [no_basiques[index] for index in negatius]

    if valors_negatius: # Apliquem regla de Bland
        variable_entrada = min(valors_negatius)
        return variable_entrada
    else:
        return None


def direccio_basica_factible(inversa_B, A_n, variable_entrada, no_basiques): #comprobar que d es negativa
    """
    Calcula la direcció bàsica factible per la transició cap a una nova solució bàsica. 
    """
    inversa_B_neg = inversa_B * -1
    index_variable_entrada = no_basiques.index(variable_entrada)
    d = np.dot(inversa_B_neg, A_n[:, index_variable_entrada])
    return d


def es_acotat(d):
    """
    Avalua si la direcció bàsica factible manté la solució dins d'un espai factible acotat.
    """
    for element in d:
        if element <= 0:  
            return False
    return True
    
def calcul_theta(X_b, d, basiques):
    """
    Calcula el valor mínim de la longitud de pas en la direcció de la SBF.
    """
    thetas_posibles = [(i, (-(X_b[i]))/d[i]) for i in range(len(d)) if d[i] < 0 and (-(X_b[i]))/d[i] > 0]
    
    if thetas_posibles:
        min_theta = min([theta for index, theta in thetas_posibles])

        # Apliquem la Regla de Bland
        index_variable_sortida = min([index for index, theta in thetas_posibles if theta == min_theta], key=lambda index: basiques[index]) 
        variable_sortida = basiques[index_variable_sortida]
        return variable_sortida, min_theta[0]
    else:
        return None, np.inf
    


def actualitzacio(basiques, no_basiques, X_b, theta, d, z_ant, r_ant, C_b, C_n, c, A_n, A_ampliada, inversa_B, variable_entrada, variable_sortida):
    """
    Actualitza les variables necessàries per poder començar la següent iteració de la fase corresponent.
    """
    # Actualitzar llistes basiques i no basiques
    index_variable_entrada = no_basiques.index(variable_entrada)
    index_variable_sortida = basiques.index(variable_sortida)

    basiques[index_variable_sortida] = variable_entrada
    no_basiques[index_variable_entrada] = variable_sortida

    # X_b
    for i in range(X_b.shape[0]):
        if basiques[i] == variable_entrada:
             X_b[i] = theta
        else:
            X_b[i] += (theta * d[i])
    
    # z
    z_nova = np.sum(z_ant) + np.sum(theta * r_ant[index_variable_entrada])
    assert z_nova < z_ant, 'La z no ha disminuït'

    # C_b i C_n
    C_b[index_variable_sortida] = c[variable_entrada - 1]
    C_n[index_variable_entrada] = c[variable_sortida - 1]

    # A_n
    A_n[:, index_variable_entrada] = A_ampliada[:, variable_sortida - 1]
    
    # inversa_B
    E = np.eye(len(d))
    E[:,index_variable_sortida] = (-d/d[index_variable_sortida]).T
    E[index_variable_sortida,index_variable_sortida] = -1 / d[index_variable_sortida]
    inversa_B = np.dot(E, inversa_B)

    return basiques, no_basiques, X_b, z_nova , C_b, C_n, A_n, inversa_B


def simplex(fase, A=None, num_iteracio=None, basiques=None, basiques_original=None, no_basiques=None, z=None, funcio_objectiu=None, b=None, inversa_B=None, A_ampliada=None):
    """
    Implementa el métode simplex trobant la solució òptima (si hi ha) que minimitza la funció objectiu.
    """
    if fase == 1:
        print("Fase 1")
        A_ampliada = afegir_variables_artificials(A)
        basiques, basiques_original, no_basiques, c, B, inversa_B, A_n, C_b, C_n, X_b, X_n, z = inicialitzacio_fase1(A, A_ampliada, b)
        num_iteracio = 1
        print(f"Valor inicial z: {z}")
        print(f"Variables bàsiques inicials: {basiques}")

    elif fase == 2:
        print("Fase 2")
        C_b, C_n, X_b, X_n, A_n, z = inicialitzacio_fase2(basiques, basiques_original, no_basiques, funcio_objectiu, b, inversa_B)
    
    while True:
        if fase == 1:
            if not np.all(X_b >= 0):
                print("El problema no té solució factible")
                return None
            
            r = costos_reduits(C_n, C_b, inversa_B, A_n)
            if es_optim(r):
                if z <= 10**-10:
                    break
                else:
                    print("El problema no té solució factible")  
                    return None
        elif fase == 2:
            r = costos_reduits(C_n, C_b, inversa_B, A_n)
            if es_optim(r):
                break
        
        variable_entrada = calcul_variable_entrada(r, no_basiques)

        d = direccio_basica_factible(inversa_B, A_n, variable_entrada, no_basiques)
        if es_acotat(d):
            print("El problema no té solució acotada")
            return None
        
        variable_sortida, theta = calcul_theta(X_b, d, basiques)

        if fase == 1:
            # Actualitzar las variables
            basiques, no_basiques, X_b, z, C_b, C_n, A_n, inversa_B = actualitzacio(basiques, no_basiques, X_b, theta, d, z, r, C_b, C_n, c, A_n, A_ampliada, inversa_B, variable_entrada, variable_sortida)
        elif fase == 2:
            # Actualitzar las variables
            basiques, no_basiques, X_b, z, C_b, C_n, A_n, inversa_B = actualitzacio(basiques, no_basiques, X_b, theta, d, z, r, C_b, C_n, funcio_objectiu, A_n, A_ampliada, inversa_B, variable_entrada, variable_sortida)

        print(f"Iteració {num_iteracio}:, variable d'entrada = {variable_entrada}, variable de sortida = {variable_sortida}, theta = {theta}, z = {z}")
        num_iteracio += 1
    
    if fase == 1:
        print(f"Solució bàsica factible trobada: Iteració {num_iteracio-1}")
        return num_iteracio, basiques, basiques_original, no_basiques, A_n, C_b, C_n, X_b, z, inversa_B, A_ampliada
    
    elif fase == 2:
        print(f"Solució òptima trobada: Iteració {num_iteracio-1}")
        print("Fi Simplex primal")
        print("Solució òptima:")
        print(f"Variables bàsiques: {basiques}")
        print(f"Valors variables bàsiques: {X_b}")
        print(f"Valor z: {z}")
        print(f"Costos reduïts: {r}")

def fase1(A):
    """
    Crida a la funció simplex per la fase 1.
    """
    return simplex(fase=1, A=A, b=b)

def fase2(num_iteracio, basiques, basiques_original, no_basiques, z, funcio_objectiu, b, inversa_B, A_ampliada):
    """
    Crida a la funció simplex per la fase 2.
    """
    return simplex(fase=2, num_iteracio=num_iteracio, basiques=basiques, basiques_original=basiques_original, no_basiques=no_basiques, z=z, funcio_objectiu=funcio_objectiu, b=b, inversa_B=inversa_B, A_ampliada=A_ampliada)


if __name__ == "__main__":

    nombre_archivo = "input.txt"

    n_alumno = int(input("Introdueix el número d'alumne: "))
    n_problema = int(input("Introdueix el número de problema: "))

    funcio_objectiu, A, b = llegir_dades(nombre_archivo, n_alumno, n_problema)

    num_restriccions = A.shape[0]
    num_variables = A.shape[1]
    b = b.reshape(-1, 1)

    print("Inicialitzant Simplex")

    resultat_fase1 = fase1(A)

    if resultat_fase1 is None:
        exit()
    else:
        num_iteracio, basiques, basiques_original, no_basiques, A_n, C_b, C_n, X_b, z, inversa_B, A_ampliada = resultat_fase1
        
    fase2(num_iteracio, basiques, basiques_original, no_basiques, z, funcio_objectiu, b, inversa_B, A_ampliada)
    exit()