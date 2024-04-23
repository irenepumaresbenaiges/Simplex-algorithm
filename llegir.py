import numpy as np
import re

def llegir_dades(n_arxiu, n_alumne, n_problema):
    with open(n_arxiu, "r") as file:
        lines = file.readlines()
        
    c = None
    A = None
    b = None

    alumne_info = f"alumno\s*{n_alumne}"
    problema_info = f"problema\s*PL\s*{n_problema}"

    found_alumne = False
    found_problema = False

    for idx, line in enumerate(lines):
        if re.search(alumne_info, line):
            found_alumne = True

        elif found_alumne and re.search(problema_info, line):
            found_problema = True

        if found_problema:
            if "c=" in line:
                if "Columns" in lines[idx + 2]:
                    c = np.array([int(x) for x in lines[idx + 4].split()] + [int(x) for x in lines[idx + 8].split()])
                else:
                    c = np.array([int(x) for x in lines[idx + 1].split()])

            elif "A=" in line:
                i = 1
                temp_A = []
                if "Columns" in lines[idx + 2]:
                    temp_A_aux = []
                    i += 3
                    while len(lines[idx + i]) > 3:
                        temp_A_aux.append([int(x) for x in lines[idx + i].split()])
                        i += 1
                    i += 3
                    for row in temp_A_aux:
                        temp_A.append(row + [int(x) for x in lines[idx + i].split()])
                        i += 1
                else:
                    while len(lines[idx + i]) > 3:
                        temp_A.append([int(x) for x in lines[idx + i].split()])
                        i += 1
                A = np.array(temp_A)

            elif "b=" in line:
                b = np.array([int(x) for x in lines[idx+1].split()])
                break  # Acaba després de troabr 'b=', assumint que ja s'han trobat totes les variables necessàries
    return c, A, b