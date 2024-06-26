# Implementació de l'Algorisme del Simplex Primal
Aquest projecte implementa l'algoritme del Simplex primal per resoldre problemes de programació lineal (PL). Es divideix en dos arxius principals: llegir.py per llegir les dades d'entrada des d'un fitxer, i simplex.py que conté la lògica de l'algoritme del Simplex i l'execució del programa.

## Arxiu: llegir.py
Aquest arxiu defineix una funció per llegir i emmagatzemar en tres variables les dades necessàries per l'execució d'un problema de programació lineal des d'un fitxer de text.

### Funcionalitat
La funció llegir_dades(n_arxiu, n_alumne, n_problema): Llegeix les dades del problema de PL especificat pel número d'alumne i el número de problema indicat per l'usuari. Retorna els coeficients de la funció objectiu c, la matriu de restriccions A, i el vector b emmagatzemats en tres variables diferents.

### Ús
Per utilitzar aquesta funció, és necessari proporcionar el nom de l'arxiu d'entrada, el número d'alumne, i el número de problema com a arguments. La funció retornarà les matrius c, A, i b que corresponen a les dades del problema.

## Arxiu: simplex.py
Aquest arxiu implementa l'algoritme del Simplex per problemes de programació lineal, incloent una fase inicial per trobar una solució bàsica factible i una segona fase per optimitzar la solució. 

### Funcionalitats Principals
- Implementació de la fase I del Simplex per trobar una solució bàsica factible.
- Implementació de la fase II del Simplex per trobar la solució òptima del problema.
- Implementació de la regla de Bland per prevenir cicles.

### Ús
Per executar el programa, és necessari córrer l'arxiu simplex.py i introduir el número d'alumne i el número de problema quan se sol·licitin. El programa llegirà les dades corresponents del problema des de l'arxiu especificat en nom_arxiu (per defecte, input.txt) utilitzant la funció definida en llegir.py. Llavors, procedirà a trobar una solució bàsica factible amb la fase I del Simplex i optimitzarà aquesta solució a la fase II.

### Exemple de Sortida
El programa imprimirà el progrés de les iteracions de l'algoritme, incloent les variables que entren i surten en cada pas, així com el valor de la funció objectiu. Al final, es mostraran els detalls de la solució òptima trobada, incloent el valor de les variables bàsiques i el valor òptim de la funció objectiu. En el cas que no s'hagi trobat SBF, s'imprimirà el seu cas (problema infactible o no acotat).