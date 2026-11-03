Dentro de la carpeta Software se encuentran las siguientes carpetas: Python, MATLAB y Webots. En estas carpetas se encuentran los scripts desarrollados y utilizados durante la implementación y validación experimental del proyecto.

Python:
El archivo principal del proyecto es “flight_purepursuit.py”, el cual implementa el sistema completo de generación y seguimiento de trayectorias basado en splines Catmull-Rom y el algoritmo Pure Pursuit.
Se recomienda descargar la carpeta completa (Code_V1) para ejecutar las pruebas correctamente, ya que el archivo principal depende de módulos adicionales relacionados con generación de trayectorias, control de seguimiento, comunicación y visualización de resultados.

Dentro de Code_V1(Lineal) y Code_V2(C-R) se encuentra únicamente el archivo 'flight_purepursuit', este se debe reemplazar por el que está en 'Code_V1'. La única diferencia es que ya tiene algunos parámetros de velocidad definidos que son los que se utilizaron para hacer las pruebas en el ecosistema Robotat.

Nota:
La carpeta 'Code_V1' corresponde al modo de vuelo estable, con una velocidad mas lenta. La carpeta 'Code_V1 (Lineal)' corresponde el método con trayectorias lineales en lugar de Catmull-Rom y la carpeta 'Code_V2(C-R)' es la versión rápida de 'Code_V1', solo se cambiaron algunos parametros de velocidades y aceleraciones.

MATLAB:
Acá se encuentra el script utilizado para leer los marcadores de los obstáculos utilizando el sistema de captura del laboratorio Robotat. 
Al ejecutar el archivos 'read_waypoints.m' este genera un archivo 'waypoints.json' el cual se debe reemplazar por el de 'Code_V1'. A partir de ese archivo se genera la trayectoria.

Nota:
Si se desea colocar los puntos de paso (waypoints) manualmente, se puede ejecutar el código de python 'create_waypoints' el cual también genera un archivo 'waypoints.json', este ya se encuentra dentro de 'Code_V1'.

Webots:
En esta carpeta se puede encontrar el controlador y el script necesario para ejecutar la simulación del vuelo.
