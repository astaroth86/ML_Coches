# Machine Learning Coches

## Proyecto para mostrar los conocimientos de machine learning

El proyecto es un modelo de regresión que calcula el precio de coches de segunda mano en base a los datos encontrados en una de las principales Web de venta de coches de segunda mano

### Para ejecutar el proyecto seguir las siguientes instrucciones:

- Ejecutar el comando: docker build -t cochesml .
- Ejecutar el comando: docker run -d --name cochesml-app -p 5000:5000 cochesml

Ahora Ya puedes acceder a http://127.0.0.1:5000 desde un navegador y comprobar el precio de los coches