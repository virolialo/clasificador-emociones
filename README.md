### Manual de Usuario: Despliegue de Proyecto Django en Local

Este manual proporciona los pasos necesarios para desplegar el proyecto "Clasificador de Emociones" en un entorno local utilizando Django.

---

#### Requisitos Previos

Antes de comenzar, asegúrate de tener instalados los siguientes componentes:

1. **MongoDB y MongoDB Compass**: Base de datos necesaria para almacenar los datos del proyecto.
2. **Git**: Herramienta para clonar el repositorio del proyecto desde GitHub.
3. **Python y pip**: Instalados en tu sistema para ejecutar el proyecto Django.
4. **NLTK**: Biblioteca de procesamiento de lenguaje natural utilizada en el proyecto.

---

#### Pasos para el Despliegue

1. **Descargar MongoDB y MongoDB Compass**:
   - Descarga e instala MongoDB y MongoDB Compass desde el sitio oficial de MongoDB según las instrucciones para tu sistema operativo.

2. **Descargar JSON con Datos**:
   - Descarga el archivo JSON que contiene los datos necesarios para el proyecto desde [enlace al archivo JSON].

3. **Crear Base de Datos y Colección**:
   - Abre MongoDB Compass.
   - Crea una nueva base de datos llamada "sentimientos".
   - Dentro de esta base de datos, crea una colección llamada "raw_data".
   
4. **Importar JSON en la Colección**:
   - Selecciona la colección "raw_data".
   - Importa los datos desde el archivo JSON descargado.

5. **Conectar con la Base de Datos**:
   - Asegúrate de que MongoDB esté en ejecución.
   - Configura la conexión del proyecto Django para que se conecte a la base de datos "sentimientos".

6. **Clonar el Repositorio del Proyecto**:
   - Abre la terminal o línea de comandos.
   - Ejecuta el siguiente comando para clonar el repositorio desde GitHub:
     ```
     git clone https://github.com/virolialo/clasificador-emociones.git
     ```
   - Cambia al directorio del proyecto:
     ```
     cd clasificador-emociones
     ```

7. **Instalar Dependencias del Proyecto**:
   - Instala las dependencias necesarias ejecutando el siguiente comando:
     ```
     pip install -r requirements.txt
     ```

8. **Preparar el Entorno de Python**:
   - Inicia un intérprete de Python ejecutando el siguiente comando:
     ```
     python
     ```
   - Descarga los recursos adicionales de NLTK necesarios para el proyecto ejecutando los siguientes comandos dentro del intérprete de Python:
     ```
     import nltk
     nltk.download('wordnet')
     nltk.download('stopwords')
     ```

9. **Ejecutar el Servidor Django**:
   - Ejecuta el siguiente comando para iniciar el servidor Django:
     ```
     python manage.py runserver
     ```

10. **Acceder al Proyecto**:
    - Abre tu navegador web y navega a la dirección proporcionada por Django en la terminal después de ejecutar el comando `runserver` (por lo general, `http://127.0.0.1:8000/`).

---

¡Ahora deberías poder acceder al proyecto "Clasificador de Emociones" en tu entorno local y comenzar a explorarlo! Si encuentras algún problema durante el proceso, no dudes en ponerte en contacto con nosotros en este correo jjavierrr20@gmail.com o buscar ayuda en la comunidad de desarrollo de Django.

