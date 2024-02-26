import re
import pymongo
from nltk.corpus import stopwords
from nltk.stem import SnowballStemmer
from nltk.stem import WordNetLemmatizer

lemmatizer = WordNetLemmatizer()

def connect_to_mongodb():
    client = pymongo.MongoClient('mongodb://localhost:27017/')
    db = client['sentimientos']  # Nombre de tu base de datos MongoDB
    return db

def get_documents():  # Esta función está fuera del alcance de la corrección actual
     db = connect_to_mongodb()
     collection = db['raw_data']  # Reemplaza 'nombre_coleccion' con el nombre de tu colección MongoDB
     documents = collection.find({}).limit(5)
     return documents

def filtrado(texto):
    # Crear un stemmer en inglés
    stemmer = SnowballStemmer('english')
    # Obtener la lista de stopwords en inglés

    stopwords_english = set(stopwords.words('english'))
     # Eliminar menciones de usuarios, URL y caracteres especiales
    text = re.sub(r'@[A-Za-z0-9_]+', '', texto)
    text = re.sub(r'http\S+', '', text)
    text = re.sub(r'[^A-Za-z0-9]', ' ', text)
    # Tokenizar el texto en palabras
   # palabras = nltk.word_tokenize(text)
    words = text.split()
    words = [lemmatizer.lemmatize(word) for word in words if word not in stopwords_english]

    # Filtrar las palabras
    palabras_filtradas = []
    for palabra in words:
        # Verificar si la palabra no está en la lista de stopwords y consiste únicamente de letras
        if palabra.lower() not in stopwords_english and palabra.isalpha():
            # Realizar el proceso de stemming en la palabra y agregarla a la lista de palabras filtradas
            palabras_filtradas.append(stemmer.stem(palabra.lower()))

    # Devolver el texto filtrado y stemmizado como una cadena de texto
    return ' '.join(palabras_filtradas)


def get_sentiment_from_documents():
    db = connect_to_mongodb()
    collection = db['raw_data']
    documents = collection.find({},{"_id": 0, "sentiment": 1}).limit(50)
    return [document['sentiment'] for document in documents]

def get_content_from_documents():
    db = connect_to_mongodb()
    collection = db['raw_data']
    documents = collection.find({}, {'_id': 0, 'content': 1}).limit(50) # Proyección para excluir _id y solo incluir content
    return [document['content'] for document in documents]




def apply_filtrado_to_content(content_list):
    filtered_content_list = []
    for content in content_list:
        filtered_content = filtrado(content)
        filtered_content_list.append(filtered_content)
    return filtered_content_list


def frecuencia(lista): 
    # Dividir el contenido en una lista de palabras
    palabras = lista.split()

    # Cuenta la cantidad de veces que aparece cada palabra
    frecuencias = {}
    for palabra in palabras:
        frecuencias[palabra] = frecuencias.get(palabra, 0) + 1

    # Calcula los porcentajes de aparición de cada palabra
    total_palabras = len(palabras)
    porcentajes = {palabra: (frecuencia / total_palabras) * 100 for palabra, frecuencia in frecuencias.items()}
    return porcentajes

def vocabulario():
    content_list = get_content_from_documents()
    # Aplicar la función filtrado al contenido
    filtered_content_list = apply_filtrado_to_content(content_list)
    combined_filtered_content = ' '.join(filtered_content_list)
    # Calcular la frecuencia de palabras en el contenido filtrado combinado
    diccionario_ordenado = dict(sorted(frecuencia(combined_filtered_content).items(), key=lambda item: item[1], reverse=True))
    vocabulario = list(diccionario_ordenado.keys())[:int((len(diccionario_ordenado)))]
    return print("Numero total de palabras distintas:", len(vocabulario))

