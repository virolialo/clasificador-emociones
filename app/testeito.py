import pandas as pd
import re
import imblearn
from gensim.models import Word2Vec
from imblearn.over_sampling import RandomOverSampler
import numpy as np
from gensim.models import Word2Vec
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.ensemble import AdaBoostClassifier, RandomForestClassifier
from sklearn.naive_bayes import GaussianNB
from sklearn.neighbors import KNeighborsClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.svm import SVC
from sklearn.metrics import classification_report, accuracy_score, f1_score
import nltk
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer
# Descargar stopwords y lematizador
nltk.download('stopwords')
nltk.download('wordnet')
stop_words = set(stopwords.words('english'))
lemmatizer = WordNetLemmatizer()
# Función para limpiar y preprocesar datos
def clean_and_preprocess(text):
    # Eliminar menciones de usuarios, URL y caracteres especiales
    text = re.sub(r'@[A-Za-z0-9_]+', '', text)
    text = re.sub(r'http\S+', '', text)
    text = re.sub(r'[^A-Za-z0-9]', ' ', text)
    # Convertir a minúsculas y dividir en palabras
    words = text.lower().split()
    # Eliminar stopwords y lematizar
    words = [lemmatizer.lemmatize(word) for word in words if word not in stop_words]
    return ' '.join(words)
# Cargar datos
file_path = "C:\\Users\\user\\Desktop\\Universidad\\3\\C2\\IA\\Procesamiento del lenguaje natural\\tweet_emotions.csv"
data = pd.read_csv(file_path)

# Mostrar primeras filas de datos
data.head()
# Información del DataFrame
data.info()
# Eliminar columna innecesaria y limpiar el dataset
data.drop(columns=['tweet_id'], inplace=True)
data.drop_duplicates(keep="first", inplace=True)
data.dropna(inplace=True)

# Aplicar función de limpieza básica a los tweets
data['cleaned_content'] = data['content'].apply(clean_and_preprocess)
data.info()
# Mostrar primeros valores del corpus procesado
data.head(n=100)
# Preparar datos para Word2Vec
w2v_data = [text.split() for text in data['cleaned_content']]


# Entrenar modelo Word2Vec
w2v_model = Word2Vec(w2v_data, vector_size=150, window=10, min_count=2, workers=4)
# Función para convertir texto en vector
def text_to_vector(text):
    words = text.split()
    word_vectors = [w2v_model.wv[word] for word in words if word in w2v_model.wv.key_to_index]
    return np.mean(word_vectors, axis=0) if word_vectors else np.zeros(w2v_model.vector_size)

# Convertir todos los textos en vectores
vectorized_texts = np.array([text_to_vector(text) for text in data['cleaned_content']])
# Inicializar y aplicar RandomOverSampler
ros = RandomOverSampler(random_state=42)
X_resampled, y_resampled = ros.fit_resample(vectorized_texts, data['sentiment'])
# Dividir datos en conjuntos de entrenamiento y prueba
X_train, X_test, y_train, y_test = train_test_split(X_resampled, y_resampled, test_size=0.2, random_state=42)

# Функция вывода результатов
def ouput_result(y_pred):
  print('Accuracy:', accuracy_score(y_test, y_pred))
  print('F1 Score:', f1_score(y_test, y_pred, average='weighted'))
  print('Classification Report TESTS:\n', classification_report(y_test, y_pred))
dt_params = {'max_depth': [10, 60]}
dt_model = GridSearchCV(DecisionTreeClassifier(), dt_params, cv=5, scoring='f1_weighted')
dt_model.fit(X_train, y_train)

y_pred = dt_model.predict(X_test)
ouput_result(y_pred)

#Decison tree goooooood
#Knn gooooooood
  
