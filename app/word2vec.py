import numpy as np
from sklearn.metrics import accuracy_score, classification_report, f1_score
from sklearn.model_selection import GridSearchCV, train_test_split
from sklearn.naive_bayes import GaussianNB
from sklearn.neighbors import KNeighborsClassifier
from sklearn.tree import DecisionTreeClassifier
from app.procesado import get_content_from_documents, apply_filtrado_to_content, get_sentiment_from_documents
from imblearn.over_sampling import RandomOverSampler
from gensim.models import Word2Vec


content_list = get_content_from_documents()
sentiment_list= get_sentiment_from_documents()
filtered_content_list = apply_filtrado_to_content(content_list)


# Crear una lista de oraciones tokenizadas a partir del conjunto de entrenamiento
sentences = [text.split() for text in filtered_content_list]

# Entrenar un modelo Word2Vec con las oraciones tokenizadas
word2vec_model = Word2Vec(sentences, vector_size=100, window=2, min_count=1, workers=8, epochs=100)

# Definir una función para obtener el vector de un texto utilizando el modelo de Word2Vec
def text_to_vector(text):
    words = text.split()
    word_vectors = [word2vec_model.wv[word] for word in words if word in word2vec_model.wv.key_to_index]
    return np.mean(word_vectors, axis=0) if word_vectors else np.zeros(word2vec_model.vector_size)

vectorized_texts = np.array([text_to_vector(text) for text in filtered_content_list])
ros = RandomOverSampler(random_state=42)
X_resampled, y_resampled = ros.fit_resample(vectorized_texts, sentiment_list)
X_train, X_test, y_train, y_test = train_test_split(X_resampled, y_resampled, test_size=0.2, random_state=42)

# Crear una instancia del clasificador y entrenarlo con los vectores de texto y etiquetas de entrenamiento
model = GaussianNB()
model.fit(X_train, y_train)

def ouput_result(y_pred):
  print('Accuracy:', accuracy_score(y_test, y_pred))
  print('F1 Score:', f1_score(y_test, y_pred, average='weighted'))
  return classification_report(y_test, y_pred, output_dict=True)

y_pred= model.predict(X_test)
ouput_result(y_pred)

dt_params = {'max_depth': [10, 60]}
dt_model = GridSearchCV(DecisionTreeClassifier(), dt_params, cv=5, scoring='f1_weighted')
dt_model.fit(X_train, y_train)

y_pred = dt_model.predict(X_test)
ouput_result(y_pred)

knn_params = {'n_neighbors': [2, 50]}
knn_model = GridSearchCV(KNeighborsClassifier(), knn_params, cv=2, scoring='f1_weighted')
knn_model.fit(X_train, y_train)

y_pred = knn_model.predict(X_test)
ouput_result(y_pred)