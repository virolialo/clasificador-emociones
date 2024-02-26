import numpy as np
from sklearn.model_selection import StratifiedKFold
from app.procesado import  get_content_from_documents, apply_filtrado_to_content, get_sentiment_from_documents
from sklearn.metrics import classification_report
import nltk
def stratify(data, categoria, randm):
    # Asignar los datos de entrada a las variables X e y
    X = data
    y = categoria
    
    # Crear una instancia de StratifiedKFold con 10 divisiones para asi poder obtener un 20% de indices de test, aleatoriedad y una semilla aleatoria
    skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=randm)
    
    # Obtener el número de divisiones de StratifiedKFold
    skf.get_n_splits(X, y)
    
    # Iterar sobre las divisiones generadas por StratifiedKFold
    for i, (train_index, test_index) in enumerate(skf.split(X, y)):
        # Almacenar los índices de entrenamiento y prueba en una lista ac
        ac = [train_index, test_index]
    
    # Devolver la lista de índices de entrenamiento y prueba
    return ac



def clasificador(data):
    content_list = get_content_from_documents()
    sentiment_list= get_sentiment_from_documents()
    filtered_content_list = apply_filtrado_to_content(content_list)
    separated_words_list = [[word for word in sentence.split()] for sentence in filtered_content_list]
    data = [(content, sentiment) for content, sentiment in zip(separated_words_list, sentiment_list)]
    # Obtener los conjuntos de entrenamiento y prueba mediante la función stratify
    split_data = stratify(data, sentiment_list, 8)
    train_set = [data[index] for index in split_data[0]]
    test_set = [data[index] for index in split_data[1]]
    print(len(test_set))
    # Crear el conjunto de entrenamiento para el clasificador Naive Bayes
    training_set = [(nltk.FreqDist(features), category) for (features, category) in train_set]
    
    # Entrenar el clasificador Naive Bayes
    classifier = nltk.NaiveBayesClassifier.train(training_set)

    # Crear el conjunto de pruebas
    test_data = [(nltk.FreqDist(features), category) for (features, category) in test_set]

    # Obtener las etiquetas reales y las predicciones del clasificador para el conjunto de pruebas
    y_true = [category for features, category in test_data]
    y_pred = [classifier.classify(features) for features, category in test_data]
    # Imprimir el informe de clasificación utilizando classification_report de scikit-learn
    return classification_report(y_true, y_pred)

