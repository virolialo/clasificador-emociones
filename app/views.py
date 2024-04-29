from contextlib import nullcontext
from django.shortcuts import redirect, render

from nltk.corpus import stopwords
from app.naiveBayes import clasificador

from app.word2vec import *


def pagina_principal(request):
    return render(request, 'pagina_principal.html')

def botones_modelos(request):
    return render(request, 'models.html')

def boton_viewNB(request):
    if request.method == 'POST':
        content_list = get_content_from_documents()
        sentiment_list = get_sentiment_from_documents()
        filtered_content_list = apply_filtrado_to_content(content_list)
        separated_words_list = [[word for word in sentence.split()] for sentence in filtered_content_list]
        data = [(content, sentiment) for content, sentiment in zip(separated_words_list, sentiment_list)]
        
        # Llamar a la función clasificador con los datos
        result_report = clasificador(data)

        # Redirige a otra vista para mostrar el resultado
        return render(request, 'resultado_templateNB.html', {'result_report': result_report})
    
    return render(request, 'models.html')



def boton_viewGNB(request):
    if request.method == 'POST':
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
        

        y_pred= model.predict(X_test)
        result_report= ouput_result(y_pred)

        # Redirige a otra vista para mostrar el resultado
        return render(request, 'resultado_templateGNB.html', {'result_report': result_report})
    
    return render(request, 'models.html')


def boton_viewDT(request):
    if request.method == 'POST':
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
        dt_params = {'max_depth': [10, 60]}
        dt_model = GridSearchCV(DecisionTreeClassifier(), dt_params, cv=5, scoring='f1_weighted')
        dt_model.fit(X_train, y_train)

        y_pred = dt_model.predict(X_test)
        result_report = ouput_result(y_pred)

        # Redirige a otra vista para mostrar el resultado
        return render(request, 'resultado_templateDT.html', {'result_report': result_report})
    
    return render(request, 'models.html')

def boton_viewKnn(request):
    if request.method == 'POST':
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

        knn_params = {'n_neighbors': [2, 50]}
        knn_model = GridSearchCV(KNeighborsClassifier(), knn_params, cv=2, scoring='f1_weighted')
        knn_model.fit(X_train, y_train)

        y_pred = knn_model.predict(X_test)
        result_report = ouput_result(y_pred)

        # Redirige a otra vista para mostrar el resultado
        return render(request, 'resultado_templateKnn.html', {'result_report': result_report})
    
    return render(request, 'models.html')

