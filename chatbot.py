import random
import json
import pickle
import numpy as np

import nltk
from nltk.stem import WordNetLemmatizer

from keras.models import load_model

lemmatizer = WordNetLemmatizer()

# Cargamos los archivos generados anteriormente
intents = json.loads(open('intents.json').read())
words = pickle.load(open('words.pkl', 'rb'))
classes = pickle.load(open('classes.pkl', 'rb'))
model = load_model('chatbot_model.h5')

context = {}
user_data = {}


def clean_up_sentence(sentence):
    sentence_words = nltk.word_tokenize(sentence)
    sentence_words = [lemmatizer.lemmatize(word) for word in sentence_words]
    return sentence_words


def bag_of_words(sentence):
    sentence_words = clean_up_sentence(sentence)
    bag = [0] * len(words)
    for w in sentence_words:
        for i, word in enumerate(words):
            if word == w:
                bag[i] = 1
    return np.array(bag)


def predict_class(sentence):
    bow = bag_of_words(sentence)
    res = model.predict(np.array([bow]))[0]
    max_index = np.where(res == np.max(res))[0][0]
    category = classes[max_index]
    return category


def get_response(tag, intents_json, user_id=''):
    list_of_intents = intents_json['intents']
    for i in list_of_intents:
        if i["tag"] == tag:
            response = random.choice(i['responses'])
            if 'context_set' in i:
                context[user_id] = i['context_set']
            return response
    return ""


def chatbot_response(msg, user_id='123'):
    ints = predict_class(msg)
    tag = ints
    res = get_response(tag, intents, user_id)
    return res


def extract_entity(text, entity):
    words = text.split()
    for word in words:
        if word.isdigit() and len(word) == 8:
            return word
    return None


while True:
    message = input("Yo: ")
    user_id = "123"  # Identificador de usuario (puede ser cualquier cosa única)

    if user_id not in context:
        context[user_id] = ""

    if context[user_id] == "appointment_dni":
        dni = extract_entity(message, 'dni')
        if dni:
            user_data[user_id] = {"dni": dni}
            context[user_id] = "appointment_details"
            response = "Gracias. Ahora, ¿cual es tu nombre y numero de telefono?"
        else:
            response = "Por favor proporciona un numero de DNI valido."
    elif context[user_id] == "appointment_details":
        # Aquí extraes nombre y teléfono y los guardas en user_data
        user_data[user_id].update({"name": message.split()[0], "phone": message.split()[-1]})
        context[user_id] = "appointment_date"
        response = "Gracias, {}. ¿Para qué dia te gustaria agendar la cita?".format(user_data[user_id]["name"])
    elif context[user_id] == "appointment_date":
        # Aquí extraes la fecha y finalizas el proceso de registro
        user_data[user_id].update({"date": message})
        context[user_id] = ""
        response = "Perfecto, {}. Tu cita para el {} ha sido agendada.".format(user_data[user_id]["name"],
                                                                               user_data[user_id]["date"])
    else:
        response = chatbot_response(message, user_id)

    print("Chatbot: " + response)
