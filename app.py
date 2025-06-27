import streamlit as st
import speech_recognition as sr
import pyttsx3
import random
import json
import pickle
import numpy as np
from keras.models import load_model
from nltk.stem import WordNetLemmatizer
from nltk.tokenize import word_tokenize
import nltk

nltk.download('punkt')

# === Initialize persistent components ===
if 'model' not in st.session_state:
    st.session_state.model = load_model('chatbot_model.h5')
    st.session_state.intents = json.loads(open('cleaned_intents.json', encoding='utf-8').read())
    st.session_state.words = pickle.load(open('words.pkl', 'rb'))
    st.session_state.classes = pickle.load(open('classes.pkl', 'rb'))
    st.session_state.recognizer = sr.Recognizer()

# Access session state
model = st.session_state.model
intents = st.session_state.intents
words = st.session_state.words
classes = st.session_state.classes
recognizer = st.session_state.recognizer
lemmatizer = WordNetLemmatizer()

# === NLP Helper Functions ===
def clean_up_sentence(sentence):
    sentence_words = word_tokenize(sentence)
    sentence_words = [lemmatizer.lemmatize(word.lower()) for word in sentence_words]
    return sentence_words

def bow(sentence, words):
    sentence_words = clean_up_sentence(sentence)
    bag = [0] * len(words)
    for s in sentence_words:
        for i, w in enumerate(words):
            if w == s:
                bag[i] = 1
    return np.array(bag)

def predict_class(sentence):
    bow_input = bow(sentence, words)
    res = model.predict(np.array([bow_input]), verbose=0)[0]
    threshold = 0.7
    results = [[i, r] for i, r in enumerate(res) if r > threshold]
    results.sort(key=lambda x: x[1], reverse=True)
    return [{'intent': classes[r[0]], 'probability': str(r[1])} for r in results]

def get_response(intents_list, intents_json):
    if intents_list:
        tag = intents_list[0]['intent']
        for i in intents_json['intents']:
            if i['tag'] == tag:
                return random.choice(i['responses']), tag
    return "I'm not sure I understand. Could you please rephrase?", "unknown"

def chat_response(msg):
    ints = predict_class(msg)
    response, tag = get_response(ints, intents)
    return response, tag

# === Voice I/O ===
def listen():
    with sr.Microphone() as source:
        recognizer.adjust_for_ambient_noise(source, duration=0.5)
        audio = recognizer.listen(source)
        try:
            text = recognizer.recognize_google(audio)
            return text
        except sr.UnknownValueError:
            st.error("Sorry, I couldn't understand that.")
            return ""
        except sr.RequestError:
            st.error("Speech service error.")
            return ""

def speak(text, gender="Female"):
    engine = pyttsx3.init()
    voices = engine.getProperty('voices')
    if gender == "Male":
        engine.setProperty('voice', voices[0].id)
    else:
        engine.setProperty('voice', voices[1].id)
    engine.setProperty("rate", 175)
    engine.setProperty("volume", 1.0)
    engine.say(text)
    engine.runAndWait()

# === Streamlit UI ===
st.set_page_config(page_title="MediTalk Voice Chatbot", page_icon="🩺", layout="centered")
st.title("🩺 MediTalk - Voice Health Assistant")
st.image("assets/icons8-chatbot-100.png", width=150)
st.markdown("Talk or type to interact with the medical chatbot.")
st.markdown("---")

mode = st.radio("Choose your input method:", ["📝 Type", "🎤 Speak"])

# --- TEXT MODE ---
if mode == "📝 Type":
    user_input = st.text_input("Enter your message:", "")
    if st.button("Send") and user_input.strip():
        response, tag = chat_response(user_input)
        st.success(f"Bot: {response}")
        speak(response)

# --- VOICE MODE ---
elif mode == "🎤 Speak":
    voice_gender = st.radio("Select Voice Gender:", ["Female", "Male"])
    if st.button("Speak Now"):
        speak("Please tell me your symptoms or question.", voice_gender)
        with st.spinner("Listening..."):
            user_input = listen()
        if user_input.strip():
            st.success(f"You said: {user_input}")
            response, tag = chat_response(user_input)

            if tag not in ['greeting', 'goodbye', 'thanks', 'unknown', 'no_match']:
                speak("Scanning our database for your symptom. Please wait...", voice_gender)
                speak("Found it. Based on our database, it looks like " + response, voice_gender)
            else:
                speak(response, voice_gender)

            st.success(f"Bot: {response}")

st.markdown("---")
st.caption("Developed by Kranthi Kumar 👨‍💻 using Streamlit, TensorFlow, and NLTK")
