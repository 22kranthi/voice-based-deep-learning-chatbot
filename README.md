# 🎤 Voice-Based Deep Learning Chatbot

This project is a voice-enabled chatbot developed using Natural Language Processing (NLP) and Deep Learning techniques. The system interprets spoken user input, classifies it into predefined intents using a trained neural network model, and responds through both text and voice output.

---

## 🔧 Features

- 🎙️ Voice-to-text input using `SpeechRecognition`
- 🧠 Intent classification using a deep learning model (Keras + TensorFlow)
- 💬 Text and voice-based responses via `pyttsx3`
- 📊 Streamlit-based web interface for demo interaction (optional)

---

## 🛠️ Technologies Used
```
- Python 3.x
- NLTK (Natural Language Toolkit)
- TensorFlow / Keras
- SpeechRecognition
- pyttsx3 (Text-to-Speech)
- Streamlit (optional UI)
- NumPy, Matplotlib
```
---

## 📁 Project Structure

voice-chatbot/
├── assets/ # Images like diagrams and screenshots
├── intents.json # Dataset with user intents, patterns, and responses
├── train.py # Script to train the model
├── model.h5 # Saved trained model
├── words.pkl # Pickled vocabulary
├── classes.pkl # Pickled intent labels
├── chat.py # Voice-based chatbot logic
├── app.py # Streamlit web interface
├── requirements.txt # Project dependencies
└── README.md # Project documentation


---

## 🚀 Getting Started

bash
pip install -r requirements.txt
python train.py          # Train the model
python chat.py           # Run the voice chatbot
streamlit run app.py     # Run the Streamlit UI (optional)

📌 Future Scope
Add multilingual support
Integrate BERT or LLMs
Deploy to cloud or mobile platforms
