import os
import streamlit as st
from dotenv import load_dotenv
from langchain_core.messages import AIMessage, HumanMessage
from langchain_core.prompts import ChatPromptTemplate
from langchain_groq import ChatGroq
from langchain_core.output_parsers import StrOutputParser
from FAQs import faq
import sounddevice as sd
import numpy as np
import faster_whisper
import soundfile as sf
from gtts import gTTS
from intent_detection import detect_intent_with_llm


load_dotenv()


def record_voice_input(duration=5, fs=44100):
    st.write("Recording...")
    recording = sd.rec(int(duration * fs), samplerate=fs, channels=1, dtype='float32')
    sd.wait()  # Wait until recording is finished
    st.write("Recording finished.")
    return recording


def transcribe_voice_input(recording, fs=44100):
    model = faster_whisper.WhisperModel("base")
    audio_data = (recording.flatten() * 32767).astype(np.int16)  
    st.write(f"Audio data length: {len(audio_data)}")  
    st.write(f"Audio data sample: {audio_data[:10]}")  
    st.write("Transcribing...")

    temp_audio_file = "temp_audio.wav"
    sf.write(temp_audio_file, audio_data, fs)

    segments, _ = model.transcribe(temp_audio_file, language="en")
    
    transcription = " ".join([segment.text for segment in segments])
    st.write(f"Transcription: {transcription}")  
    return transcription, temp_audio_file


def get_response(user_query: str, chat_history: list):
  
  chat_history_formatted = "\n".join([f"User: {message.content}" if isinstance(message, HumanMessage) else f"AI: {message.content}" for message in chat_history])
  
  template = """
    You are a customer support agent for a bank. You are interacting with a user who is asking you questions about banking services and transactions.
    Based on the context and conversation history provided, generate a helpful and accurate response to the user's query. Take the conversation history into account.

    Context:
    - You are knowledgeable about various banking services such as account management, loan services, credit and debit card issues, and transaction inquiries.
    - You provide clear and concise information and can guide users through different banking processes.
    - Maintain a polite and professional tone.

    I want you to answer most questions for this common {faq} rather than use your own thinking.

    Conversation History: {chat_history}

    For example:
    Question: How can I apply for a personal loan?
    Response: To apply for a personal loan, you can visit our website and fill out the application form under the "Loans" section. You will need to provide your personal details, employment information, and income details. Alternatively, you can visit any of our branches and apply in person.

    Question: I lost my debit card. What should I do?
    Response: If you have lost your debit card, please call our customer service immediately at [customer service number] to report the loss and block the card. You can also log in to your online banking account and report the lost card under the "Card Services" section. A new card will be issued to you within 5-7 business days.

    
    

    Your turn:

    Question: {question}
    Response:
    """
  formatted_prompt = template.format(
        chat_history=chat_history_formatted,
        question=user_query,
        faq = faq

    )
  llm = ChatGroq(model="mixtral-8x7b-32768", temperature=0)
  response = llm.invoke(formatted_prompt)
  return response.content

def text_to_speech(text: str, lang: str = 'en'):
    tts = gTTS(text=text, lang=lang)
    tts.save("response.mp3")
    return "response.mp3"


if "chat_history" not in st.session_state:
    st.session_state.chat_history = [
      AIMessage(content="Hello! I'm a Customer Support assistant. How can I help you."),
    ]



st.set_page_config(page_title="Chat with Customer Support", page_icon=":speech_balloon:")

st.title("Chat with Cutomer Support")

for message in st.session_state.chat_history:
    if isinstance(message, AIMessage):
        with st.chat_message("AI"):
            st.markdown(message.content)
    elif isinstance(message, HumanMessage):
        with st.chat_message("Human"):
            st.markdown(message.content)

user_query = st.chat_input("Type a message...")

voice_input = st.button("Send Voice Input")
temp_audio_file = None
if voice_input:
    recording = record_voice_input()
    st.write(f"Recording shape: {recording.shape}")  
    transcription, temp_audio_file = transcribe_voice_input(recording)
    user_query = transcription
    st.write(f"Transcribed Voice Input: {user_query}")
    st.audio(temp_audio_file)  


if user_query is not None and user_query.strip() != "":
    st.session_state.chat_history.append(HumanMessage(content=user_query))
    
    with st.chat_message("Human"):
        st.markdown(user_query)
        
    with st.chat_message("AI"):
        response = get_response(user_query, st.session_state.chat_history)
        st.markdown(response)
        if voice_input:
            audio_file = text_to_speech(response)
            st.audio(audio_file)
    st.session_state.chat_history.append(AIMessage(content=response))

    detected_intent = detect_intent_with_llm(user_query)
    st.write(f"Detected Intent: {detected_intent}")