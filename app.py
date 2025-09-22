import streamlit as st
import numpy as np
import sounddevice as sd
import vosk
import json
import pyttsx3
import io

@st.cache_resource
def load_vosk_model():
    """Loads the Vosk model and returns the recognizer."""
    try:
        model = vosk.Model("vosk-model-small-en-us-0.15") 
        return vosk.KaldiRecognizer(model, 16000)
    except Exception as e:
        st.error(f"Error loading Vosk model: {e}. Make sure the 'vosk-model-en-us-0.22' folder is in your project directory.")
        return None

def speak_text(text):
    """Speaks the given text using pyttsx3."""
    engine = pyttsx3.init()
    engine.say(text)
    engine.runAndWait()

def main():
    st.title("🗣️ Pronunciation Checker")
    st.write("Type a word or phrase, record your pronunciation, and see if the model recognizes it!")

    
    if 'recording_state' not in st.session_state:
        st.session_state.recording_state = 'stopped'
    if 'recognized_text' not in st.session_state:
        st.session_state.recognized_text = ""

    recognizer = load_vosk_model()
    if not recognizer:
        return

    user_text = st.text_input("Enter the word or phrase to pronounce:", "Hello world")
    
    st.markdown("---")

    st.subheader("Pronunciation & Recording")

    col1, col2 = st.columns([1, 1])

    with col1:
        if st.button("🔊 Read Aloud"):
            if user_text:
                with st.spinner(f"Reading out loud: '{user_text}'"):
                    speak_text(user_text)
            else:
                st.warning("Please enter some text to read.")

    with col2:
        record_button = st.button("🎤 Record")
        if record_button:
            st.session_state.recording_state = 'recording'
            st.session_state.audio_data = []

    if st.session_state.recording_state == 'recording':
        st.info("Recording for 5 seconds... 🎙️")
        
        with sd.InputStream(samplerate=16000, channels=1, dtype='int16') as stream:
            try:
                for _ in range(0, int(5 * 16000 / 1024)):
                    data = stream.read(1024)
                    st.session_state.audio_data.append(data[0])
            except Exception as e:
                st.error(f"Error during audio recording: {e}")
            finally:
                st.session_state.recording_state = 'processing'
                st.rerun() 

    if st.session_state.recording_state == 'processing':
        if st.session_state.audio_data:
            st.info("Processing recorded audio... 🧠")
            audio_bytes = b''.join([d.tobytes() for d in st.session_state.audio_data])

            recognizer.AcceptWaveform(audio_bytes)
            result = json.loads(recognizer.FinalResult())
            recognized_text = result.get('text', '')
            st.session_state.recognized_text = recognized_text
            st.session_state.user_text = user_text
            
            st.session_state.recording_state = 'stopped'
            st.rerun() 

    st.markdown("---")

    st.subheader("Results")

    if 'recognized_text' in st.session_state and st.session_state.recognized_text:
        recognized_text = st.session_state.recognized_text.lower().strip()
        user_text = st.session_state.user_text.lower().strip()

        st.write(f"**Your pronunciation was recognized as:** '{recognized_text}'")

        if recognized_text == user_text:
            st.success("✅ **Correct!** Your pronunciation matches the text.")
        else:
            st.error("❌ **Incorrect.** Your pronunciation did not match the text.")
            st.info(f"The expected text was: '{user_text}'")

if __name__ == "__main__":
    main()