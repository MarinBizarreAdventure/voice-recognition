import streamlit as st
import numpy as np
import sounddevice as sd
import vosk
import json
import pyttsx3
import io
import Levenshtein

ACCURACY_THRESHOLD_PERCENTAGE = 70

@st.cache_resource
def load_vosk_model():
    """Loads the Vosk model and returns the recognizer."""
    try:
        model = vosk.Model("vosk-model-small-en-us-0.15") 
        return vosk.KaldiRecognizer(model, 16000)
    except Exception as e:
        st.error(f"Error loading Vosk model: {e}. Make sure the 'vosk-model-small-en-us-0.15' folder is in your project directory.")
        return None

def speak_text(text):
    """Speaks the given text using pyttsx3."""
    engine = pyttsx3.init()
    engine.say(text)
    engine.runAndWait()

def clean_text(text):
    """
    Cleans text by removing punctuation and converting to lowercase.
    """
    return text.lower().strip()

def calculate_word_accuracy(transcribed_text, expected_text):
    """
    Calculates the percentage of correctly recognized words using Levenshtein distance.
    This provides a more flexible score than a simple word-for-word match.
    """
    transcribed_words = clean_text(transcribed_text).split()
    expected_words = clean_text(expected_text).split()

    if not expected_words:
        return 100.0 if not transcribed_words else 0.0
    if not transcribed_words:
        return 0.0

    total_similarity = 0
    
    # Create a list of flags to track matched words
    matched_flags = [False] * len(expected_words)
    
    # Iterate through each word in the transcribed text
    for trans_word in transcribed_words:
        best_match_score = 0
        best_match_index = -1
        
        # Find the best matching, unmatched word in the expected text
        for i, exp_word in enumerate(expected_words):
            if not matched_flags[i]:
                distance = Levenshtein.distance(trans_word, exp_word)
                # Normalize the distance to get a similarity score between 0 and 1
                similarity = 1.0 - (distance / max(len(trans_word), len(exp_word)))
                
                if similarity > best_match_score:
                    best_match_score = similarity
                    best_match_index = i
        
        if best_match_score > 0:
            total_similarity += best_match_score
            if best_match_index != -1:
                matched_flags[best_match_index] = True

    return (total_similarity / len(expected_words)) * 100

def main():
    st.title("🗣️ Pronunciation Checker")
    st.write(f"Type a word or phrase, record your pronunciation, and see if the model recognizes it!")
    st.write(f"Pronunciation is considered 'correct' if the accuracy is above {ACCURACY_THRESHOLD_PERCENTAGE}%.")

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
        recognized_text = st.session_state.recognized_text
        user_text = st.session_state.user_text

        # Calculate word accuracy
        word_accuracy = calculate_word_accuracy(recognized_text, user_text)

        st.write(f"**Your pronunciation was recognized as:** '{recognized_text}'")

        if word_accuracy >= 99.99:
            st.success(f"✅ **Perfect!** Your pronunciation matches the text with a score of {word_accuracy:.2f}%.")
        elif word_accuracy >= ACCURACY_THRESHOLD_PERCENTAGE:
            st.warning(f"🟡 **Correct with minor errors.** Your pronunciation scored {word_accuracy:.2f}% which is above the threshold of {ACCURACY_THRESHOLD_PERCENTAGE}%.")
            st.info(f"The expected text was: '{user_text}'")
        else:
            st.error(f"❌ **Incorrect.** Your pronunciation scored {word_accuracy:.2f}% which is below the threshold of {ACCURACY_THRESHOLD_PERCENTAGE}%.")
            st.info(f"The expected text was: '{user_text}'")

if __name__ == "__main__":
    main()