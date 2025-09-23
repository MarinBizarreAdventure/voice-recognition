import streamlit as st
import numpy as np
import sounddevice as sd
import vosk
import json
import Levenshtein
import torch
from transformers import AutoProcessor, CsmForConditionalGeneration

# --- Configuration ---
# Set a threshold for what is considered a "correct" pronunciation.
ACCURACY_THRESHOLD_PERCENTAGE = 70

# Define the Marvis TTS model for the transformers library
MARVIS_MODEL_ID = "Marvis-AI/marvis-tts-250m-v0.1-transformers"

# Map user-friendly names to speaker IDs. The transformers version uses numeric IDs.
# Let's assume [0] is female and [1] is male.
MARVIS_VOICES = {
    "Female": "[0]",
    "Male": "[1]"
}

# --- Model Loading Functions ---

@st.cache_resource
def load_vosk_model():
    """Loads the Vosk ASR model and returns the recognizer."""
    try:
        model = vosk.Model("vosk-model-small-en-us-0.15")
        return vosk.KaldiRecognizer(model, 16000)
    except Exception as e:
        st.error(f"Error loading Vosk model: {e}. Make sure 'vosk-model-small-en-us-0.15' is in your project directory.")
        return None

@st.cache_resource
def load_marvis_model():
    """
    Loads the Marvis TTS model and processor from Hugging Face using the transformers library.
    This is a stable, cross-platform method.
    """
    try:
        device = "cuda" if torch.cuda.is_available() else "cpu"

        processor = AutoProcessor.from_pretrained(MARVIS_MODEL_ID)
        model = CsmForConditionalGeneration.from_pretrained(MARVIS_MODEL_ID).to(device)
        
        return model, processor, device
    except Exception as e:
        # We'll catch the error in the main app block to display it
        st.session_state.model_load_error = e
        return None, None, None

def speak_text(text, model, processor, device, speaker_id):
    """
    Generates and plays speech using the Marvis TTS model with transformers.
    """
    try:
        # Prepend the speaker ID to the text, as required by the model
        text_with_speaker = f"{speaker_id}{text}"

        # Process the text to create model inputs
        inputs = processor(text_with_speaker, add_special_tokens=True, return_tensors="pt").to(device)
        
        # The model may not use token_type_ids
        if "token_type_ids" in inputs:
            inputs.pop("token_type_ids")

        # Generate the audio waveform
        with st.spinner("Generating audio..."):
            audio_tensor = model.generate(**inputs, output_audio=True)
        
        # Move tensor to CPU, convert to NumPy array for playback
        waveform = audio_tensor[0].cpu().numpy()
        sample_rate = 24_000  # The model's native sample rate
        
        st.audio(waveform, sample_rate=sample_rate)
        
    except Exception as e:
        st.error(f"An unexpected error occurred during audio generation: {e}")

# --- Utility and Accuracy Functions (Unchanged) ---

def clean_text(text):
    """Cleans text by removing punctuation and converting to lowercase."""
    return text.lower().strip()

def calculate_word_accuracy(transcribed_text, expected_text):
    """Calculates word accuracy using Levenshtein distance."""
    transcribed_words = clean_text(transcribed_text).split()
    expected_words = clean_text(expected_text).split()

    if not expected_words: return 100.0 if not transcribed_words else 0.0
    if not transcribed_words: return 0.0

    total_similarity = 0
    matched_flags = [False] * len(expected_words)
    
    for trans_word in transcribed_words:
        best_match_score, best_match_index = 0, -1
        for i, exp_word in enumerate(expected_words):
            if not matched_flags[i]:
                distance = Levenshtein.distance(trans_word, exp_word)
                similarity = 1.0 - (distance / max(len(trans_word), len(exp_word)))
                if similarity > best_match_score:
                    best_match_score, best_match_index = similarity, i
        if best_match_score > 0 and best_match_index != -1:
            total_similarity += best_match_score
            matched_flags[best_match_index] = True

    return (total_similarity / len(expected_words)) * 100

# --- Main Streamlit App ---

def main():
    st.title("Pronunciation Checker")
    st.write("Powered by **Vosk** (Speech Recognition) and **Marvis TTS** (Speech Synthesis via Transformers).")
    
    if 'recording_state' not in st.session_state:
        st.session_state.recording_state = 'stopped'
    if 'model_load_error' not in st.session_state:
        st.session_state.model_load_error = None

    # Load models
    vosk_recognizer = load_vosk_model()
    
    # Load the Marvis model and display UI feedback here, outside the cached function.
    with st.spinner(f"Loading Marvis TTS model: {MARVIS_MODEL_ID}..."):
        marvis_model, marvis_processor, device = load_marvis_model()

    if st.session_state.model_load_error:
        st.error(f"Error loading Marvis TTS model: {st.session_state.model_load_error}")
        st.error("Please check your internet connection and ensure torch/transformers are installed.")
        return
    else:
        st.toast("Marvis TTS model loaded successfully! 🎉")
        st.info(f"Using device for TTS: {device}")
    
    if not vosk_recognizer or not marvis_model:
        st.warning("One or more models could not be loaded. Please check the logs.")
        return

    user_text = st.text_input("Enter the word or phrase to pronounce:", "Hello world, this is a test.")
    st.markdown("---")
    st.subheader("Pronunciation & Recording")

    col1, col2, col3 = st.columns([2, 1, 1])
    with col1:
        selected_voice_name = st.selectbox("Choose a voice:", options=list(MARVIS_VOICES.keys()))
        selected_speaker_id = MARVIS_VOICES[selected_voice_name]
    with col2:
        if st.button("Read Aloud", use_container_width=True):
            if user_text:
                speak_text(user_text, marvis_model, marvis_processor, device, selected_speaker_id)
    with col3:
        if st.button("Record", use_container_width=True):
            st.session_state.recording_state = 'recording'
            st.session_state.audio_data = []
            st.rerun()

    # Recording and Processing Logic
    if st.session_state.recording_state == 'recording':
        st.info("Recording for 5 seconds...")
        with sd.InputStream(samplerate=16000, channels=1, dtype='int16') as stream:
            try:
                for _ in range(int(5 * 16000 / 1024)):
                    data, _ = stream.read(1024)
                    st.session_state.audio_data.append(data)
            finally:
                st.session_state.recording_state = 'processing'
                st.rerun()

    if st.session_state.recording_state == 'processing':
        if st.session_state.get('audio_data'):
            st.info("Processing recorded audio...")
            audio_bytes = b''.join(st.session_state.audio_data)
            vosk_recognizer.AcceptWaveform(audio_bytes)
            result = json.loads(vosk_recognizer.FinalResult())
            st.session_state.recognized_text = result.get('text', '')
            st.session_state.user_text = user_text
            st.session_state.recording_state = 'stopped'
            st.session_state.audio_data = []
            st.rerun()

    # Display Results
    st.markdown("---")
    st.subheader("Results")
    if st.session_state.get('recognized_text'):
        recognized_text = st.session_state.recognized_text
        user_text = st.session_state.user_text
        word_accuracy = calculate_word_accuracy(recognized_text, user_text)

        st.write(f"**Recognized as:** '{recognized_text}'")
        if word_accuracy >= 99.99:
            st.success(f"✅ **Perfect!** Score: {word_accuracy:.2f}%.")
        elif word_accuracy >= ACCURACY_THRESHOLD_PERCENTAGE:
            st.warning(f"🟡 **Correct.** Score: {word_accuracy:.2f}%. (Expected: '{user_text}')")
        else:
            st.error(f"❌ **Incorrect.** Score: {word_accuracy:.2f}%. (Expected: '{user_text}')")

if __name__ == "__main__":
    main()

