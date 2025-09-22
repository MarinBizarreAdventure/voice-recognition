import vosk
import json
import soundfile as sf
import os
import re
import pandas as pd
from scipy.signal import resample
import numpy as np
import Levenshtein 

# The path to the downloaded and extracted dataset
DATASET_PATH = "LJSpeech-1.1"

# The path to the Vosk model
VOSK_MODEL_PATH = "vosk-model-small-en-us-0.15"

def clean_text(text):
    """
    Cleans text by removing punctuation and converting to lowercase for a fair comparison.
    """
    return re.sub(r'[^\w\s]', '', text).lower().strip()

def calculate_word_accuracy(transcribed_text, expected_text):
    """
    Calculates the percentage of correctly recognized words using a
    Levenshtein-based similarity score.
    """
    transcribed_words = clean_text(transcribed_text).split()
    expected_words = clean_text(expected_text).split()

    if not expected_words:
        return 100.0 if not transcribed_words else 0.0
    if not transcribed_words:
        return 0.0

    total_similarity = 0
    
    matched_flags = [False] * len(expected_words)
    
    for trans_word in transcribed_words:
        best_match_score = 0
        best_match_index = -1
        
        for i, exp_word in enumerate(expected_words):
            if not matched_flags[i]:
                distance = Levenshtein.distance(trans_word, exp_word)
                similarity = 1.0 - (distance / max(len(trans_word), len(exp_word)))
                
                if similarity > best_match_score:
                    best_match_score = similarity
                    best_match_index = i
        
        if best_match_score > 0:
            total_similarity += best_match_score
            if best_match_index != -1:
                matched_flags[best_match_index] = True 
    return (total_similarity / len(expected_words)) * 100

def run_test():
    """
    Reads the LJ Speech dataset, runs each audio clip through the Vosk model,
    and calculates the overall transcription accuracy with two metrics.
    """
    try:
        model = vosk.Model(VOSK_MODEL_PATH)
        recognizer = vosk.KaldiRecognizer(model, 16000)
    except Exception as e:
        print(f"Error loading Vosk model: {e}")
        print(f"Please ensure the '{VOSK_MODEL_PATH}' folder is in your project directory.")
        return

    try:
        metadata_path = os.path.join(DATASET_PATH, 'metadata.csv')
        df = pd.read_csv(metadata_path, sep='|', header=None, names=['id', 'transcription', 'normalized_transcription'])
    except FileNotFoundError:
        print(f"Error: '{metadata_path}' file not found.")
        print(f"Please ensure the '{DATASET_PATH}' folder is in the correct location.")
        return

    print("--- Starting Model Pronunciation Accuracy Test ---")

    fully_correct_count = 0
    minor_errors_count = 0
    incorrect_count = 0
    total_count = len(df)
    
    test_limit = min(total_count, 50) 
    df_subset = df.head(test_limit)

    for index, row in df_subset.iterrows():
        audio_file_name = row['id'] + '.wav'
        audio_path = os.path.join(DATASET_PATH, 'wavs', audio_file_name)
        expected_text = row['normalized_transcription']

        if not os.path.exists(audio_path):
            print(f"Warning: Audio file not found for ID {row['id']}. Skipping...")
            continue

        try:
            data, samplerate = sf.read(audio_path)
            
            if samplerate != 16000:
                num_samples = int(len(data) * 16000 / samplerate)
                data = resample(data, num_samples)
                
            if data.ndim > 1:
                data = data[:, 0]
            
            data_16bit = np.int16(data / np.max(np.abs(data)) * 32767)

            recognizer.AcceptWaveform(data_16bit.tobytes())
            
            transcription_result = json.loads(recognizer.FinalResult())
            transcribed_text = transcription_result.get('text', '')

            word_accuracy = calculate_word_accuracy(transcribed_text, expected_text)

            if word_accuracy >= 99.99:
                fully_correct_count += 1
                print(f"[{index + 1}/{test_limit}] ✅ Fully Correct (100.00%): '{expected_text}'")
            elif word_accuracy >= 70:
                minor_errors_count += 1
                print(f"[{index + 1}/{test_limit}] 🟡 Minor Errors ({word_accuracy:.2f}%): '{expected_text}'")
                print(f"    -> Vosk transcribed: '{transcribed_text}'")
            else:
                incorrect_count += 1
                print(f"[{index + 1}/{test_limit}] ❌ Incorrect ({word_accuracy:.2f}%): '{expected_text}'")
                print(f"    -> Vosk transcribed: '{transcribed_text}'")

        except Exception as e:
            print(f"Error processing {audio_file_name}: {e}")
            continue

    print("\n" + "="*40)
    print("        Final Accuracy Report")
    print("="*40)
    
    fully_correct_rate = (fully_correct_count / test_limit) * 100 if test_limit > 0 else 0
    minor_errors_rate = (minor_errors_count / test_limit) * 100 if test_limit > 0 else 0
    incorrect_rate = (incorrect_count / test_limit) * 100 if test_limit > 0 else 0
    
    print(f"Total Sentences Tested: {test_limit}")
    print(f"Fully Correct: {fully_correct_count} ({fully_correct_rate:.2f}%)")
    print(f"Minor Errors (>=70%): {minor_errors_count} ({minor_errors_rate:.2f}%)")
    print(f"Incorrect (<70%): {incorrect_count} ({incorrect_rate:.2f}%)")

if __name__ == "__main__":
    run_test()