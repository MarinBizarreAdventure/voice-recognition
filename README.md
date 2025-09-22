
# English Pronunciation Checker and Evaluation Suite

This project provides a **Streamlit web application** for real-time English pronunciation practice and a separate script to evaluate a speech recognition model's accuracy on a real-world dataset. The app uses the **Vosk toolkit** for offline speech recognition.

---

## Features
- **Interactive Web App (`app.py`)**  
  An intuitive interface where a user can speak a word or phrase, have the model transcribe it, and see a real-time correctness score.

- **Model Accuracy Test (`test_model_accuracy.py`)**  
  A powerful script that evaluates the Vosk model's performance on a large dataset of English audio, providing detailed accuracy metrics.

- **Offline Functionality**  
  All speech recognition and text-to-speech tasks are handled offline, requiring no internet connection after the initial setup.



## Setup Instructions

### 1. Create and Activate a Virtual Environment

```bash
# Create a virtual environment
python -m venv venv

# Activate the environment
# On macOS and Linux
source venv/bin/activate

# On Windows
.\venv\Scripts\activate
```

### 2. Install Python Libraries

Generate and install dependencies:

```bash
# Generate requirements.txt
pip freeze > requirements.txt

# Install dependencies
pip install -r requirements.txt
```

### 3. Download Required Files

#### Vosk Model

* Download from [Vosk Models](https://alphacephei.com/vosk/models)
* Recommended: `vosk-model-small-en-us-0.15` (40 MB)
* Extract and place the folder in your project root

#### LJ Speech Dataset

* Download from [LJ Speech Dataset](https://keithito.com/LJ-Speech-Dataset/)
* Extract and place `LJSpeech-1.1` in your project root

Your project structure should look like this:

```
your-project-folder/
├── app.py
├── test_model_accuracy.py
├── requirements.txt
├── venv/
├── vosk-model-small-en-us-0.15/   <-- Model folder
└── LJSpeech-1.1/                  <-- Dataset folder
```

---

## How to Use

### A. Run the Pronunciation App

```bash
streamlit run app.py
```

Your browser will open. Enter a sentence, click **Read Aloud**, then **Record** to test your pronunciation.

### B. Run the Model Accuracy Test

```bash
python test_model_accuracy.py
```

The script processes audio files and prints results in the terminal. It may take several minutes.

---

## Customizing the Test

Edit `test_model_accuracy.py` to change the number of test cases:

```python
# Default (tests up to 50 samples)
test_limit = min(total_count, 50)
```

Modify the number (e.g., `100`) or remove `min(total_count, 50)` to test the entire dataset.

---

## Understanding the Accuracy Report

The evaluation script outputs results in three categories:

* **Fully Correct**: Exact match (case/punctuation normalized)
* **Minor Errors**: Word-level accuracy ≥ 70% (small mistakes allowed)
* **Incorrect**: Word-level accuracy < 70%

---

## Troubleshooting

* **`AttributeError: module 'streamlit' has no attribute 'experimental_rerun'`**
  Use `st.rerun()` instead.

* **`Error during audio recording: ...`**
  Ensure your microphone is configured and drivers are installed.

* **`Warning: Audio file has a different sample rate`**
  This is normal. Audio is automatically resampled to 16kHz for Vosk.

---
