
# English Pronunciation Checker and Evaluation Suite

This project provides a **Streamlit web application** for real-time English pronunciation practice and a separate script to evaluate a speech recognition model's accuracy on a real-world dataset. The app uses the **Vosk toolkit** for offline speech recognition.

---

## Features

- **Interactive Web App (`app.py`)**: An intuitive interface where a user can type a phrase, listen to a natural-sounding AI voice read it, and then record their own pronunciation for an instant accuracy score.
- **High-Quality Text-to-Speech**: Uses the Marvis TTS model to generate clear, expressive audio, allowing users to hear the correct pronunciation before they practice.
- **Offline Speech Recognition**: The Vosk ASR model runs entirely locally, ensuring privacy and functionality without a constant internet connection.
- **Model Accuracy Test (`test_model_accuracy.py`)**: A powerful script that evaluates the Vosk model's performance on the LJ Speech dataset, providing detailed accuracy metrics.

**Note**: The application requires an internet connection on the first run to download the Marvis TTS model and its dependencies from Hugging Face. After the initial download, it can run offline.



## Setup Instructions

### 1. Create and Activate a Virtual Environment

It is highly recommended to use a Python version between 3.9 and 3.12, as newer versions may have compatibility issues with some libraries.

```bash
# Create a virtual environment (e.g., with Python 3.11)
python3.11 -m venv venv

# Activate the environment
# On macOS and Linux
source venv/bin/activate

# On Windows
.\venv\Scripts\activate
```

### 2. Install Python Libraries

Install the required dependencies from the `requirements.txt` file.

```bash
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

## Results

Below are some sample evaluation outputs from the `test_model_accuracy.py` script:


```
\[1971/2000] ✅ Fully Correct (100.00%): 'I propose to return now to the subject of Newgate executions,'
\[1972/2000] 🟡 Minor Errors (78.80%): 'which we left at the time of the discontinuance of the long-practiced procession to Tyburn.'
-> Vosk transcribed: 'which we left at the time of the discontinuance of the law practice procession to tiber'
\[1973/2000] ✅ Fully Correct (100.00%): 'The reasons for this change were fully set forth in a previous chapter.'
\[1974/2000] 🟡 Minor Errors (95.33%): 'The terrible spectacle was as demoralizing to the public, for whose admonition it was intended,'
-> Vosk transcribed: 'the terrible spectacle was is demoralizing to the public for who's admonition it was intended'
\[1975/2000] ✅ Fully Correct (100.00%): 'as the exposure was brutal and cruel towards the principal actors.'
...
\[2000/2000] ✅ Fully Correct (100.00%): 'when, in pursuance of an order issued by the Recorder to the sheriffs of Middlesex and the keeper of His Majesty's jail, Newgate,'

```

### Final Accuracy Report
```

=====================
Final Accuracy Report
=====================

Total Sentences Tested: 2000
Fully Correct: 740 (37.00%)
Minor Errors (>=70%): 1190 (59.50%)
Incorrect (<70%): 56 (2.80%)

```

✅ **Fully Correct**: Exact match (case/punctuation normalized)  
🟡 **Minor Errors**: Word-level accuracy ≥ 70%  
❌ **Incorrect**: Word-level accuracy < 70%  

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

Edit `ACCURACY_THRESHOLD_PERCENTAGE ` to change the percentage of corectness to count as corect:

```python
# Default is 70 (10 line in app.py)
ACCURACY_THRESHOLD_PERCENTAGE = 70

```
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
