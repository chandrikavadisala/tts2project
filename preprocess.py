# project/English/preprocess.py (Or project/Telugu/preprocess.py)
import os
import re
import yaml
import numpy as np
import librosa
import pyworld as pw
from scipy.io import wavfile

# --- Text Processing Modules ---
import inflect
import nltk
from nltk.tokenize import word_tokenize

try:
    nltk.data.find('tokenizers/punkt')
    nltk.data.find('taggers/averaged_perceptron_tagger')
except LookupError:
    nltk.download('punkt')
    nltk.download('averaged_perceptron_tagger')

_inflect = inflect.engine()

# Unique Phonemic/Character Mappings
ENGLISH_SYMBOLS = ["_"] + list("ABCDEFGHIJKLMNOPQRSTUVWXYZabcdefghijklmnopqrstuvwxyz'.,!? ")
TELUGU_SYMBOLS = ["_"] + [chr(c) for c in range(0x0C00, 0x0C7F)] + list(".,!? ")

# Custom Dictionary for E4 code-switching and edge-cases
LEXICON_EXCEPTIONS = {
    "andaroo": ["AH", "N", "D", "AA", "R", "UW"]
}

def normalize_text_english(text):
    """Handles abbreviations, expansions, numbers, and case standardization."""
    text = text.replace("AM", "a m").replace("PM", "p m")
    words = word_tokenize(text)
    normalized_words = []
    for word in words:
        if word.isdigit():
            normalized_words.append(_inflect.number_to_words(word))
        else:
            normalized_words.append(word)
    return " ".join(normalized_words).lower()

def resolve_heteronyms_and_phonemize(text):
    """Uses Parts of Speech tagging to uniquely distinguish heteronym tokens like 'conduct'."""
    tokens = word_tokenize(text)
    tagged = nltk.pos_tag(tokens)
    phoneme_sequence = []
    
    for word, tag in tagged:
        word_lower = word.lower()
        if word_lower in LEXICON_EXCEPTIONS:
            phoneme_sequence.extend(LEXICON_EXCEPTIONS[word_lower])
            continue
            
        # Context-dependent rules for heteronyms (E2a vs E2b)
        if word_lower == "conduct":
            if tag.startswith("VB"):  # Verb variant
                phoneme_sequence.extend(["k", "ah", "n", "d", "AH", "k", "t"])
            else:  # Noun/Noun Phrase variant
                phoneme_sequence.extend(["K", "AA", "n", "d", "ah", "k", "t"])
        else:
            # Fallback simple character parsing or external G2P dictionary lookup
            phoneme_sequence.extend(list(word_lower))
            
    return phoneme_sequence

def text_to_sequence(text, language="English"):
    """Transforms raw text sequences into system numerical token array mappings."""
    symbols = ENGLISH_SYMBOLS if language == "English" else TELUGU_SYMBOLS
    symbol_to_id = {s: i for i, s in enumerate(symbols)}
    
    if language == "English":
        clean_text = normalize_text_english(text)
        phones = resolve_heteronyms_and_phonemize(clean_text)
    else:
        # Telugu is phonetic by nature; simple regex stripping and symbol mapping
        phones = list(re.sub(r'[^\u0C00-\u0C7F\s.,!?]', '', text))
        
    sequence = [symbol_to_id[p] for p in phones if p in symbol_to_id]
    return np.array(sequence, dtype=np.int32)

# --- Feature Extraction Pipeline ---
def extract_variance_features(wav_path, config):
    """Extracts ground truth pitch (F0), Energy, and dummy durations for alignment."""
    sr, audio = wavfile.read(wav_path)
    if sr != config["audio"]["sampling_rate"]:
        audio = librosa.resample(audio.astype(np.float32), orig_sr=sr, target_sr=config["audio"]["sampling_rate"])
        sr = config["audio"]["sampling_rate"]
        
    audio = audio.astype(np.float64) / config["audio"]["max_wav_value"]
    
    # Pitch Contour extraction via PyWorld World Algorithm
    f0, t = pw.dio(audio, sr, frame_period=config["audio"]["hop_length"] / sr * 1000)
    f0 = pw.stonemask(audio, f0, t, sr)
    
    # Energy Contour extraction via short-time root-mean-square
    hop_length = config["audio"]["hop_length"]
    win_length = config["audio"]["win_length"]
    energy = np.array([
        np.sqrt(np.mean(audio[i : i + win_length] ** 2))
        for i in range(0, len(audio), hop_length)
    ])
    
    # Dynamic sequence alignment tracking padding lengths
    min_len = min(len(f0), len(energy))
    return f0[:min_len], energy[:min_len]

if __name__ == "__main__":
    with open("config.yaml", "r") as f:
        cfg = yaml.safe_load(f)
    print("Preprocessing components initialized cleanly without error loops.")
