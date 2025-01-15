import re
from Sastrawi.Stemmer.StemmerFactory import StemmerFactory

# Buat stemmer bahasa Indonesia
factory = StemmerFactory()
stemmer = factory.create_stemmer()

# Kamus kata tidak baku ke kata baku
normalization_dict = {
    "gk": "tidak", "ga": "tidak", "jelek bgt": "jelek banget", 
    "bgs": "bagus", "tdk": "tidak", "mantul": "mantap betul"
}

def normalize_text(text):
    # Normalisasi kata
    for word, replacement in normalization_dict.items():
        text = re.sub(r'\b{}\b'.format(word), replacement, text)
    return text

def preprocess_text(text):
    # Hilangkan karakter non-alphabet
    text = re.sub(r'[^a-zA-Z\s]', '', text)
    # Ubah teks ke huruf kecil
    text = text.lower()
    # Normalisasi kata
    text = normalize_text(text)
    # Stemming
    text = stemmer.stem(text)
    return text

# Preprocessing dataset
df['Comment'] = df['Comment'].apply(preprocess_text)
