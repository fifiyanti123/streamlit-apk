import re
from Sastrawi.Stemmer.StemmerFactory import StemmerFactory
import streamlit as st
import pandas as pd
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.naive_bayes import MultinomialNB
from sklearn.model_selection import train_test_split
import joblib
import os

# Buat stemmer bahasa Indonesia
factory = StemmerFactory()
stemmer = factory.create_stemmer()

# Kamus kata tidak baku ke kata baku
normalization_dict = {"gk": "tidak", "ga": "tidak", "jelek bgt": "jelek banget", "bgs": "bagus", "tdk": "tidak", "mantul": "mantap betul"}

def normalize_text(text):
    for word, replacement in normalization_dict.items():
        text = re.sub(r'\b{}\b'.format(word), replacement, text)
    return text

def preprocess_text(text):
    text = re.sub(r'[^a-zA-Z\s]', '', text)  # Hilangkan karakter non-alphabet
    text = text.lower()  # Ubah teks ke huruf kecil
    text = normalize_text(text)  # Normalisasi kata
    text = stemmer.stem(text)  # Stemming
    return text

# Daftar kata positif dan negatif
positive_words = ["bagus", "mantap", "keren", "puas", "baik"]
negative_words = ["jelek", "buruk", "tidak", "mengecewakan", "parah"]

def predict_sentiment_with_logic(Comment):
    # Preprocess komentar
    Comment = preprocess_text(Comment)
    if any(word in Comment for word in positive_words):
        return "positif"
    elif any(word in Comment for word in negative_words):
        return "negatif"
    else:
        return predict_sentiment(Comment)  # Default ke model

def predict_sentiment(Comment):
    try:
        model = joblib.load("models/sentiment_model.pkl")
        vectorizer = joblib.load("models/vectorizer.pkl")
    except FileNotFoundError:
        st.error("Model belum dilatih. Harap latih model terlebih dahulu.")
        return None

    comment_vector = vectorizer.transform([Comment])
    sentiment = model.predict(comment_vector)
    return sentiment[0]

# Streamlit App
st.title("Aplikasi Analisis Sentimen Komentar")

# Input komentar
user_input = st.text_area("Masukkan komentar Anda di sini:", height=150)

# Tombol Analisis Sentimen
if st.button("Analisis Sentimen"):
    if user_input.strip():
        sentiment = predict_sentiment_with_logic(user_input)
        if sentiment:
            color = "green" if sentiment == "positif" else "red"
            emoji = "😊" if sentiment == "positif" else "😞"
            st.markdown(f"<h3 style='color:{color};'>{emoji} Sentimen: **{sentiment.capitalize()}**</h3>", unsafe_allow_html=True)
    else:
        st.error("Harap masukkan komentar.")

# Sidebar untuk melatih ulang model
dataset_path = "dataset_sentiment.xlsx"  # Nama file dataset Anda
uploaded_file = st.sidebar.file_uploader("Unggah dataset Excel", type="xlsx")

def train_model_from_dataset(dataset_path):
    if not os.path.exists(dataset_path):
        st.error(f"Dataset tidak ditemukan di {dataset_path}. Harap unggah dataset terlebih dahulu.")
        return

    # Membaca dataset
    df = pd.read_excel(dataset_path)

    # Pastikan kolom 'Comment' dan 'sentiment' ada di dataset
    if 'Comment' not in df.columns or 'sentiment' not in df.columns:
        st.error("Dataset harus memiliki kolom 'Comment' dan 'sentiment'.")
        return

    # Proses vectorizer dan pelatihan model
    vectorizer = CountVectorizer()
    X = vectorizer.fit_transform(df['Comment'])  # Kolom komentar
    y = df['sentiment']  # Kolom label sentimen

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    # Model Naive Bayes
    model = MultinomialNB()
    model.fit(X_train, y_train)

    # Simpan model dan vectorizer
    joblib.dump(model, "models/sentiment_model.pkl")
    joblib.dump(vectorizer, "models/vectorizer.pkl")

if uploaded_file:
    with open(dataset_path, "wb") as f:
        f.write(uploaded_file.read())
    st.sidebar.success("Dataset berhasil diunggah.")

if st.sidebar.button("Latih Ulang Model"):
    with st.spinner('Model sedang dilatih...'):
        train_model_from_dataset(dataset_path)
    st.sidebar.success("Model berhasil dilatih ulang.")

# Fitur tambahan: Menampilkan data training dan evaluasi model
if st.sidebar.button("Tampilkan Data Training"):
    if os.path.exists(dataset_path):
        df = pd.read_excel(dataset_path)
        st.write(df.head())  # Menampilkan 5 baris pertama data

# Visualisasi sentimen dari dataset
if st.sidebar.button("Visualisasi Sentimen Dataset"):
    if os.path.exists(dataset_path):
        try:
            df = pd.read_excel(dataset_path)
            if 'sentiment' not in df.columns:
                st.error("Kolom 'sentiment' tidak ditemukan di dataset. Harap unggah dataset yang sesuai.")
            else:
                sentiment_counts = df['sentiment'].value_counts()
                st.bar_chart(sentiment_counts)
        except Exception as e:
            st.error(f"Gagal membaca dataset: {e}")

