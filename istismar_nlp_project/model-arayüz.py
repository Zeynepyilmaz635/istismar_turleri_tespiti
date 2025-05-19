# -*- coding: utf-8 -*-
"""
Created on Sun May 18 14:13:17 2025

@author: pc
"""

import pandas as pd
import re
import nltk
import pickle
import numpy as np
from nltk.corpus import stopwords
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split
import tensorflow as tf
from tensorflow.keras.preprocessing.text import Tokenizer
from tensorflow.keras.preprocessing.sequence import pad_sequences

# nltk yi indirme işlemi 
nltk.download('stopwords')

# stopwords seti
stop_words = set(stopwords.words('turkish'))

# ön işleme
def clean_text(text):
    text = str(text).lower()
    text = re.sub(r'[^\w\sçğıöşü]', '', text)  # türkçe karakterleri korumak için
    words = text.split()
    filtered_words = [w for w in words if w not in stop_words]
    return " ".join(filtered_words)

# Veri seti
df = pd.read_csv("istismar_turleri_veri_seti_2.csv")  

# datasetindeki her bir satırı temzle
df['cleaned_text'] = df['text'].apply(clean_text)

# Etiketleri encode et
le = LabelEncoder()
df['label_enc'] = le.fit_transform(df['label'])

# Tokenizer 
max_words = 7000  
max_len = 50


tokenizer = Tokenizer(num_words=max_words)
tokenizer.fit_on_texts(df['cleaned_text'])

sequences = tokenizer.texts_to_sequences(df['cleaned_text'])
X = pad_sequences(sequences, maxlen=max_len)

y = tf.keras.utils.to_categorical(df['label_enc'])

# Veri setini train test olark böl 
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Modeli oluştur -- LSTM tabanlı
model = tf.keras.Sequential([
    tf.keras.layers.Embedding(input_dim=max_words, output_dim=128, input_length=max_len),
    tf.keras.layers.SpatialDropout1D(0.3),
    tf.keras.layers.LSTM(64, return_sequences=True),
    tf.keras.layers.LSTM(32),
    tf.keras.layers.Dense(32, activation='relu'),
    tf.keras.layers.Dropout(0.3),
    tf.keras.layers.Dense(y.shape[1], activation='softmax')
])

model.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['accuracy'])
model.summary()

# modeli burda fit ediyoruz öğrenme sağlanıyor
history = model.fit(X_train, y_train, epochs=8, batch_size=64, validation_split=0.1)

# doğruluk
loss, accuracy = model.evaluate(X_test, y_test)
print(f"Test Doğruluğu: %{accuracy*100:.2f}")

# Modeli, tokenizer'ı ve label encoder'ı kaydet
model.save("my_istismar_model.h5")

with open("my_tokenizer.pickle", "wb") as handle:
    pickle.dump(tokenizer, handle)

with open("my_label_encoder.pickle", "wb") as handle:
    pickle.dump(le, handle)









history = model.fit(X_train, y_train, epochs=10, batch_size=64, validation_split=0.1)

# Eğitim sonuçlarını görelm
import matplotlib.pyplot as plt

plt.figure(figsize=(12,5))

plt.subplot(1,2,1)
plt.plot(history.history['loss'], label='Eğitim Kaybı')
plt.plot(history.history['val_loss'], label='Doğrulama Kaybı')
plt.xlabel('Epoch')
plt.ylabel('Kayıp')
plt.legend()
plt.title('Kayıp Grafiği')

plt.subplot(1,2,2)
plt.plot(history.history['accuracy'], label='Eğitim Doğruluğu')
plt.plot(history.history['val_accuracy'], label='Doğrulama Doğruluğu')
plt.xlabel('Epoch')
plt.ylabel('Doğruluk')
plt.legend()
plt.title('Doğruluk Grafiği')

plt.show()


















import matplotlib.pyplot as plt

label_counts = df['label'].value_counts()

plt.figure(figsize=(8,6))
label_counts.plot(kind='bar', color='skyblue')
plt.title('Sınıf Dağılımı')
plt.xlabel('Sınıf')
plt.ylabel('Örnek Sayısı')
plt.show()






from collections import Counter
import nltk

all_words = ' '.join(df['cleaned_text']).split()
word_counts = Counter(all_words)
common_words = word_counts.most_common(20)

words = [w[0] for w in common_words]
counts = [w[1] for w in common_words]

plt.figure(figsize=(10,6))
plt.bar(words, counts, color='coral')
plt.title('En Sık Kullanılan 20 Kelime')
plt.xticks(rotation=45)
plt.show()







import string

all_text = ''.join(df['cleaned_text'])
letters = list(string.ascii_lowercase + "çğıöşü")

letter_counts = {letter: all_text.count(letter) for letter in letters}

plt.figure(figsize=(10,6))
plt.bar(letter_counts.keys(), letter_counts.values(), color='orange')
plt.title('Harf Frekans Dağılımı')
plt.xlabel('Harf')
plt.ylabel('Frekans')
plt.show()











from sklearn.feature_extraction.text import TfidfVectorizer
import numpy as np

tfidf = TfidfVectorizer(max_features=20)
X_tfidf = tfidf.fit_transform(df['cleaned_text'])

features = tfidf.get_feature_names_out()
scores = np.asarray(X_tfidf.mean(axis=0)).ravel()

plt.figure(figsize=(10,6))
plt.bar(features, scores, color='darkcyan')
plt.title('En Yüksek Ortalama TF-IDF Skoruna Sahip Kelimeler')
plt.xticks(rotation=45)
plt.show()






"""
BURDA ARAYÜZ TASARIMINA BAŞLAYALM
"""
import sys
import re
import pickle
import numpy as np
from PyQt5 import QtWidgets
import tensorflow as tf
from tensorflow.keras.preprocessing.sequence import pad_sequences
import nltk
from nltk.corpus import stopwords

nltk.download('stopwords')
stop_words = set(stopwords.words('turkish'))

class IstismarTahminApp(QtWidgets.QWidget):
    def __init__(self):
        super().__init__()
        self.init_ui()
        self.load_model()

    def init_ui(self):
        self.setWindowTitle("İstismar Türü Tahmin")
        self.resize(500, 400)
        layout = QtWidgets.QVBoxLayout()

        self.textbox = QtWidgets.QTextEdit()
        self.textbox.setPlaceholderText("Buraya metni girin ve noktalama işaretleri ile cümleyi ayırmaya özen gösterin . Her cümle ayrı satırda tahmin edilecektir...")
        layout.addWidget(self.textbox)

        self.predict_btn = QtWidgets.QPushButton("Tahmin Et")
        self.predict_btn.clicked.connect(self.predict)
        layout.addWidget(self.predict_btn)

        self.result_area = QtWidgets.QTextEdit()
        self.result_area.setReadOnly(True)
        layout.addWidget(self.result_area)

        self.setLayout(layout)

    def load_model(self):
        self.model = tf.keras.models.load_model("my_istismar_model.h5")
        with open("my_tokenizer.pickle", "rb") as f:
            self.tokenizer = pickle.load(f)
        with open("my_label_encoder.pickle", "rb") as f:
            self.label_encoder = pickle.load(f)

        self.max_len = 50

    def clean_text(self, text):
        text = text.lower()
        text = re.sub(r'[^\w\sçğıöşü]', '', text)
        words = text.split()
        filtered_words = [w for w in words if w not in stop_words]
        return " ".join(filtered_words)

    def predict(self):
        raw_text = self.textbox.toPlainText().strip()
        if not raw_text:
            self.result_area.setPlainText("Lütfen metin girin.")
            return

        # Cümleleri ayır (Nokta, ünlem, soru işareti, yeni satır vb ile)
        sentences = re.split(r'[.!?]\s*|\n+', raw_text)
        sentences = [s.strip() for s in sentences if s.strip()]

        results = []
        for cümle in sentences:
            cleaned = self.clean_text(cümle)
            seq = self.tokenizer.texts_to_sequences([cleaned])
            padded = pad_sequences(seq, maxlen=self.max_len)
            pred = self.model.predict(padded)
            class_idx = np.argmax(pred)
            label = self.label_encoder.inverse_transform([class_idx])[0]
            confidence = np.max(pred)*100
            results.append(f"Cümle: \"{cümle}\"\nTahmin: {label} (Güven: %{confidence:.2f})\n")

        self.result_area.setPlainText("\n".join(results))


if __name__ == "__main__":
    app = QtWidgets.QApplication(sys.argv)
    pencere = IstismarTahminApp()
    pencere.show()
    sys.exit(app.exec_())


