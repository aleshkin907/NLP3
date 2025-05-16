import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report
import re

# 1. Загрузка данных
data = pd.read_csv('../assets/spam.csv', encoding='latin-1')[['v1', 'v2']]
data.columns = ['label', 'text']

# 2. Предобработка текста)
def preprocess_text(text):
    text = text.lower()  # приведение к нижнему регистру
    text = re.sub(r'\W+', ' ', text)  # удаление не-алфавитных символов
    return text

# Применяем предобработку ко всем текстам
data['text_clean'] = data['text'].apply(preprocess_text)

# 3. Векторизация текста
tfidf_vectorizer = TfidfVectorizer()
X = tfidf_vectorizer.fit_transform(data['text_clean'])  # Векторизуем очищенный текст
y = data['label']

# 4. Разделение данных
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# 5. Обучение модели
model = RandomForestClassifier()
model.fit(X_train, y_train)

# 6. Оценка модели
y_pred = model.predict(X_test)
print(classification_report(y_test, y_pred))