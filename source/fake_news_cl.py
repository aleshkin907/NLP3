import pandas as pd
import re
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import classification_report
from tqdm import tqdm
import nltk


nltk.download('stopwords')
nltk.download('wordnet')

def preprocess_text(text):
    text = re.sub(r'\W+', ' ', str(text))  # Удаление специальных символов
    text = text.lower()  # Приведение к нижнему регистру
    words = text.split()
    lemmatizer = WordNetLemmatizer()
    words = [lemmatizer.lemmatize(word) for word in words if word not in stopwords.words('english')]
    return ' '.join(words)

# Загрузка данных (по 100 строк из каждого файла)
true_news = pd.read_csv('../assets/True.csv', nrows=100)
false_news = pd.read_csv('../assets/Fake.csv', nrows=100)

# Добавление меток и объединение
true_news['label'] = 'real'
false_news['label'] = 'fake'
data = pd.concat([true_news, false_news], axis=0)

# Предобработка текста
data['cleaned_text'] = data['text'].apply(preprocess_text)

# Разделение данных
X = data['cleaned_text']
y = data['label']
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Векторизация
vectorizer = TfidfVectorizer()
X_train_vectors = vectorizer.fit_transform(X_train)
X_test_vectors = vectorizer.transform(X_test)

# Обучение модели с выводом прогресса через verbose
model = LogisticRegression(max_iter=1000, verbose=1)  # verbose=1 для встроенного прогресса
model.fit(X_train_vectors, y_train)

# Оценка
y_pred = model.predict(X_test_vectors)
print(classification_report(y_test, y_pred))