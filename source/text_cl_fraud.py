import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report
import re

# 1. Сбор данных
data = pd.read_csv('../assets/spam.csv', encoding='latin-1')[['v1', 'v2']]
data.columns = ['label', 'text']
texts = data['text']  # Колонка с текстами
labels = data['label']  # Колонка с метками (spam / ham)

# 2. Предобработка данных
# В данном примере пропущен процесс очистки, но его стоит сделать перед векторизацией
tfidf_vectorizer = TfidfVectorizer()
X = tfidf_vectorizer.fit_transform(texts)
y = labels

X = X.str.lower()
X = re.sub(r'\W+', ' ', X)

# 3. Разделение данных
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# 4. Обучение модели
model = RandomForestClassifier()
model.fit(X_train, y_train)

# 5. Оценка модели
y_pred = model.predict(X_test)
print(classification_report(y_test, y_pred))