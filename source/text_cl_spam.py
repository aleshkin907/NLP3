import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.naive_bayes import MultinomialNB
from sklearn.metrics import classification_report, confusion_matrix

# Загрузка данных
data = pd.read_csv('../assets/spam.csv', encoding='latin-1')[['v1', 'v2']]
data.columns = ['label', 'text']
X = data['text']  # Колонка с текстами
y = data['label']  # Колонка с метками (spam / ham)

# Предобработка (пример минимальной предобработки)
X = X.str.lower()  # Привести к нижнему регистру

# Векторизация
vectorizer = TfidfVectorizer()
X_vectorized = vectorizer.fit_transform(X)

# Разделение на обучающую и тестовую выборки
X_train, X_test, y_train, y_test = train_test_split(X_vectorized, y, test_size=0.2, random_state=42)

# Обучение модели
model = MultinomialNB()
model.fit(X_train, y_train)

# Предсказание
y_pred = model.predict(X_test)

# Оценка
print(confusion_matrix(y_test, y_pred))
print(classification_report(y_test, y_pred))
