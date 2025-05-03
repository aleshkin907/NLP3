import pandas as pd
import re
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import classification_report
from tqdm import tqdm


# Пример функции для предобработки текста
def preprocess_text(text):
    text = re.sub(r'\W+', ' ', text)  # Удаление специальных символов
    text = text.lower()  # Приведение к нижнему регистру
    words = text.split()
    lemmatizer = WordNetLemmatizer()
    words = [lemmatizer.lemmatize(word) for word in words if word not in stopwords.words('english')]

    return ' '.join(words)


# Загрузка и предобработка данных
true_news = pd.read_csv('../assets/True.csv')
false_news = pd.read_csv('../assets/Fake.csv')

# Добавляем метки
true_news['label'] = 'real'
false_news['label'] = 'fake'

# Объединяем датасеты
data = pd.concat([true_news, false_news], axis=0)

# Перемешиваем данные
# data = data.sample(frac=1, random_state=42).reset_index(drop=True)
data['cleaned_text'] = data['text'].apply(preprocess_text)

X = data['cleaned_text']
y = data['label']  # Предполагается, что в датасете есть колонка с метками (fake/real)
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

vectorizer = TfidfVectorizer()
X_train_vectors = vectorizer.fit_transform(X_train)
X_test_vectors = vectorizer.transform(X_test)

model = LogisticRegression(max_iter=1000)
model.fit(X_train_vectors, y_train)

print("sdsdsd")
class ProgressCallback:
    def __init__(self, total_iterations):
        self.pbar = tqdm(total=total_iterations, desc="Итерации обучения")
        self.current_iter = 0

    def __call__(self, coef, intercept, iteration):
        self.pbar.update(iteration - self.current_iter)
        self.current_iter = iteration


model.set_params(verbose=1, callback=ProgressCallback(model.max_iter))
model.fit(X_train_vectors, y_train)

y_pred = model.predict(X_test_vectors)
print(classification_report(y_test, y_pred))
