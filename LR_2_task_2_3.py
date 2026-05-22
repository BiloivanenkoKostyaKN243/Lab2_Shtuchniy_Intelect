import os
import numpy as np
from sklearn.preprocessing import LabelEncoder


def find_income_file():
    file_names = ["income_data.txt", "income data.txt", "adult.data"]
    for file_name in file_names:
        if os.path.exists(file_name):
            return file_name
    raise FileNotFoundError("Файл income_data.txt не знайдено. Поклади його в одну папку з програмою.")


def is_number(value):
    value = value.strip()
    if value.startswith("-"):
        value = value[1:]
    return value.isdigit()


def load_income_data(max_per_class=1000):
    input_file = find_income_file()
    rows = []
    first_class_count = 0
    second_class_count = 0

    with open(input_file, "r", encoding="utf-8") as file:
        for line in file:
            if first_class_count >= max_per_class and second_class_count >= max_per_class:
                break
            if "?" in line:
                continue

            row = line.strip().rstrip(".").split(", ")
            if len(row) != 15:
                continue

            if row[-1] == "<=50K" and first_class_count < max_per_class:
                rows.append(row)
                first_class_count += 1
            elif row[-1] == ">50K" and second_class_count < max_per_class:
                rows.append(row)
                second_class_count += 1

    if not rows:
        raise ValueError("Не вдалося прочитати дані з income_data.txt.")

    rows = np.array(rows)
    encoded_data = np.empty(rows.shape)
    encoders = {}

    for column_index in range(rows.shape[1]):
        column_values = rows[:, column_index]
        if is_number(column_values[0]):
            encoded_data[:, column_index] = column_values.astype(int)
        else:
            encoder = LabelEncoder()
            encoded_data[:, column_index] = encoder.fit_transform(column_values)
            encoders[column_index] = encoder

    X = encoded_data[:, :-1].astype(float)
    y = encoded_data[:, -1].astype(int)
    return X, y, encoders

from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
from sklearn.model_selection import train_test_split
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC

X, y, encoders = load_income_data(max_per_class=800)

print("Дані завантажено. Кількість рядків:", len(X))
print("Для швидкого запуску використано 800 прикладів на кожен клас.")

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=5)

classifier = make_pipeline(
    StandardScaler(),
    SVC(kernel="sigmoid",  cache_size=500, max_iter=5000)
)

print("Навчання моделі...")
classifier.fit(X_train, y_train)

predictions = classifier.predict(X_test)

accuracy = accuracy_score(y_test, predictions)
precision = precision_score(y_test, predictions, average="weighted", zero_division=0)
recall = recall_score(y_test, predictions, average="weighted", zero_division=0)
f1 = f1_score(y_test, predictions, average="weighted", zero_division=0)

print("Результати SVM з сигмоїдальним ядром")
print("Акуратність:", round(accuracy * 100, 2), "%")
print("Точність:", round(precision * 100, 2), "%")
print("Повнота:", round(recall * 100, 2), "%")
print("F1-міра:", round(f1 * 100, 2), "%")
