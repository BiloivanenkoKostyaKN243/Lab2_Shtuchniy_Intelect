import numpy as np
from sklearn import preprocessing
from sklearn.model_selection import train_test_split, cross_val_score, StratifiedKFold
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, classification_report
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
from sklearn.naive_bayes import GaussianNB
from sklearn.svm import SVC

input_file = "income_data.txt"

data_rows = []
class_one_count = 0
class_two_count = 0
max_datapoints = 1200

with open(input_file, "r", encoding="utf-8") as file:
    for line in file:
        if class_one_count >= max_datapoints and class_two_count >= max_datapoints:
            break

        if "?" in line:
            continue

        row = line.strip().split(", ")

        if row[-1] == "<=50K" and class_one_count < max_datapoints:
            data_rows.append(row)
            class_one_count += 1

        elif row[-1] == ">50K" and class_two_count < max_datapoints:
            data_rows.append(row)
            class_two_count += 1

data_rows = np.array(data_rows)

encoded_data = np.empty(data_rows.shape)

for column_index in range(data_rows.shape[1]):
    column_values = data_rows[:, column_index]

    if column_values[0].replace("-", "").isdigit():
        encoded_data[:, column_index] = column_values.astype(int)
    else:
        encoder = preprocessing.LabelEncoder()
        encoded_data[:, column_index] = encoder.fit_transform(column_values)

X = encoded_data[:, :-1].astype(int)
y = encoded_data[:, -1].astype(int)

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=5
)

models = [
    ("LR", LogisticRegression(solver="liblinear", max_iter=2000)),
    ("LDA", LinearDiscriminantAnalysis()),
    ("KNN", KNeighborsClassifier()),
    ("CART", DecisionTreeClassifier()),
    ("NB", GaussianNB()),
    ("SVM", SVC(gamma="auto", cache_size=500, max_iter=5000))
]

best_name = ""
best_f1 = 0

print("Порівняння моделей для набору income_data.txt")
print("Для швидшого запуску використано 1200 прикладів на кожен клас.")

for model_name, model in models:
    print("Навчання моделі", model_name + "...")
    model.fit(X_train, y_train)
    predictions = model.predict(X_test)

    accuracy = accuracy_score(y_test, predictions)
    precision = precision_score(y_test, predictions, average="weighted")
    recall = recall_score(y_test, predictions, average="weighted")
    f1 = f1_score(y_test, predictions, average="weighted")

    print()
    print("Модель:", model_name)
    print("Акуратність:", round(accuracy * 100, 2), "%")
    print("Точність:", round(precision * 100, 2), "%")
    print("Повнота:", round(recall * 100, 2), "%")
    print("F1-міра:", round(f1 * 100, 2), "%")

    if f1 > best_f1:
        best_f1 = f1
        best_name = model_name

print()
print("Найкраща модель за F1-мірою:", best_name)
print("Найкраще значення F1:", round(best_f1 * 100, 2), "%")
