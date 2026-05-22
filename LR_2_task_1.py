import numpy as np
from sklearn import preprocessing
from sklearn.svm import LinearSVC
from sklearn.multiclass import OneVsOneClassifier
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score

input_file = "income_data.txt"

data_rows = []
class_one_count = 0
class_two_count = 0
max_datapoints = 25000

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

label_encoders = []
encoded_data = np.empty(data_rows.shape)

for column_index in range(data_rows.shape[1]):
    column_values = data_rows[:, column_index]

    if column_values[0].replace("-", "").isdigit():
        encoded_data[:, column_index] = column_values.astype(int)
    else:
        encoder = preprocessing.LabelEncoder()
        encoded_data[:, column_index] = encoder.fit_transform(column_values)
        label_encoders.append((column_index, encoder))

X = encoded_data[:, :-1].astype(int)
y = encoded_data[:, -1].astype(int)

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=5
)

classifier = OneVsOneClassifier(LinearSVC(random_state=0, max_iter=20000))
classifier.fit(X_train, y_train)

predictions = classifier.predict(X_test)

accuracy = accuracy_score(y_test, predictions)
precision = precision_score(y_test, predictions, average="weighted")
recall = recall_score(y_test, predictions, average="weighted")
f1 = f1_score(y_test, predictions, average="weighted")
cross_f1 = cross_val_score(classifier, X, y, scoring="f1_weighted", cv=3)

print("Результати класифікації SVM з лінійним ядром")
print("Акуратність:", round(accuracy * 100, 2), "%")
print("Точність:", round(precision * 100, 2), "%")
print("Повнота:", round(recall * 100, 2), "%")
print("F1-міра:", round(f1 * 100, 2), "%")
print("F1-міра крос-валідації:", round(cross_f1.mean() * 100, 2), "%")

input_data = [
    "37", "Private", "215646", "HS-grad", "9", "Never-married",
    "Handlers-cleaners", "Not-in-family", "White", "Male",
    "0", "0", "40", "United-States"
]

encoded_input = np.zeros(len(input_data), dtype=int)
encoder_map = dict(label_encoders)

for column_index, value in enumerate(input_data):
    if value.replace("-", "").isdigit():
        encoded_input[column_index] = int(value)
    else:
        encoded_input[column_index] = encoder_map[column_index].transform([value])[0]

predicted_class = classifier.predict([encoded_input])[0]
target_encoder = encoder_map[data_rows.shape[1] - 1]
predicted_label = target_encoder.inverse_transform([predicted_class])[0]

print("Передбачений клас для тестової точки:", predicted_label)
