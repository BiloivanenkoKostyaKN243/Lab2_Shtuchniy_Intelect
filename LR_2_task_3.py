import numpy as np
from pandas import read_csv
from pandas.plotting import scatter_matrix
from matplotlib import pyplot
from sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split, cross_val_score, StratifiedKFold
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
from sklearn.naive_bayes import GaussianNB
from sklearn.svm import SVC

iris_dataset = load_iris()

print("Ключі iris_dataset:")
print(iris_dataset.keys())
print("Опис набору даних:")
print(iris_dataset["DESCR"])
print("Назви класів:", iris_dataset["target_names"])
print("Назви ознак:", iris_dataset["feature_names"])
print("Тип масиву data:", type(iris_dataset["data"]))
print("Форма масиву data:", iris_dataset["data"].shape)
print("Перші п'ять рядків даних:")
print(iris_dataset["data"][:5])
print("Тип масиву target:", type(iris_dataset["target"]))
print("Відповіді:")
print(iris_dataset["target"])

url = "https://raw.githubusercontent.com/jbrownlee/Datasets/master/iris.csv"
column_names = ["sepal-length", "sepal-width", "petal-length", "petal-width", "class"]
dataset = read_csv(url, names=column_names)

print("Розмір датасету:")
print(dataset.shape)
print("Перші 20 рядків:")
print(dataset.head(20))
print("Статистичний опис:")
print(dataset.describe())
print("Розподіл за класами:")
print(dataset.groupby("class").size())

dataset.plot(kind="box", subplots=True, layout=(2, 2), sharex=False, sharey=False)
pyplot.savefig("iris_boxplot.png")
pyplot.close()

dataset.hist()
pyplot.savefig("iris_histogram.png")
pyplot.close()

scatter_matrix(dataset)
pyplot.savefig("iris_scatter_matrix.png")
pyplot.close()

array = dataset.values
X = array[:, 0:4]
y = array[:, 4]

X_train, X_validation, y_train, y_validation = train_test_split(
    X, y, test_size=0.20, random_state=1
)

models = [
    ("LR", LogisticRegression(solver="liblinear", multi_class="ovr")),
    ("LDA", LinearDiscriminantAnalysis()),
    ("KNN", KNeighborsClassifier()),
    ("CART", DecisionTreeClassifier()),
    ("NB", GaussianNB()),
    ("SVM", SVC(gamma="auto"))
]

results = []
names = []

print("Порівняння моделей:")
for model_name, model in models:
    kfold = StratifiedKFold(n_splits=10, random_state=1, shuffle=True)
    cv_results = cross_val_score(model, X_train, y_train, cv=kfold, scoring="accuracy")
    results.append(cv_results)
    names.append(model_name)
    print(model_name + ":", round(cv_results.mean(), 4), "стандартне відхилення:", round(cv_results.std(), 4))

pyplot.boxplot(results, labels=names)
pyplot.title("Порівняння алгоритмів")
pyplot.savefig("iris_algorithm_comparison.png")
pyplot.close()

model = SVC(gamma="auto")
model.fit(X_train, y_train)
predictions = model.predict(X_validation)

print("Акуратність на контрольній вибірці:")
print(accuracy_score(y_validation, predictions))
print("Матриця помилок:")
print(confusion_matrix(y_validation, predictions))
print("Звіт класифікації:")
print(classification_report(y_validation, predictions))

new_flower = np.array([[5, 2.9, 1, 0.2]])
new_prediction = model.predict(new_flower)

print("Форма масиву нової квітки:", new_flower.shape)
print("Прогноз для нової квітки:", new_prediction[0])
