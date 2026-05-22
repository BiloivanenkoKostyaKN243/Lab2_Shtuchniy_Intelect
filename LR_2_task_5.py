import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn import metrics
from sklearn.datasets import load_iris
from sklearn.linear_model import RidgeClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import confusion_matrix

iris = load_iris()
X = iris.data
y = iris.target

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.3, random_state=0
)

classifier = RidgeClassifier(tol=1e-2, solver="sag")
classifier.fit(X_train, y_train)

predictions = classifier.predict(X_test)

accuracy = metrics.accuracy_score(y_test, predictions)
precision = metrics.precision_score(y_test, predictions, average="weighted")
recall = metrics.recall_score(y_test, predictions, average="weighted")
f1 = metrics.f1_score(y_test, predictions, average="weighted")
kappa = metrics.cohen_kappa_score(y_test, predictions)
matthews = metrics.matthews_corrcoef(y_test, predictions)

print("Результати класифікатора Ridge")
print("Акуратність:", np.round(accuracy, 4))
print("Точність:", np.round(precision, 4))
print("Повнота:", np.round(recall, 4))
print("F1-міра:", np.round(f1, 4))
print("Коефіцієнт Коена Каппа:", np.round(kappa, 4))
print("Коефіцієнт кореляції Метьюза:", np.round(matthews, 4))
print("Звіт класифікації:")
print(metrics.classification_report(y_test, predictions, target_names=iris.target_names))

matrix = confusion_matrix(y_test, predictions)

sns.set()
sns.heatmap(matrix.T, square=True, annot=True, fmt="d", cbar=False)
plt.xlabel("Справжній клас")
plt.ylabel("Передбачений клас")
plt.title("Матриця помилок RidgeClassifier")
plt.savefig("Confusion.jpg")
plt.close()
