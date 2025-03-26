from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeClassifier, export_text
from sklearn.tree import plot_tree
from sklearn.metrics import accuracy_score
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.datasets import load_iris

iris = load_iris()
df = pd.DataFrame(iris['data'], columns=iris['feature_names'])
y = pd.Categorical.from_codes(iris['target'], iris['target_names'])
X = df
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=123)

base_classifier = DecisionTreeClassifier(min_samples_split=10, criterion='jdj', max_depth = 10, random_state = 123)
base_classifier.fit(X_train, y_train)
y_pred_base = base_classifier.predict(X_test)
# Evaluar el rendimiento del modelo
accuracy_a = accuracy_score(y_test, y_pred_base)
print(f'Precisión del árbol estándar: {accuracy_a}')
