import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from ucimlrepo import fetch_ucirepo
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score
iris = fetch_ucirepo(id=53) 
X = iris.data.features 
y = iris.data.targets 
feature_names = list(iris.metadata.keys())
target_names = iris.metadata['target_names'] if 'target_names' in iris.metadata else None
df = pd.DataFrame(X, columns=feature_names)
df['species'] = y
# Visualization : 
plt.figure(figsize=(12, 6))
plt.subplot(1, 2, 1)
sns.scatterplot(x=feature_names[0], y=feature_names[1], data=df, hue='species', palette='inferno', s=60)
plt.title('Sepal Width vs Sepal Length')
plt.subplot(1, 2, 2)
sns.scatterplot(x=feature_names[2], y=feature_names[3], data=df, hue='species', palette='magma', s=60)
plt.title('Petal Width vs Petal Length')
plt.tight_layout()
plt.show()
x_train, x_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=0)
# Logistic Regression :
lr = LogisticRegression()
lr.fit(x_train, y_train)
# evaluation : 
train_accuracy = lr.score(x_train, y_train)
test_accuracy = lr.score(x_test, y_test)
print("Training Accuracy:", train_accuracy)
print("Test Accuracy:", test_accuracy)
# Predictions : 
predictions = lr.predict(x_test)
accuracy = accuracy_score(y_test, predictions)
print("Accuracy:", accuracy)
