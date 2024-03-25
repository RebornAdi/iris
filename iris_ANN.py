import numpy as np
import pandas as pd
from ucimlrepo import fetch_ucirepo
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense
from tensorflow.keras.utils import to_categorical
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
import matplotlib.pyplot as plt
iris = fetch_ucirepo(id=53)
X = iris.data.features
y = iris.data.targets
X = np.array(X)
y = np.array(y)
print("Shape of X:", X.shape)
print("Shape of y:", y.shape)
# Spliting
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)
y_train_categorical = to_categorical(y_train)
y_test_categorical = to_categorical(y_test)
# Building of model : 
model = Sequential([
    Dense(10, activation='relu', input_shape=(X_train_scaled.shape[1],)),
    Dense(10, activation='relu'),
    Dense(3, activation='softmax')
])
# Compiling : 
model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])
# Training : 
model.fit(X_train_scaled, y_train_categorical, epochs=100, batch_size=5, verbose=0)
# Evaluation : 
print("\nArtificial Neural Network:")
test_loss, test_accuracy = model.evaluate(X_test_scaled, y_test_categorical, verbose=0)
print("Accuracy:", test_accuracy)
# Prediction : 
y_pred_nn = np.argmax(model.predict(X_test_scaled), axis=-1)
# Classification : 
print("Classification Report:")
print(classification_report(y_test, y_pred_nn))
print("Confusion Matrix:")
print(confusion_matrix(y_test, y_pred_nn))
# Plotting :
plt.figure(figsize=(8, 6))
cm = confusion_matrix(y_test, y_pred_nn)
plt.imshow(cm, interpolation='nearest', cmap=plt.cm.Reds)
plt.title('Confusion Matrix for Neural Network')
plt.colorbar()
tick_marks = np.arange(len(iris.metadata['target_names']))
plt.xticks(tick_marks, iris.metadata['target_names'], rotation=45)
plt.yticks(tick_marks, iris.metadata['target_names'])
plt.xlabel('Predicted Label')
plt.ylabel('True Label')
plt.show()
