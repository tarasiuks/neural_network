import numpy as np
import h5py
from sklearn.neural_network import MLPClassifier
from sklearn.metrics import accuracy_score

train_dataset = h5py.File('./data/train_catvnoncat.h5', "r")
test_dataset = h5py.File('./data/test_catvnoncat.h5', "r")

X_train = np.array(train_dataset["train_set_x"][:]) / 255.0
y_train = np.array(train_dataset["train_set_y"][:]).reshape(-1, 1)

X_test = np.array(test_dataset["test_set_x"][:]) / 255.0
y_test = np.array(test_dataset["test_set_y"][:]).reshape(-1, 1)

y_train = y_train.reshape(-1)
y_test = y_test.reshape(-1)

model_1 = MLPClassifier(hidden_layer_sizes=(100,), solver='sgd', activation='logistic', max_iter=300, random_state=42)
model_1.fit(X_train.reshape(X_train.shape[0], -1), y_train)
train_accuracy_1 = accuracy_score(y_train, model_1.predict(X_train.reshape(X_train.shape[0], -1)))
test_accuracy_1 = accuracy_score(y_test, model_1.predict(X_test.reshape(X_test.shape[0], -1)))

model_2 = MLPClassifier(hidden_layer_sizes=(3, 3), solver='sgd', activation='relu', max_iter=300, random_state=42)
model_2.fit(X_train.reshape(X_train.shape[0], -1), y_train)
train_accuracy_2 = accuracy_score(y_train, model_2.predict(X_train.reshape(X_train.shape[0], -1)))
test_accuracy_2 = accuracy_score(y_test, model_2.predict(X_test.reshape(X_test.shape[0], -1)))

model_3 = MLPClassifier(hidden_layer_sizes=(2, 7, 10), solver='sgd', activation='relu', max_iter=300, random_state=42)
model_3.fit(X_train.reshape(X_train.shape[0], -1), y_train)
train_accuracy_3 = accuracy_score(y_train, model_3.predict(X_train.reshape(X_train.shape[0], -1)))
test_accuracy_3 = accuracy_score(y_test, model_3.predict(X_test.reshape(X_test.shape[0], -1)))


print(f"Train accuracy (1 layer): {train_accuracy_1}")
print(f"Test accuracy (1 layer): {test_accuracy_1}")

print(f"Train accuracy (2 layers): {train_accuracy_2}")
print(f"Test accuracy (2 layers): {test_accuracy_2}")

print(f"Train accuracy (3 layers): {train_accuracy_3}")
print(f"Test accuracy (3 layers): {test_accuracy_3}")
