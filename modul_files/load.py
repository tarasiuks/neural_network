import numpy as np
import h5py

train_dataset = h5py.File('../data/train_catvnoncat.h5', "r")
test_dataset = h5py.File('../data/test_catvnoncat.h5', "r")

X_train = np.array(train_dataset["train_set_x"][:]) / 255.0
y_train = np.array(train_dataset["train_set_y"][:]).reshape(-1, 1)

X_test = np.array(test_dataset["test_set_x"][:]) / 255.0
y_test = np.array(test_dataset["test_set_y"][:]).reshape(-1, 1)

y_train = y_train.reshape(-1)
y_test = y_test.reshape(-1)

print("X_train shape:", X_train.shape)
print("y_train shape:", y_train.shape)
print("X_test shape:", X_test.shape)
print("y_test shape:", y_test.shape)