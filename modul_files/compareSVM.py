from sklearn.svm import SVC

svm_model = SVC()
svm_model.fit(X_train.reshape(X_train.shape[0], -1), y_train)

train_accuracy_svm = accuracy_score(y_train, svm_model.predict(X_train.reshape(X_train.shape[0], -1)))
test_accuracy_svm = accuracy_score(y_test, svm_model.predict(X_test.reshape(X_test.shape[0], -1)))

print(f"Train accuracy (SVM): {train_accuracy_svm}")
print(f"Test accuracy (SVM): {test_accuracy_svm}")