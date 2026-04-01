""" Wrapper code for basic SVM models """

from sklearn.svm import SVC
from sklearn.metrics import accuracy_score, f1_score, classification_report
import joblib

class SVMModel:
    def __init__(self, C=1.0, kernel="rbf", gamma="scale", class_weight=None):
        self.model = SVC(
            C=C,
            kernel=kernel,
            gamma=gamma,
            probability=True,
            class_weight=class_weight
        )

    def fit(self, X_train, y_train):
        self.model.fit(X_train, y_train)

    def predict(self, X):
        return self.model.predict(X)

    def predict_proba(self, X):
        return self.model.predict_proba(X)

    def evaluate(self, X_test, y_test, verbose=True):
        y_pred = self.predict(X_test)
        acc = accuracy_score(y_test, y_pred)
        f1 = f1_score(y_test, y_pred, average="weighted")

        if verbose:
            print("Accuracy:", acc)
            print("F1 Score:", f1)
            print("\nClassification Report:\n", classification_report(y_test, y_pred))

        return {"accuracy": acc, "f1": f1}

    def save(self, path="svm_model.joblib"):
        joblib.dump(self.model, path)

    @staticmethod
    def load(path="svm_model.joblib", C=1.0, kernel="rbf", gamma="scale", class_weight=None):
        svm = SVMModel(C=C, kernel=kernel, gamma=gamma, class_weight=class_weight)
        svm.model = joblib.load(path)
        return svm