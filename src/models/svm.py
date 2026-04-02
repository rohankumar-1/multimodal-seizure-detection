""" Wrapper code for basic SVM models """

from sklearn.svm import SVC
from sklearn.metrics import accuracy_score, f1_score, classification_report, roc_auc_score, precision_score, recall_score
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
        y_prob = self.predict_proba(X_test)[:, 1]
        auc = roc_auc_score(y_test, y_prob)
        acc = accuracy_score(y_test, y_pred)
        f1 = f1_score(y_test, y_pred, average="weighted")
        precision = precision_score(y_test, y_pred, average="weighted")
        recall = recall_score(y_test, y_pred, average="weighted")

        if verbose:
            print(f"AUC: {auc:.4f}")
            print(f"Accuracy: {acc:.4f}")
            print(f"F1 Score: {f1:.4f}")
            print(f"Precision: {precision:.4f}")
            print(f"Recall: {recall:.4f}")

        return {"auc": auc, "accuracy": acc, "f1": f1, "precision": precision, "recall": recall}

    def save(self, path="svm_model.joblib"):
        joblib.dump(self.model, path)

    @staticmethod
    def load(path="svm_model.joblib", C=1.0, kernel="rbf", gamma="scale", class_weight=None):
        svm = SVMModel(C=C, kernel=kernel, gamma=gamma, class_weight=class_weight)
        svm.model = joblib.load(path)
        return svm