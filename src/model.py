from sklearn.neural_network import MLPClassifier
from sklearn.metrics import accuracy_score


def create_model():
    return MLPClassifier(hidden_layer_sizes=(64,), max_iter=20, random_state=42)


def train_model(model, X_train, y_train):
    model.fit(X_train, y_train)
    return model


def evaluate_model(model, X_test, y_test):
    y_pred = model.predict(X_test)
    acc = accuracy_score(y_test, y_pred)
    return acc
