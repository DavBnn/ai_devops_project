import numpy as np
from model import create_model, train_model, evaluate_model


def test_model_training():
    # Crea dati finti per test
    X_train = np.random.rand(100, 784)
    y_train = np.random.randint(0, 10, 100)

    model = create_model()
    model = train_model(model, X_train, y_train)

    assert model is not None
    print("test_model_training passato")


def test_model_accuracy():
    X = np.random.rand(100, 784)
    y = np.random.randint(0, 10, 100)

    model = create_model()
    model = train_model(model, X, y)

    acc = evaluate_model(model, X, y)
    assert 0 <= acc <= 1
    print("test_model_accuracy passato")


if __name__ == "__main__":
    test_model_training()
    test_model_accuracy()
