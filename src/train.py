import os
import matplotlib.pyplot as plt
import numpy as np
from PIL import Image

DATASET_PATH = "dataset/notMNIST_small"


def load_images(path, max_images_per_class=100):
    images = []
    labels = []
    classes = sorted(os.listdir(path))
    for label, cls in enumerate(classes):
        cls_folder = os.path.join(path, cls)
        if not os.path.isdir(cls_folder):
            continue
        count = 0
        for fname in os.listdir(cls_folder):
            if count >= max_images_per_class:
                break
            try:
                img_path = os.path.join(cls_folder, fname)
                img = Image.open(img_path).convert("L").resize((28, 28))
                images.append(np.array(img))
                labels.append(label)
                count += 1
            except:
                continue
    return np.array(images), np.array(labels), classes


if __name__ == "__main__":
    X, y, class_names = load_images(DATASET_PATH)
    print("Immagini caricate:", X.shape)
    print("Classi:", class_names)

    # Mostra 10 immagini
    fig, axes = plt.subplots(1, 10, figsize=(12, 2))
    for i in range(10):
        axes[i].imshow(X[i], cmap='gray')
        axes[i].axis('off')
        axes[i].set_title(class_names[y[i]])
    plt.tight_layout()
    plt.show()

from sklearn.model_selection import train_test_split
from model import create_model, train_model, evaluate_model

# Flatten immagini e normalizza
X = X.reshape((X.shape[0], -1)) / 255.0

# Split train/test
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Crea e addestra il modello
model = create_model()
model = train_model(model, X_train, y_train)

# Valuta il modello
acc = evaluate_model(model, X_test, y_test)
print(f"Accuratezza sul test set: {acc:.2f}")
