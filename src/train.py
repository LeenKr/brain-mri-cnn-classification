import tensorflow as tf
import matplotlib.pyplot as plt
import numpy as np
from sklearn.metrics import confusion_matrix
import seaborn as sns
from datetime import datetime
import os

# ======================
# Constants
# ======================
IMG_SIZE = (256, 256)
BATCH_SIZE = 32
SEED = 123
EPOCHS = 15

# ✅ Change only this if you want higher glioma recall (fewer FN)
THRESHOLD = 0.30

# ✅ Helps reduce FN by making the model care more about glioma (class 1)
CLASS_WEIGHT = {0: 1.0, 1: 3.0}
 # 0=healthy, 1=glioma


# ======================
# Data Loader
# ======================
def load_data(data_dir="dataset"):
    train_ds = tf.keras.utils.image_dataset_from_directory(
        data_dir,
        labels="inferred",
        label_mode="binary",
        class_names=["healthy", "glioma"],
        image_size=IMG_SIZE,
        batch_size=BATCH_SIZE,
        shuffle=True,
        seed=SEED,
        validation_split=0.20,
        subset="training"
    )

    test_ds = tf.keras.utils.image_dataset_from_directory(
        data_dir,
        labels="inferred",
        label_mode="binary",
        class_names=["healthy", "glioma"],
        image_size=IMG_SIZE,
        batch_size=BATCH_SIZE,
        shuffle=True,   # ✅ FIX: must match train shuffle for correct split
        seed=SEED,
        validation_split=0.20,
        subset="validation"
    )

    AUTOTUNE = tf.data.AUTOTUNE
    train_ds = train_ds.prefetch(buffer_size=AUTOTUNE)
    test_ds  = test_ds.prefetch(buffer_size=AUTOTUNE)
    return train_ds, test_ds


def count_labels(ds, name):
    c0, c1 = 0, 0
    for _, y in ds:
        y = y.numpy().reshape(-1).astype(int)
        c0 += int((y == 0).sum())
        c1 += int((y == 1).sum())
    print(f"{name} -> healthy(0): {c0}, glioma(1): {c1}")


# ======================
# Plot Training History
# ======================
def plot_training_history(history, save_path="results/training_history.png"):
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5))

    ax1.plot(history.history['accuracy'], label='Training Accuracy', marker='o')
    ax1.plot(history.history['val_accuracy'], label='Validation Accuracy', marker='s')
    ax1.set_title('Model Accuracy', fontsize=14, fontweight='bold')
    ax1.set_xlabel('Epoch', fontsize=12)
    ax1.set_ylabel('Accuracy', fontsize=12)
    ax1.legend(loc='lower right')
    ax1.grid(True, alpha=0.3)

    ax2.plot(history.history['loss'], label='Training Loss', marker='o')
    ax2.plot(history.history['val_loss'], label='Validation Loss', marker='s')
    ax2.set_title('Model Loss', fontsize=14, fontweight='bold')
    ax2.set_xlabel('Epoch', fontsize=12)
    ax2.set_ylabel('Loss', fontsize=12)
    ax2.legend(loc='upper right')
    ax2.grid(True, alpha=0.3)

    plt.tight_layout()
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    print(f"Training history plot saved to {save_path}")
    plt.close()


# ======================
# Confusion Matrix Plot
# ======================
def plot_confusion_matrix(y_true, y_pred, save_path="results/confusion_matrix.png"):
    # ✅ FIX: always force 2x2 matrix
    cm = confusion_matrix(y_true, y_pred, labels=[0, 1])

    plt.figure(figsize=(8, 6))
    sns.heatmap(
        cm, annot=True, fmt='d', cmap='Blues',
        xticklabels=['Healthy', 'Glioma'],
        yticklabels=['Healthy', 'Glioma'],
        cbar_kws={'label': 'Count'}
    )

    plt.title('Confusion Matrix', fontsize=14, fontweight='bold')
    plt.xlabel('Predicted Label', fontsize=12)
    plt.ylabel('True Label', fontsize=12)

    plt.text(
        0.5, -0.15,
        f'TN: {cm[0,0]}  |  FP: {cm[0,1]}  |  FN: {cm[1,0]}  |  TP: {cm[1,1]}',
        ha='center', transform=plt.gca().transAxes, fontsize=10
    )

    plt.tight_layout()
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    print(f"Confusion matrix saved to {save_path}")
    plt.close()

    return cm


# ======================
# Main
# ======================
os.makedirs("results", exist_ok=True)

train_ds, test_ds = load_data("dataset")
print(f"Loaded {len(train_ds)} batches for training and {len(test_ds)} batches for testing.")
count_labels(train_ds, "TRAIN")
count_labels(test_ds, "TEST")

model = tf.keras.Sequential([
    tf.keras.layers.InputLayer(input_shape=IMG_SIZE + (3,)),
    tf.keras.layers.Rescaling(1.0/255),
    tf.keras.layers.RandomFlip("horizontal"),
    tf.keras.layers.RandomRotation(0.1),
    tf.keras.layers.RandomZoom(0.1),
    tf.keras.layers.Conv2D(32, (3,3), activation="relu"),
    tf.keras.layers.MaxPooling2D(pool_size=(2,2)),
    tf.keras.layers.Conv2D(64, (3,3), activation="relu"),
    tf.keras.layers.MaxPooling2D(pool_size=(2,2)),
    tf.keras.layers.Conv2D(128, (3,3), activation="relu"),
    tf.keras.layers.GlobalAveragePooling2D(),
    tf.keras.layers.Dense(64, activation="relu"),
    tf.keras.layers.Dropout(0.5),
    tf.keras.layers.Dense(1, activation="sigmoid")
])

model.compile(
    optimizer=tf.keras.optimizers.Adam(),
    loss=tf.keras.losses.BinaryCrossentropy(),
    metrics=[
        tf.keras.metrics.BinaryAccuracy(name="accuracy"),
        tf.keras.metrics.Precision(name="precision"),
        tf.keras.metrics.Recall(name="recall"),
        tf.keras.metrics.AUC(name="auc")
    ]
)

history = model.fit(
    train_ds,
    validation_data=test_ds,
    epochs=EPOCHS,
    class_weight=CLASS_WEIGHT   # ✅ helps reduce FN by prioritizing glioma
)

model.save("results/brain_tumor_classifier.keras")
print("Training complete. Model saved to results/brain_tumor_classifier.keras")

plot_training_history(history)

print("\nGenerating confusion matrix...")
y_true, y_pred = [], []

for images, labels in test_ds:
    probs = model.predict(images, verbose=0)
    preds = (probs >= THRESHOLD).astype(int).flatten()  # ✅ lower threshold => higher recall
    y_true.append(labels.numpy().astype(int).flatten())
    y_pred.append(preds)

y_true = np.concatenate(y_true)
y_pred = np.concatenate(y_pred)

timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
cm_path = f"results/confusion_matrix_{timestamp}.png"
cm = plot_confusion_matrix(y_true, y_pred, cm_path)
plot_confusion_matrix(y_true, y_pred, "results/confusion_matrix_latest.png")

print("\nConfusion Matrix (True vs Predicted labels):")
print(cm)
print(f"TN: {cm[0,0]} | FP: {cm[0,1]} | FN: {cm[1,0]} | TP: {cm[1,1]}")

tn, fp, fn, tp = cm[0,0], cm[0,1], cm[1,0], cm[1,1]
accuracy = (tp + tn) / (tp + tn + fp + fn)
precision = tp / (tp + fp) if (tp + fp) > 0 else 0
recall = tp / (tp + fn) if (tp + fn) > 0 else 0
f1_score = 2 * (precision * recall) / (precision + recall) if (precision + recall) > 0 else 0

print(f"\nMetrics from Confusion Matrix (threshold={THRESHOLD}):")
print(f"Accuracy:  {accuracy:.4f}")
print(f"Precision: {precision:.4f}")
print(f"Recall:    {recall:.4f}")
print(f"F1-Score:  {f1_score:.4f}")
