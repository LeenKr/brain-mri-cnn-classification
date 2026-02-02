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

# Keep SAME as Exp1 for fair comparison
THRESHOLD = 0.30
CLASS_WEIGHT = {0: 1.0, 1: 3.0}  # 0=healthy, 1=glioma

# Experiment naming (avoids overwriting Exp1 outputs)
EXP_NAME = "exp2_regularized"
RESULTS_DIR = os.path.join("results", EXP_NAME)

# Regularization strength
L2 = tf.keras.regularizers.l2(1e-4)

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

    val_ds = tf.keras.utils.image_dataset_from_directory(
        data_dir,
        labels="inferred",
        label_mode="binary",
        class_names=["healthy", "glioma"],
        image_size=IMG_SIZE,
        batch_size=BATCH_SIZE,
        shuffle=True,   # keep consistent with training for split reproducibility
        seed=SEED,
        validation_split=0.20,
        subset="validation"
    )

    AUTOTUNE = tf.data.AUTOTUNE
    train_ds = train_ds.prefetch(buffer_size=AUTOTUNE)
    val_ds   = val_ds.prefetch(buffer_size=AUTOTUNE)
    return train_ds, val_ds


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
def plot_training_history(history, save_path):
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
def plot_confusion_matrix(y_true, y_pred, save_path):
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
# Model Builder (Experiment 2)
# ======================
def build_model():
    inputs = tf.keras.Input(shape=IMG_SIZE + (3,))

    x = tf.keras.layers.Rescaling(1.0 / 255)(inputs)

    # same augmentation as Exp1
    x = tf.keras.layers.RandomFlip("horizontal")(x)
    x = tf.keras.layers.RandomRotation(0.1)(x)
    x = tf.keras.layers.RandomZoom(0.1)(x)

    # ---- Block 1 (32) ----
    x = tf.keras.layers.Conv2D(32, (3, 3), padding="same", use_bias=False,
                               kernel_regularizer=L2)(x)
    x = tf.keras.layers.BatchNormalization()(x)
    x = tf.keras.layers.ReLU()(x)

    x = tf.keras.layers.Conv2D(32, (3, 3), padding="same", use_bias=False,
                               kernel_regularizer=L2)(x)
    x = tf.keras.layers.BatchNormalization()(x)
    x = tf.keras.layers.ReLU()(x)

    x = tf.keras.layers.MaxPooling2D((2, 2))(x)
    x = tf.keras.layers.SpatialDropout2D(0.20)(x)

    # ---- Block 2 (64) ----
    x = tf.keras.layers.Conv2D(64, (3, 3), padding="same", use_bias=False,
                               kernel_regularizer=L2)(x)
    x = tf.keras.layers.BatchNormalization()(x)
    x = tf.keras.layers.ReLU()(x)

    x = tf.keras.layers.Conv2D(64, (3, 3), padding="same", use_bias=False,
                               kernel_regularizer=L2)(x)
    x = tf.keras.layers.BatchNormalization()(x)
    x = tf.keras.layers.ReLU()(x)

    x = tf.keras.layers.MaxPooling2D((2, 2))(x)
    x = tf.keras.layers.SpatialDropout2D(0.25)(x)

    # ---- Block 3 (128) ----
    x = tf.keras.layers.Conv2D(128, (3, 3), padding="same", use_bias=False,
                               kernel_regularizer=L2)(x)
    x = tf.keras.layers.BatchNormalization()(x)
    x = tf.keras.layers.ReLU()(x)

    x = tf.keras.layers.Conv2D(128, (3, 3), padding="same", use_bias=False,
                               kernel_regularizer=L2)(x)
    x = tf.keras.layers.BatchNormalization()(x)
    x = tf.keras.layers.ReLU()(x)

    x = tf.keras.layers.MaxPooling2D((2, 2))(x)
    x = tf.keras.layers.SpatialDropout2D(0.30)(x)

    # Head
    x = tf.keras.layers.GlobalAveragePooling2D()(x)

    x = tf.keras.layers.Dense(128, use_bias=False, kernel_regularizer=L2)(x)
    x = tf.keras.layers.BatchNormalization()(x)
    x = tf.keras.layers.ReLU()(x)

    x = tf.keras.layers.Dropout(0.5)(x)
    outputs = tf.keras.layers.Dense(1, activation="sigmoid")(x)

    model = tf.keras.Model(inputs, outputs, name="cnn_exp2_regularized")
    return model


# ======================
# Main
# ======================
def main():
    os.makedirs(RESULTS_DIR, exist_ok=True)

    train_ds, val_ds = load_data("dataset")
    print(f"Loaded {len(train_ds)} batches for training and {len(val_ds)} batches for validation.")
    count_labels(train_ds, "TRAIN")
    count_labels(val_ds, "VAL")

    model = build_model()

    model.compile(
        optimizer=tf.keras.optimizers.Adam(),
        loss=tf.keras.losses.BinaryCrossentropy(),
        metrics=[
            tf.keras.metrics.BinaryAccuracy(name="accuracy"),
            tf.keras.metrics.Precision(name="precision"),
            tf.keras.metrics.Recall(name="recall"),
            tf.keras.metrics.AUC(name="auc"),
        ]
    )

    history = model.fit(
        train_ds,
        validation_data=val_ds,
        epochs=EPOCHS,
        class_weight=CLASS_WEIGHT
    )

    # Save model
    model_path = os.path.join(RESULTS_DIR, "brain_tumor_classifier_exp2.keras")
    model.save(model_path)
    print(f"Training complete. Model saved to {model_path}")

    # Save training curves
    history_path = os.path.join(RESULTS_DIR, "training_history.png")
    plot_training_history(history, history_path)

    # Confusion matrix (threshold applied)
    print("\nGenerating confusion matrix...")
    y_true, y_pred = [], []

    for images, labels in val_ds:
        probs = model.predict(images, verbose=0)
        preds = (probs >= THRESHOLD).astype(int).flatten()
        y_true.append(labels.numpy().astype(int).flatten())
        y_pred.append(preds)

    y_true = np.concatenate(y_true)
    y_pred = np.concatenate(y_pred)

    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    cm_path = os.path.join(RESULTS_DIR, f"confusion_matrix_{timestamp}.png")
    cm = plot_confusion_matrix(y_true, y_pred, cm_path)

    cm_latest = os.path.join(RESULTS_DIR, "confusion_matrix_latest.png")
    plot_confusion_matrix(y_true, y_pred, cm_latest)

    print("\nConfusion Matrix (True vs Predicted labels):")
    print(cm)
    print(f"TN: {cm[0,0]} | FP: {cm[0,1]} | FN: {cm[1,0]} | TP: {cm[1,1]}")

    tn, fp, fn, tp = cm[0,0], cm[0,1], cm[1,0], cm[1,1]
    acc = (tp + tn) / (tp + tn + fp + fn)
    prec = tp / (tp + fp) if (tp + fp) > 0 else 0
    rec = tp / (tp + fn) if (tp + fn) > 0 else 0
    f1 = 2 * (prec * rec) / (prec + rec) if (prec + rec) > 0 else 0

    print(f"\nMetrics from Confusion Matrix (threshold={THRESHOLD}):")
    print(f"Accuracy:  {acc:.4f}")
    print(f"Precision: {prec:.4f}")
    print(f"Recall:    {rec:.4f}")
    print(f"F1-Score:  {f1:.4f}")

    # Useful for report: last epoch train/val accuracy
    train_acc_last = history.history["accuracy"][-1]
    val_acc_last = history.history["val_accuracy"][-1]
    print(f"\nTraining Accuracy (last epoch):   {train_acc_last:.4f}")
    print(f"Validation Accuracy (last epoch): {val_acc_last:.4f}")


if __name__ == "__main__":
    main()
