import os
import numpy as np
import tensorflow as tf
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import confusion_matrix
from datetime import datetime

from data_loader import load_data, count_labels, IMG_SIZE

# ======================
# Settings
# ======================
BATCH_SIZE = 32
EPOCHS_PHASE1 = 8
EPOCHS_PHASE2 = 8

# Keep the same logic you used in Exp1 (higher glioma recall)
THRESHOLD = 0.30
CLASS_WEIGHT = {0: 1.0, 1: 3.0}

RESULTS_DIR = os.path.join("results", "pretrained_effnetb0")
os.makedirs(RESULTS_DIR, exist_ok=True)

# EfficientNet preprocessing
preprocess_input = tf.keras.applications.efficientnet.preprocess_input


# ======================
# Plot Training History
# ======================
def plot_training_history(history, save_path):
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5))

    ax1.plot(history["accuracy"], label="Training Accuracy", marker="o")
    ax1.plot(history["val_accuracy"], label="Testing Accuracy", marker="s")
    ax1.set_title("Model Accuracy", fontsize=14, fontweight="bold")
    ax1.set_xlabel("Epoch")
    ax1.set_ylabel("Accuracy")
    ax1.legend()
    ax1.grid(True, alpha=0.3)

    ax2.plot(history["loss"], label="Training Loss", marker="o")
    ax2.plot(history["val_loss"], label="Testing Loss", marker="s")
    ax2.set_title("Model Loss", fontsize=14, fontweight="bold")
    ax2.set_xlabel("Epoch")
    ax2.set_ylabel("Loss")
    ax2.legend()
    ax2.grid(True, alpha=0.3)

    plt.tight_layout()
    plt.savefig(save_path, dpi=300, bbox_inches="tight")
    plt.close()
    print(f"Saved training history -> {save_path}")


# ======================
# Confusion Matrix Plot
# ======================
def plot_confusion_matrix(y_true, y_pred, save_path, title):
    cm = confusion_matrix(y_true, y_pred, labels=[0, 1])

    plt.figure(figsize=(8, 6))
    sns.heatmap(
        cm, annot=True, fmt="d", cmap="Blues",
        xticklabels=["Healthy", "Glioma"],
        yticklabels=["Healthy", "Glioma"],
        cbar_kws={"label": "Count"}
    )

    plt.title(title, fontsize=14, fontweight="bold")
    plt.xlabel("Predicted Label")
    plt.ylabel("True Label")

    plt.text(
        0.5, -0.15,
        f"TN: {cm[0,0]} | FP: {cm[0,1]} | FN: {cm[1,0]} | TP: {cm[1,1]}",
        ha="center", transform=plt.gca().transAxes
    )

    plt.tight_layout()
    plt.savefig(save_path, dpi=300, bbox_inches="tight")
    plt.close()
    print(f"Saved confusion matrix -> {save_path}")
    return cm


# ======================
# Build Pretrained Model
# ======================
def build_model():
    inputs = tf.keras.Input(shape=IMG_SIZE + (3,))

    # augmentation
    x = tf.keras.layers.RandomFlip("horizontal")(inputs)
    x = tf.keras.layers.RandomRotation(0.1)(x)
    x = tf.keras.layers.RandomZoom(0.1)(x)

    # preprocessing for EfficientNet
    x = tf.keras.layers.Lambda(preprocess_input)(x)

    base_model = tf.keras.applications.EfficientNetB0(
        include_top=False,
        weights="imagenet",
        input_tensor=x
    )

    base_model.trainable = False  # phase 1 freeze

    y = tf.keras.layers.GlobalAveragePooling2D()(base_model.output)
    y = tf.keras.layers.Dropout(0.5)(y)
    outputs = tf.keras.layers.Dense(1, activation="sigmoid")(y)

    model = tf.keras.Model(inputs, outputs, name="pretrained_effnetb0")
    return model, base_model


def main():
    # Load data (80% train, 20% testing)
    train_ds, test_ds = load_data("dataset")
    print(f"Train batches: {len(train_ds)} | Test batches: {len(test_ds)}")
    count_labels(train_ds, "TRAIN")
    count_labels(test_ds, "TEST")

    model, base_model = build_model()

    # callbacks
    early_stop = tf.keras.callbacks.EarlyStopping(
        monitor="val_loss", patience=3, restore_best_weights=True
    )
    reduce_lr = tf.keras.callbacks.ReduceLROnPlateau(
        monitor="val_loss", factor=0.5, patience=2, min_lr=1e-6, verbose=1
    )

    # ======================
    # Phase 1: Train Head Only
    # ======================
    model.compile(
        optimizer=tf.keras.optimizers.Adam(learning_rate=1e-3),
        loss=tf.keras.losses.BinaryCrossentropy(),
        metrics=[
            tf.keras.metrics.BinaryAccuracy(name="accuracy"),
            tf.keras.metrics.Precision(name="precision"),
            tf.keras.metrics.Recall(name="recall"),
            tf.keras.metrics.AUC(name="auc")
        ]
    )

    print("\n=== Phase 1: Feature Extraction (backbone frozen) ===")
    hist1 = model.fit(
        train_ds,
        validation_data=test_ds,
        epochs=EPOCHS_PHASE1,
        class_weight=CLASS_WEIGHT,
        callbacks=[early_stop, reduce_lr]
    )

    # ======================
    # Phase 2: Fine-tuning
    # ======================
    print("\n=== Phase 2: Fine-tuning (unfreeze last layers) ===")
    base_model.trainable = True

    # unfreeze only last 30 layers
    for layer in base_model.layers[:-30]:
        layer.trainable = False

    model.compile(
        optimizer=tf.keras.optimizers.Adam(learning_rate=1e-5),
        loss=tf.keras.losses.BinaryCrossentropy(),
        metrics=[
            tf.keras.metrics.BinaryAccuracy(name="accuracy"),
            tf.keras.metrics.Precision(name="precision"),
            tf.keras.metrics.Recall(name="recall"),
            tf.keras.metrics.AUC(name="auc")
        ]
    )

    hist2 = model.fit(
        train_ds,
        validation_data=test_ds,
        epochs=EPOCHS_PHASE2,
        class_weight=CLASS_WEIGHT,
        callbacks=[early_stop, reduce_lr]
    )

    # Combine histories (for one plot)
    history = {}
    for k in hist1.history.keys():
        history[k] = hist1.history[k] + hist2.history.get(k, [])

    # Save model
    model_path = os.path.join(RESULTS_DIR, "brain_tumor_classifier_pretrained_effnetb0.keras")
    model.save(model_path)
    print(f"\nSaved model -> {model_path}")

    # Save training curves
    plot_training_history(history, os.path.join(RESULTS_DIR, "training_history.png"))

    # ======================
    # Confusion Matrix (threshold)
    # ======================
    y_true, y_pred = [], []

    for images, labels in test_ds:
        probs = model.predict(images, verbose=0).flatten()
        preds = (probs >= THRESHOLD).astype(int)
        y_true.append(labels.numpy().astype(int).flatten())
        y_pred.append(preds)

    y_true = np.concatenate(y_true)
    y_pred = np.concatenate(y_pred)

    ts = datetime.now().strftime("%Y%m%d_%H%M%S")
    cm = plot_confusion_matrix(
        y_true, y_pred,
        os.path.join(RESULTS_DIR, f"confusion_matrix_{ts}.png"),
        title=f"Confusion Matrix (threshold={THRESHOLD})"
    )
    plot_confusion_matrix(
        y_true, y_pred,
        os.path.join(RESULTS_DIR, "confusion_matrix_latest.png"),
        title=f"Confusion Matrix (threshold={THRESHOLD})"
    )

    tn, fp, fn, tp = cm[0,0], cm[0,1], cm[1,0], cm[1,1]
    acc = (tp + tn) / (tp + tn + fp + fn)
    prec = tp / (tp + fp) if (tp + fp) else 0
    rec = tp / (tp + fn) if (tp + fn) else 0
    f1 = 2 * (prec * rec) / (prec + rec) if (prec + rec) else 0

    print("\nMetrics from Confusion Matrix:")
    print(cm)
    print(f"TN:{tn} FP:{fp} FN:{fn} TP:{tp}")
    print(f"Accuracy:  {acc:.4f}")
    print(f"Precision: {prec:.4f}")
    print(f"Recall:    {rec:.4f}")
    print(f"F1-Score:  {f1:.4f}")


if __name__ == "__main__":
    main()