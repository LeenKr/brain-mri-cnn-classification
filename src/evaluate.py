import numpy as np
import tensorflow as tf
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import confusion_matrix
from datetime import datetime
import os

from data_loader import load_data, count_labels

def plot_confusion_matrix(y_true, y_pred, save_path="results/confusion_matrix_eval.png"):
    # ✅ FIX: force 2x2 matrix
    cm = confusion_matrix(y_true, y_pred, labels=[0, 1])

    plt.figure(figsize=(8, 6))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues',
                xticklabels=['Healthy', 'Glioma'],
                yticklabels=['Healthy', 'Glioma'],
                cbar_kws={'label': 'Count'})

    plt.title('Confusion Matrix - Evaluation', fontsize=14, fontweight='bold')
    plt.xlabel('Predicted Label', fontsize=12)
    plt.ylabel('True Label', fontsize=12)

    plt.text(0.5, -0.15, f'TN: {cm[0,0]}  |  FP: {cm[0,1]}  |  FN: {cm[1,0]}  |  TP: {cm[1,1]}',
             ha='center', transform=plt.gca().transAxes, fontsize=10)

    plt.tight_layout()
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    print(f"Confusion matrix saved to {save_path}")
    plt.close()

    return cm

os.makedirs("results", exist_ok=True)

model = tf.keras.models.load_model("results/brain_tumor_classifier.keras")

_, test_ds = load_data("dataset")

# ✅ Optional check (recommended once)
count_labels(test_ds, "TEST")

loss, accuracy, precision, recall, auc = model.evaluate(test_ds, verbose=0)
print(f"Test Accuracy:  {accuracy:.4f}")
print(f"Test Precision: {precision:.4f}")
print(f"Test Recall:    {recall:.4f}")
print(f"Test AUC:       {auc:.4f}")

y_true, y_pred = [], []

for images, labels in test_ds:
    probs = model.predict(images, verbose=0)
    preds = (probs >= 0.5).astype(int).flatten()
    y_true.append(labels.numpy().astype(int).flatten())
    y_pred.append(preds)

y_true = np.concatenate(y_true)
y_pred = np.concatenate(y_pred)

timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
cm_path = f"results/confusion_matrix_eval_{timestamp}.png"
cm = plot_confusion_matrix(y_true, y_pred, cm_path)
plot_confusion_matrix(y_true, y_pred, "results/confusion_matrix_eval_latest.png")

print("\nConfusion Matrix (True vs Predicted labels):")
print(cm)
print(f"TN: {cm[0,0]} | FP: {cm[0,1]} | FN: {cm[1,0]} | TP: {cm[1,1]}")

tn, fp, fn, tp = cm[0,0], cm[0,1], cm[1,0], cm[1,1]
cm_accuracy = (tp + tn) / (tp + tn + fp + fn)
cm_precision = tp / (tp + fp) if (tp + fp) > 0 else 0
cm_recall = tp / (tp + fn) if (tp + fn) > 0 else 0
f1_score = 2 * (cm_precision * cm_recall) / (cm_precision + cm_recall) if (cm_precision + cm_recall) > 0 else 0

print(f"\nMetrics from Confusion Matrix:")
print(f"Accuracy:  {cm_accuracy:.4f}")
print(f"Precision: {cm_precision:.4f}")
print(f"Recall:    {cm_recall:.4f}")
print(f"F1-Score:  {f1_score:.4f}")
