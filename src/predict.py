import numpy as np
import tensorflow as tf
import argparse
from data_loader import IMG_SIZE

# Parse command-line arguments for image path and optional threshold
parser = argparse.ArgumentParser(description="Brain Tumor MRI Classification")
parser.add_argument('--image', type=str, required=True, help="Path to input MRI image")
parser.add_argument('--threshold', type=float, default=0.5, help="Threshold for classifying as tumor (default=0.5)")
args = parser.parse_args()

image_path = args.image
threshold = args.threshold

# Load the trained model
model = tf.keras.models.load_model("results/brain_tumor_classifier.keras")

# Load and preprocess the image
img = tf.keras.utils.load_img(image_path, target_size=IMG_SIZE)
img_array = tf.keras.utils.img_to_array(img)
img_array = np.expand_dims(img_array, axis=0)  # create batch dimension, shape (1, H, W, 3)

# Get prediction probability for class "glioma"
pred_prob = model.predict(img_array, verbose=0)[0][0]   # model outputs a probability (sigmoid)
# Determine class based on threshold
pred_label = "glioma (tumor)" if pred_prob >= threshold else "healthy"
confidence = pred_prob if pred_label.startswith("glioma") else (1 - pred_prob)

# Print the prediction and confidence
print(f"Model confidence that the image is a tumor: {pred_prob*100:.2f}%")
print(f"--> Predicted class = '{pred_label}' (using threshold {threshold}) with confidence {confidence*100:.2f}%")