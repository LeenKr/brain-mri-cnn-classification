import tensorflow as tf
import numpy as np

IMG_SIZE = (256, 256)
BATCH_SIZE = 32
SEED = 123

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
        shuffle=True,   # ✅ FIX: must be True (same as train) so split is correct
        seed=SEED,
        validation_split=0.20,
        subset="validation"
    )

    AUTOTUNE = tf.data.AUTOTUNE
    train_ds = train_ds.prefetch(buffer_size=AUTOTUNE)
    test_ds  = test_ds.prefetch(buffer_size=AUTOTUNE)
    return train_ds, test_ds


# ✅ Optional (recommended): quick sanity check to confirm both classes exist in test_ds
def count_labels(ds, name="DATASET"):
    c0, c1 = 0, 0
    for _, y in ds:
        y = y.numpy().reshape(-1).astype(int)
        c0 += int((y == 0).sum())
        c1 += int((y == 1).sum())
    print(f"{name} -> healthy(0): {c0}, glioma(1): {c1}")
    return c0, c1
