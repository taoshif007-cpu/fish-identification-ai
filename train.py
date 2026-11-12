"""Step 2: train the fish classifier (MobileNetV2 transfer learning).

Run in Google Colab (see notebooks/fish_training_colab.ipynb) or anywhere with TensorFlow:
    python train.py --data dataset --out models

Outputs (in --out):
    fish_model.keras, class_names.json, training_curves.png, confusion_matrix.png,
    classification_report.csv, training_summary.json
To add a new species later: add a folder in dataset_raw, a row in species_info.csv, re-run
prepare_dataset.py and this script. The class list is read from the folders automatically.
"""
import argparse
import json
import sys
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
import tensorflow as tf
from sklearn.metrics import classification_report, confusion_matrix
from tensorflow import keras
from tensorflow.keras import layers

IMG_SIZE = 224


def load_datasets(data_dir, batch_size, seed):
    data_dir = Path(data_dir)
    for split in ("train", "val", "test"):
        if not (data_dir / split).is_dir():
            sys.exit(f"ERROR: '{data_dir / split}' not found. Run prepare_dataset.py first.")

    def load(split, shuffle):
        return keras.utils.image_dataset_from_directory(
            data_dir / split, labels="inferred", label_mode="int",
            image_size=(IMG_SIZE, IMG_SIZE), batch_size=batch_size,
            shuffle=shuffle, seed=seed)

    train_ds, val_ds, test_ds = load("train", True), load("val", False), load("test", False)
    class_names = train_ds.class_names
    if val_ds.class_names != class_names or test_ds.class_names != class_names:
        sys.exit("ERROR: train/val/test folders contain different species.")
    return train_ds, val_ds, test_ds, class_names


def build_model(num_classes):
    """Augmentation + rescaling are INSIDE the model, so the web app just feeds raw 0-255 pixels."""
    augmentation = keras.Sequential([
        layers.RandomFlip("horizontal"),
        layers.RandomRotation(0.08),
        layers.RandomZoom(0.15),
        layers.RandomContrast(0.15),
        layers.RandomBrightness(0.15),
    ], name="augmentation")

    base = keras.applications.MobileNetV2(
        input_shape=(IMG_SIZE, IMG_SIZE, 3), include_top=False, weights="imagenet")
    base.trainable = False

    inputs = keras.Input(shape=(IMG_SIZE, IMG_SIZE, 3), name="image")
    x = augmentation(inputs)
    x = layers.Rescaling(1.0 / 127.5, offset=-1.0, name="mobilenet_rescale")(x)  # MobileNetV2 expects [-1, 1]
    x = base(x, training=False)
    x = layers.GlobalAveragePooling2D()(x)
    x = layers.Dropout(0.3)(x)
    outputs = layers.Dense(num_classes, activation="softmax", name="species")(x)
    return keras.Model(inputs, outputs), base


def compute_class_weights(train_ds, num_classes):
    labels = np.concatenate([y.numpy() for _, y in train_ds])
    counts = np.bincount(labels, minlength=num_classes).astype(float)
    weights = counts.sum() / (num_classes * np.maximum(counts, 1))
    return {i: float(w) for i, w in enumerate(weights)}


def plot_history(histories, out_path):
    acc = sum((h["accuracy"] for h in histories), [])
    val_acc = sum((h["val_accuracy"] for h in histories), [])
    loss = sum((h["loss"] for h in histories), [])
    val_loss = sum((h["val_loss"] for h in histories), [])
    boundary = len(histories[0]["accuracy"])
    fig, axes = plt.subplots(1, 2, figsize=(12, 4))
    for ax, tr, va, title in ((axes[0], acc, val_acc, "Accuracy"), (axes[1], loss, val_loss, "Loss")):
        ax.plot(tr, label="train")
        ax.plot(va, label="validation")
        ax.axvline(boundary - 1, color="grey", linestyle="--", label="start fine-tuning")
        ax.set_title(title)
        ax.set_xlabel("epoch")
        ax.legend()
    plt.tight_layout()
    plt.savefig(out_path, dpi=150)
    plt.close()


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", default="dataset")
    ap.add_argument("--out", default="models")
    ap.add_argument("--batch", type=int, default=32)
    ap.add_argument("--epochs", type=int, default=12, help="epochs for the first (frozen) stage")
    ap.add_argument("--finetune-epochs", type=int, default=10)
    ap.add_argument("--seed", type=int, default=42)
    args = ap.parse_args()

    tf.keras.utils.set_random_seed(args.seed)
    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    print("TensorFlow version:", tf.__version__, "| GPU:", bool(tf.config.list_physical_devices("GPU")))

    train_ds, val_ds, test_ds, class_names = load_datasets(args.data, args.batch, args.seed)
    num_classes = len(class_names)
    print("Species (classes):", class_names)
    class_weights = compute_class_weights(train_ds, num_classes)

    autotune = tf.data.AUTOTUNE
    train_ds, val_ds, test_eval = train_ds.prefetch(autotune), val_ds.prefetch(autotune), test_ds.prefetch(autotune)

    model, base = build_model(num_classes)
    callbacks = [
        keras.callbacks.EarlyStopping(monitor="val_loss", patience=4, restore_best_weights=True),
        keras.callbacks.ReduceLROnPlateau(monitor="val_loss", factor=0.3, patience=2, min_lr=1e-7),
    ]

    # ---- Stage 1: train only the new classification head ----
    model.compile(optimizer=keras.optimizers.Adam(1e-3),
                  loss="sparse_categorical_crossentropy", metrics=["accuracy"])
    print("\n=== Stage 1: training the new top layer ===")
    h1 = model.fit(train_ds, validation_data=val_ds, epochs=args.epochs,
                   class_weight=class_weights, callbacks=callbacks)

    # ---- Stage 2: fine-tune the last layers of MobileNetV2 with a tiny learning rate ----
    base.trainable = True
    for layer in base.layers[:-30]:
        layer.trainable = False
    for layer in base.layers:                      # keep BatchNorm frozen (more stable)
        if isinstance(layer, layers.BatchNormalization):
            layer.trainable = False
    model.compile(optimizer=keras.optimizers.Adam(1e-5),
                  loss="sparse_categorical_crossentropy", metrics=["accuracy"])
    print("\n=== Stage 2: fine-tuning ===")
    h2 = model.fit(train_ds, validation_data=val_ds, epochs=args.finetune_epochs,
                   class_weight=class_weights, callbacks=callbacks)

    plot_history([h1.history, h2.history], out / "training_curves.png")

    # ---- Save the model FIRST so a later error can never lose your training ----
    model.save(out / "fish_model.keras")
    with open(out / "class_names.json", "w", encoding="utf-8") as f:
        json.dump(class_names, f, indent=2)
    print("Model saved to", out / "fish_model.keras")

    # ---- Evaluation on the untouched TEST set ----
    print("\n=== Evaluating on the test set ===")
    test_loss, test_acc = model.evaluate(test_eval, verbose=0)
    y_true = np.concatenate([y.numpy() for _, y in test_ds])
    y_prob = model.predict(test_eval, verbose=0)
    y_pred = y_prob.argmax(axis=1)

    all_labels = list(range(num_classes))
    report = classification_report(y_true, y_pred, labels=all_labels, target_names=class_names,
                                   output_dict=True, zero_division=0)
    pd.DataFrame(report).T.to_csv(out / "classification_report.csv")
    print(classification_report(y_true, y_pred, labels=all_labels, target_names=class_names, zero_division=0))

    cm = confusion_matrix(y_true, y_pred, labels=all_labels)
    plt.figure(figsize=(1.0 * num_classes + 4, 1.0 * num_classes + 3))
    sns.heatmap(cm, annot=True, fmt="d", cmap="Blues", xticklabels=class_names, yticklabels=class_names)
    plt.xlabel("Predicted")
    plt.ylabel("True")
    plt.title("Confusion matrix (test set)")
    plt.tight_layout()
    plt.savefig(out / "confusion_matrix.png", dpi=150)
    plt.close()

    with open(out / "training_summary.json", "w", encoding="utf-8") as f:
        json.dump({"tensorflow_version": tf.__version__, "img_size": IMG_SIZE,
                   "test_accuracy": float(test_acc), "test_loss": float(test_loss),
                   "classes": class_names}, f, indent=2)
    print(f"\nTest accuracy: {test_acc:.3f}")
    print(f"Saved model and reports to '{out}'. Train with TensorFlow {tf.__version__} - "
          f"requirements.txt must use the same version.")


if __name__ == "__main__":
    main()
