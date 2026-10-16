from pathlib import Path
import argparse
import json
import numpy as np
import tensorflow as tf
from sklearn.model_selection import train_test_split

IMAGE_SIZE = (224, 224)
BATCH_SIZE = 32
SEED = 42
SUPPORTED_EXTENSIONS = {".jpg", ".jpeg", ".png", ".bmp"}

def locate_class_folders(data_dir: Path):
    """Find a directory whose immediate subfolders are the six TrashNet classes."""
    expected = {"cardboard", "glass", "metal", "paper", "plastic", "trash"}
    candidates = [data_dir] + [p for p in data_dir.rglob("*") if p.is_dir()]
    for folder in candidates:
        subdirs = {p.name.lower(): p for p in folder.iterdir() if p.is_dir()} if folder.exists() else {}
        if expected.issubset(set(subdirs.keys())):
            return folder, [subdirs[name] for name in sorted(expected)]
    raise FileNotFoundError(
        "Could not find the six class folders: cardboard, glass, metal, paper, plastic, trash. "
        "Point --data_dir to the extracted TrashNet repository or its dataset-resized folder."
    )

def collect_images(class_dirs):
    paths, labels, class_names = [], [], []
    for class_index, class_dir in enumerate(class_dirs):
        class_names.append(class_dir.name.lower())
        for path in class_dir.rglob("*"):
            if path.is_file() and path.suffix.lower() in SUPPORTED_EXTENSIONS:
                paths.append(str(path))
                labels.append(class_index)
    if not paths:
        raise ValueError("No supported images were found in the class folders.")
    return np.array(paths), np.array(labels, dtype=np.int32), class_names

def make_dataset(paths, labels, training=False):
    path_ds = tf.data.Dataset.from_tensor_slices((paths, labels))

    def load_image(path, label):
        image_bytes = tf.io.read_file(path)
        image = tf.io.decode_image(image_bytes, channels=3, expand_animations=False)
        image.set_shape([None, None, 3])
        image = tf.image.resize(image, IMAGE_SIZE)
        image = tf.cast(image, tf.float32)  # 0..255; model contains its own rescaling layer
        return image, label

    ds = path_ds.map(load_image, num_parallel_calls=tf.data.AUTOTUNE)
    if training:
        ds = ds.shuffle(min(len(paths), 2000), seed=SEED, reshuffle_each_iteration=True)
    return ds.batch(BATCH_SIZE).prefetch(tf.data.AUTOTUNE)

def main():
    parser = argparse.ArgumentParser(description="Train a six-class TrashNet image classifier.")
    parser.add_argument("--data_dir", required=True, help="Path to extracted TrashNet repository or dataset-resized folder.")
    parser.add_argument("--output_dir", default="models", help="Folder to save model and class names.")
    parser.add_argument("--epochs", type=int, default=8, help="Training epochs for the classification head.")
    args = parser.parse_args()

    data_dir = Path(args.data_dir).expanduser().resolve()
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    _, class_dirs = locate_class_folders(data_dir)
    paths, labels, class_names = collect_images(class_dirs)
    print(f"Found {len(paths)} images in classes: {class_names}")

    train_paths, temp_paths, train_labels, temp_labels = train_test_split(
        paths, labels, test_size=0.30, random_state=SEED, stratify=labels
    )
    val_paths, test_paths, val_labels, test_labels = train_test_split(
        temp_paths, temp_labels, test_size=0.50, random_state=SEED, stratify=temp_labels
    )
    print(f"Train: {len(train_paths)} | Validation: {len(val_paths)} | Test: {len(test_paths)}")

    train_ds = make_dataset(train_paths, train_labels, training=True)
    val_ds = make_dataset(val_paths, val_labels)
    test_ds = make_dataset(test_paths, test_labels)

    # The model rescales input pixels from [0, 255] to [-1, 1] internally.
    inputs = tf.keras.Input(shape=(224, 224, 3), name="image")
    x = tf.keras.layers.Rescaling(scale=1.0 / 127.5, offset=-1.0, name="pixel_rescaling")(inputs)
    base_model = tf.keras.applications.MobileNetV2(
        input_shape=(224, 224, 3), include_top=False, weights="imagenet"
    )
    base_model.trainable = False
    x = base_model(x, training=False)
    x = tf.keras.layers.GlobalAveragePooling2D()(x)
    x = tf.keras.layers.Dropout(0.25)(x)
    outputs = tf.keras.layers.Dense(len(class_names), activation="softmax", name="class_probabilities")(x)
    model = tf.keras.Model(inputs, outputs)

    model.compile(
        optimizer=tf.keras.optimizers.Adam(learning_rate=1e-3),
        loss="sparse_categorical_crossentropy",
        metrics=["accuracy"],
    )
    callbacks = [
        tf.keras.callbacks.EarlyStopping(monitor="val_loss", patience=3, restore_best_weights=True),
    ]
    model.summary()
    model.fit(train_ds, validation_data=val_ds, epochs=args.epochs, callbacks=callbacks)

    test_loss, test_accuracy = model.evaluate(test_ds, verbose=1)
    print(f"\nHeld-out test loss: {test_loss:.4f}")
    print(f"Held-out test accuracy: {test_accuracy:.4f}")

    # Produce a confusion matrix and macro F1 score when scikit-learn is available.
    y_true, y_pred = [], []
    for batch_images, batch_labels in test_ds:
        probabilities = model.predict(batch_images, verbose=0)
        y_true.extend(batch_labels.numpy().tolist())
        y_pred.extend(np.argmax(probabilities, axis=1).tolist())
    try:
        from sklearn.metrics import classification_report, confusion_matrix, f1_score
        print("\nClassification report:")
        print(classification_report(y_true, y_pred, target_names=class_names, zero_division=0))
        print("Macro F1:", round(float(f1_score(y_true, y_pred, average="macro")), 4))
        print("Confusion matrix (rows=true, columns=predicted):")
        print(confusion_matrix(y_true, y_pred))
    except ImportError:
        print("scikit-learn metrics unavailable; install scikit-learn to view detailed metrics.")

    model_path = output_dir / "waste_classifier.keras"
    model.save(model_path)
    with open(output_dir / "class_names.json", "w", encoding="utf-8") as f:
        json.dump(class_names, f, indent=2)
    with open(output_dir / "evaluation.json", "w", encoding="utf-8") as f:
        json.dump({"test_loss": float(test_loss), "test_accuracy": float(test_accuracy),
                   "num_images": int(len(paths)), "class_names": class_names}, f, indent=2)
    print(f"\nSaved model to: {model_path.resolve()}")
    print(f"Saved class labels and evaluation results in: {output_dir.resolve()}")

if __name__ == "__main__":
    main()
