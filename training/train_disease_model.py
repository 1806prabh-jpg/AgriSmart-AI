
import json
import random
from pathlib import Path

import numpy as np
import tensorflow as tf
from sklearn.model_selection import train_test_split

# Project paths
BASE_DIR = Path(__file__).resolve().parent.parent
DATA_DIR = BASE_DIR / "data" / "plant_village" / "tfds"
MODEL_DIR = BASE_DIR / "models"

MODEL_DIR.mkdir(parents=True, exist_ok=True)

# Training settings
IMAGE_SIZE = (160, 160)
BATCH_SIZE = 16
MAX_IMAGES_PER_CLASS = 300
EPOCHS = 5
SEED = 42

random.seed(SEED)
np.random.seed(SEED)
tf.random.set_seed(SEED)

# Limit CPU usage
tf.config.threading.set_intra_op_parallelism_threads(4)
tf.config.threading.set_inter_op_parallelism_threads(2)

print("Searching for PlantVillage dataset...")

roots = list(
    DATA_DIR.rglob("Plant_leave_diseases_dataset_without_augmentation")
)

if not roots:
    raise FileNotFoundError("PlantVillage image folder not found.")

dataset_root = roots[0]

# Collect a balanced selection of images
image_paths = []
image_labels = []
class_names = []

folders = sorted([p for p in dataset_root.iterdir() if p.is_dir()])

for folder in folders:
    images = [
        p for p in folder.iterdir()
        if p.suffix.lower() in [".jpg", ".jpeg", ".png"]
    ]

    if len(images) < 2:
        continue

    selected = random.sample(
        images,
        min(len(images), MAX_IMAGES_PER_CLASS)
    )

    label = len(class_names)
    class_names.append(folder.name)

    for image in selected:
        image_paths.append(str(image))
        image_labels.append(label)

print("Classes:", len(class_names))
print("Selected images:", len(image_paths))

# Split into training and validation sets
train_paths, val_paths, train_labels, val_labels = train_test_split(
    image_paths,
    image_labels,
    test_size=0.20,
    random_state=SEED,
    stratify=image_labels
)

# Load and preprocess images
def load_image(path, label):
    image = tf.io.read_file(path)
    image = tf.io.decode_image(
        image, channels=3, expand_animations=False
    )
    image.set_shape([None, None, 3])
    image = tf.image.resize(image, IMAGE_SIZE)
    image = tf.keras.applications.mobilenet_v2.preprocess_input(image)
    return image, label


def make_dataset(paths, labels, training=False):
    dataset = tf.data.Dataset.from_tensor_slices(
        (paths, labels)
    )

    if training:
        dataset = dataset.shuffle(
            buffer_size=len(paths),
            seed=SEED
        )

    dataset = dataset.map(
        load_image,
        num_parallel_calls=2
    )

    dataset = dataset.batch(BATCH_SIZE)
    dataset = dataset.prefetch(1)

    return dataset


train_ds = make_dataset(train_paths, train_labels, training=True)
val_ds = make_dataset(val_paths, val_labels)

print("Training images:", len(train_paths))
print("Validation images:", len(val_paths))

# Load pretrained MobileNetV2
print("Loading MobileNetV2...")

base_model = tf.keras.applications.MobileNetV2(
    input_shape=(160, 160, 3),
    include_top=False,
    weights="imagenet"
)

# Freeze pretrained layers
base_model.trainable = False

# Build classification model
inputs = tf.keras.Input(shape=(160, 160, 3))

x = base_model(inputs, training=False)
x = tf.keras.layers.GlobalAveragePooling2D()(x)
x = tf.keras.layers.Dropout(0.25)(x)

outputs = tf.keras.layers.Dense(
    len(class_names),
    activation="softmax"
)(x)

model = tf.keras.Model(inputs, outputs)

model.compile(
    optimizer=tf.keras.optimizers.Adam(learning_rate=0.001),
    loss="sparse_categorical_crossentropy",
    metrics=["accuracy"]
)

model.summary()

# Stop early if validation stops improving
early_stop = tf.keras.callbacks.EarlyStopping(
    monitor="val_loss",
    patience=2,
    restore_best_weights=True
)

# Train model
print("\nStarting disease model training...")

history = model.fit(
    train_ds,
    validation_data=val_ds,
    epochs=EPOCHS,
    callbacks=[early_stop]
)

# Save trained model
model_path = MODEL_DIR / "plant_disease_model.keras"
model.save(model_path)

# Save class names in prediction order
labels_path = MODEL_DIR / "disease_classes.json"

with open(labels_path, "w", encoding="utf-8") as file:
    json.dump(class_names, file, indent=4)

# Save evaluation metrics
metrics = {
    "classes": len(class_names),
    "selected_images": len(image_paths),
    "training_images": len(train_paths),
    "validation_images": len(val_paths),
    "final_training_accuracy": float(
        history.history["accuracy"][-1]
    ),
    "final_validation_accuracy": float(
        history.history["val_accuracy"][-1]
    )
}

with open(
    MODEL_DIR / "disease_metrics.json",
    "w",
    encoding="utf-8"
) as file:
    json.dump(metrics, file, indent=4)

print("\nTraining completed successfully!")
print("Model saved:", model_path)
print("Class names saved:", labels_path)
print("Metrics:", metrics)
