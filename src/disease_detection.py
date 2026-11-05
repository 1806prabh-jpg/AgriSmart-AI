
import json
from functools import lru_cache
from pathlib import Path

import numpy as np
import tensorflow as tf
from PIL import Image


BASE_DIR = Path(__file__).resolve().parent.parent
MODEL_PATH = BASE_DIR / "models" / "plant_disease_model.keras"
CLASSES_PATH = BASE_DIR / "models" / "disease_classes.json"

IMAGE_SIZE = (160, 160)


@lru_cache(maxsize=1)
def load_disease_model():
    """Load the trained model and its class names once."""

    if not MODEL_PATH.exists():
        raise FileNotFoundError("Disease model not found.")

    if not CLASSES_PATH.exists():
        raise FileNotFoundError("Disease class names not found.")

    model = tf.keras.models.load_model(MODEL_PATH)

    with open(CLASSES_PATH, "r", encoding="utf-8") as file:
        class_names = json.load(file)

    return model, class_names


def predict_disease(uploaded_image):
    """Predict plant disease from an uploaded PIL image."""

    model, class_names = load_disease_model()

    # Convert image to RGB
    image = uploaded_image.convert("RGB")

    # Resize image
    image = image.resize(IMAGE_SIZE)

    # Convert image to NumPy array
    image_array = np.array(image, dtype=np.float32)

    # Apply MobileNetV2 preprocessing
    image_array = tf.keras.applications.mobilenet_v2.preprocess_input(
        image_array
    )

    # Add batch dimension
    image_array = np.expand_dims(image_array, axis=0)

    # Make prediction
    predictions = model.predict(image_array, verbose=0)[0]

    # Get top 3 predictions
    top_indices = np.argsort(predictions)[-3:][::-1]

    results = []

    for index in top_indices:
        raw_name = class_names[int(index)]

        if "___" in raw_name:
            crop, condition = raw_name.split("___", 1)
        else:
            crop = "Unknown"
            condition = raw_name

        results.append({
            "crop": crop.replace("_", " "),
            "condition": condition.replace("_", " "),
            "class_name": raw_name,
            "confidence": round(float(predictions[index]) * 100, 2)
        })

    best_result = results[0]

    return {
        "crop": best_result["crop"],
        "condition": best_result["condition"],
        "confidence": best_result["confidence"],
        "is_background": (
            best_result["class_name"] == "Background_without_leaves"
        ),
        "top_predictions": results
    }
