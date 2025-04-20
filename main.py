from google.cloud import storage
import tensorflow as tf
import numpy as np
import pandas as pd
from PIL import Image
import os
from flask import jsonify
import functions_framework
from transformers import TFAutoModelForImageClassification, ViTFeatureExtractor

# === Configuration ===
BUCKET_NAME = "aurora-project"  # ✅ Replace with your actual bucket
MODEL_FOLDER = "models"
CSV_PATH = "models/aurora_products_B.csv"
TMP_MODEL_DIR = "/tmp/vit_model"
TMP_CSV_PATH = "/tmp/products.csv"

# === Labels and Critical Conditions ===
LABEL_TO_NAME = {
    "LABEL_0": "Acne",
    "LABEL_1": "Milia",
    "LABEL_2": "Hyperpigmentation",
    "LABEL_3": "Wrinkles",
    "LABEL_4": "Keratosis",
    "LABEL_5": "Oily Skin",
    "LABEL_6": "Dry Skin",
    "LABEL_7": "Normal",
    "LABEL_8": "Non-Wrinkled Skin"
}
CRITICAL_CONDITIONS = ["Acne", "Milia", "Keratosis", "Hyperpigmentation"]

# === Global Variables ===
model = None
feature_extractor = None
df = None

def download_blob(bucket_name, source_blob_name, destination_file_name):
    client = storage.Client()
    bucket = client.bucket(bucket_name)
    blob = bucket.blob(source_blob_name)
    blob.download_to_filename(destination_file_name)

def load_resources():
    global model, feature_extractor, df

    if model is None or feature_extractor is None:
        os.makedirs(TMP_MODEL_DIR, exist_ok=True)
        download_blob(BUCKET_NAME, f"{MODEL_FOLDER}/config.json", f"{TMP_MODEL_DIR}/config.json")
        download_blob(BUCKET_NAME, f"{MODEL_FOLDER}/tf_model.h5", f"{TMP_MODEL_DIR}/tf_model.h5")
        download_blob(BUCKET_NAME, f"{MODEL_FOLDER}/preprocessor_config.json", f"{TMP_MODEL_DIR}/preprocessor_config.json")
        
        model = TFAutoModelForImageClassification.from_pretrained(TMP_MODEL_DIR)
        feature_extractor = ViTFeatureExtractor.from_pretrained(TMP_MODEL_DIR)

    if df is None:
        download_blob(BUCKET_NAME, CSV_PATH, TMP_CSV_PATH)
        df = pd.read_csv(TMP_CSV_PATH)

@functions_framework.http
def predict(request):
    load_resources()

    if 'file' not in request.files:
        return jsonify({"error": "No image file provided"}), 400

    file = request.files['file']
    image = Image.open(file.stream).convert("RGB").resize((224, 224))
    inputs = feature_extractor(images=[np.array(image)], return_tensors='tf')

    logits = model(**inputs).logits
    probs = tf.nn.softmax(logits, axis=1).numpy()[0]

    top_idx = np.argmax(probs)
    confidence = float(probs[top_idx])
    raw_label = model.config.id2label[top_idx]
    condition = LABEL_TO_NAME.get(raw_label, raw_label)

    result = {
        "condition": condition,
        "confidence": round(confidence, 4)
    }

    matches = df[df['Targets'].str.lower().str.contains(condition.lower(), na=False)]
    recommendations = matches[matches['Product'].notna()].head(3).to_dict(orient='records')

    if confidence >= 0.99:
        result["recommendation_type"] = "products"
        result["recommendations"] = recommendations
    elif confidence < 0.90 and condition in CRITICAL_CONDITIONS:
        result["recommendation_type"] = "refer"
        result["message"] = "Model is not confident and condition is critical. Please consult a dermatologist."
        result["recommendations"] = recommendations
    else:
        result["recommendation_type"] = "cautious_products"
        result["message"] = "Model is moderately confident. Use recommended products with care."
        result["recommendations"] = recommendations

    return jsonify(result)
