import io
import numpy as np
import tensorflow as tf
from fastapi import FastAPI, UploadFile, File, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from PIL import Image
import colorsys

app = FastAPI(title="Crop Disease Detection API")


app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


model = None
IMG_SIZE = 224

@app.on_event("startup")
def load_model():
    global model
    try:
        model = tf.keras.models.load_model("plant_disease_model.h5", compile=False)
        print("✅ MODEL LOADED SUCCESSFULLY")
    except Exception as e:
        print("❌ MODEL FAILED TO LOAD:", e)


CLASS_NAMES = [
    "Apple Scab", "Apple Black Rot", "Apple Cedar Rust", "Apple Healthy",
    "Blueberry Healthy", "Cherry Powdery Mildew", "Cherry Healthy",
    "Corn Cercospora Leaf Spot", "Corn Common Rust", "Corn Northern Leaf Blight", "Corn Healthy",
    "Grape Black Rot", "Grape Esca (Black Measles)", "Grape Leaf Blight", "Grape Healthy",
    "Orange Haunglongbing (Citrus Greening)", "Peach Bacterial Spot", "Peach Healthy",
    "Pepper Bell Bacterial Spot", "Pepper Bell Healthy", "Potato Early Blight", "Potato Late Blight", "Potato Healthy",
    "Raspberry Healthy", "Soybean Healthy", "Squash Powdery Mildew", "Strawberry Leaf Scorch", "Strawberry Healthy",
    "Tomato Bacterial Spot", "Tomato Early Blight", "Tomato Late Blight", "Tomato Leaf Mold", "Tomato Septoria Leaf Spot",
    "Tomato Spider Mites", "Tomato Target Spot", "Tomato Yellow Leaf Curl Virus", "Tomato Mosaic Virus", "Tomato Healthy"
]

ADVICE_MAP = {
    "Apple Scab": "Prune infected parts and apply fungicide.",
    "Apple Black Rot": "Remove mummified fruits and prune dead branches.",
    "Apple Cedar Rust": "Remove nearby cedar hosts and apply preventive fungicide.",
    "Cherry Powdery Mildew": "Increase air circulation and apply sulfur fungicide.",
    "Corn Cercospora Leaf Spot": "Rotate crops and use resistant varieties.",
    "Corn Common Rust": "Apply fungicide early and use resistant hybrids.",
    "Corn Northern Leaf Blight": "Manage crop residue and plant resistant varieties.",
    "Grape Black Rot": "Remove infected fruits and leaves from vines.",
    "Grape Esca (Black Measles)": "Remove infected vines and protect pruning wounds.",
    "Grape Leaf Blight": "Maintain sanitation and apply appropriate fungicides.",
    "Orange Haunglongbing (Citrus Greening)": "Control psyllids and remove infected trees.",
    "Peach Bacterial Spot": "Avoid overhead watering and plant resistant cultivars.",
    "Pepper Bell Bacterial Spot": "Use clean seeds and rotate crops.",
    "Potato Early Blight": "Remove infected leaves and avoid overhead irrigation.",
    "Potato Late Blight": "Improve drainage and destroy infected plants.",
    "Squash Powdery Mildew": "Improve airflow and plant resistant varieties.",
    "Strawberry Leaf Scorch": "Remove infected foliage and maintain field sanitation.",
    "Tomato Bacterial Spot": "Avoid overhead watering and use copper sprays.",
    "Tomato Early Blight": "Prune lower leaves and apply mulch to reduce splash.",
    "Tomato Late Blight": "Remove infected plants immediately and improve air circulation.",
    "Tomato Leaf Mold": "Reduce humidity and increase plant spacing.",
    "Tomato Septoria Leaf Spot": "Remove infected debris and avoid wet foliage.",
    "Tomato Spider Mites": "Use insecticidal soap or predatory mites.",
    "Tomato Target Spot": "Improve airflow and apply fungicides if needed.",
    "Tomato Yellow Leaf Curl Virus": "Control whiteflies and remove infected plants.",
    "Tomato Mosaic Virus": "Sanitize tools and use resistant varieties."
}


@app.get("/")
def home():
    return {"message": "Crop Disease Detection API Running"}


def is_plant_image(image):
    """
    Check if the image likely contains a plant/leaf by analyzing green color dominance.
    Uses HSV color space to detect green-dominant pixels typical in plant imagery.
    Returns True if the image appears to be a plant, False otherwise.
    """
    # Resize to a small size for faster analysis
    small = image.resize((100, 100))
    pixels = np.array(small)

    green_pixel_count = 0
    total_pixels = pixels.shape[0] * pixels.shape[1]

    for row in pixels:
        for pixel in row:
            r, g, b = pixel[0] / 255.0, pixel[1] / 255.0, pixel[2] / 255.0
            h, s, v = colorsys.rgb_to_hsv(r, g, b)
            h_deg = h * 360

            # Green hue range: roughly 35° to 150° in HSV
            # Also require minimum saturation and brightness to avoid grays/whites
            if 35 <= h_deg <= 150 and s > 0.15 and v > 0.15:
                green_pixel_count += 1

    green_ratio = green_pixel_count / total_pixels
    # If at least 10% of pixels are green-ish, it's likely a plant image
    return green_ratio >= 0.10


# Minimum confidence threshold (%) — below this, the image is likely not a plant
CONFIDENCE_THRESHOLD = 40.0


def preprocess_image(image):
    image = image.resize((IMG_SIZE, IMG_SIZE))
    img = np.array(image) / 255.0
    img = np.expand_dims(img, axis=0)
    return img


@app.post("/predict")
async def predict(file: UploadFile = File(...)):

    if model is None:
        raise HTTPException(status_code=500, detail="Model not loaded on server")

    if file.content_type and not file.content_type.startswith("image/") and file.content_type != "application/octet-stream":
        print(f"Warning: Received file with content_type {file.content_type}")

    try:
        contents = await file.read()
        image = Image.open(io.BytesIO(contents)).convert("RGB")

        # Step 1: Check if the image looks like a plant/leaf using color analysis
        if not is_plant_image(image):
            return {
                "is_plant": False,
                "error": "This does not appear to be a plant or leaf image. Please upload a clear image of a plant leaf for disease detection."
            }

        processed = preprocess_image(image)

        prediction = model.predict(processed)
        index = int(np.argmax(prediction))
        confidence = float(np.max(prediction)) * 100

        # Step 2: Check if the model is confident enough about the prediction
        if confidence < CONFIDENCE_THRESHOLD:
            return {
                "is_plant": False,
                "error": "Could not identify a known crop disease. The uploaded image may not be a supported plant leaf. Please upload a clear image of a plant leaf."
            }

        disease = CLASS_NAMES[index]
        advice = ADVICE_MAP.get(disease, "Consult an agricultural expert for treatment.")

        return {
            "is_plant": True,
            "disease": disease,
            "confidence": f"{confidence:.2f}%",
            "advice": advice
        }

    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))
