from fastapi import FastAPI, HTTPException, Query
from fastapi.middleware.cors import CORSMiddleware
import cv2
import numpy as np
import mediapipe as mp
from sklearn.cluster import KMeans
import colorsys
import requests
from io import BytesIO
from PIL import Image

app = FastAPI()

# Enable CORS (Allow frontend to call the backend)
app.add_middleware(
    CORSMiddleware,
    allow_origins=["https://www.tailorloop.com"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Load Mediapipe Face Detection
mp_face_detection = mp.solutions.face_detection

def get_face_region(img):
    with mp_face_detection.FaceDetection(model_selection=1, min_detection_confidence=0.5) as face_detection:
        results = face_detection.process(cv2.cvtColor(img, cv2.COLOR_BGR2RGB))
        if results.detections:
            bbox = results.detections[0].location_data.relative_bounding_box
            h, w, _ = img.shape
            x1, y1 = int(bbox.xmin * w), int(bbox.ymin * h)
            x2, y2 = int((bbox.xmin + bbox.width) * w), int((bbox.ymin + bbox.height) * h)
            return (x1, y1, x2, y2)
    return None

def get_avg_color(region):
    return np.mean(region, axis=(0, 1)).astype(int)

def rgb_to_lab(rgb):
    return cv2.cvtColor(np.uint8([[rgb]]), cv2.COLOR_RGB2LAB)[0][0]

def rgb_to_hsv(rgb):
    return colorsys.rgb_to_hsv(*np.array(rgb, dtype=np.float32) / 255.0)

def classify_skin_tone(rgb):
    l = rgb_to_lab(rgb)[0]
    return "Fair" if l > 75 else "Medium" if l > 50 else "Dark"

def classify_undertone(rgb):
    h, s, _ = rgb_to_hsv(rgb)
    h *= 360
    if s < 0.25:
        return "Neutral"
    return "Cool" if 180 <= h <= 270 else "Warm"

def classify_intensity(rgb):
    _, s, v = rgb_to_hsv(rgb)
    return "High" if s > 0.5 and v > 0.6 else "Low"

def classify_contrast(skin_rgb, hair_hsv):
    return "High" if abs(rgb_to_lab(skin_rgb)[0] - hair_hsv[2]) > 50 else "Low"

def determine_season(skin_tone, undertone, intensity, contrast):
    if skin_tone == "Fair":
        if undertone == "Warm":
            if intensity == "High" and contrast == "High":
                return "Bright Spring"
            elif intensity == "Low":
                return "Light Spring"
            else:
                return "True Spring"
        elif undertone == "Cool":
            if intensity == "High" or contrast == "High":
                return "Bright Winter"
            elif intensity == "Low":
                return "Soft Summer"
            else:
                return "True Summer"
        else:
            if intensity == "Low":
                return "Light Spring"
            else:
                return "Bright Spring"
    
    elif skin_tone == "Medium":
        if undertone == "Warm":
            if intensity == "Low":
                return "Soft Autumn"
            elif contrast == "High":
                return "True Autumn"
            else:
                return "Light Spring"
        elif undertone == "Cool":
            if contrast == "High":
                return "Bright Winter"
            elif intensity == "Low":
                return "Soft Summer"
            else:
                return "True Summer"
        else:
            if intensity == "Low":
                return "Soft Autumn"
            else:
                return "True Autumn"
    
    elif skin_tone == "Dark":
        if undertone == "Warm":
            if intensity == "Low":
                return "Deep Autumn"
            else:
                return "True Autumn"
        elif undertone == "Cool":
            if contrast == "High":
                return "True Winter"
            elif intensity == "Low":
                return "Deep Winter"
            else:
                return "Bright Winter"
        else:
            if intensity == "Low":
                return "Deep Winter"
            else:
                return "Bright Winter"
    
    return "Unknown Season"

@app.get("/")
def read_root():
    return {"message": "12-Season Analysis API is running!"}

@app.get("/analyze/")
def analyze_image(image_url: str = Query(..., description="URL of the image to analyze")):
    """
    Secure Proxy API: Fetches the image from the given URL, processes it, 
    and returns skin tone, undertone, intensity, contrast, and season.
    """
    try:
        # Securely fetch the image (frontend never sees the external URL)
        response = requests.get(image_url, timeout=10)
        if response.status_code != 200:
            raise HTTPException(status_code=400, detail="Failed to download image.")

        img = np.array(Image.open(BytesIO(response.content)).convert("RGB"))
        img = cv2.cvtColor(img, cv2.COLOR_RGB2BGR)

    except Exception as e:
        raise HTTPException(status_code=400, detail=f"Error processing image: {str(e)}")

    face_region = get_face_region(img)
    if not face_region:
        raise HTTPException(status_code=400, detail="No face detected in the image.")

    x1, y1, x2, y2 = face_region
    cheek_region = img[y1 + (y2 - y1) // 3 : y2 - (y2 - y1) // 3, x1 + (x2 - x1) // 3 : x2 - (x2 - x1) // 3]
    hair_region = img[max(0, y1 - 50):y1, x1:x2]

    skin_rgb = get_avg_color(cv2.cvtColor(cheek_region, cv2.COLOR_BGR2RGB))
    hair_hsv = KMeans(n_clusters=1).fit(cv2.cvtColor(hair_region, cv2.COLOR_BGR2HSV).reshape(-1, 3)).cluster_centers_[0]

    skin_tone = classify_skin_tone(skin_rgb)
    undertone = classify_undertone(skin_rgb)
    intensity = classify_intensity(skin_rgb)
    contrast = classify_contrast(skin_rgb, hair_hsv)

    user_season = determine_season(skin_tone, undertone, intensity, contrast)

    return {
        "Skin Tone": skin_tone,
        "Undertone": undertone,
        "Intensity": intensity,
        "Contrast": contrast,
        "Season": user_season
    }
