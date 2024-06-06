from fastapi import FastAPI, File, UploadFile, HTTPException, Request
from fastapi.middleware.cors import CORSMiddleware
import uvicorn
import numpy as np
from io import BytesIO
from PIL import Image, UnidentifiedImageError
import tensorflow as tf
import os

app = FastAPI()
origins = [
    "http://localhost",
    "http://localhost:3000",
]

app.add_middleware(
    CORSMiddleware,
    allow_origins=origins,
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

@app.get("/ping")
async def ping():
    return {"message": "hello, I am alive"}

print("Current Working Directory:", os.getcwd())

MODEL_PATH = "C:/Users/Dell/Documents/ML + Dev/Potato Project/api/potatoes.h5"
if not os.path.exists(MODEL_PATH):
    print(f"Model file not found at: {MODEL_PATH}")
MODEL = tf.keras.models.load_model(MODEL_PATH)

CLASS_NAMES = ["Early Blight", "Late Blight", "Healthy"]

def read_file_as_image(data: bytes) -> np.ndarray:
    try:
        image = Image.open(BytesIO(data))
        print(f"Image format: {image.format}, size: {image.size}, mode: {image.mode}")
        return np.array(image)
    except UnidentifiedImageError as e:
        print(f"Error: {e}")
        raise ValueError("Uploaded file is not a valid image.")
    except Exception as e:
        print(f"Unexpected error: {e}")
        raise

@app.post("/predict")
async def predict(request: Request, file: UploadFile = File(...)):
    print(f"Received file: {file.filename}")
    
    # Print request headers for debugging
    headers = request.headers
    print(f"Request Headers: {headers}")
    
    # Debugging content_type
    print(f"File content_type: {file.content_type}")
    
    if file.content_type is None:
        print("File content type is None")
        raise HTTPException(status_code=400, detail="Could not determine the file content type. Make sure you are uploading an image file.")
    
    if not file.content_type.startswith("image/"):
        print(f"File content type '{file.content_type}' does not start with 'image/'")
        raise HTTPException(status_code=400, detail="Uploaded file is not an image")
    
    data = await file.read()
    print(f"File size: {len(data)} bytes")
    
    if len(data) == 0:
        raise HTTPException(status_code=400, detail="Uploaded file is empty")

    try:
        image = read_file_as_image(data)
        if image is None:
            raise ValueError("Could not convert file to image.")
    except ValueError as e:
        raise HTTPException(status_code=400, detail=str(e))
    
    img_batch = np.expand_dims(image, 0)
    
    predictions = MODEL.predict(img_batch)
    predicted_class = CLASS_NAMES[np.argmax(predictions[0])]
    confidence = np.max(predictions[0])
    
    return {
        'class': predicted_class,
        'confidence': float(confidence)
    }

if __name__ == "__main__":
    uvicorn.run(app, host='127.0.0.1', port=8000)
