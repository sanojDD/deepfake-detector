import torch
import torch.nn as nn
from fastapi import FastAPI, UploadFile, File
from torchvision import models
import io
import gc
from utils import process_and_predict
from fastapi.middleware.cors import CORSMiddleware

app = FastAPI()

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"], # Change to your Vercel URL later
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

def load_v2_model():
    # Load architecture without weights to save initial RAM
    model = models.efficientnet_b0(weights=None)
    num_ftrs = model.classifier[1].in_features
    model.classifier[1] = nn.Sequential(
        nn.Linear(num_ftrs, 512),
        nn.ReLU(),
        nn.Dropout(0.3),
        nn.Linear(512, 2)
    )
    # Map to CPU
    model.load_state_dict(torch.load("deepfake_detection_v2.pt", map_location='cpu'))
    model.eval()
    return model

@app.get("/")
def home():
    return {"message": "Deepfake Detection API is Online"}

@app.post("/predict")
async def predict(file: UploadFile = File(...)):
    # Load model locally for this request
    detector_model = load_v2_model()
    
    try:
        image_data = io.BytesIO(await file.read())
        # The inference happens here
        result, confidence = process_and_predict(image_data, detector_model)
        return {"prediction": result, "confidence": confidence}
    
    except Exception as e:
        return {"prediction": "Error", "confidence": str(e)}
        
    finally:
        # Clear model from memory
        del detector_model
        gc.collect()