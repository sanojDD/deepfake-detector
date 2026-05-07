import torch
import torch.nn as nn
from fastapi import FastAPI, UploadFile, File
from torchvision import models
import io
from utils import process_and_predict
from fastapi.middleware.cors import CORSMiddleware

app = FastAPI()

app.add_middleware(
    CORSMiddleware,
    allow_origins=[
        "http://localhost:5173",  # Your React dev URL
        "http://127.0.0.1:5173",
    ],
    allow_credentials=True,
    allow_methods=["*"],  # Allows GET, POST, etc.
    allow_headers=["*"],  # Allows all headers
)

def load_v2_model():
    model = models.efficientnet_b0()
    num_ftrs = model.classifier[1].in_features
    model.classifier[1] = nn.Sequential(
        nn.Linear(num_ftrs, 512),
        nn.ReLU(),
        nn.Dropout(0.3),
        nn.Linear(512, 2)
    )
    model.load_state_dict(torch.load("deepfake_detection_v2.pt", map_location='cpu'))
    model.eval()
    return model

detector_model = load_v2_model()

@app.get("/")
def home():
    return {"message": "Deepfake Detection API is Online"}

@app.post("/predict")
async def predict(file: UploadFile = File(...)):
    image_data = io.BytesIO(await file.read())
    result, confidence = process_and_predict(image_data, detector_model)
    return {"prediction": result, "confidence": confidence}
