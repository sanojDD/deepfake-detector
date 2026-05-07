import torch
from PIL import Image
from facenet_pytorch import MTCNN
import gc

# Force CPU for compatibility
device = torch.device('cpu')

def process_and_predict(image_bytes, model):
    # Initialize MTCNN inside the function to save RAM
    mtcnn = MTCNN(image_size=224, margin=20, keep_all=False, device=device)
    
    try:
        img = Image.open(image_bytes).convert('RGB')
        face = mtcnn(img)
        
        if face is None: 
            return None, "No face detected"

        face_tensor = face.unsqueeze(0).to(device)
        
        with torch.no_grad():
            outputs = model(face_tensor)
            probs = torch.nn.functional.softmax(outputs, dim=1)
            confidence, predicted = torch.max(probs, 1)
            
            label = "REAL" if predicted.item() == 1 else "FAKE"
            conf_score = f"{confidence.item() * 100:.2f}%"
            
        return label, conf_score
    
    finally:
        # Clean up local references to free RAM
        del mtcnn
        gc.collect()