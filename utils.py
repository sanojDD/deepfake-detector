import torch
from PIL import Image
from facenet_pytorch import MTCNN

# Force CPU for compatibility
device = torch.device('cpu')
mtcnn = MTCNN(image_size=224, margin=20, keep_all=False, device=device)

def process_and_predict(image_bytes, model):
    img = Image.open(image_bytes).convert('RGB')
    face = mtcnn(img)
    
    if face is None: 
        return None, "No face detected"

    face_tensor = face.unsqueeze(0).to(device)
    
    with torch.no_grad():
        outputs = model(face_tensor)
        probs = torch.nn.functional.softmax(outputs, dim=1)
        confidence, predicted = torch.max(probs, 1)
        
        # 0 = FAKE, 1 = REAL (matches your 99.97% test)
        label = "REAL" if predicted.item() == 1 else "FAKE"
        conf_score = f"{confidence.item() * 100:.2f}%"
        
    return label, conf_score
