import torch
import torch.nn as nn
from torchvision import models, transforms
from PIL import Image
import os
from pathlib import Path

BASE_DIR = Path(__file__).resolve().parent.parent
MODEL_PATH = BASE_DIR / "models" / "rice.pth"
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model = models.convnext_small()

numfeatures = model.classifier[2].in_features
model.classifier[2] = nn.Linear(numfeatures, 3)

if not MODEL_PATH.exists():
    raise FileNotFoundError(f"Model checkpoint not found: {MODEL_PATH}")

model.load_state_dict(torch.load(MODEL_PATH, map_location=device))
model = model.to(device)
model.eval()

result_map = { 
    'brown_spot': 'Positive (Brown Spot)',
    'healthy': 'Negative (Healthy)',
    'unknown': 'Unknown'
}

if MODEL_PATH.exists():
    print("Model loaded successfully.")
else:
    print(f"Model file not found: {MODEL_PATH}")

transform = transforms.Compose([  
    transforms.Resize(256, interpolation=transforms.InterpolationMode.BICUBIC),
    transforms.CenterCrop(224),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
])
image_dir = "images"
class_names = ['brown_spot', 'healthy', 'unknown']

def predict_image(image_dir, model, class_names):
    image = Image.open(image_dir).convert('RGB')
    image = transform(image).unsqueeze(0).to(device)
    with torch.no_grad():
        outputs = model(image)
        probability = torch.nn.functional.softmax(outputs, dim=1)[0]
        confidence_score = torch.max(probability) * 100
        predicted = torch.argmax(probability).item()
        folder_name = class_names[predicted]
        if confidence_score < 75:
            folder_name = 'unknown'
        return result_map.get(folder_name, 'Unknown'), confidence_score

if os.path.exists(image_dir):
    image_files = [f for f in os.listdir(image_dir) if f.lower().endswith(('.jpg', '.jpeg', '.png'))]
    
    if not image_files:
        print(f"No image files found in {image_dir}")
    else:
        print(f"Found {len(image_files)} images\n")
        for img_file in image_files:
            img_full_path = os.path.join(image_dir, img_file)
            try:
                result, confidence = predict_image(img_full_path, model, class_names)
                print(f"Image: {img_file}")
                print(f"Classfication: {result}, Confidence: {confidence:.2f}%\n")
            except Exception as e:
                print("Error processing")
else:
    print("Path does not exist")