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

recommendations = {
    'brown_spot': [
        "🍄 Spray fungicide (Mancozeb or Propiconazole) on affected plants immediately best applied early morning or late afternoon.",
        "🌾 Stop or reduce urea fertilizer application. Too much nitrogen weakens the plant and worsens the disease.",
        "💧 Keep water level at 2–5 cm in the field. Do not let the soil dry out completely.",
        "🚜 After harvest, burn or bury leftover plant stalks to stop the disease from spreading to the next crop.",
        "🌱 For your next planting, switch to resistant varieties like NSIC Rc222 or PSB Rc18 to reduce the risk of brown spot.",
        "🔄 Consider crop rotation next season plant mung beans or corn to break the disease cycle and let the soil recover.",
    ],
    'healthy': [
        "✅ Your rice plant is healthy keep up the good work!",
        "👁️ Check your field weekly for early signs of brown spot: oval brown patches with yellow edges on the leaves.",
        "💧 Maintain steady water levels and continue your regular fertilizer schedule.",
    ],
    'unknown': [
        "⚠️ The image is not clear enough for a confident result.",
        "📷 Try taking a new photo in a well-lit area and make sure the rice leaf is clearly visible.",
        "🔍 If you suspect your plant is sick, reach out to your local agriculturist or extension worker for help.",
    ]
}

reasoning = {
    'brown_spot': "The AI detected visual patterns consistent with Brown Spot disease oval to circular brown lesions typically surrounded by yellow halos on the leaf surface. These are characteristic markers of Bipolaris oryzae fungal infection, which thrives in conditions of poor soil nutrition and excessive moisture.",
    'healthy': "The AI found no signs of disease. The leaf shows uniform green pigmentation with no visible lesions, discoloration, or abnormal spotting. The color and texture patterns closely match those of a well-nourished, disease-free rice plant.",
    'unknown': "The AI could not identify a clear pattern with sufficient confidence. This may be due to image quality, unusual lighting, or the leaf not being clearly visible. A cleaner photo taken in natural daylight will give better results."
}

def get_recommendation(folder_name):
    return recommendations.get(folder_name, recommendations['unknown'])

def get_reasoning(folder_name):
    return reasoning.get(folder_name, reasoning['unknown'])

transform = transforms.Compose([
    transforms.Resize(256, interpolation=transforms.InterpolationMode.BICUBIC),
    transforms.CenterCrop(224),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
])
image_dir = "images"
class_names = ['brown_spot', 'healthy', 'unknown']

def predict_image(image_input, model, class_names):
    image = Image.open(image_input).convert('RGB')
    image = transform(image).unsqueeze(0).to(device)
    with torch.no_grad():
        outputs = model(image)
        probability = torch.nn.functional.softmax(outputs, dim=1)[0]
        confidence_score = torch.max(probability) * 100
        predicted = torch.argmax(probability).item()
        folder_name = class_names[predicted]
        if confidence_score < 75:
            folder_name = 'unknown'
        recommendation = get_recommendation(folder_name)
        reason = get_reasoning(folder_name)
        return result_map.get(folder_name, 'Unknown'), confidence_score, recommendation, reason

if os.path.exists(image_dir):
    image_files = [f for f in os.listdir(image_dir) if f.lower().endswith(('.jpg', '.jpeg', '.png'))]
    if not image_files:
        print(f"No image files found in {image_dir}")
    else:
        print(f"Found {len(image_files)} images\n")
        for img_file in image_files:
            img_full_path = os.path.join(image_dir, img_file)
            try:
                result, confidence, recommendation, reason = predict_image(img_full_path, model, class_names)
                print(f"Image: {img_file}")
                print(f"Classification: {result}, Confidence: {confidence:.2f}%")
                print(f"Reasoning: {reason}")
                print("Recommendations:")
                for r in recommendation:
                    print(f"  {r}")
                print()
            except Exception as e:
                print(f"Error processing {img_file}: {e}")
else:
    print("Path does not exist")