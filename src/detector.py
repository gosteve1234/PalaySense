import torch
import torch.nn as nn
from torchvision import models, transforms
from PIL import Image
from pathlib import Path

BASE_DIR = Path(__file__).resolve().parent.parent
MODEL_PATH = BASE_DIR / "models" / "rice.pth"
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

if not MODEL_PATH.exists():
    raise FileNotFoundError(f"Model checkpoint not found: {MODEL_PATH}")

# Load checkpoint — classes and img_size are saved inside it
checkpoint = torch.load(MODEL_PATH, map_location=device)
class_names = checkpoint["classes"]        # e.g. ['Healthy', 'Light', 'Moderate', 'Severe']
img_size    = checkpoint.get("img_size", 224)

model = models.convnext_small()
in_features = model.classifier[2].in_features
model.classifier[2] = nn.Linear(in_features, len(class_names))
model.load_state_dict(checkpoint["model_state"])
model = model.to(device)
model.eval()

# ── Labels ────────────────────────────────────────────────────────────────────

result_map = {
    "Healthy":  "Negative (Healthy)",
    "Light":    "Positive – Light Brown Spot",
    "Moderate": "Positive – Moderate Brown Spot",
    "Severe":   "Positive – Severe Brown Spot",
    "unknown":  "Unknown",
}

# ── Recommendations ───────────────────────────────────────────────────────────

recommendations = {
    "Healthy": [
        "✅ Your rice plant is healthy — keep up the good work!",
        "👁️ Check your field weekly for early signs of brown spot: oval brown patches with yellow edges on the leaves.",
        "💧 Maintain steady water levels and continue your regular fertilizer schedule.",
    ],
    "Light": [
        "🔍 Early-stage brown spot detected. Monitor the affected plants daily for any spreading.",
        "🌾 Slightly reduce urea/nitrogen fertilizer — excess nitrogen weakens plants and encourages fungal growth.",
        "💧 Keep water level at 2–5 cm; avoid letting the soil dry out completely.",
        "🌱 Consider applying a preventive fungicide (e.g., Mancozeb) if the affected area grows.",
    ],
    "Moderate": [
        "🍄 Apply a systemic fungicide (Propiconazole or Mancozeb) promptly, best done early morning or late afternoon.",
        "🌾 Stop or significantly reduce urea fertilizer application to slow disease progression.",
        "💧 Keep water level at 2–5 cm and avoid water stress.",
        "🚜 Remove and dispose of heavily infected leaves away from the field.",
        "🔄 Plan for crop rotation next season (e.g., mung beans or corn) to break the disease cycle.",
    ],
    "Severe": [
        "🚨 Severe brown spot detected. Act immediately to prevent total crop loss.",
        "🍄 Spray a systemic fungicide (Propiconazole or Tricyclazole) as soon as possible — repeat after 7–10 days.",
        "🌾 Halt all nitrogen/urea fertilizer application at once.",
        "🚜 After harvest, burn or deeply bury all leftover plant stalks to prevent the disease from surviving to the next crop.",
        "🌱 For your next planting, switch to resistant varieties like NSIC Rc222 or PSB Rc18.",
        "👨‍🌾 Contact your local agriculturist or extension worker for on-site assessment and support.",
    ],
    "unknown": [
        "⚠️ The image is not clear enough for a confident result.",
        "📷 Try taking a new photo in a well-lit area and make sure the rice leaf is clearly visible.",
        "🔍 If you suspect your plant is sick, reach out to your local agriculturist or extension worker for help.",
    ],
}

# ── Reasoning ─────────────────────────────────────────────────────────────────

reasoning = {
    "Healthy": (
        "The AI found no signs of disease. The leaf shows uniform green pigmentation with no visible "
        "lesions, discoloration, or abnormal spotting — closely matching a well-nourished, disease-free rice plant."
    ),
    "Light": (
        "The AI detected faint early-stage patterns associated with Brown Spot disease: small, sparse "
        "oval lesions that may have slight brown discoloration. The infection is limited and has not "
        "yet spread significantly across the leaf surface."
    ),
    "Moderate": (
        "The AI identified a moderate presence of Brown Spot disease markers — multiple oval to circular "
        "brown lesions, some surrounded by yellow halos, covering a noticeable portion of the leaf. "
        "These are characteristic of an active Bipolaris oryzae fungal infection."
    ),
    "Severe": (
        "The AI detected extensive Brown Spot lesions covering a large area of the leaf surface. "
        "Dense clusters of brown, necrotic spots with yellow halos are present, indicating advanced "
        "Bipolaris oryzae infection. Immediate treatment is critical to prevent further crop damage."
    ),
    "unknown": (
        "The AI could not identify a clear pattern with sufficient confidence. This may be due to "
        "image quality, unusual lighting, or the leaf not being clearly visible. A cleaner photo "
        "taken in natural daylight will give better results."
    ),
}

# ── Transform (uses img_size from checkpoint) ─────────────────────────────────

transform = transforms.Compose([
    transforms.Resize(int(img_size * 1.15), interpolation=transforms.InterpolationMode.BICUBIC),
    transforms.CenterCrop(img_size),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
])

# ── Helpers ───────────────────────────────────────────────────────────────────

def get_recommendation(label: str) -> list[str]:
    return recommendations.get(label, recommendations["unknown"])

def get_reasoning(label: str) -> str:
    return reasoning.get(label, reasoning["unknown"])

# ── Prediction ────────────────────────────────────────────────────────────────

def predict_image(image_input) -> tuple[str, float, list[str], str]:
    """
    Returns (result_label, confidence_pct, recommendations, reasoning).
    Falls back to 'unknown' when confidence < 75 %.
    """
    image = Image.open(image_input).convert("RGB")
    tensor = transform(image).unsqueeze(0).to(device)

    with torch.no_grad():
        outputs = model(tensor)
        probability = torch.nn.functional.softmax(outputs, dim=1)[0]

    confidence_score = torch.max(probability).item() * 100
    predicted_idx    = torch.argmax(probability).item()
    label            = class_names[predicted_idx]

    if confidence_score < 75:
        label = "unknown"

    return (
        result_map.get(label, "Unknown"),
        confidence_score,
        get_recommendation(label),
        get_reasoning(label),
    )
