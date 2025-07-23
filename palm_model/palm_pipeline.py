from PIL import Image
import torch
import torch.nn as nn
from torchvision import transforms, models
import base64
from io import BytesIO

# === Load MobileNetV2 ===
PALM_MODEL_PATH = "./models/mobilenet_model.pth"

# Use pretrained MobileNetV2 structure
model = models.mobilenet_v2(weights=None)  # No pretrained weights
model.classifier[1] = nn.Linear(model.last_channel, 2)  # Assuming binary output
model.load_state_dict(torch.load(PALM_MODEL_PATH, map_location='cpu'))
model.eval()

# Image transform
transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
])

def pil_to_base64(pil_image):
    """Convert PIL image to base64 string"""
    buffered = BytesIO()
    pil_image.save(buffered, format="JPEG")
    img_str = base64.b64encode(buffered.getvalue()).decode()
    return img_str

def run_palm_pipeline(image_pil, palm_side="unknown"):
    """
    Accepts a PIL palm image and returns detailed prediction results
    
    Args:
        image_pil: PIL Image of palm
        palm_side: "left" or "right" for identification
    
    Returns:
        dict: Detailed palm analysis results
    """
    try:
        # Resize image for consistent processing
        processed_image = image_pil.resize((224, 224))
        
        # Run prediction
        image_tensor = transform(image_pil).unsqueeze(0)
        with torch.no_grad():
            output = model(image_tensor)
            probs = torch.softmax(output, dim=1)
            class_idx = torch.argmax(probs).item()
            confidence = float(probs[0][class_idx].item())

        prediction = "anemic" if class_idx == 1 else "non_anemic"
        
        return {
            "model_type": "palm_analysis",
            "palm_side": palm_side,
            "prediction": prediction,
            "confidence": round(confidence * 100, 2),
            "raw_probabilities": {
                "non_anemic": round(float(probs[0][0].item()) * 100, 2),
                "anemic": round(float(probs[0][1].item()) * 100, 2)
            },
            "processed_image_base64": pil_to_base64(processed_image),
            "original_image_base64": pil_to_base64(image_pil),
            "success": True
        }

    except Exception as e:
        return {
            "model_type": "palm_analysis",
            "palm_side": palm_side,
            "success": False,
            "error": str(e),
            "prediction": "error",
            "confidence": 0.0
        }

################################################################################################################################
