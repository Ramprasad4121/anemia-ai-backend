import os
import cv2
import torch
from PIL import Image
from torchvision import transforms
from ultralytics import YOLO
import numpy as np
import traceback
import base64
from io import BytesIO

torch.serialization.add_safe_globals([torch.nn.Module])

# === Setup paths ===
YOLO_MODEL_PATH = "models/yolov8_model.pt"
RESNET_MODEL_PATH = "models/resnet101_anemia_model1.pth"

# === Load models globally ===
yolo_model = None
model = None
transform = None
class_names = ["Non Anemia", "Anemia"]
models_loaded = False

def load_models():
    global yolo_model, model, transform, models_loaded
    
    try:
        # Check if model files exist
        if not os.path.exists(YOLO_MODEL_PATH):
            print(f"❌ YOLO model not found at: {YOLO_MODEL_PATH}")
            return False
            
        if not os.path.exists(RESNET_MODEL_PATH):
            print(f"❌ ResNet model not found at: {RESNET_MODEL_PATH}")
            return False
        
        print("📁 Model files found, loading...")
        
        # Load YOLO model
        print("🔄 Loading YOLO model...")
        yolo_model = YOLO(YOLO_MODEL_PATH)
        print("✅ YOLO model loaded successfully")
        
        # Load ResNet Model
        print("🔄 Loading ResNet model...")
        checkpoint = torch.load(RESNET_MODEL_PATH, map_location='cpu')
        
        from torchvision.models import resnet101
        model = resnet101(pretrained=False)
        model.fc = torch.nn.Linear(model.fc.in_features, 2)
        model.load_state_dict(checkpoint['model_state_dict'])
        model.eval()
        print("✅ ResNet model loaded successfully")
        
        # Setup preprocessing
        transform = transforms.Compose([
            transforms.Resize((224, 224)),
            transforms.ToTensor(),
        ])
        
        models_loaded = True
        print("🎉 All eye models loaded successfully!")
        return True
        
    except Exception as e:
        print(f"❌ Error loading eye models: {e}")
        print(f"Full traceback: {traceback.format_exc()}")
        models_loaded = False
        return False

# Initialize models
load_models()

def pil_to_base64(pil_image):
    """Convert PIL image to base64 string"""
    buffered = BytesIO()
    pil_image.save(buffered, format="JPEG")
    img_str = base64.b64encode(buffered.getvalue()).decode()
    return img_str

def numpy_to_base64(numpy_array):
    """Convert numpy array to base64 string"""
    pil_image = Image.fromarray(numpy_array)
    return pil_to_base64(pil_image)

# === Helper: Crop conjunctiva using YOLO ===
def crop_conjunctiva(image_input):
    if not models_loaded:
        print("❌ Models not loaded, cannot crop conjunctiva")
        return None, None
        
    try:
        # Handle both file paths and PIL Image objects
        if isinstance(image_input, str):
            # It's a file path
            if not os.path.exists(image_input):
                print(f"❌ Image file not found: {image_input}")
                return None, None
                
            image = cv2.imread(image_input)
            if image is None:
                print(f"❌ Could not load image: {image_input}")
                return None, None
            print(f"🔄 Processing image from path: {image_input}")
            
            # For YOLO prediction with file path
            results = yolo_model.predict(image_input, conf=0.05, verbose=False)
            
        elif hasattr(image_input, 'mode'):
            # It's a PIL Image object
            print("🔄 Processing PIL Image object")
            # Ensure consistent and correct conversion for YOLO input
            image_array = np.array(image_input.convert("RGB"))
            image = cv2.cvtColor(image_array, cv2.COLOR_RGB2BGR)
            results = yolo_model.predict(image_array, conf=0.05, verbose=False)
            
        else:
            print(f"❌ Unsupported image input type: {type(image_input)}")
            return None, None
        
        if len(results) == 0 or results[0].boxes is None:
            print(f"❌ No detection results")
            return None, None
            
        boxes = results[0].boxes.xyxy.cpu().numpy()
        print(f"🔍 Detected {len(boxes)} conjunctiva boxes")

        if len(boxes) == 0:
            print("⚠️ No conjunctiva detected - fallback resizing")
            return None, None

        # Only take first detection
        x1, y1, x2, y2 = map(int, boxes[0])
        print(f"📍 Detected conjunctiva at: ({x1}, {y1}, {x2}, {y2})")
        
        cropped = image[y1:y2, x1:x2]
        cropped_rgb = cv2.cvtColor(cropped, cv2.COLOR_BGR2RGB)
        
        detection_info = {
            "bounding_box": [x1, y1, x2, y2],
            "confidence": float(results[0].boxes.conf[0].item()) if results[0].boxes.conf is not None and len(results[0].boxes.conf) > 0 else 1.0
        }
        
        print(f"✅ Successfully cropped conjunctiva")
        return cropped_rgb, detection_info
        
    except Exception as e:
        print(f"❌ Error processing image: {e}")
        print(f"Full traceback: {traceback.format_exc()}")
        return None, None

# === Helper: Classify cropped image ===
def classify_anemia(cropped_img):
    if not models_loaded:
        print("❌ Models not loaded, cannot classify")
        return "Unknown", 0.0, None
        
    try:
        pil_img = Image.fromarray(cropped_img)
        input_tensor = transform(pil_img).unsqueeze(0)
        
        with torch.no_grad():
            output = model(input_tensor)
            probabilities = torch.nn.functional.softmax(output, dim=1)
            confidence, pred = torch.max(probabilities, 1)
            
            result = class_names[pred.item()]
            conf_score = confidence.item()
            
            raw_probs = {
                "non_anemic": round(float(probabilities[0][0].item()) * 100, 2),
                "anemic": round(float(probabilities[0][1].item()) * 100, 2)
            }
            
            print(f"🔍 Classification result: {result} (confidence: {conf_score:.2f})")
            return result, conf_score, raw_probs
            
    except Exception as e:
        print(f"❌ Error classifying image: {e}")
        print(f"Full traceback: {traceback.format_exc()}")
        return "Unknown", 0.0, None

# === Single eye processing function ===
def process_single_eye(eye_input, eye_side="unknown"):
    """
    Process a single eye image and return detailed results
    
    Args:
        eye_input: Path to eye image OR PIL Image object
        eye_side: "left" or "right" for identification
    
    Returns:
        dict: Detailed eye analysis results
    """
    if not models_loaded:
        return {
            "model_type": "eye_analysis",
            "eye_side": eye_side,
            "success": False,
            "error": "eye_model_not_available",
            "prediction": "error",
            "confidence": 0.0
        }
    
    try:
        # Check if it's a valid input
        is_valid_input = False
        if isinstance(eye_input, str) and os.path.exists(eye_input):
            is_valid_input = True
            print(f"👁️ Processing {eye_side} eye from path: {eye_input}")
        elif hasattr(eye_input, 'mode'):  # PIL Image check
            is_valid_input = True
            print(f"👁️ Processing {eye_side} eye from PIL Image")
        
        if not is_valid_input:
            return {
                "model_type": "eye_analysis",
                "eye_side": eye_side,
                "success": False,
                "error": f"Invalid input type: {type(eye_input)}",
                "prediction": "error",
                "confidence": 0.0
            }
        
        # Get original image as base64
        if isinstance(eye_input, str):
            original_pil = Image.open(eye_input)
        else:
            original_pil = eye_input
        
        # Crop conjunctiva
        cropped_conjunctiva, detection_info = crop_conjunctiva(eye_input)
        print(f"🔍 Detection Info: {detection_info}")
        if cropped_conjunctiva is None:
            print(f"⚠️ No conjunctiva detected in {eye_side} eye.")
            return {
                "model_type": "eye_analysis",
                "eye_side": eye_side,
                "success": False,
                "error": "no_conjunctiva_detected",
                "prediction": "error",
                "confidence": 0.0,
                "cropped_conjunctiva_base64": None
            }
        
        # Classify anemia
        result, confidence, raw_probs = classify_anemia(cropped_conjunctiva)
        
        if result == "Unknown":
            return {
                "model_type": "eye_analysis",
                "eye_side": eye_side,
                "success": False,
                "error": "classification_failed",
                "prediction": "error",
                "confidence": 0.0,
                "cropped_conjunctiva_base64": numpy_to_base64(cropped_conjunctiva)
            }
        
        prediction = "anemic" if result == "Anemia" else "non_anemic"
        
        return {
            "model_type": "eye_analysis",
            "eye_side": eye_side,
            "success": True,
            "prediction": prediction,
            "confidence": round(confidence * 100, 2),
            "raw_probabilities": raw_probs,
            "detection_info": detection_info,
            "cropped_conjunctiva_base64": numpy_to_base64(cropped_conjunctiva)
        }
        
    except Exception as e:
        print(f"❌ Error processing {eye_side} eye: {e}")
        return {
            "model_type": "eye_analysis",
            "eye_side": eye_side,
            "success": False,
            "error": str(e),
            "prediction": "error",
            "confidence": 0.0
        }

# === Main function for API integration ===
def run_eye_pipeline(left_eye_input, right_eye_input):
    """
    Process both eye images and return detailed results
    
    Args:
        left_eye_input: Path to left eye image OR PIL Image object
        right_eye_input: Path to right eye image OR PIL Image object
    
    Returns:
        dict: Results containing detailed predictions for both eyes
    """
    print("🚀 Starting eye pipeline...")
    
    if not models_loaded:
        return {
            "model_type": "eye_analysis",
            "success": False,
            "error": "eye_model_not_available",
            "left_eye": None,
            "right_eye": None,
            "overall_prediction": "error",
            "overall_confidence": 0.0
        }
    
    # Process individual eyes
    left_result = process_single_eye(left_eye_input, "left") if left_eye_input is not None else None
    right_result = process_single_eye(right_eye_input, "right") if right_eye_input is not None else None
    
    # Calculate overall result
    valid_predictions = []
    valid_confidences = []
    
    if left_result and left_result["success"] and left_result["prediction"] != "error":
        if left_result["prediction"] == "anemic":
            valid_predictions.append(1)
        else:
            valid_predictions.append(0)
        valid_confidences.append(left_result["confidence"] / 100.0)
    
    if right_result and right_result["success"] and right_result["prediction"] != "error":
        if right_result["prediction"] == "anemic":
            valid_predictions.append(1)
        else:
            valid_predictions.append(0)
        valid_confidences.append(right_result["confidence"] / 100.0)
    
    # Determine final diagnosis
    if not valid_predictions:
        overall_prediction = "error"
        overall_confidence = 0.0
        diagnosis_note = "No valid eye predictions available"
    else:
        anemic_count = valid_predictions.count(1)
        non_anemic_count = valid_predictions.count(0)
        overall_prediction = "anemic" if anemic_count > non_anemic_count else "non_anemic"
        overall_confidence = round(np.mean(valid_confidences) * 100, 2)
        diagnosis_note = f"Based on {len(valid_predictions)} eyes: {anemic_count} anemic, {non_anemic_count} non-anemic"
    
    return {
        "model_type": "eye_analysis",
        "success": True,
        "left_eye": left_result,
        "right_eye": right_result,
        "overall_prediction": overall_prediction,
        "overall_confidence": overall_confidence,
        "diagnosis_note": diagnosis_note,
        "eyes_processed": len(valid_predictions)
    }

# === For standalone testing ===
def process_both_eyes():
    """Original function for standalone testing"""
    left_path = "test_images/left_eye.jpeg"
    right_path = "test_images/right_eye.jpeg"
    
    result = run_eye_pipeline(left_path, right_path)
    
    if not result["success"]:
        print(f"⚠️ Error: {result.get('error', 'Unknown error')}")
    else:
        print(f"Overall Prediction: {result['overall_prediction']}")
        print(f"Overall Confidence: {result['overall_confidence']}%")

# === Debugging function ===
def debug_models():
    """Debug function to check model status"""
    print("=== EYE MODEL DEBUG INFO ===")
    print(f"Models loaded: {models_loaded}")
    print(f"YOLO model path: {YOLO_MODEL_PATH}")
    print(f"ResNet model path: {RESNET_MODEL_PATH}")
    print(f"YOLO model exists: {os.path.exists(YOLO_MODEL_PATH)}")
    print(f"ResNet model exists: {os.path.exists(RESNET_MODEL_PATH)}")
    print(f"YOLO model object: {yolo_model}")
    print(f"ResNet model object: {model}")
    print("===========================")

if __name__ == "__main__":
    debug_models()
    process_both_eyes()
