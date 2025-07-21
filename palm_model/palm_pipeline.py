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
##for single palm image 


# from PIL import Image
# import torch
# import torch.nn as nn
# from torchvision import transforms, models
# import numpy as np
# import cv2
# import os

# # === Load MobileNetV2 ===
# PALM_MODEL_PATH = "./models/mobilenet_model.pth"

# # Use pretrained MobileNetV2 structure
# model = models.mobilenet_v2(weights=None)  # No pretrained weights
# model.classifier[1] = nn.Linear(model.last_channel, 2)  # Assuming binary output
# model.load_state_dict(torch.load(PALM_MODEL_PATH, map_location='cpu'))
# model.eval()

# # Image transform
# transform = transforms.Compose([
#     transforms.Resize((224, 224)),
#     transforms.ToTensor(),
# ])

# def detect_and_crop_palm(image_pil):
#     """
#     Detect and crop palm region from the image
#     For now, this is a simple center crop - you can replace with actual palm detection
    
#     Args:
#         image_pil: PIL Image object
        
#     Returns:
#         PIL Image: Cropped palm region
#     """
#     try:
#         # Convert PIL to numpy array for processing
#         img_array = np.array(image_pil)
        
#         # Get image dimensions
#         height, width = img_array.shape[:2]
        
#         # Simple center crop approach (replace with actual palm detection if available)
#         # Assuming palm is in the center 60% of the image
#         crop_size = min(height, width) * 0.6
#         center_x, center_y = width // 2, height // 2
        
#         # Calculate crop boundaries
#         half_size = int(crop_size // 2)
#         x1 = max(0, center_x - half_size)
#         y1 = max(0, center_y - half_size)
#         x2 = min(width, center_x + half_size)
#         y2 = min(height, center_y + half_size)
        
#         # Crop the image
#         cropped_array = img_array[y1:y2, x1:x2]
        
#         # Convert back to PIL Image
#         cropped_pil = Image.fromarray(cropped_array)
        
#         print(f"✅ Palm cropped successfully: {cropped_pil.size}")
#         return cropped_pil
        
#     except Exception as e:
#         print(f"❌ Error cropping palm: {e}")
#         # Return original image if cropping fails
#         return image_pil

# def run_palm_pipeline(image_pil):
#     """
#     Process palm image for anemia detection
    
#     Args:
#         image_pil: PIL Image object containing palm image
        
#     Returns:
#         dict: Results containing prediction, confidence, and cropped palm
#     """
#     try:
#         print("🚀 Starting palm pipeline...")
        
#         # Detect and crop palm region
#         cropped_palm = detect_and_crop_palm(image_pil)
        
#         # Transform the cropped palm for model input
#         image_tensor = transform(cropped_palm).unsqueeze(0)
        
#         # Run inference
#         with torch.no_grad():
#             output = model(image_tensor)
#             probs = torch.softmax(output, dim=1)
#             class_idx = torch.argmax(probs).item()
#             confidence = float(probs[0][class_idx].item())

#         # Determine result
#         result = "anemic" if class_idx == 1 else "non-anemic"
        
#         print(f"🤚 Palm result: {result} (confidence: {confidence:.2f})")
        
#         return {
#             "result": result,
#             "confidence": round(confidence, 2),
#             "cropped_palm": cropped_palm  # PIL Image for base64 conversion
#         }
        
#     except Exception as e:
#         print(f"❌ Error in palm pipeline: {e}")
#         return {
#             "error": f"palm_processing_failed: {str(e)}",
#             "result": "unknown",
#             "confidence": 0.0,
#             "cropped_palm": None
#         }

# # === Alternative: Advanced palm detection function (if you have palm detection model) ===
# def detect_palm_with_model(image_pil, detection_model=None):
#     """
#     Advanced palm detection using a detection model (YOLO/etc.)
#     Replace detect_and_crop_palm() with this if you have a palm detection model
    
#     Args:
#         image_pil: PIL Image object
#         detection_model: Palm detection model (YOLO, etc.)
        
#     Returns:
#         PIL Image: Cropped palm region or original image if detection fails
#     """
#     if detection_model is None:
#         print("⚠️ No detection model provided, falling back to center crop")
#         return detect_and_crop_palm(image_pil)
    
#     try:
#         # Convert PIL to numpy array
#         img_array = np.array(image_pil)
        
#         # Run palm detection (example for YOLO-like model)
#         # results = detection_model.predict(img_array, conf=0.25, verbose=False)
        
#         # Extract bounding box and crop
#         # This is pseudocode - replace with actual detection logic
#         # if len(results) > 0 and results[0].boxes is not None:
#         #     boxes = results[0].boxes.xyxy.cpu().numpy()
#         #     if len(boxes) > 0:
#         #         x1, y1, x2, y2 = map(int, boxes[0])
#         #         cropped = img_array[y1:y2, x1:x2]
#         #         return Image.fromarray(cropped)
        
#         # Fallback to center crop if detection fails
#         return detect_and_crop_palm(image_pil)
        
#     except Exception as e:
#         print(f"❌ Error in advanced palm detection: {e}")
#         return detect_and_crop_palm(image_pil)

# # === Legacy function for backward compatibility ===
# def run_palm_pipeline_legacy(image_pil):
#     """
#     Legacy palm pipeline without cropped palm return
#     For backward compatibility with existing code
#     """
#     try:
#         image_tensor = transform(image_pil).unsqueeze(0)
#         with torch.no_grad():
#             output = model(image_tensor)
#             probs = torch.softmax(output, dim=1)
#             class_idx = torch.argmax(probs).item()
#             confidence = float(probs[0][class_idx].item())

#         return {
#             "result": "anemic" if class_idx == 1 else "non-anemic",
#             "confidence": round(confidence, 2)
#         }
        
#     except Exception as e:
#         print(f"❌ Error in legacy palm pipeline: {e}")
#         return {
#             "result": "unknown",
#             "confidence": 0.0
#         }

# # === Testing function ===
# def test_palm_pipeline():
#     """Test function for palm pipeline"""
#     # Test with a dummy image
#     try:
#         # Create a test image or load from file
#         test_image_path = "test_images/palm_test.jpg"
        
#         if os.path.exists(test_image_path):
#             test_image = Image.open(test_image_path).convert("RGB")
#             print(f"📁 Testing with image: {test_image_path}")
#         else:
#             # Create a dummy RGB image for testing
#             test_image = Image.new('RGB', (640, 480), color='pink')
#             print("🖼️ Testing with dummy pink image")
        
#         # Test the pipeline
#         result = run_palm_pipeline(test_image)
        
#         if "error" in result:
#             print(f"⚠️ Error: {result['error']}")
#         else:
#             print(f"Palm Result: {result['result']}")
#             print(f"Confidence: {result['confidence']}")
#             print(f"Cropped palm available: {result['cropped_palm'] is not None}")
#             if result['cropped_palm']:
#                 print(f"Cropped palm size: {result['cropped_palm'].size}")
                
#     except Exception as e:
#         print(f"❌ Test failed: {e}")

# if __name__ == "__main__":
#     print("=== Palm Pipeline Test ===")
#     test_palm_pipeline()