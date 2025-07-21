import numpy as np
import cv2
from PIL import Image
from ultralytics import YOLO
import tensorflow as tf
import torch
import torch.nn as nn
from torchvision import transforms
import os
import base64
from io import BytesIO

# Model paths
YOLO_PATH = "./models/yolov8_nail_best.pt"
CLASSIFIER_PATH = "./models/nail_cnn_classifier_model.h5"
ANEMIA_MODEL_PATH = "./models/anemia_cnn_model2.pth"

# Load YOLO and Keras models only once
yolo_model = YOLO(YOLO_PATH)
nail_classifier = tf.keras.models.load_model(CLASSIFIER_PATH)

# Define PyTorch model
class SimpleCNN(nn.Module):
    def __init__(self, num_classes=2):
        super(SimpleCNN, self).__init__()
        self.features = nn.Sequential(
            nn.Conv2d(3, 16, 3, 1, 1),
            nn.ReLU(),
            nn.MaxPool2d(2, 2),
            nn.Conv2d(16, 32, 3, 1, 1),
            nn.ReLU(),
            nn.MaxPool2d(2, 2),
        )
        self.classifier = nn.Sequential(
            nn.Flatten(),
            nn.Linear(32 * 56 * 56, 128),
            nn.ReLU(),
            nn.Linear(128, num_classes)
        )
    
    def forward(self, x):
        x = self.features(x)
        x = self.classifier(x)
        return x

# Load anemia model
anemia_model = SimpleCNN()
anemia_model.load_state_dict(torch.load(ANEMIA_MODEL_PATH, map_location=torch.device("cpu")))
anemia_model.eval()

# Preprocessing
anemia_transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor()
])

def pil_to_base64(pil_image):
    """Convert PIL image to base64 string"""
    buffered = BytesIO()
    pil_image.save(buffered, format="JPEG")
    img_str = base64.b64encode(buffered.getvalue()).decode()
    return img_str

def run_nail_pipeline(image_pil):
    """
    Accepts a PIL hand image, crops nails, detects polish,
    runs anemia model, and returns detailed results for each nail.
    """
    try:
        results = yolo_model(image_pil)
        boxes = results[0].boxes.xyxy.cpu().numpy()

        nail_results = []
        valid_predictions = []
        valid_confidences = []

        for i, box in enumerate(boxes):
            x1, y1, x2, y2 = map(int, box)
            nail_crop = image_pil.crop((x1, y1, x2, y2))

            # Check polish
            resized = nail_crop.resize((128, 128))
            arr = np.array(resized) / 255.0
            arr = np.expand_dims(arr, axis=0)
            pred = nail_classifier.predict(arr, verbose=0)
            class_idx = np.argmax(pred)
            polish_confidence = float(np.max(pred))

            nail_info = {
                "nail_id": i + 1,
                "bounding_box": [x1, y1, x2, y2],
                "has_polish": class_idx == 1,
                "polish_confidence": round(polish_confidence * 100, 2),
                "cropped_image_base64": pil_to_base64(nail_crop)
            }

            # Only plain nails (class 0) for anemia prediction
            if class_idx == 0:
                nail_tensor = anemia_transform(nail_crop).unsqueeze(0)
                with torch.no_grad():
                    output = anemia_model(nail_tensor)
                    probs = torch.softmax(output, dim=1)
                    anemia_class = torch.argmax(probs).item()
                    confidence = probs[0][anemia_class].item()

                    nail_info["anemia_prediction"] = "anemic" if anemia_class == 1 else "non_anemic"
                    nail_info["anemia_confidence"] = round(confidence * 100, 2)
                    nail_info["used_for_diagnosis"] = True

                    valid_predictions.append(anemia_class)
                    valid_confidences.append(confidence)
            else:
                nail_info["anemia_prediction"] = "skipped_due_to_polish"
                nail_info["anemia_confidence"] = 0.0
                nail_info["used_for_diagnosis"] = False

            nail_results.append(nail_info)

        # Calculate overall result
        if len(valid_predictions) == 0:
            overall_result = "non_anemic"  # Default when no valid nails
            overall_confidence = 0.0
            diagnosis_note = "No plain nails detected for analysis"
        else:
            anemic_count = valid_predictions.count(1)
            non_anemic_count = valid_predictions.count(0)
            overall_result = "anemic" if anemic_count > non_anemic_count else "non_anemic"
            overall_confidence = round(np.mean(valid_confidences) * 100, 2)
            diagnosis_note = f"Based on {len(valid_predictions)} plain nails: {anemic_count} anemic, {non_anemic_count} non-anemic"

        return {
            "model_type": "nail_analysis",
            "total_nails_detected": len(boxes),
            "nails_used_for_diagnosis": len(valid_predictions),
            "individual_nails": nail_results,
            "overall_prediction": overall_result,
            "overall_confidence": overall_confidence,
            "diagnosis_note": diagnosis_note,
            "success": True
        }

    except Exception as e:
        return {
            "model_type": "nail_analysis",
            "success": False,
            "error": str(e),
            "overall_prediction": "error",
            "overall_confidence": 0.0
        }

##########################################################################################################################################################################################

# import numpy as np
# import cv2
# from PIL import Image
# from ultralytics import YOLO
# import tensorflow as tf
# import torch
# import torch.nn as nn
# from torchvision import transforms
# import os
# import base64
# from io import BytesIO

# # Model paths
# YOLO_PATH = "./models/yolov8_nail_best.pt"
# CLASSIFIER_PATH = "./models/nail_cnn_classifier_model.h5"
# ANEMIA_MODEL_PATH = "./models/anemia_cnn_model2.pth"

# # Load YOLO and Keras models only once
# yolo_model = YOLO(YOLO_PATH)
# nail_classifier = tf.keras.models.load_model(CLASSIFIER_PATH)

# # Define PyTorch model
# class SimpleCNN(nn.Module):
#     def __init__(self, num_classes=2):
#         super(SimpleCNN, self).__init__()
#         self.features = nn.Sequential(
#             nn.Conv2d(3, 16, 3, 1, 1),
#             nn.ReLU(),
#             nn.MaxPool2d(2, 2),
#             nn.Conv2d(16, 32, 3, 1, 1),
#             nn.ReLU(),
#             nn.MaxPool2d(2, 2),
#         )
#         self.classifier = nn.Sequential(
#             nn.Flatten(),
#             nn.Linear(32 * 56 * 56, 128),
#             nn.ReLU(),
#             nn.Linear(128, num_classes)
#         )
    
#     def forward(self, x):
#         x = self.features(x)
#         x = self.classifier(x)
#         return x

# # Load anemia model
# anemia_model = SimpleCNN()
# anemia_model.load_state_dict(torch.load(ANEMIA_MODEL_PATH, map_location=torch.device("cpu")))
# anemia_model.eval()

# # Preprocessing
# anemia_transform = transforms.Compose([
#     transforms.Resize((224, 224)),
#     transforms.ToTensor()
# ])

# def pil_to_base64(pil_image):
#     """Convert PIL image to base64 string"""
#     buffer = BytesIO()
#     pil_image.save(buffer, format='JPEG')
#     img_str = base64.b64encode(buffer.getvalue()).decode()
#     return img_str

# def run_nail_pipeline(image_pil):
#     """
#     Accepts a PIL hand image, crops nails, detects polish,
#     runs anemia model, and returns final diagnosis with base64 cropped images.
#     """
#     try:
#         results = yolo_model(image_pil)
#         boxes = results[0].boxes.xyxy.cpu().numpy()

#         anemia_preds = []
#         confidences = []
#         cropped_nails = []  # Store cropped nail info
#         all_crops = []      # Store all detected nails (even polished ones)

#         for i, box in enumerate(boxes):
#             x1, y1, x2, y2 = map(int, box)
#             nail_crop = image_pil.crop((x1, y1, x2, y2))

#             # Convert to base64
#             nail_base64 = pil_to_base64(nail_crop)
            
#             # Check polish
#             resized = nail_crop.resize((128, 128))
#             arr = np.array(resized) / 255.0
#             arr = np.expand_dims(arr, axis=0)
#             pred = nail_classifier.predict(arr, verbose=0)
#             class_idx = np.argmax(pred)
#             polish_confidence = float(np.max(pred))
            
#             nail_info = {
#                 "nail_id": i + 1,
#                 "coordinates": {"x1": x1, "y1": y1, "x2": x2, "y2": y2},
#                 "base64_image": nail_base64,
#                 "polish_status": "polished" if class_idx == 1 else "plain",
#                 "polish_confidence": round(polish_confidence, 3)
#             }

#             # Only plain nails (class 0) for anemia detection
#             if class_idx == 0:
#                 nail_tensor = anemia_transform(nail_crop).unsqueeze(0)
#                 with torch.no_grad():
#                     output = anemia_model(nail_tensor)
#                     probs = torch.softmax(output, dim=1)
#                     anemia_class = torch.argmax(probs).item()
#                     confidence = probs[0][anemia_class].item()

#                     anemia_preds.append(anemia_class)
#                     confidences.append(confidence)
                    
#                     nail_info["anemia_prediction"] = "anemic" if anemia_class == 1 else "non-anemic"
#                     nail_info["anemia_confidence"] = round(confidence, 3)
                    
#                     cropped_nails.append(nail_info)
#             else:
#                 nail_info["anemia_prediction"] = "not_analyzed_polished"
#                 nail_info["anemia_confidence"] = 0.0
            
#             all_crops.append(nail_info)

#         # Final diagnosis
#         if len(anemia_preds) == 0:
#             final_result = {
#                 "result": "non-anemic", 
#                 "confidence": 0.0,
#                 "reason": "No plain nails detected for analysis",
#                 "total_nails_detected": len(all_crops),
#                 "plain_nails_analyzed": 0,
#                 "all_detected_nails": all_crops,
#                 "analyzed_nails": []
#             }
#         else:
#             final = 1 if anemia_preds.count(1) > anemia_preds.count(0) else 0
#             avg_conf = float(np.mean(confidences))
            
#             final_result = {
#                 "result": "anemic" if final == 1 else "non-anemic",
#                 "confidence": round(avg_conf, 3),
#                 "total_nails_detected": len(all_crops),
#                 "plain_nails_analyzed": len(cropped_nails),
#                 "anemic_count": anemia_preds.count(1),
#                 "non_anemic_count": anemia_preds.count(0),
#                 "all_detected_nails": all_crops,
#                 "analyzed_nails": cropped_nails
#             }

#         return final_result

#     except Exception as e:
#         return {
#             "result": "error", 
#             "confidence": 0.0, 
#             "error": str(e),
#             "all_detected_nails": [],
#             "analyzed_nails": []
#         }