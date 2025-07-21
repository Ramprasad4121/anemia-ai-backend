

#  AI-Based Anemia Detection System

An AI-powered backend system that detects anemia from user-uploaded images of **nails**, **palms**, and **eyes**. This project uses deep learning models, including YOLOv8 and ResNet101 etc, to analyze visual features and deliver a final diagnosis using a weighted voting mechanism.

---
## Note:  This project is under active development. models are not fully reliable yet. Do not consider current predictions for clinical use.

## Note: if you face any issues while local set-up , reach me out @ ramprasadgoud34@gmail.com

##  Project Structure

```
anemia-ai-backend/
├── app_combined.py              # Flask API entrypoint
├── combined_pipeline.py         # Orchestrates nail, palm, and eye pipelines
├── eye_pipeline.py              # Eye detection and classification
├── nail_pipeline.py             # Nail detection and classification
├── palm_pipeline.py            # Palm classification logic
├── models/                      # Pretrained models (YOLO + ResNet)
├── test_images/                 # Sample images for testing
├── requirements.txt             # Python dependencies
└── README.md                    # You're here!
```



##  How It Works

1. **User uploads 3 images**: nails, palms, and eyes.
2. **Flask API** (`app_combined.py`) receives the images.
3. **Combined Pipeline** (`combined_pipeline.py`) runs each:
   - `nail_pipeline.py`
   - `palm_pipeline.py`
   - `eye_pipeline.py`
4. Each model returns a prediction (`anemic` or `non_anemic`).
5. **Weighted Majority Voting** calculates final diagnosis.
6. A JSON response is sent back to the frontend.


##  Installation Guide

###  Requirements

- Python 3.9+
- pip
- Git
- CUDA (Optional, for GPU acceleration)

###  Setup Instructions (read this clearly and install step-by-step)

```bash
# Clone the repository
git clone https://github.com/YOUR_USERNAME/anemia-ai-backend.git
cd anemia-ai-backend

# (Optional but recommended) Create a virtual environment
python -m venv venv
source venv/bin/activate  # For Linux/macOS
venv\Scripts\activate     # For Windows

# Install dependencies
pip install -r requirements.txt

# Make sure YOLOv8 is installed (from ultralytics) (use ai to install ultralytics if you face issue)
pip install ultralytics
```

###  Download Pretrained Models

Place the following models into the `models/` directory:

```
models/
├── yolov8_model.pt                 # For nail and eye detection
├── resnet101_anemia_model1.pth     # For eye classification
├── palm_cnn_model.pth              # For palm classification
├── nail_resnet_model.pth           # For nail classification
```

> **Note**: You can contact the maintainers to obtain these models or retrain your own.

---

##  Testing the API

Start the server:

```bash
python app_combined.py
```

Use Postman or any HTTP client to send a `POST` request:

**URL**: `http://localhost:2000/predict_anemia`

**Body (form-data)**:
- `nail_image`
- `palm_left`
- `palm_right`
- `eye_left`
- `eye_right`

Each should be an image file.


The final result is computed based on highest weighted prediction:
```json
"final_prediction_3_models_combined": "non_anemic",
"confidence": 82.14,
"method": "weighted_majority_vote"
```



##  License

This project is licensed under the **MIT License**.
