
from flask import Flask, request, jsonify
from PIL import Image
import io
import traceback
from combined_pipeline import process_all_images, print_final_summary
import base64
import numpy as np

app = Flask(__name__)

# Configure Flask for larger file uploads (60MB per image)
app.config['MAX_CONTENT_LENGTH'] = 300 * 1024 * 1024  # 300MB total (5 images × 60MB each)

def convert_to_serializable(obj):
    if isinstance(obj, (np.bool_, np.integer, np.floating)):
        return obj.item()
    elif isinstance(obj, dict):
        return {k: convert_to_serializable(v) for k, v in obj.items()}
    elif isinstance(obj, list):
        return [convert_to_serializable(elem) for elem in obj]
    else:
        return obj

def validate_and_process_image(file_obj, image_name):
    """
    Validate and convert uploaded file to PIL Image
    
    Args:
        file_obj: Flask file object from request.files
        image_name: Name of the image for error reporting
    
    Returns:
        PIL.Image or None if invalid
    """
    try:
        if file_obj and file_obj.filename != '':
            # Check individual file size (60MB limit per image)
            file_obj.seek(0, 2)  # Seek to end of file
            file_size = file_obj.tell()
            file_obj.seek(0)  # Reset to beginning
            
            max_size_mb = 60
            max_size_bytes = max_size_mb * 1024 * 1024
            
            if file_size > max_size_bytes:
                print(f"❌ {image_name} too large: {file_size / (1024*1024):.1f}MB (max: {max_size_mb}MB)")
                return None
            
            # Read the image file
            image_data = file_obj.read()
            # Convert to PIL Image
            pil_image = Image.open(io.BytesIO(image_data))
            # Convert to RGB if necessary (in case of RGBA, grayscale, etc.)
            if pil_image.mode != 'RGB':
                pil_image = pil_image.convert('RGB')
            print(f"✅ Successfully loaded {image_name}: {pil_image.size} ({file_size / (1024*1024):.1f}MB)")
            return pil_image
        else:
            print(f"⚠️ No {image_name} provided")
            return None
    except Exception as e:
        print(f"❌ Error processing {image_name}: {e}")
        return None

@app.errorhandler(413)
def too_large(e):
    """Handle file too large error"""
    return jsonify({
        "error": "File too large",
        "message": "Total upload size exceeds 300MB limit. Each image should be under 60MB.",
        "success": False
    }), 413

@app.route('/predict_anemia', methods=['POST'])
def predict_anemia():
    """
    Main API endpoint for anemia prediction
    
    Expected form-data fields:
    - nail_image: Hand image file (max 60MB)
    - left_palm: Left palm image file (max 60MB)
    - right_palm: Right palm image file (max 60MB)
    - left_eye: Left eye image file (max 60MB)
    - right_eye: Right eye image file (max 60MB)
    """
    try:
        print("🚀 Starting anemia prediction API...")
        
        # Validate that at least one image is provided
        files = request.files
        if not any(key in files for key in ['nail_image', 'left_palm', 'right_palm', 'left_eye', 'right_eye']):
            return jsonify({
                "error": "No images provided",
                "message": "Please provide at least one image: nail_image, left_palm, right_palm, left_eye, or right_eye",
                "success": False
            }), 400
        
        # Process each image
        nail_image = validate_and_process_image(files.get('nail_image'), 'nail_image')
        left_palm = validate_and_process_image(files.get('left_palm'), 'left_palm')
        right_palm = validate_and_process_image(files.get('right_palm'), 'right_palm')
        left_eye = validate_and_process_image(files.get('left_eye'), 'left_eye')
        right_eye = validate_and_process_image(files.get('right_eye'), 'right_eye')
        
        # Check if at least one image was successfully processed
        images_provided = [nail_image, left_palm, right_palm, left_eye, right_eye]
        valid_images = [img for img in images_provided if img is not None]
        
        if not valid_images:
            return jsonify({
                "error": "No valid images could be processed",
                "message": "All provided images failed to load. Please check image formats and file sizes (max 60MB per image) and try again.",
                "success": False
            }), 400
        
        print(f"📊 Processing {len(valid_images)} valid images...")
        
        # Run the combined analysis
        results = process_all_images(
            nail_image=nail_image,
            left_palm=left_palm,
            right_palm=right_palm,
            left_eye=left_eye,
            right_eye=right_eye
        )
        
        # Convert all numpy types in results to native python types
        results = convert_to_serializable(results)
        
        # Print summary to console
        final_summary = print_final_summary(results)
        
        # Prepare API response
        api_response = {
            "success": True,
            "timestamp": None,  # You can add timestamp if needed
            "analysis_summary": results["analysis_summary"],
            "detailed_results": {
                "nail_analysis": results["nail_analysis"],
                "palm_analysis": results["palm_analysis"],
                "eye_analysis": results["eye_analysis"]
            },
            "final_prediction_3_models_combined": {
                "prediction": str(results["final_diagnosis"]["prediction"]),
                "confidence": float(results["final_diagnosis"]["confidence"]),
                "confidence_level": str(results["final_diagnosis"]["confidence_level"])
            }
        }
        
        print("✅ API response prepared successfully")
        return jsonify(api_response), 200
        
    except Exception as e:
        error_message = str(e)
        error_trace = traceback.format_exc()
        
        print(f"❌ API Error: {error_message}")
        print(f"📍 Full traceback: {error_trace}")
        
        return jsonify({
            "success": False,
            "error": "Internal server error",
            "message": error_message,
            "details": "Check server logs for full error details"
        }), 500

@app.route('/health', methods=['GET'])
def health_check():
    """Health check endpoint"""
    return jsonify({
        "status": "healthy",
        "message": "Anemia detection API is running",
        "endpoints": {
            "/predict_anemia": "POST - Main prediction endpoint",
            "/health": "GET - Health check"
        }
    }), 200

@app.route('/', methods=['GET'])
def home():
    """Home endpoint with API information"""
    return jsonify({
        "message": "Anemia Detection API",
        "version": "1.0.0",
        "description": "AI-powered anemia detection using nail, palm, and eye images",
        "usage": {
            "endpoint": "/predict_anemia",
            "method": "POST",
            "content_type": "multipart/form-data",
            "required_fields": "At least one of: nail_image, left_palm, right_palm, left_eye, right_eye",
            "response": "Comprehensive anemia analysis with individual model results and final diagnosis"
        },
        "supported_formats": ["JPEG", "JPG", "PNG"],
        "max_file_size": "60MB per image (300MB total)"
    }), 200

if __name__ == '__main__':
    print("🏥 Starting Anemia Detection API Server...")
    print("📡 Available endpoints:")
    print("   POST /predict_anemia - Main prediction endpoint")
    print("   GET  /health        - Health check")
    print("   GET  /              - API information")
    print("🚀 Server starting on http://localhost:2000")
    print("📁 Max file size: 60MB per image, 300MB total")
    
    app.run(
        host='0.0.0.0',  # Allow external connections
        port=2000,
        debug=True  # Remove in production
    )