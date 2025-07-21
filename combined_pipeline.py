from PIL import Image
from nailbed_model.nail_pipeline import run_nail_pipeline
from palm_model.palm_pipeline import run_palm_pipeline
from eye_model.eye_pipeline import run_eye_pipeline
import numpy as np
from collections import Counter

# Helper function to make results JSON serializable
def make_json_serializable(obj):
    if isinstance(obj, (np.integer, np.int_, np.int64)):
        return int(obj)
    elif isinstance(obj, (np.floating, np.float_, np.float32)):
        return float(obj)
    elif isinstance(obj, (np.bool_, bool)):
        return bool(obj)
    elif isinstance(obj, dict):
        return {k: make_json_serializable(v) for k, v in obj.items()}
    elif isinstance(obj, list):
        return [make_json_serializable(i) for i in obj]
    else:
        return obj

def process_all_images(nail_image=None, left_palm=None, right_palm=None, left_eye=None, right_eye=None):
    """
    Main orchestrator function that processes all images and returns comprehensive results
    
    Args:
        nail_image: PIL Image of hand/nails
        left_palm: PIL Image of left palm
        right_palm: PIL Image of right palm  
        left_eye: PIL Image of left eye
        right_eye: PIL Image of right eye
    
    Returns:
        dict: Comprehensive analysis results with final diagnosis
    """
    
    results = {
        "analysis_summary": {
            "total_images_processed": 0,
            "models_used": [],
            "processing_success": True
        },
        "nail_analysis": None,
        "palm_analysis": {
            "left_palm": None,
            "right_palm": None,
            "overall_prediction": "not_processed",
            "overall_confidence": 0.0
        },
        "eye_analysis": None,
        "final_diagnosis": {
            "prediction": "unknown",
            "confidence": 0.0,
            "method": "majority_vote",
            "breakdown": {},
            "recommendation": ""
        }
    }
    
    all_predictions = []  # For final majority vote
    all_confidences = []
    model_contributions = {}
    
    # Process Nail Image
    if nail_image is not None:
        print("🔄 Processing nail image...")
        try:
            nail_result = run_nail_pipeline(nail_image)
            # Remove any full/original image base64 fields (keep only cropped/thumbs)
            if isinstance(nail_result, dict):
                nail_result = {
                    k: v for k, v in nail_result.items()
                    if not (isinstance(k, str) and ("original" in k and "base64" in k))
                }
                # For nested per-nail results (if present), clean those too
                if "nails" in nail_result and isinstance(nail_result["nails"], list):
                    for nail in nail_result["nails"]:
                        keys_to_remove = [key for key in nail if "original" in key and "base64" in key]
                        for key in keys_to_remove:
                            nail.pop(key, None)
            results["nail_analysis"] = nail_result
            results["analysis_summary"]["models_used"].append("nail_detection")
            results["analysis_summary"]["total_images_processed"] += 1
            
            if nail_result["success"] and nail_result["overall_prediction"] != "error":
                prediction = 1 if nail_result["overall_prediction"] == "anemic" else 0
                confidence = nail_result["overall_confidence"] / 100.0
                
                # Weight nail predictions by number of nails used
                nail_weight = max(1, nail_result.get("nails_used_for_diagnosis", 1))
                for _ in range(nail_weight):
                    all_predictions.append(prediction)
                    all_confidences.append(confidence)
                
                model_contributions["nails"] = {
                    "prediction": nail_result["overall_prediction"],
                    "confidence": nail_result["overall_confidence"],
                    "weight": nail_weight,
                    "nails_analyzed": nail_result.get("nails_used_for_diagnosis", 0)
                }
                
        except Exception as e:
            results["nail_analysis"] = {"success": False, "error": str(e)}
            print(f"❌ Nail processing error: {e}")
    
    # Process Palm Images
    palm_predictions = []
    palm_confidences = []
    
    if left_palm is not None:
        print("🔄 Processing left palm...")
        try:
            left_palm_result = run_palm_pipeline(left_palm, "left")
            # Remove any full/original image base64 fields (keep only cropped/thumbs)
            if isinstance(left_palm_result, dict):
                left_palm_result = {
                    k: v for k, v in left_palm_result.items()
                    if not (isinstance(k, str) and ("original" in k and "base64" in k))
                }
            results["palm_analysis"]["left_palm"] = left_palm_result
            results["analysis_summary"]["total_images_processed"] += 1
            
            if left_palm_result["success"] and left_palm_result["prediction"] != "error":
                prediction = 1 if left_palm_result["prediction"] == "anemic" else 0
                confidence = left_palm_result["confidence"] / 100.0
                palm_predictions.append(prediction)
                palm_confidences.append(confidence)
                all_predictions.append(prediction)
                all_confidences.append(confidence)
                
        except Exception as e:
            results["palm_analysis"]["left_palm"] = {"success": False, "error": str(e)}
            print(f"❌ Left palm processing error: {e}")
    
    if right_palm is not None:
        print("🔄 Processing right palm...")
        try:
            right_palm_result = run_palm_pipeline(right_palm, "right")
            # Remove any full/original image base64 fields (keep only cropped/thumbs)
            if isinstance(right_palm_result, dict):
                right_palm_result = {
                    k: v for k, v in right_palm_result.items()
                    if not (isinstance(k, str) and ("original" in k and "base64" in k))
                }
            results["palm_analysis"]["right_palm"] = right_palm_result
            results["analysis_summary"]["total_images_processed"] += 1
            
            if right_palm_result["success"] and right_palm_result["prediction"] != "error":
                prediction = 1 if right_palm_result["prediction"] == "anemic" else 0
                confidence = right_palm_result["confidence"] / 100.0
                palm_predictions.append(prediction)
                palm_confidences.append(confidence)
                all_predictions.append(prediction)
                all_confidences.append(confidence)
                
        except Exception as e:
            results["palm_analysis"]["right_palm"] = {"success": False, "error": str(e)}
            print(f"❌ Right palm processing error: {e}")
    
    # Calculate palm overall result
    if palm_predictions:
        results["analysis_summary"]["models_used"].append("palm_analysis")
        anemic_palms = palm_predictions.count(1)
        non_anemic_palms = palm_predictions.count(0)
        palm_overall = "anemic" if anemic_palms > non_anemic_palms else "non_anemic"
        palm_confidence = np.mean(palm_confidences) * 100
        
        results["palm_analysis"]["overall_prediction"] = palm_overall
        results["palm_analysis"]["overall_confidence"] = round(palm_confidence, 2)
        
        model_contributions["palms"] = {
            "prediction": palm_overall,
            "confidence": round(palm_confidence, 2),
            "weight": len(palm_predictions),
            "palms_analyzed": len(palm_predictions)
        }
    
    # Process Eye Images
    if left_eye is not None or right_eye is not None:
        print("🔄 Processing eye images...")
        try:
            eye_result = run_eye_pipeline(left_eye, right_eye)
            # Remove any full/original image base64 fields (keep only cropped/thumbs)
            if isinstance(eye_result, dict):
                # Remove top-level original image base64
                eye_result = {
                    k: v for k, v in eye_result.items()
                    if not (isinstance(k, str) and ("original" in k and "base64" in k))
                }
                # Remove from left_eye/right_eye fields if present
                for eye_side in ["left_eye", "right_eye"]:
                    if eye_side in eye_result and isinstance(eye_result[eye_side], dict):
                        keys_to_remove = [key for key in eye_result[eye_side] if "original" in key and "base64" in key]
                        for key in keys_to_remove:
                            eye_result[eye_side].pop(key, None)
            results["eye_analysis"] = eye_result
            results["analysis_summary"]["total_images_processed"] += (1 if left_eye else 0) + (1 if right_eye else 0)
            
            if eye_result["success"] and eye_result["overall_prediction"] != "error":
                results["analysis_summary"]["models_used"].append("eye_analysis")
                prediction = 1 if eye_result["overall_prediction"] == "anemic" else 0
                confidence = eye_result["overall_confidence"] / 100.0
                
                # Weight eye predictions by number of eyes processed
                eye_weight = eye_result.get("eyes_processed", 1)
                for _ in range(eye_weight):
                    all_predictions.append(prediction)
                    all_confidences.append(confidence)
                
                model_contributions["eyes"] = {
                    "prediction": eye_result["overall_prediction"],
                    "confidence": eye_result["overall_confidence"],
                    "weight": eye_weight,
                    "eyes_analyzed": eye_weight
                }
                
        except Exception as e:
            results["eye_analysis"] = {"success": False, "error": str(e)}
            print(f"❌ Eye processing error: {e}")
    
    # Calculate Final Diagnosis
    if all_predictions:
        # Majority vote
        anemic_votes = all_predictions.count(1)
        non_anemic_votes = all_predictions.count(0)
        total_votes = len(all_predictions)
        
        final_prediction = "anemic" if anemic_votes > non_anemic_votes else "non_anemic"
        final_confidence = np.mean(all_confidences) * 100
        
        # Create breakdown
        breakdown = {
            "total_predictions": total_votes,
            "anemic_votes": anemic_votes,
            "non_anemic_votes": non_anemic_votes,
            "anemic_percentage": round((anemic_votes / total_votes) * 100, 1),
            "non_anemic_percentage": round((non_anemic_votes / total_votes) * 100, 1),
            "model_contributions": model_contributions
        }
        
        # Generate recommendation
        confidence_level = "high" if final_confidence > 80 else "moderate" if final_confidence > 60 else "low"
        
        if final_prediction == "anemic":
            if confidence_level == "high":
                recommendation = "Strong indication of anemia detected. Recommend immediate medical consultation for blood tests and proper diagnosis."
            elif confidence_level == "moderate":
                recommendation = "Moderate indication of anemia detected. Recommend medical consultation for blood tests to confirm."
            else:
                recommendation = "Weak indication of anemia detected. Consider medical consultation for comprehensive evaluation."
        else:
            if confidence_level == "high":
                recommendation = "Strong indication of normal hemoglobin levels. Continue regular health monitoring."
            elif confidence_level == "moderate":
                recommendation = "Moderate indication of normal hemoglobin levels. Regular health checkups recommended."
            else:
                recommendation = "Inconclusive results. Recommend medical consultation for proper evaluation."
        
        results["final_diagnosis"] = {
            "prediction": final_prediction,
            "confidence": round(final_confidence, 2),
            "confidence_level": confidence_level,
            "method": "weighted_majority_vote",
            "breakdown": breakdown,
            "recommendation": recommendation
        }
        
        print(f"🎯 FINAL DIAGNOSIS: {final_prediction.upper()} ({final_confidence:.1f}% confidence)")
        print(f"📊 Votes: {anemic_votes} anemic, {non_anemic_votes} non-anemic (total: {total_votes})")
        
    else:
        results["analysis_summary"]["processing_success"] = False
        results["final_diagnosis"] = {
            "prediction": "error",
            "confidence": 0.0,
            "confidence_level": "none",
            "method": "no_valid_predictions",
            "breakdown": {"error": "No valid predictions from any model"},
            "recommendation": "Unable to analyze images. Please ensure images are clear and try again, or consult a healthcare professional."
        }
        print("❌ No valid predictions available for final diagnosis")
    
    return make_json_serializable(results)


def print_final_summary(results):
    """
    Print a formatted summary for console/API response
    """
    print("\n" + "="*80)
    print("🏥 ANEMIA DETECTION ANALYSIS SUMMARY")
    print("="*80)
    
    # Analysis Summary
    summary = results["analysis_summary"]
    print(f"📊 Images Processed: {summary['total_images_processed']}")
    print(f"🤖 Models Used: {', '.join(summary['models_used'])}")
    print(f"✅ Processing Success: {summary['processing_success']}")
    
    # Individual Results
    print(f"\n📍 INDIVIDUAL MODEL RESULTS:")
    
    # Nails
    if results["nail_analysis"]:
        nail = results["nail_analysis"]
        if nail["success"]:
            print(f"   💅 Nails: {nail['overall_prediction'].upper()} ({nail['overall_confidence']}% confidence)")
            print(f"      └─ {nail['total_nails_detected']} nails detected, {nail['nails_used_for_diagnosis']} used for analysis")
        else:
            print(f"   💅 Nails: ERROR - {nail.get('error', 'Unknown error')}")
    
    # Palms
    palm = results["palm_analysis"]
    if palm["left_palm"] or palm["right_palm"]:
        print(f"   🤚 Palms: {palm['overall_prediction'].upper()} ({palm['overall_confidence']}% confidence)")
        if palm["left_palm"]:
            left = palm["left_palm"]
            status = f"{left['prediction'].upper()} ({left['confidence']}%)" if left["success"] else "ERROR"
            print(f"      ├─ Left: {status}")
        if palm["right_palm"]:
            right = palm["right_palm"]
            status = f"{right['prediction'].upper()} ({right['confidence']}%)" if right["success"] else "ERROR"
            print(f"      └─ Right: {status}")
    
    # Eyes
    if results["eye_analysis"]:
        eye = results["eye_analysis"]
        if eye["success"]:
            print(f"   👁️ Eyes: {eye['overall_prediction'].upper()} ({eye['overall_confidence']}% confidence)")
            if eye["left_eye"]:
                left = eye["left_eye"]
                status = f"{left['prediction'].upper()} ({left['confidence']}%)" if left["success"] else "ERROR"
                print(f"      ├─ Left: {status}")
            if eye["right_eye"]:
                right = eye["right_eye"]
                status = f"{right['prediction'].upper()} ({right['confidence']}%)" if right["success"] else "ERROR"
                print(f"      └─ Right: {status}")
        else:
            print(f"   👁️ Eyes: ERROR - {eye.get('error', 'Unknown error')}")
    
    # Final Diagnosis
    diagnosis = results["final_diagnosis"]
    print(f"\n🎯 FINAL DIAGNOSIS:")
    print(f"   Result: {diagnosis['prediction'].upper()}")
    print(f"   Confidence: {diagnosis['confidence']}% ({diagnosis['confidence_level']} confidence)")
    print(f"   Method: {diagnosis['method']}")
    
    if "breakdown" in diagnosis and "anemic_votes" in diagnosis["breakdown"]:
        breakdown = diagnosis["breakdown"]
        print(f"   Vote Breakdown: {breakdown['anemic_votes']} anemic, {breakdown['non_anemic_votes']} non-anemic")
        print(f"   Total Predictions: {breakdown['total_predictions']}")
    
    print(f"\n💡 RECOMMENDATION:")
    print(f"   {diagnosis['recommendation']}")
    
    print("="*80)
    
    return {
        "final_prediction": diagnosis['prediction'],
        "final_confidence": diagnosis['confidence'],
        "recommendation": diagnosis['recommendation']
    }  

