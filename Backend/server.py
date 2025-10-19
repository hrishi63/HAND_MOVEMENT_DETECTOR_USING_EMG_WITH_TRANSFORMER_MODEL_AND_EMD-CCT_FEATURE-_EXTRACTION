"""
Flask Server for 2-Channel EMG Hand Gesture Recognition
FIXED: JSON serialization for boolean values
"""
from flask import Flask, jsonify
from flask_cors import CORS
import numpy as np

# ========== MODEL SELECTION ==========
USE_2_CHANNEL = False  # Set to False to use old 10-channel model

if USE_2_CHANNEL:
    print("🔵 Loading 2-CHANNEL model...")
    from predict_2ch import predict_clench_2ch as predict_function
else:
    print("🔴 Loading 10-CHANNEL model...")
    from predict import predict_finger_movement as predict_function

# ========== FLASK APP ==========
app = Flask(__name__)
CORS(app)  # Allow React on localhost:3000

@app.route('/prediction')
def get_prediction():
    """
    Main prediction endpoint
    Returns: [bool, False, False, False, False]
    - First element: True = clench, False = open
    - Other 4 elements: Reserved for future finger detection
    """
    try:
        prediction = predict_function()
        
        # Convert to Python native types for JSON serialization
        prediction_json = [bool(x) for x in prediction]
        
        return jsonify(prediction_json)
    
    except Exception as e:
        print(f"❌ Prediction error: {e}")
        import traceback
        traceback.print_exc()
        
        # Return safe default on error
        return jsonify([False, False, False, False, False])

@app.route('/signal')
def get_signal():
    """
    Dummy EMG signal endpoint (for visualization)
    Returns: List of 100 float values
    """
    # Generate dummy signal with small amplitude
    signal = (np.random.randn(100) * 0.02).tolist()
    return jsonify(signal)

@app.route('/status')
def get_status():
    """
    System status endpoint
    Returns: Model info and configuration
    """
    model_info = {
        "model_type": "2-channel" if USE_2_CHANNEL else "10-channel",
        "input_features": 18 if USE_2_CHANNEL else 90,
        "channels": 2 if USE_2_CHANNEL else 10,
        "ready": True
    }
    return jsonify(model_info)

@app.route('/health')
def health_check():
    """
    Health check endpoint
    """
    return jsonify({"status": "ok", "message": "Server is running"})

if __name__ == '__main__':
    print("\n" + "="*60)
    print("🚀 EMG Hand Gesture Recognition Server")
    print("="*60)
    if USE_2_CHANNEL:
        print("📊 Model: 2-Channel Transformer (18-D features)")
        print("🔌 Hardware: Arduino + 2 EMG sensors")
    else:
        print("📊 Model: 10-Channel Transformer (90-D features)")
        print("🔌 Hardware: NinaPro DB1 simulation")
    print("="*60)
    print("✅ Server starting on http://127.0.0.1:5000")
    print("✅ Endpoints:")
    print("   - /prediction  (Main EMG prediction)")
    print("   - /signal      (Dummy signal data)")
    print("   - /status      (Model configuration)")
    print("   - /health      (Health check)")
    print("="*60)
    print("\n💡 Press Ctrl+C to stop the server\n")
   
    app.run(debug=True, port=5000)