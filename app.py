from flask import Flask, render_template, request
from tensorflow.keras.utils import load_img, img_to_array
from keras.models import load_model
import numpy as np
import os
import matplotlib.pyplot as plt
from tf_keras_vis.gradcam import Gradcam
from tf_keras_vis.utils.model_modifiers import ReplaceToLinear
from tf_keras_vis.utils import normalize
import cv2
from PIL import Image
import uuid

app = Flask(__name__)

# Ensure static directory exists
os.makedirs('static', exist_ok=True)

# Load trained model
try:
    model = load_model('models/trained_model_pneumonia.h5')
    print("Model loaded successfully!")
except Exception as e:
    print(f"Error loading model: {e}")
    model = None

def generate_gradcam(image_path, model, penultimate_layer='target_conv_layer'):
    """Generate GradCAM visualization for the given image."""
    try:
        # Load and preprocess image
        img = load_img(image_path, target_size=(224, 224), color_mode='rgb')
        img_array = img_to_array(img) / 255.0
        img_array = np.expand_dims(img_array, axis=0)
        
        # Initialize Gradcam with your model
        gradcam = Gradcam(model)
        
        # Score function
        def score(output):
            return output[:, 0]
        
        # Generate heatmap
        heatmap = gradcam(score, img_array, penultimate_layer=penultimate_layer)
        heatmap = normalize(heatmap)[0]  # shape (224, 224)
        
        # Resize heatmap and original image to same size
        heatmap_resized = cv2.resize(heatmap, (224, 224))
        heatmap_colored = cv2.applyColorMap(np.uint8(255 * heatmap_resized), cv2.COLORMAP_JET)
        original_img = img_to_array(img)  # shape (224, 224, 3)
        overlayed = heatmap_colored * 0.4 + original_img
        
        # Save overlayed image
        overlayed = np.uint8(overlayed)
        filename = f'gradcam_overlay_{uuid.uuid4().hex}.png'
        gradcam_path = os.path.join('static', filename)
        cv2.imwrite(gradcam_path, cv2.cvtColor(overlayed, cv2.COLOR_RGB2BGR))
        
        return gradcam_path
    
    except Exception as e:
        print(f"Error generating GradCAM: {e}")
        return None

@app.route('/', methods=['GET', 'POST'])
def predict():
    """Main route for prediction and visualization."""
    if request.method == 'POST':
        # Check if model is loaded
        if model is None:
            return render_template('index.html', 
                                 error="Model not loaded. Please check server logs.")
        
        try:
            # Get uploaded file
            imagefile = request.files.get('imagefile')
            if not imagefile or imagefile.filename == '':
                return render_template('index.html', 
                                     error="No file selected. Please upload an image.")
            
            # Save uploaded file
            filename = f"{uuid.uuid4().hex}_{imagefile.filename}"
            image_path = os.path.join('static', filename)
            imagefile.save(image_path)
            
            # Preprocess and predict
            img = load_img(image_path, target_size=(224, 224), color_mode='rgb')
            x = img_to_array(img) / 255.0
            x = np.expand_dims(x, axis=0)
            
            # Make prediction
            prediction = model.predict(x)
            prob = prediction[0][0]
            
            # Determine label and confidence
            if prob >= 0.5:
                label = 'Pneumonia Detected'
                confidence = prob * 100
            else:
                label = 'Normal'
                confidence = (1 - prob) * 100
            
            classification = f'{label} ({confidence:.2f}%)'
            
            # Generate GradCAM visualization
            gradcam_path = generate_gradcam(image_path, model, 
                                          penultimate_layer='target_conv_layer')
            
            if gradcam_path is None:
                gradcam_path = image_path  # Fallback to original image
            
            return render_template('index.html', 
                                 prediction=classification,
                                 imagePath=image_path, 
                                 gradcamPath=gradcam_path)
        
        except Exception as e:
            print(f"Error during prediction: {e}")
            return render_template('index.html', 
                                 error=f"Error processing image: {str(e)}")
    
    return render_template('index.html')

@app.route('/health')
def health_check():
    """Health check endpoint."""
    return {"status": "healthy", "model_loaded": model is not None}

# Error handlers
@app.errorhandler(404)
def not_found_error(error):
    return render_template('index.html', error="Page not found"), 404

@app.errorhandler(500)
def internal_error(error):
    return render_template('index.html', error="Internal server error"), 500

if __name__ == '__main__':
    # Get port from environment variable (Render provides this)
    port = int(os.environ.get('PORT', 5000))
    
    # Get host from environment variable (default to 0.0.0.0 for deployment)
    host = os.environ.get('HOST', '0.0.0.0')
    
    # Set debug mode based on environment
    debug_mode = os.environ.get('FLASK_ENV', 'production') == 'development'
    
    print(f"Starting Flask app on {host}:{port}")
    print(f"Debug mode: {debug_mode}")
    
    # CRITICAL: Bind to 0.0.0.0 and use environment PORT
    app.run(host=host, port=port, debug=debug_mode)
