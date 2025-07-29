from flask import Flask, render_template, request
from tensorflow.keras.utils import load_img, img_to_array
from keras.models import load_model
import numpy as np
import os
import cv2
import uuid
import matplotlib.pyplot as plt
from tf_keras_vis.gradcam import Gradcam
from tf_keras_vis.utils.model_modifiers import ReplaceToLinear
from tf_keras_vis.utils import normalize

# Set dynamic port for Render or fallback to 10000 for local
port = int(os.environ.get("PORT", 10000))

# Ensure 'static' folder exists
os.makedirs('static', exist_ok=True)

app = Flask(__name__)

# Load the trained model
model = load_model('models/trained_model_pneumonia.h5')

# Grad-CAM generation function
def generate_gradcam(image_path, model, penultimate_layer='target_conv_layer'):
    # Load and preprocess image
    img = load_img(image_path, target_size=(224, 224))
    original_img = np.array(img).astype('float32') / 255.0
    img_array = img_to_array(img) / 255.0
    img_array = np.expand_dims(img_array, axis=0)

    # GradCAM setup
    gradcam = Gradcam(model, model_modifier=ReplaceToLinear(), clone=True)

    # Scoring function (assuming binary classification)
    def score(output):
        return output[:, 0]

    # Generate Grad-CAM heatmap
    heatmap = gradcam(score, img_array, penultimate_layer=penultimate_layer)
    heatmap = normalize(heatmap[0])  # (224, 224)
    heatmap = np.power(heatmap, 2.5)  # Enhance activations

    # Apply colormap and blend
    jet_cmap = plt.cm.jet(heatmap)[..., :3]
    overlay = np.clip(original_img + jet_cmap * 0.7, 0, 1)

    # Save overlay image
    filename = f'gradcam_overlay_{uuid.uuid4().hex}.png'
    gradcam_path = os.path.join('static', filename)
    plt.imsave(gradcam_path, overlay)

    return gradcam_path

# Home page
@app.route('/')
def home():
    return render_template('index.html')

# Analyze page (GET and POST)
@app.route('/analyze', methods=['GET', 'POST'])
def analyze():
    if request.method == 'POST':
        imagefile = request.files['file']
        filename = f"upload_{uuid.uuid4().hex}.jpg"
        image_path = os.path.join('static', filename)
        imagefile.save(image_path)

        img = load_img(image_path, target_size=(224, 224), color_mode='rgb')
        x = img_to_array(img) / 255.0
        x = np.expand_dims(x, axis=0)
        prediction = model.predict(x)
        prob = prediction[0][0]
        label = 'Positive' if prob >= 0.5 else 'Negative'
        confidence = prob * 100 if prob >= 0.5 else (1 - prob) * 100
        classification = f'{label} ({confidence:.2f}%)'

        gradcam_path = generate_gradcam(image_path, model)

        return render_template('analyze.html',
                               prediction=classification,
                               original_image_path='/' + image_path,
                               gradcam_path='/' + gradcam_path)

    return render_template('analyze.html')

# About page
@app.route('/about')
def about():
    return render_template('about.html')

# Prevention page
@app.route('/prevention')
def prevention():
    return render_template('prevention.html')

# Contact page
@app.route('/contact')
def contact():
    return render_template('contact.html')

# Entry point
if __name__ == '__main__':
    app.run(host='0.0.0.0', port=port)  # Works both locally and on Render
