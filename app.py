from flask import Flask, render_template, request
from tensorflow.keras.utils import load_img, img_to_array
from keras.models import load_model
import numpy as np
import os
import matplotlib.pyplot as plt
from tf_keras_vis.gradcam import Gradcam  # Ensure correct import
from tf_keras_vis.utils.model_modifiers import ReplaceToLinear
from tf_keras_vis.utils import normalize
import cv2
from PIL import Image
import uuid

app = Flask(__name__)

# Load trained model
model = load_model('models/trained_model_pneumonia.h5')

# Set up GradCAM

def generate_gradcam(image_path, model, penultimate_layer='target_conv_layer'):
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


# Routes
@app.route('/', methods=['GET', 'POST'])
def predict():
    if request.method == 'POST':
        imagefile = request.files['imagefile']
        filename = imagefile.filename
        image_path = os.path.join('static', filename)
        imagefile.save(image_path)

        # Preprocess and predict
        img = load_img(image_path, target_size=(224, 224), color_mode='rgb')
        x = img_to_array(img) / 255.0
        x = np.expand_dims(x, axis=0)
        prediction = model.predict(x)
        prob = prediction[0][0]
        label = 'Positive' if prob >= 0.5 else 'Negative'
        confidence = prob * 100 if prob >= 0.5 else (1 - prob) * 100
        classification = f'{label} ({confidence:.2f}%)'

        # Generate GradCAM visualization
        gradcam_path = generate_gradcam(image_path, model, penultimate_layer='target_conv_layer')

        return render_template('index.html', prediction=classification,
                               imagePath=image_path, gradcamPath=gradcam_path)

    return render_template('index.html')

if __name__ == '__main__':
    app.run(debug=True)
