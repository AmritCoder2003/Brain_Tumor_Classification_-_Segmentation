from flask import Flask, request, render_template, redirect, url_for
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing.image import img_to_array, load_img
import numpy as np
import os
import cv2
from werkzeug.utils import secure_filename
from PIL import Image

app = Flask(__name__)

UPLOAD_FOLDER = 'static/uploads'
SEGMENTED_FOLDER = 'static/segmented'
os.makedirs(UPLOAD_FOLDER, exist_ok=True)
os.makedirs(SEGMENTED_FOLDER, exist_ok=True)

app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER
app.config['SEGMENTED_FOLDER'] = SEGMENTED_FOLDER

# Load models
classification_model = load_model("models/model12.h5")
segmentation_model = load_model("models/modelsegmentation.h5", compile=False)
class_labels = ['pituitary', 'notumor', 'meningioma', 'glioma']


@app.route('/')
def index():
    return render_template('index.html')


@app.route('/predict', methods=['POST'])
def predict():
    results = []
    files = request.files.getlist('images')

    for file in files:
        if file and file.filename != '':
            filename = secure_filename(file.filename)
            filepath = os.path.join(app.config['UPLOAD_FOLDER'], filename)
            file.save(filepath)

            img = load_img(filepath, target_size=(128, 128))
            img_array = img_to_array(img) / 255.0
            img_array = np.expand_dims(img_array, axis=0)

            predictions = classification_model.predict(img_array)
            predicted_class_index = np.argmax(predictions, axis=1)[0]
            confidence_score = float(np.max(predictions, axis=1)[0]) * 100

            result_text = "No Tumor" if class_labels[predicted_class_index] == 'notumor' else f"Tumor: {class_labels[predicted_class_index]}"

            results.append({
                "filename": filename,
                "prediction": result_text,
                "confidence": f"{confidence_score:.2f}"
            })

    return render_template('index.html', results=results)


@app.route('/segmentation', methods=['GET', 'POST'])
def segmentation():
    segmented_results = []

    if request.method == 'POST':
        files = request.files.getlist('images')

        for file in files:
            if file and file.filename != '':
                filename = secure_filename(file.filename)
                filepath = os.path.join(app.config['UPLOAD_FOLDER'], filename)
                file.save(filepath)

                # Load image in grayscale and resize to 128x128
                img = load_img(filepath, color_mode='grayscale', target_size=(128, 128))
                img_array = img_to_array(img) / 255.0  # shape: (128, 128, 1)
                img_array = np.expand_dims(img_array, axis=0)  # shape: (1, 128, 128, 1)

                # Predict segmentation mask
                mask = segmentation_model.predict(img_array)[0]  # shape: (128, 128, 1)
                mask = (mask.squeeze() * 255).astype(np.uint8)  # shape: (128, 128)

                # Apply color map to mask
                mask_colored = cv2.applyColorMap(mask, cv2.COLORMAP_JET)  # shape: (128, 128, 3)

                # Load original image, resize to 128x128
                original_img = Image.open(filepath).convert('RGB')
                original_resized = original_img.resize((128, 128))
                original_array = np.array(original_resized)

                # Convert to BGR for OpenCV compatibility
                original_bgr = cv2.cvtColor(original_array, cv2.COLOR_RGB2BGR)

                # Ensure both arrays have the same shape
                if original_bgr.shape != mask_colored.shape:
                    print("Resizing mask to match original image")
                    mask_colored = cv2.resize(mask_colored, (original_bgr.shape[1], original_bgr.shape[0]))

                # Blend original image with mask
                overlay = cv2.addWeighted(original_bgr, 0.7, mask_colored, 0.3, 0)

                # Save result
                segmented_filename = f"seg_{filename}"
                segmented_path = os.path.join(app.config['SEGMENTED_FOLDER'], segmented_filename)
                cv2.imwrite(segmented_path, overlay)

                segmented_results.append({
                    "filename": filename,
                    "segmented": segmented_filename
                })

    return render_template('segmentation.html', results=segmented_results)


if __name__ == '__main__':
    app.run(debug=True)
