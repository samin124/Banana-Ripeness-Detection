from flask import Flask, render_template, request
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing import image
import numpy as np
import os

app = Flask(__name__)

# Load your model
model = load_model("banana_ripeness_model_finetuned.h5")

# Folder to store uploaded images
UPLOAD_FOLDER = os.path.join('static', 'uploads')

app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER

# Class labels
class_names = ['A (Unripe)', 'B (Mid-ripe)', 'C (Ripe)', 'D (Overripe)']

@app.route('/', methods=['GET', 'POST'])
def index():
    prediction = None
    filename = None
    probabilities = []

    if request.method == 'POST':
        file = request.files['file']
        if file:
            filename = file.filename
            filepath = os.path.join(app.config['UPLOAD_FOLDER'], filename)
            file.save(filepath)

            # Preprocess the image
            img = image.load_img(filepath, target_size=(224, 224))
            img_array = image.img_to_array(img) / 255.0
            img_array = np.expand_dims(img_array, axis=0)

            # Predict
            preds = model.predict(img_array)[0]

            # Convert to list of native Python floats for JSON serialization
            probabilities = list(map(lambda x: float(round(x * 100, 2)), preds.tolist()))

            # Get top prediction
            predicted_class = class_names[np.argmax(preds)]
            confidence = round(np.max(preds) * 100, 2)
            prediction = f"{predicted_class} (Confidence: {confidence}%)"

    return render_template(
        'index.html',
        prediction=prediction,
        filename=filename,
        class_names=class_names,
        probabilities=probabilities
    )

