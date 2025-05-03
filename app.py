from flask import Flask, request, render_template
import os
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing import image
import numpy as np

# Initialize Flask app
app = Flask(__name__)

# Load the trained model
model = load_model("tomato_disease_model.h5")

# Class labels (same as in training)
class_labels = ['Bacterial Spot', 'Early Blight', 'Healthy', 'Late Blight', 'Leaf Mold',
                'Septoria Leaf Spot', 'Spider Mites', 'Target Spot', 'Mosaic Virus', 'Yellow Leaf Curl Virus']

# Set folder to save uploaded images (using 'static/uploads')
UPLOAD_FOLDER = "static/uploads/"
app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER
os.makedirs(UPLOAD_FOLDER, exist_ok=True)

def predict_image(img_path):
    """Load and preprocess the image, and make a prediction."""
    img = image.load_img(img_path, target_size=(128, 128))
    img_array = image.img_to_array(img) / 255.0
    img_array = np.expand_dims(img_array, axis=0)

    # Predict the class
    prediction = model.predict(img_array)
    predicted_class = class_labels[np.argmax(prediction)]
    confidence = np.max(prediction) * 100
    return predicted_class, confidence

@app.route("/", methods=["GET", "POST"])
def index():
    """Handle file upload and prediction."""
    prediction_result = None
    confidence_score = None
    uploaded_file_url = None

    if request.method == "POST":
        file = request.files.get("file")
        if file:
            filename = file.filename
            file_path = os.path.join(app.config['UPLOAD_FOLDER'], filename)
            file.save(file_path)

            # Get the prediction
            predicted_class, confidence = predict_image(file_path)
            prediction_result = f"Predicted Disease: {predicted_class}"
            confidence_score = f"Confidence: {confidence:.2f}%"
            uploaded_file_url = f"/static/uploads/{filename}"  # Adjusted for static path

    return render_template("index.html", 
                           prediction_result=prediction_result,
                           confidence_score=confidence_score,
                           uploaded_file_url=uploaded_file_url)

if __name__ == "__main__":
    app.run(debug=True)