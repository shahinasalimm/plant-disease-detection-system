import numpy as np
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing import image
import os

# Load the trained model
model = load_model("tomato_disease_model.h5")

# Class labels (based on the flow_from_directory class indices)
class_labels = ['Bacterial Spot', 'Early Blight', 'Healthy', 'Late Blight', 'Leaf Mold',
                'Septoria Leaf Spot', 'Spider Mites', 'Target Spot', 'Mosaic Virus', 'Yellow Leaf Curl Virus']

def predict_image(img_path):
    # Load and preprocess the image
    img = image.load_img(img_path, target_size=(128, 128))
    img_array = image.img_to_array(img) / 255.0
    img_array = np.expand_dims(img_array, axis=0)

    # Predict the disease
    prediction = model.predict(img_array)
    predicted_class = class_labels[np.argmax(prediction)]
    confidence = np.max(prediction) * 100
    return predicted_class, confidence


# Test with a sample image (provide the path)
test_image_path = "test_images/sample.jpg"  # Replace with your test image path
if os.path.exists(test_image_path):
    predicted_class, confidence = predict_image(test_image_path)
    print(f"✅ Predicted Disease: {predicted_class} (Confidence: {confidence:.2f}%)")
else:
    print(f"❌ Image not found: {test_image_path}")
