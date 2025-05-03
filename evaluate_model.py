import tensorflow as tf
from tensorflow.keras.models import load_model #type: ignore
from tensorflow.keras.preprocessing.image import ImageDataGenerator #type: ignore
from sklearn.metrics import classification_report, confusion_matrix
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np

# Load the trained model
model_path = "tomato_disease_model.h5"  # Update this if your model is saved elsewhere
model = load_model(model_path)

# Path to validation data
val_data_path = "Dataset/val"  # Adjust path as per your folder structure

# ImageDataGenerator for validation set
val_datagen = ImageDataGenerator(rescale=1.0/255.0)
val_generator = val_datagen.flow_from_directory(
    val_data_path,
    target_size=(128, 128),
    batch_size=32,
    class_mode='categorical',
    shuffle=False
)

# Evaluate the model on validation data
loss, accuracy = model.evaluate(val_generator)
print(f"Validation Loss: {loss:.4f}")
print(f"Validation Accuracy: {accuracy:.4f}")

# Get class labels from the generator
class_names = list(val_generator.class_indices.keys())
print(f"Class Labels: {class_names}")

# Predict on validation set
y_pred = model.predict(val_generator)
y_pred_classes = np.argmax(y_pred, axis=1)  # Convert predictions to class indices
y_true = val_generator.classes  # True labels

# Classification Report
print("\nClassification Report:")
print(classification_report(y_true, y_pred_classes, target_names=class_names))

# Confusion Matrix
conf_matrix = confusion_matrix(y_true, y_pred_classes)

# Plot Confusion Matrix
plt.figure(figsize=(10, 8))
sns.heatmap(conf_matrix, annot=True, cmap="Blues", fmt="d", xticklabels=class_names, yticklabels=class_names)
plt.xlabel("Predicted Labels")
plt.ylabel("True Labels")
plt.title("Confusion Matrix")
plt.show()

# Optional: Plot training and validation accuracy/loss if history is saved
try:
    import pickle
    # Load history (update the path if your training history is saved)
    with open("history.pkl", "rb") as file:
        history = pickle.load(file)
    
    # Plot Training vs. Validation Accuracy
    plt.figure(figsize=(10, 5))
    plt.plot(history['accuracy'], label='Training Accuracy')
    plt.plot(history['val_accuracy'], label='Validation Accuracy')
    plt.xlabel('Epochs')
    plt.ylabel('Accuracy')
    plt.title('Training vs Validation Accuracy')
    plt.legend()
    plt.show()

    # Plot Training vs. Validation Loss
    plt.figure(figsize=(10, 5))
    plt.plot(history['loss'], label='Training Loss')
    plt.plot(history['val_loss'], label='Validation Loss')
    plt.xlabel('Epochs')
    plt.ylabel('Loss')
    plt.title('Training vs Validation Loss')
    plt.legend()
    plt.show()

except FileNotFoundError:
    print("Training history not found. Skipping accuracy/loss plot.")
