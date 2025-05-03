# Updated train_model.py
import os
import tensorflow as tf
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.applications import MobileNetV2
from tensorflow.keras.layers import Dense, Flatten, Dropout
from tensorflow.keras.models import Model
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.callbacks import EarlyStopping, ModelCheckpoint

# Set dataset paths
train_dir = "Dataset/train"
val_dir = "Dataset/val"

# Check if dataset folders exist
if not os.path.exists(train_dir) or not os.path.exists(val_dir):
    raise FileNotFoundError("Dataset folders not found. Check your paths.")

# Data augmentation and normalization
train_datagen = ImageDataGenerator(
    rescale=1.0 / 255,
    horizontal_flip=True,
    zoom_range=0.3,
    rotation_range=20,
    shear_range=0.2,
    width_shift_range=0.2,
    height_shift_range=0.2
)
val_datagen = ImageDataGenerator(rescale=1.0 / 255)

# Load images from directory
train_set = train_datagen.flow_from_directory(train_dir, target_size=(128, 128), batch_size=32, class_mode="categorical")
val_set = val_datagen.flow_from_directory(val_dir, target_size=(128, 128), batch_size=32, class_mode="categorical")

# Print class-to-index mapping
print("Class-to-Index Mapping:", train_set.class_indices)

# Load Pretrained MobileNetV2 (without top layers)
base_model = MobileNetV2(input_shape=(128, 128, 3), include_top=False, weights="imagenet")

# Freeze all base model layers initially
base_model.trainable = False

# Add custom layers
x = Flatten()(base_model.output)
x = Dense(128, activation="relu")(x)
x = Dropout(0.3)(x)
output_layer = Dense(10, activation="softmax")(x)  # 10 classes

# Create the model
model = Model(inputs=base_model.input, outputs=output_layer)

# Compile the model
model.compile(optimizer=Adam(learning_rate=1e-3), loss="categorical_crossentropy", metrics=["accuracy"])

# Train the top layers first (base model frozen)
model.fit(train_set, epochs=10, validation_data=val_set)

# Unfreeze some layers for fine-tuning
for layer in base_model.layers[-20:]:
    layer.trainable = True

# Recompile with a lower learning rate for fine-tuning
model.compile(optimizer=Adam(learning_rate=1e-4), loss="categorical_crossentropy", metrics=["accuracy"])

# Define callbacks for early stopping and model checkpointing
checkpoint = ModelCheckpoint("best_model.h5", monitor="val_accuracy", save_best_only=True)
early_stopping = EarlyStopping(monitor="val_loss", patience=5, restore_best_weights=True)

# Fine-tune the model
model.fit(train_set, epochs=25, validation_data=val_set, callbacks=[checkpoint, early_stopping])

# Save the fine-tuned model
model.save("tomato_disease_model.h5")
print("\u2705 Fine-tuned model saved as tomato_disease_model.h5")