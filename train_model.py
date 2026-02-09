import tensorflow as tf
from tensorflow.keras.layers import Input, Conv2D, MaxPooling2D, UpSampling2D
from tensorflow.keras.models import Model
from tensorflow.keras.preprocessing.image import ImageDataGenerator
import os
import numpy as np

# --- Configuration ---
IMG_SIZE = 128
CHANNELS = 3
EPOCHS = 30 
BATCH_SIZE = 32
# --- CHANGE THIS PATH TO POINT TO THE NEW IMAGE FOLDER ---
# The ImageDataGenerator needs the root path to find the sub-class folder.
TRAINING_ROOT_DIR = 'data/' 
# The new sub-directory created by extract_faces.py to store the images
REAL_CLASS_SUBDIR = 'Real_Frames' 
MODEL_PATH = 'deepfake_cae_model.h5'

# 1. Convolutional Autoencoder Model Definition (No changes needed)
def build_cae(input_shape):
    # ... (Keep the existing build_cae function) ...
    input_img = Input(shape=input_shape)

    # === Encoder ===
    x = Conv2D(32, (3, 3), activation='relu', padding='same')(input_img)
    x = MaxPooling2D((2, 2), padding='same')(x) 
    x = Conv2D(64, (3, 3), activation='relu', padding='same')(x)
    encoded = MaxPooling2D((2, 2), padding='same')(x) # Bottleneck

    # === Decoder ===
    x = Conv2D(64, (3, 3), activation='relu', padding='same')(encoded)
    x = UpSampling2D((2, 2))(x) 
    x = Conv2D(32, (3, 3), activation='relu', padding='same')(x)
    x = UpSampling2D((2, 2))(x) 
    decoded = Conv2D(CHANNELS, (3, 3), activation='sigmoid', padding='same')(x) 

    autoencoder = Model(input_img, decoded)
    autoencoder.compile(optimizer='adam', loss='mse') 
    
    return autoencoder

# 2. Data Generator (UPDATED function call to use the new path)
def create_data_generator(root_dir, target_size, batch_size, target_class):
    """Creates a data generator, filtering to load ONLY images from the target_class folder."""
    datagen = ImageDataGenerator(rescale=1./255) 
    
    generator = datagen.flow_from_directory(
        root_dir,
        # IMPORTANT: Now looks in the 'Real_Frames' sub-folder
        classes=[target_class], 
        target_size=target_size,
        batch_size=batch_size,
        class_mode='input',  # X=Y for autoencoders
        shuffle=True
    )
    return generator

# 3. Main Training Block
if __name__ == '__main__':
    # ... (Keep the existing main training block, it uses the updated variables) ...
    
    # 1. Create Data Generator
    target_size = (IMG_SIZE, IMG_SIZE)
    train_generator = create_data_generator(
        TRAINING_ROOT_DIR,  # 'data/'
        target_size, 
        BATCH_SIZE,
        REAL_CLASS_SUBDIR   # 'Real_Frames'
    )

    # 2. Build and Compile Model
    input_shape = (IMG_SIZE, IMG_SIZE, CHANNELS)
    autoencoder = build_cae(input_shape)
    autoencoder.summary()
    
    # 3. Training
    print(f"\nStarting Autoencoder Training on ONLY '{REAL_CLASS_SUBDIR}' images...")
    
    if train_generator.samples == 0:
        print(f"ERROR: No images found in {os.path.join(TRAINING_ROOT_DIR, REAL_CLASS_SUBDIR)}. Run 'python extract_faces.py' first.")
    else:
        history = autoencoder.fit(
            train_generator,
            steps_per_epoch=train_generator.samples // BATCH_SIZE,
            epochs=EPOCHS
        )

        # 4. Save the model
        autoencoder.save(MODEL_PATH)
        print(f"\nTraining finished. Model saved as {MODEL_PATH}")