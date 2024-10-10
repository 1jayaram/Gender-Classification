import os
import cv2
import numpy as np
import tensorflow as tf
from sklearn.model_selection import train_test_split
from tensorflow.keras import layers, models
from tensorflow.keras.utils import to_categorical

# Set the dataset directory
dataset_dir = r'C:\Users\USER\Downloads\Gender-Detection-master\Gender-Detection-master\gender_dataset_face\Full Dataset'

# Verify dataset directory exists
if not os.path.exists(dataset_dir):
    print(f"The directory {dataset_dir} does not exist. Please check the path.")
    exit()

# Load data function with enhanced debugging
def load_data(dataset_dir):
    images = []
    labels = []
    label_map = {}
    
    for label in os.listdir(dataset_dir):
        label_dir = os.path.join(dataset_dir, label)
        if os.path.isdir(label_dir):
            print(f"Processing label: {label}")
            label_map[label] = len(label_map)  # Assigning label an index
            for filename in os.listdir(label_dir):
                img_path = os.path.join(label_dir, filename)
                if not filename.lower().endswith(('.png', '.jpg', '.jpeg', '.bmp', '.tiff')):
                    print(f"Skipping non-image file: {img_path}")
                    continue
                try:
                    image = cv2.imread(img_path)
                    if image is not None:
                        image = cv2.resize(image, (100, 100))  # Resize to 100x100
                        images.append(image)
                        labels.append(label_map[label])
                    else:
                        print(f"Warning: Could not read image {img_path}")
                except Exception as e:
                    print(f"Error reading {img_path}: {e}")
        else:
            print(f"Skipping non-directory item: {label_dir}")
    
    return np.array(images), np.array(labels), label_map

# Load images and labels
images, labels, label_map = load_data(dataset_dir)

# Check if any images were loaded
print(f"Number of images loaded: {len(images)}")
print(f"Number of labels loaded: {len(labels)}")

# Ensure you have data to split
if len(images) == 0 or len(labels) == 0:
    print("No images or labels loaded. Please check your dataset directory.")
else:
    # Split the data into training and testing sets
    X_train, X_test, y_train, y_test = train_test_split(images, labels, test_size=0.2, random_state=42)

    # Preprocess the data
    X_train = X_train.astype('float32') / 255.0
    X_test = X_test.astype('float32') / 255.0
    y_train = to_categorical(y_train, num_classes=len(label_map))
    y_test = to_categorical(y_test, num_classes=len(label_map))

    # Build the model
    model = models.Sequential()
    model.add(layers.Conv2D(32, (3, 3), activation='relu', input_shape=(100, 100, 3)))
    model.add(layers.MaxPooling2D(pool_size=(2, 2)))
    model.add(layers.Conv2D(64, (3, 3), activation='relu'))
    model.add(layers.MaxPooling2D(pool_size=(2, 2)))
    model.add(layers.Conv2D(128, (3, 3), activation='relu'))
    model.add(layers.MaxPooling2D(pool_size=(2, 2)))
    model.add(layers.Flatten())
    model.add(layers.Dense(128, activation='relu'))
    model.add(layers.Dense(len(label_map), activation='softmax'))

    # Compile the model
    model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])

    # Train the model
    model.fit(X_train, y_train, epochs=10, batch_size=32, validation_data=(X_test, y_test))

    # Evaluate the model
    test_loss, test_acc = model.evaluate(X_test, y_test)
    print(f'Test accuracy: {test_acc}')

    # Save the model
    model.save('gender_detection_model.h5')
