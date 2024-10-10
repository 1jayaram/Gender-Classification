import tensorflow as tf
from tensorflow.keras.preprocessing import image
import numpy as np
import gradio as gr

# Load a pre-trained model (replace with the correct path to your model)
model = tf.keras.models.load_model(r'C:\Users\USER\Downloads\Gender-Detection-master\Gender-Detection-master\gender_dataset_face\gender_classification_model.h5')

# Define a function to predict gender
def classify_gender(img):
    img = img.resize((224, 224))  # Resize image to match model input size
    img_array = image.img_to_array(img)
    img_array = np.expand_dims(img_array, axis=0)  # Add batch dimension
    img_array /= 255.0  # Normalize

    prediction = model.predict(img_array)[0][0]  # Assuming the model outputs probabilities
    if prediction < 0.5:
        return "Man"
    else:
        return "Woman"

# Create Gradio Interface
interface = gr.Interface(fn=classify_gender, 
                         inputs=gr.Image(type="pil"),  # Updated syntax
                         outputs="text",
                         title="Gender Detection",
                         description="Upload an image to detect gender")

# Launch the interface
interface.launch(share=True)