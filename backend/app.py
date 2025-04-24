from flask import Flask, request, jsonify
import numpy as np
import tensorflow as tf
from tensorflow.keras.preprocessing import image
import os
import logging
from werkzeug.utils import secure_filename

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

app = Flask(__name__)

# Define the directory where uploaded images will be temporarily stored
UPLOAD_FOLDER = 'uploads'
app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER

# Ensure the upload directory exists
if not os.path.exists(UPLOAD_FOLDER):
    os.makedirs(UPLOAD_FOLDER)

# Load the trained model
try:
    model = tf.keras.models.load_model('plant_model.h5')
    logging.info("Model loaded successfully.")
except IOError:
    logging.critical("Failed to load the model. Ensure 'plant_model.h5' exists in the correct location.")
    # Consider exiting the application if the model is crucial
    # raise SystemExit("Model loading failed.")  # Or sys.exit(1)
    model = None  # Set model to None to prevent further errors

# Define class labels (replace with your actual class names)
CLASS_LABELS = ['class1', 'class2', 'class3']

def preprocess_image(img_path, target_size=(224, 224)):
    """
    Loads and preprocesses an image for model input.

    Args:
        img_path (str): Path to the image file.
        target_size (tuple): The desired size of the image.

    Returns:
        numpy.ndarray: Preprocessed image as a numpy array, or None on error.
    """
    try:
        img = image.load_img(img_path, target_size=target_size)
        img_array = image.img_to_array(img) / 255.0  # Normalize to [0, 1]
        img_array = np.expand_dims(img_array, axis=0)  # Add batch dimension
        return img_array
    except FileNotFoundError:
        logging.error(f"Image file not found: {img_path}")
        return None
    except Exception as e:
        logging.error(f"Error preprocessing image: {e}")
        return None

def predict_class(model, img_array, class_labels):
    """
    Predicts the class of the input image.

    Args:
        model: The trained TensorFlow Keras model.
        img_array (numpy.ndarray): Preprocessed image.
        class_labels (list): List of class labels.

    Returns:
        str: Predicted class name, or None if prediction fails.
    """
    if model is None:
        logging.error("Cannot predict without a loaded model.")
        return None

    try:
        predictions = model.predict(img_array)
        predicted_class_index = np.argmax(predictions, axis=1)[0]
        return class_labels[predicted_class_index]
    except Exception as e:
        logging.error(f"Error during prediction: {e}")
        return None

@app.route('/predict', methods=['POST'])
def predict():
    """
    Handles image upload and prediction.

    Returns:
        jsonify: JSON response containing the predicted class.
    """
    if 'file' not in request.files:
        logging.warning("No file provided in request.")
        return jsonify({"error": "No file provided"}), 400

    file = request.files['file']

    if file.filename == '':
        logging.warning("No selected file.")
        return jsonify({"error": "No selected file"}), 400

    # Sanitize the filename
    filename = secure_filename(file.filename)
    if not filename:
        logging.error("Invalid filename.")
        return jsonify({"error": "Invalid filename"}), 400

    img_path = os.path.join(app.config['UPLOAD_FOLDER'], filename)

    try:
        file.save(img_path)  # Save the file to the upload directory
        logging.info(f"File saved successfully to {img_path}")
    except Exception as e:
        logging.error(f"Error saving the uploaded file: {e}")
        return jsonify({"error": "Failed to save the file"}), 500

    img_array = preprocess_image(img_path)
    if img_array is None:
        os.remove(img_path)  # Remove the corrupted file
        return jsonify({"error": "Failed to preprocess the image"}), 400

    predicted_class = predict_class(model, img_array, CLASS_LABELS)
    os.remove(img_path)  # Remove the temporary file

    if predicted_class:
        return jsonify({"class": predicted_class})
    else:
        return jsonify({"error": "Failed to predict the class"}), 500

if __name__ == '__main__':
    app.run(debug=True)