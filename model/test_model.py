import os
import numpy as np
import matplotlib.pyplot as plt
import tensorflow as tf
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.models import load_model
import logging

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

def create_test_generator(test_dataset_path, target_size=(224, 224), batch_size=32):
    """
    Creates an ImageDataGenerator for the test dataset.

    Args:
        test_dataset_path (str): Path to the directory containing the test images.
        target_size (tuple, optional): The target size for the images. Defaults to (224, 224).
        batch_size (int, optional): The batch size for the generator. Defaults to 32.

    Returns:
        ImageDataGenerator: The test data generator.
                          Returns None if an error occurs.
    """
    try:
        test_datagen = ImageDataGenerator(rescale=1.0 / 255)  # Only rescale for testing

        test_generator = test_datagen.flow_from_directory(
            test_dataset_path,
            target_size=target_size,
            batch_size=batch_size,
            class_mode="categorical",
            shuffle=False  # No shuffling for evaluation
        )

        return test_generator

    except FileNotFoundError:
        logging.error(f"Test dataset directory not found: {test_dataset_path}")
        return None
    except Exception as e:
        logging.error(f"Error creating test generator: {e}")
        return None

def load_trained_model(model_path):
    """
    Loads the trained model from the given path.

    Args:
        model_path (str): Path to the trained model file (.h5).

    Returns:
        Model: The loaded TensorFlow Keras model.
               Returns None if an error occurs.
    """
    try:
        model = load_model(model_path)
        logging.info(f"Model loaded successfully from {model_path}")
        return model
    except FileNotFoundError:
        logging.error(f"Model file not found: {model_path}")
        return None
    except Exception as e:
        logging.error(f"Error loading model: {e}")
        return None

def evaluate_model(model, test_generator):
    """
    Evaluates the model on the test set.

    Args:
        model (Model): The trained TensorFlow Keras model.
        test_generator (ImageDataGenerator): The test data generator.

    Returns:
        tuple: A tuple containing (test_loss, test_accuracy).
               Returns (None, None) if an error occurs.
    """
    try:
        test_loss, test_acc = model.evaluate(test_generator, verbose=1)
        logging.info(f"Test Loss: {test_loss:.4f}, Test Accuracy: {test_acc * 100:.2f}%")
        return test_loss, test_acc
    except Exception as e:
        logging.error(f"Error evaluating model: {e}")
        return None, None

def predict_and_visualize(model, test_generator, class_labels, num_samples=9):
    """
    Makes predictions on a few samples from the test set and visualizes the results.

    Args:
        model (Model): The trained TensorFlow Keras model.
        test_generator (ImageDataGenerator): The test data generator.
        class_labels (list): List of class labels.
        num_samples (int, optional): Number of samples to visualize. Defaults to 9.
    """
    try:
        # Get a batch of images and labels from the test generator
        images, labels = next(test_generator)

        # Make predictions
        predictions = model.predict(images)
        predicted_classes = np.argmax(predictions, axis=1)
        true_classes = np.argmax(labels, axis=1)  # Get true class indices

        # Display sample test images with predictions
        fig, axes = plt.subplots(3, 3, figsize=(10, 10))
        axes = axes.ravel()

        for i in range(num_samples):
            axes[i].imshow(images[i])
            axes[i].set_title(f"Predicted: {class_labels[predicted_classes[i]]}\nActual: {class_labels[true_classes[i]]}")
            axes[i].axis("off")

        plt.tight_layout()
        plt.show()

    except Exception as e:
        logging.error(f"Error during prediction and visualization: {e}")

if __name__ == "__main__":
    test_dataset_path = r"C:\Users\DELL\Documents\mini proj\dataset\test"  # Replace with your actual test dataset path
    model_path = "plant_model.h5"
    target_size = (224, 224)
    batch_size = 32

    # Create test generator
    test_generator = create_test_generator(test_dataset_path, target_size=target_size, batch_size=batch_size)

    if test_generator:
        # Load trained model
        model = load_trained_model(model_path)

        if model:
            # Get class labels
            class_labels = list(test_generator.class_indices.keys())
            logging.info(f"Class Labels: {class_labels}")

            # Evaluate model
            test_loss, test_acc = evaluate_model(model, test_generator)

            if test_loss is not None and test_acc is not None:
                # Predict and visualize results
                predict_and_visualize(model, test_generator, class_labels)
            else:
                logging.error("Model evaluation failed.")
        else:
            logging.error("Model loading failed.")
    else:
        logging.error("Test generator creation failed.")