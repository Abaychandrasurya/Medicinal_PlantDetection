import os
import numpy as np
import matplotlib.pyplot as plt
import tensorflow as tf
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.models import Sequential, load_model
from tensorflow.keras.layers import Conv2D, MaxPooling2D, Flatten, Dense, Dropout
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.applications import MobileNetV2
from tensorflow.keras.models import Model
import logging

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

def create_generators(dataset_path, validation_split=0.2, target_size=(224, 224), batch_size=32):
    """
    Creates ImageDataGenerator objects for training and validation.

    Args:
        dataset_path (str): Path to the directory containing the image dataset.
        validation_split (float, optional): Fraction of data to use for validation. Defaults to 0.2.
        target_size (tuple, optional): The target size for the images. Defaults to (224, 224).
        batch_size (int, optional): The batch size for the generators. Defaults to 32.

    Returns:
        tuple: A tuple containing the train_generator and validation_generator.
               Returns (None, None) if an error occurs.
    """
    try:
        datagen = ImageDataGenerator(
            rescale=1.0 / 255,  # Normalize images
            rotation_range=30,
            width_shift_range=0.2,
            height_shift_range=0.2,
            shear_range=0.2,
            zoom_range=0.2,
            horizontal_flip=True,
            validation_split=validation_split  # Splitting data for validation
        )

        train_generator = datagen.flow_from_directory(
            dataset_path,
            target_size=target_size,
            batch_size=batch_size,
            class_mode="categorical",
            subset="training"
        )

        validation_generator = datagen.flow_from_directory(
            dataset_path,
            target_size=target_size,
            batch_size=batch_size,
            class_mode="categorical",
            subset="validation"
        )

        return train_generator, validation_generator

    except FileNotFoundError:
        logging.error(f"Dataset directory not found: {dataset_path}")
        return None, None
    except Exception as e:
        logging.error(f"Error creating ImageDataGenerators: {e}")
        return None, None

def create_model(input_shape, num_classes):
    """
    Creates the MobileNetV2-based model.

    Args:
        input_shape (tuple): Shape of the input images (e.g., (224, 224, 3)).
        num_classes (int): Number of classes in the dataset.

    Returns:
        Model: The compiled TensorFlow Keras model.
               Returns None if an error occurs.
    """
    try:
        # Load MobileNetV2 as a base model
        base_model = MobileNetV2(weights="imagenet", include_top=False, input_shape=input_shape)

        # Freeze base model layers
        for layer in base_model.layers:
            layer.trainable = False

        # Add custom classification layers
        x = Flatten()(base_model.output)
        x = Dense(256, activation="relu")(x)
        x = Dropout(0.5)(x)
        output = Dense(num_classes, activation="softmax")(x)  # Changed variable name

        # Create model
        model = Model(inputs=base_model.input, outputs=output)  # Changed variable name
        model.compile(optimizer=Adam(learning_rate=0.0001), loss="categorical_crossentropy", metrics=["accuracy"])

        return model

    except Exception as e:
        logging.error(f"Error creating model: {e}")
        return None

def train_model(model, train_generator, validation_generator, epochs=15):
    """
    Trains the given model.

    Args:
        model (Model): The TensorFlow Keras model to train.
        train_generator (ImageDataGenerator): Generator for training data.
        validation_generator (ImageDataGenerator): Generator for validation data.
        epochs (int, optional): Number of training epochs. Defaults to 15.

    Returns:
        History: The training history object.
                 Returns None if an error occurs.
    """
    try:
        history = model.fit(
            train_generator,
            validation_data=validation_generator,
            epochs=epochs,
            verbose=1
        )
        return history
    except Exception as e:
        logging.error(f"Error training model: {e}")
        return None

def save_model(model, filepath="plant_model.h5"):
    """
    Saves the trained model to a file.

    Args:
        model (Model): The trained TensorFlow Keras model.
        filepath (str, optional): Path to save the model. Defaults to "plant_model.h5".
    """
    try:
        model.save(filepath)
        logging.info(f"Model saved as {filepath}")
    except Exception as e:
        logging.error(f"Error saving model: {e}")

def plot_training_results(history):
    """
    Plots the training and validation accuracy.

    Args:
        history (History): The training history object.
    """
    try:
        plt.plot(history.history["accuracy"], label="Train Accuracy")
        plt.plot(history.history["val_accuracy"], label="Validation Accuracy")
        plt.xlabel("Epochs")
        plt.ylabel("Accuracy")
        plt.legend()
        plt.show()
    except Exception as e:
        logging.error(f"Error plotting training results: {e}")

if __name__ == "__main__":
    dataset_path = r"C:\projects\Medicinal-PlantDetection\training_plant_images"  # Replace with your actual dataset path
    target_size = (224, 224)
    batch_size = 32
    epochs = 15
    model_filename = "plant_model.h5"

    # Create data generators
    train_generator, validation_generator = create_generators(dataset_path, target_size=target_size, batch_size=batch_size)

    if train_generator and validation_generator:
        # Get number of classes
        num_classes = train_generator.num_classes
        logging.info(f"Classes detected: {train_generator.class_indices}")

        # Create model
        model = create_model(target_size + (3,), num_classes)  # Input shape: (224, 224, 3)

        if model:
            # Train model
            history = train_model(model, train_generator, validation_generator, epochs=epochs)

            if history:
                # Save model
                save_model(model, model_filename)

                # Plot training results
                plot_training_results(history)
            else:
                logging.error("Model training failed.")
        else:
            logging.error("Model creation failed.")
    else:
        logging.error("Data generator creation failed.")