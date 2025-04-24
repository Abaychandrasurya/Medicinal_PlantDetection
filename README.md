# Medicinal Plant Identification System 🌿

This project identifies medicinal plants using deep learning and provides information about their uses via a Flask backend and MongoDB.

## 🔧 Features
- Upload plant images and identify species using a trained CNN (MobileNetV2).
- Use a responsive frontend for user interaction.

Happy Identifying! 🌱
''' TEXT
Medicinal-Plant-Detection/
│
├── backend/
│   ├── input.py     # Flask backend for image upload and classification
│   └── app.py       # Flask app for serving the model
│
├── frontend/
│   └── index.html   # HTML frontend for user interface
│
├── model/
│   ├── train_model.py   # Script to train the plant classification model
│   ├── test_model.py    # Script to test the trained model
│   ├── medicinal_plant.py
│   │   ├── classify.py  # Script for image classification (TensorFlow)
│   │   ├── cnn.py       # Script for defining a CNN model (TensorFlow)
│   │   └── edgeDetection.py # Script for edge detection
│   └── plant_model.h5   # The trained model file (will be created after training)
│
├── utils/
│   └── edgeDetection.py # Utility script for edge detection (duplicate)
│
├── tf_files/          # Directory for TensorFlow-related files (labels, graph)
│   ├── retrained_graph.pb   # Trained TensorFlow graph
│   └── retrained_labels.txt  # Labels for the plant classes
│
├── README.md          # Project README file
│
└── classify.py      # Script for image classification (duplicate)
'''

Steps to Run the Project

Prerequisites:

Python 3.x: Ensure you have Python 3.x installed.
pip: Python's package installer.
TensorFlow: Install TensorFlow (CPU or GPU version). pip install tensorflow or pip install tensorflow-gpu
Flask: Install Flask. pip install Flask
NumPy: Install NumPy. pip install numpy
opencv-python: Install OpenCV. pip install opencv-python
Pillow: Install Pillow . pip install Pillow
PyMongo: Install PyMongo . pip install PyMongo

Matplotlib: Install Matplotlib (if you want to run training scripts). pip install matplotlib

Setup and Execution Guide
1. Prepare Your Environment
- Ensure you have Python 3.12 installed (TensorFlow does not yet support Python 3.13).
- Set up a virtual environment to manage dependencies:python3.12 -m venv tensorflow_env

- Activate the virtual environment:tensorflow_env\Scripts\activate  # On Windows
source tensorflow_env/bin/activate  # On macOS/Linux

- Upgrade pip and install required dependencies:pip install --upgrade pip setuptools
pip install tensorflow Flask pymongo


2. Prepare Your Dataset
- Place your medicinal plant images in a directory. The structure should be:dataset/
├── Aloe_Vera/
│   ├── image1.jpg
│   ├── image2.jpg
├── Tulsi/
│   ├── image1.jpg
│   ├── image2.jpg

- Modify the dataset_path variable in model/train_model.py:dataset_path = r"path/to/your/plant/images"  # Replace with your path


3. Train the Model (Optional)
- If you want to train or retrain the model, run:python model/train_model.py

- This will create the plant_model.h5 file in the model/ directory.

4. Prepare TensorFlow Files (If Necessary)
- Ensure tf_files/ contains:retrained_graph.pb  # TensorFlow graph
retrained_labels.txt  # Plant labels

- If missing, generate them using a TensorFlow retraining script (retrain.py).

5. Start the Backend
- Navigate to backend/ and run:python backend/input.py  # OR
python backend/app.py

- Flask server starts at http://127.0.0.1:5000/.

6. Open the Frontend
- Open frontend/index.html in your web browser.

7. Use the Application
- Log in (basic authentication setup).
- Upload or capture plant images.
- Click "Identify Plant" to classify images.
- View results and predictions.



