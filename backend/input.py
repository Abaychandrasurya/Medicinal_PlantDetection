import os
from flask import Flask, render_template, request, redirect, url_for, send_from_directory
from werkzeug.utils import secure_filename
from pymongo import MongoClient
from pymongo.errors import ConnectionFailure, PyMongoError
import logging

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

UPLOAD_FOLDER = 'upload'
ALLOWED_EXTENSIONS = {'jpg', 'jpeg', 'JPG', 'JPEG'}
DATABASE_HOST = 'localhost'
DATABASE_PORT = 27017
DATABASE_NAME = 'plant'
DATABASE_COLLECTION = 'values'

app = Flask(__name__)
app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER

def connect_to_db():
    """Establishes a connection to the MongoDB database."""
    try:
        client = MongoClient(host=DATABASE_HOST, port=DATABASE_PORT)
        db = client[DATABASE_NAME]
        collection = db[DATABASE_COLLECTION]
        return client, collection
    except ConnectionFailure as e:
        logging.critical(f"Could not connect to MongoDB: {e}")
        return None, None

def allowed_file(filename):
    """Checks if the given filename has an allowed extension."""
    return '.' in filename and filename.rsplit('.', 1)[1] in ALLOWED_EXTENSIONS

def save_file(file):
    """Saves the uploaded file to the upload folder."""
    if not os.path.exists(app.config['UPLOAD_FOLDER']):
        os.makedirs(app.config['UPLOAD_FOLDER'])
    filename = secure_filename(file.filename)
    filepath = os.path.join(app.config['UPLOAD_FOLDER'], filename)
    try:
        file.save(filepath)
        return filepath, filename
    except Exception as e:
        logging.error(f"Error saving file: {e}")
        return None, None

@app.route('/', methods=['GET', 'POST'])
def upload_file():
    """Handles file uploads and redirects to the classification process."""

    if request.method == 'POST':
        # Check if the post request has the file part
        if 'file' not in request.files:
            logging.warning("No file part in the request")
            return redirect(request.url)
        file = request.files['file']
        # If user does not select file, browser also
        # submit an empty part without filename
        if file.filename == '':
            logging.warning("No file selected")
            return redirect(request.url)
        if file and allowed_file(file.filename):
            filepath, filename = save_file(file)
            if filepath:
                return redirect(url_for('classification_process', filename=filename, filepath=filepath))
            else:
                return "Error saving file", 500
        else:
            return "Invalid file type", 400
    return '''
    <!doctype html>
    <html>
    <head>
      <title>Input Image Page</title>
    </head>
    <body>
      <h1>Choose Image</h1>
      <form action="" method=post enctype=multipart/form-data>
        <p>
          <input type=file name=file>
          <input type=submit value=Upload>
        </p>
      </form>
    </body>
    </html>
    '''

@app.route('/classify/<filepath>')
def classification_process(filepath):
    """Processes the uploaded image, performs classification, and retrieves plant data from the database."""

    try:
        image_data = tf.io.read_file(filepath)
        image = tf.image.decode_image(image_data, channels=3)
        image = tf.image.resize(image, [224, 224])  # Resize if needed
        image = tf.expand_dims(image, 0)
    except tf.errors.InvalidArgumentError:
        logging.error(f"Invalid image data for file: {filepath}")
        return "Invalid image file", 400
    except Exception as e:
        logging.error(f"Error processing image: {e}")
        return "Error processing image", 500

    try:
        label_lines = [line.rstrip() for line in open("tf_files/retrained_labels.txt")]
    except FileNotFoundError:
        logging.error("Labels file not found")
        return "Labels file not found", 500
    except Exception as e:
        logging.error(f"Error reading labels file: {e}")
        return "Error reading labels file", 500

    try:
        with open("tf_files/retrained_graph.pb", 'rb') as f:
            graph_def = tf.compat.v1.GraphDef()
            graph_def.ParseFromString(f.read())
            tf.import_graph_def(graph_def, name='')
    except FileNotFoundError:
        logging.error("Graph file not found")
        return "Graph file not found", 500
    except Exception as e:
        logging.error(f"Error loading graph: {e}")
        return "Error loading graph", 500

    try:
        with tf.compat.v1.Session() as sess:
            softmax_tensor = sess.graph.get_tensor_by_name('final_result:0')
            predictions = sess.run(softmax_tensor, {'DecodeJpeg/contents:0': image_data})
            top_k = predictions[0].argsort()[-len(predictions[0]):][::-1]
            arr = {"predictions": {}}
            arr1 = []
            for node_id in top_k:
                human_string = label_lines[node_id]
                score = predictions[0][node_id]
                arr['predictions'][human_string] = str(score)
                arr1.append(human_string)
            arr['result'] = arr1[0]
            confidence = int(round(float(arr["predictions"][arr1[0]]) * 100))
            arr['result_confidence'] = str(confidence) + "%"

            return redirect(url_for('show_output_page', plant=arr1[0]))

    except Exception as e:
        logging.error(f"Error during classification: {e}")
        return "Error during classification", 500

@app.route('/Output/<plant>')
def show_output_page(plant):
    """Displays the output page with plant information."""

    plant_data = get_plant_data(plant.title())
    if plant_data:
        return render_template('boutput.html',
                               hPlantName=plant.title(),
                               hBotanicalName='Botanical Name:',
                               botanicalName=plant_data.get('botanicalName', 'N/A'),
                               hFamily='Family:',
                               family=plant_data.get('family', 'N/A'),
                               hAbout='About:',
                               about=plant_data.get('about', 'N/A'),
                               hMedicinalUses='Medicinal Uses:',
                               medicinalUses=plant_data.get('medicinalUses', 'N/A'))
    else:
        return "Plant data not found", 404

def get_plant_data(plant_name):
    """Retrieves plant data from the database."""

    client, collection = connect_to_db()
    if client:
        try:
            plant_data = collection.find_one({"plantName": plant_name})
            client.close()
            return plant_data
        except PyMongoError as e:
            logging.error(f"Database error: {e}")
            return None
    else:
        return None

@app.route('/getBotanicalName/<plant>')
def get_botanical_name(plant):
    """Retrieves the botanical name of the plant."""

    plant_data = get_plant_data(plant.title())
    if plant_data:
        return jsonify({'botanicalName': plant_data.get('botanicalName', 'N/A')})
    else:
        return jsonify({'error': 'Botanical name not found'}), 404

@app.route('/getFamily/<plant>')
def get_family(plant):
    """Retrieves the family of the plant."""

    plant_data = get_plant_data(plant.title())
    if plant_data:
        return jsonify({'family': plant_data.get('family', 'N/A')})
    else:
        return jsonify({'error': 'Family not found'}), 404

@app.route('/getAbout/<plant>')
def get_about(plant):
    """Retrieves information about the plant."""

    plant_data = get_plant_data(plant.title())
    if plant_data:
        return jsonify({'about': plant_data.get('about', 'N/A')})
    else:
        return jsonify({'error': 'About information not found'}), 404

@app.route('/getMedicinalValues/<plant>')
def get_medicinal_values(plant):
    """Retrieves the medicinal uses of the plant."""

    plant_data = get_plant_data(plant.title())
    if plant_data:
        return jsonify({'medicinalUses': plant_data.get('medicinalUses', 'N/A')})
    else:
        return jsonify({'error': 'Medicinal uses not found'}), 404

if __name__ == "__main__":
    app.run(debug=True)