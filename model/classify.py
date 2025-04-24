import tensorflow as tf
import sys
import os
import json
import logging

# Set TensorFlow logging level (0: all, 1: INFO, 2: WARNING, 3: ERROR)
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'

def classify_image(image_path, label_path, graph_path):
    """
    Classifies an image using a TensorFlow graph.

    Args:
        image_path (str): Path to the image file.
        label_path (str): Path to the labels file.
        graph_path (str): Path to the TensorFlow graph file (.pb).

    Returns:
        dict: A dictionary containing classification predictions.
    """

    try:
        # 1. Load image data
        with open(image_path, 'rb') as image_file:
            image_data = image_file.read()
    except FileNotFoundError:
        logging.error(f"Error: Image file not found at {image_path}")
        return None
    except Exception as e:
        logging.error(f"Error reading image file: {e}")
        return None

    try:
        # 2. Load labels
        with open(label_path, 'r') as label_file:
            label_lines = [line.rstrip() for line in label_file]
    except FileNotFoundError:
        logging.error(f"Error: Labels file not found at {label_path}")
        return None
    except Exception as e:
        logging.error(f"Error reading labels file: {e}")
        return None

    try:
        # 3. Load the TensorFlow graph
        with tf.io.gfile.GFile(graph_path, 'rb') as graph_file:
            graph_def = tf.compat.v1.GraphDef()  # Corrected this line
            graph_def.ParseFromString(graph_file.read())
            tf.import_graph_def(graph_def, name='')  # Import the graph
    except FileNotFoundError:
        logging.error(f"Error: Graph file not found at {graph_path}")
        return None
    except Exception as e:
        logging.error(f"Error loading graph: {e}")
        return None

    try:
        # 4. Run the classification
        with tf.compat.v1.Session() as sess:  # And this line
            # Get the softmax tensor
            softmax_tensor = sess.graph.get_tensor_by_name('final_result:0')  # TODO: Make tensor name configurable
            # Run the inference
            predictions = sess.run(softmax_tensor, {'DecodeJpeg/contents:0': image_data})  # TODO: Make input tensor name configurable

            # Process the results
            top_k = predictions[0].argsort()[-len(predictions[0]):][::-1]  # Get top K results
            results = {"predictions": {}}
            for node_id in top_k:
                human_string = label_lines[node_id]
                score = float(predictions[0][node_id])  # Ensure score is a Python float
                results["predictions"][human_string] = f"{score:.4f}"  # Format the score

            results["result"] = label_lines[top_k[0]]  # Top prediction
            confidence = int(round(float(predictions[0][top_k[0]]) * 100))
            results["result_confidence"] = f"{confidence}%"

            return results

    except Exception as e:
        logging.error(f"Error during TensorFlow session: {e}")
        return None


if __name__ == "__main__":
    # Configure logging
    logging.basicConfig(level=logging.INFO,
                        format='%(asctime)s - %(levelname)s - %(message)s')

    # Validate command-line arguments
    if len(sys.argv) != 2:
        logging.error("Usage: python classify.py <image_path>")
        sys.exit(1)

    image_path = sys.argv[1]
    label_path = "tf_files/retrained_labels.txt"  # TODO: Make these configurable via arguments
    graph_path = "tf_files/retrained_graph.pb"

    results = classify_image(image_path, label_path, graph_path)

    if results:
        print(json.dumps(results, indent=2))  # Print the JSON results with indentation
    else:
        print("Classification failed.")