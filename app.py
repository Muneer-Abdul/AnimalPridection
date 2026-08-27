# ============================================================
# Animal Classifier - Flask Application
# ============================================================
# This application:
# 1. Loads a trained TensorFlow/Keras model
# 2. Accepts an image from the frontend
# 3. Preprocesses the image
# 4. Uses the model to classify the image
# 5. Returns the prediction and confidence as JSON
#
# Expected classes:
#   - Animal
#   - Not Animal
# ============================================================


# ============================================================
# IMPORTS
# ============================================================

import os

# Reduce TensorFlow informational messages.
# This MUST be set before importing TensorFlow.
os.environ["TF_CPP_MIN_LOG_LEVEL"] = "2"

import traceback

import numpy as np

from PIL import Image, UnidentifiedImageError

from flask import Flask, render_template, request, jsonify

import tensorflow as tf


# ============================================================
# TENSORFLOW RESOURCE LIMITS
# ============================================================
# These settings reduce the number of CPU threads TensorFlow
# uses. This can help prevent excessive CPU usage, especially
# on smaller computers or free cloud hosting platforms.
# ============================================================

try:
    tf.config.threading.set_intra_op_parallelism_threads(1)
    tf.config.threading.set_inter_op_parallelism_threads(1)
except Exception:
    # If TensorFlow threading has already been configured,
    # continue without stopping the application.
    pass


# ============================================================
# FLASK APPLICATION
# ============================================================

app = Flask(__name__)


# ============================================================
# SETTINGS
# ============================================================

# Get the directory where this app.py file is located.
# This makes the model path more reliable.
BASE_DIR = os.path.dirname(os.path.abspath(__file__))

# Path to the trained Keras model.
MODEL_PATH = os.path.join(
    BASE_DIR,
    "animal_classifier.keras"
)

# Image size expected by the model.
IMAGE_SIZE = (224, 224)

# Maximum upload size: 10 MB.
app.config["MAX_CONTENT_LENGTH"] = 10 * 1024 * 1024


# ============================================================
# CLASS NAMES
# ============================================================
# IMPORTANT:
#
# These class names MUST match the order used when the model
# was trained.
#
# For a two-output model:
#
#   index 0 = Animal
#   index 1 = Not Animal
#
# If your training code used the opposite order, change these.
# ============================================================

CLASSES = [
    "Animal",
    "Not Animal"
]


# ============================================================
# LOAD MODEL
# ============================================================

print("======================================")
print("ANIMAL CLASSIFIER STARTING")
print("======================================")

model = None

try:

    # --------------------------------------------------------
    # Check whether the model file exists
    # --------------------------------------------------------

    if not os.path.exists(MODEL_PATH):

        print("MODEL FILE NOT FOUND!")
        print("Expected location:")
        print(MODEL_PATH)

    else:

        print("Loading model...")
        print("Model path:")
        print(MODEL_PATH)

        # Load the trained Keras model.
        # compile=False is used because we only need the model
        # for prediction.
        model = tf.keras.models.load_model(
            MODEL_PATH,
            compile=False
        )

        print("MODEL LOADED SUCCESSFULLY!")

        print(
            "INPUT SHAPE:",
            model.input_shape
        )

        print(
            "OUTPUT SHAPE:",
            model.output_shape
        )

except Exception as e:

    print("======================================")
    print("MODEL LOAD ERROR")
    print("======================================")

    print(str(e))

    traceback.print_exc()

    model = None


# ============================================================
# HOME ROUTE
# ============================================================
# When the user visits:
#
# http://localhost:5000/
#
# Flask displays index.html from the templates folder.
# ============================================================

@app.route("/")
def home():

    return render_template("index.html")


# ============================================================
# HEALTH CHECK ROUTE
# ============================================================
# This route can be used to check whether the Flask server
# and TensorFlow model are working.
#
# Visit:
# http://localhost:5000/health
# ============================================================

@app.route("/health")
def health():

    return jsonify({
        "status": "ok",
        "model_loaded": model is not None
    })


# ============================================================
# IMAGE PREPROCESSING FUNCTION
# ============================================================
# This function:
#
# 1. Opens the uploaded image
# 2. Converts it to RGB
# 3. Resizes it to 224 x 224
# 4. Converts it to NumPy
# 5. Converts pixel values from 0-255 to 0-1
# 6. Adds the batch dimension
#
# Final shape:
#
# (1, 224, 224, 3)
# ============================================================

def preprocess_image(file):

    # Open the uploaded image.
    image = Image.open(file)

    print(
        "Original image:",
        image.size,
        image.mode
    )

    # Convert the image to RGB.
    # This ensures that grayscale/RGBA images are converted
    # into the 3-channel format expected by most CNN models.
    image = image.convert("RGB")

    # Resize image to the model's expected size.
    image = image.resize(
        IMAGE_SIZE
    )

    print(
        "Resized image:",
        image.size
    )

    # Convert PIL image to NumPy array.
    data = np.asarray(
        image,
        dtype=np.float32
    )

    print(
        "NumPy shape:",
        data.shape
    )

    # Normalize pixel values.
    #
    # Original:
    # 0 - 255
    #
    # After normalization:
    # 0 - 1
    #
    # IMPORTANT:
    # This must match the preprocessing used during training.
    data = data / 255.0

    # Add batch dimension.
    #
    # Before:
    # (224, 224, 3)
    #
    # After:
    # (1, 224, 224, 3)
    data = np.expand_dims(
        data,
        axis=0
    )

    print(
        "Final model input shape:",
        data.shape
    )

    return data


# ============================================================
# PREDICTION ROUTE
# ============================================================
# The frontend should send:
#
# POST /predict
#
# with a multipart/form-data field named:
#
# image
#
# Example JavaScript:
#
# formData.append("image", file);
# ============================================================

@app.route("/predict", methods=["POST"])
def predict():

    print("")
    print("######################################")
    print("PREDICT REQUEST")
    print("######################################")

    try:

        # ====================================================
        # 1. CHECK MODEL
        # ====================================================

        if model is None:

            print("MODEL IS NOT LOADED")

            return jsonify({
                "success": False,
                "error": "Model is not loaded"
            }), 500


        # ====================================================
        # 2. CHECK FILE
        # ====================================================

        print(
            "Received files:",
            list(request.files.keys())
        )

        # Check whether the frontend sent an "image" field.
        if "image" not in request.files:

            print("NO IMAGE FIELD RECEIVED")

            return jsonify({
                "success": False,
                "error": "No image field received"
            }), 400


        # Get the uploaded file.
        file = request.files["image"]


        # Check whether a file was actually selected.
        if file.filename == "":

            print("NO FILE SELECTED")

            return jsonify({
                "success": False,
                "error": "No file selected"
            }), 400


        print(
            "Filename:",
            file.filename
        )


        # ====================================================
        # 3. OPEN AND PREPROCESS IMAGE
        # ====================================================

        print("Opening image...")

        try:

            data = preprocess_image(file)

        except UnidentifiedImageError:

            print("INVALID IMAGE")

            return jsonify({
                "success": False,
                "error": "The uploaded file is not a valid image"
            }), 400


        # ====================================================
        # 4. RUN MODEL
        # ====================================================

        print("")
        print("**************************************")
        print("CALLING MODEL")
        print("**************************************")


        # Run the model directly.
        #
        # training=False tells TensorFlow that this is
        # prediction/inference rather than training.
        prediction = model(
            data,
            training=False
        )


        # Convert TensorFlow tensor to NumPy.
        prediction = prediction.numpy()


        print("**************************************")
        print("MODEL FINISHED")
        print("**************************************")


        print(
            "Raw prediction:",
            prediction
        )


        # ====================================================
        # 5. CONVERT PREDICTION TO NUMPY
        # ====================================================

        prediction = np.asarray(
            prediction
        )

        print(
            "Prediction shape:",
            prediction.shape
        )


        # Flatten the prediction.
        #
        # Example:
        #
        # [[0.91]]
        #
        # becomes:
        #
        # [0.91]
        #
        # Or:
        #
        # [[0.10, 0.90]]
        #
        # becomes:
        #
        # [0.10, 0.90]
        prediction = prediction.flatten()


        # Make sure the model returned something.
        if prediction.size == 0:

            return jsonify({
                "success": False,
                "error": "Model returned an empty prediction"
            }), 500


        # ====================================================
        # 6. INTERPRET MODEL OUTPUT
        # ====================================================
        #
        # We support:
        #
        # A. Binary model:
        #    [0.85]
        #
        # B. Two-class model:
        #    [0.15, 0.85]
        # ====================================================


        # ----------------------------------------------------
        # BINARY CLASSIFIER
        # ----------------------------------------------------

        if prediction.size == 1:

            probability = float(
                prediction[0]
            )

            print(
                "Binary probability:",
                probability
            )


            # Make sure probability is within 0-1.
            #
            # This prevents invalid confidence values if the
            # model returns something unexpected.
            probability = float(
                np.clip(
                    probability,
                    0.0,
                    1.0
                )
            )


            # IMPORTANT:
            #
            # This assumes:
            #
            # probability >= 0.5
            #       = Animal
            #
            # probability < 0.5
            #       = Not Animal
            #
            # This MUST match your training labels.

            if probability >= 0.5:

                label = "Animal"

                confidence = (
                    probability * 100
                )

            else:

                label = "Not Animal"

                confidence = (
                    (1.0 - probability)
                    * 100
                )


        # ----------------------------------------------------
        # TWO-CLASS CLASSIFIER
        # ----------------------------------------------------

        else:

            # Make sure there are at least two classes.
            if prediction.size < 2:

                return jsonify({
                    "success": False,
                    "error": "Invalid model output"
                }), 500


            # Find the class with the highest probability.
            index = int(
                np.argmax(prediction)
            )


            # Check that the class index exists.
            if index >= len(CLASSES):

                label = "Unknown"

                confidence = 0.0

            else:

                label = CLASSES[index]

                # Get the probability of the selected class.
                confidence = float(
                    prediction[index]
                )

                # Convert to percentage.
                confidence = confidence * 100

                # Keep confidence between 0 and 100.
                confidence = float(
                    np.clip(
                        confidence,
                        0.0,
                        100.0
                    )
                )


        # ====================================================
        # 7. PRINT RESULT
        # ====================================================

        print("")
        print("======================================")
        print("PREDICTION RESULT")
        print("======================================")

        print(
            "Label:",
            label
        )

        print(
            "Confidence:",
            round(
                confidence,
                2
            ),
            "%"
        )


        # ====================================================
        # 8. RETURN JSON RESPONSE
        # ====================================================

        response = {

            "success": True,

            "prediction": label,

            "confidence": round(
                confidence,
                2
            )

        }


        print(
            "Sending JSON:",
            response
        )


        return jsonify(
            response
        )


    # ========================================================
    # HANDLE INVALID IMAGE
    # ========================================================

    except UnidentifiedImageError:

        print("")
        print("INVALID IMAGE ERROR")

        return jsonify({

            "success": False,

            "error": "The uploaded file is not a valid image"

        }), 400


    # ========================================================
    # HANDLE OTHER ERRORS
    # ========================================================

    except Exception as e:

        print("")
        print("######################################")
        print("PREDICTION EXCEPTION")
        print("######################################")

        print(
            repr(e)
        )

        traceback.print_exc()


        return jsonify({

            "success": False,

            "error": str(e)

        }), 500


# ============================================================
# HANDLE LARGE FILE UPLOADS
# ============================================================
# This prevents the application from crashing when someone
# uploads a file larger than MAX_CONTENT_LENGTH.
# ============================================================

@app.errorhandler(413)
def request_entity_too_large(error):

    return jsonify({

        "success": False,

        "error": "File is too large. Maximum size is 10 MB."

    }), 413


# ============================================================
# START APPLICATION
# ============================================================

if __name__ == "__main__":

    # Use the PORT environment variable if it exists.
    #
    # Otherwise, use port 5000.
    port = int(
        os.environ.get(
            "PORT",
            5000
        )
    )


    print("")
    print("======================================")
    print("SERVER STARTING")
    print("======================================")

    print(
        "Running on port:",
        port
    )

    print(
        "Model:",
        MODEL_PATH
    )


    # Start Flask.
    #
    # host="0.0.0.0" allows the application to be accessed
    # from other devices or cloud hosting environments.
    app.run(
        host="0.0.0.0",
        port=port,
        debug=False
    )