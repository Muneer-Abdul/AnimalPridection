# ============================================================
# CAT CLASSIFIER - FLASK APPLICATION
# ============================================================
#
# This application:
#
# 1. Loads the trained TensorFlow/Keras model
# 2. Accepts an image from the website
# 3. Preprocesses the uploaded image
# 4. Sends the image to the trained model
# 5. Classifies the image as:
#
#       Cat
#       Other Animal
#
# 6. Returns the prediction and confidence as JSON
#
# IMPORTANT:
#
# The trained model already contains this preprocessing layer:
#
#     Rescaling(1./127.5, offset=-1)
#
# Therefore, we DO NOT divide the image by 255 in this file.
#
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
#
# Limit TensorFlow CPU usage.
# This is useful when running the application on a laptop
# or a smaller cloud server.
# ============================================================

try:

    tf.config.threading.set_intra_op_parallelism_threads(1)

    tf.config.threading.set_inter_op_parallelism_threads(1)

except Exception:

    # If TensorFlow has already configured its threading,
    # simply continue running.
    pass


# ============================================================
# FLASK APPLICATION
# ============================================================

app = Flask(__name__)


# ============================================================
# PROJECT DIRECTORY
# ============================================================
#
# Find the folder where this app.py file is located.
#
# This makes the application work even if it is started
# from another directory.
# ============================================================

BASE_DIR = os.path.dirname(
    os.path.abspath(__file__)
)


# ============================================================
# MODEL SETTINGS
# ============================================================

MODEL_PATH = os.path.join(
    BASE_DIR,
    "animal_classifier.keras"
)


# ============================================================
# IMAGE SETTINGS
# ============================================================

# Your model was trained using 224 x 224 images.

IMAGE_SIZE = (
    224,
    224
)


# ============================================================
# UPLOAD SETTINGS
# ============================================================

# Maximum image upload size:
# 10 MB

app.config["MAX_CONTENT_LENGTH"] = (
    10 * 1024 * 1024
)


# ============================================================
# CLASS NAMES
# ============================================================
#
# Your dataset contains:
#
#     cat/
#     not_cat/
#
# TensorFlow sorts these alphabetically:
#
#     cat     -> index 0
#     not_cat -> index 1
#
# Therefore:
#
#     0 -> Cat
#     1 -> Other Animal
#
# ============================================================

CLASSES = [
    "Cat",
    "Other Animal"
]


# ============================================================
# LOAD TRAINED MODEL
# ============================================================

print()
print("======================================")
print("CAT CLASSIFIER STARTING")
print("======================================")


model = None


try:

    # --------------------------------------------------------
    # Check whether the model exists
    # --------------------------------------------------------

    if not os.path.exists(MODEL_PATH):

        print()
        print("ERROR: MODEL FILE NOT FOUND!")
        print()
        print("Expected model location:")
        print(MODEL_PATH)

    else:

        print()
        print("Loading model...")
        print()
        print("Model path:")
        print(MODEL_PATH)


        # ----------------------------------------------------
        # Load Keras model
        # ----------------------------------------------------
        #
        # compile=False is used because this application
        # only needs the model for prediction.
        # ----------------------------------------------------

        model = tf.keras.models.load_model(
            MODEL_PATH,
            compile=False
        )


        print()
        print("MODEL LOADED SUCCESSFULLY!")
        print()


        # ----------------------------------------------------
        # Display model information
        # ----------------------------------------------------

        print(
            "Input shape:",
            model.input_shape
        )

        print(
            "Output shape:",
            model.output_shape
        )


        print(
            "Classes:",
            CLASSES
        )


except Exception as e:

    print()
    print("======================================")
    print("MODEL LOAD ERROR")
    print("======================================")

    print(
        str(e)
    )

    traceback.print_exc()

    model = None


# ============================================================
# HOME PAGE
# ============================================================
#
# When the user visits:
#
#     http://localhost:5000
#
# Flask loads:
#
#     templates/index.html
#
# ============================================================

@app.route("/")
def home():

    return render_template(
        "index.html"
    )


# ============================================================
# HEALTH CHECK
# ============================================================
#
# Visit:
#
#     http://localhost:5000/health
#
# Example response:
#
# {
#     "status": "ok",
#     "model_loaded": true
# }
#
# ============================================================

@app.route("/health")
def health():

    return jsonify({

        "status": "ok",

        "model_loaded":
            model is not None,

        "classes":
            CLASSES

    })


# ============================================================
# IMAGE PREPROCESSING
# ============================================================
#
# IMPORTANT:
#
# The trained model already contains:
#
#     Rescaling(1./127.5, offset=-1)
#
# Therefore this function MUST NOT do:
#
#     data = data / 255.0
#
# We leave the image values in the original 0-255 range.
#
# ============================================================

def preprocess_image(file):

    # --------------------------------------------------------
    # Open uploaded image
    # --------------------------------------------------------

    image = Image.open(file)


    print(
        "Original image:",
        image.size,
        image.mode
    )


    # --------------------------------------------------------
    # Convert image to RGB
    # --------------------------------------------------------
    #
    # Some images may be:
    #
    #     Grayscale
    #     RGBA
    #     Palette
    #
    # The model expects 3 RGB channels.
    # --------------------------------------------------------

    image = image.convert(
        "RGB"
    )


    # --------------------------------------------------------
    # Resize image
    # --------------------------------------------------------

    image = image.resize(
        IMAGE_SIZE
    )


    print(
        "Resized image:",
        image.size
    )


    # --------------------------------------------------------
    # Convert PIL image to NumPy
    # --------------------------------------------------------
    #
    # Pixel values remain:
    #
    #     0 - 255
    #
    # DO NOT divide by 255 here.
    # --------------------------------------------------------

    data = np.asarray(
        image,
        dtype=np.float32
    )


    print(
        "NumPy shape:",
        data.shape
    )


    print(
        "Pixel minimum:",
        np.min(data)
    )


    print(
        "Pixel maximum:",
        np.max(data)
    )


    # --------------------------------------------------------
    # IMPORTANT
    # --------------------------------------------------------
    #
    # NO:
    #
    #     data = data / 255.0
    #
    # The Keras model performs the required rescaling itself.
    #
    # --------------------------------------------------------


    # --------------------------------------------------------
    # Add batch dimension
    # --------------------------------------------------------
    #
    # Before:
    #
    #     (224, 224, 3)
    #
    # After:
    #
    #     (1, 224, 224, 3)
    #
    # --------------------------------------------------------

    data = np.expand_dims(
        data,
        axis=0
    )


    print(
        "Final model input:",
        data.shape
    )


    return data


# ============================================================
# PREDICTION ROUTE
# ============================================================
#
# The frontend sends:
#
#     POST /predict
#
# with:
#
#     image
#
# ============================================================

@app.route(
    "/predict",
    methods=["POST"]
)
def predict():

    print()
    print("######################################")
    print("PREDICT REQUEST")
    print("######################################")


    try:

        # ====================================================
        # 1. CHECK MODEL
        # ====================================================

        if model is None:

            print(
                "MODEL IS NOT LOADED"
            )

            return jsonify({

                "success": False,

                "error":
                    "Model is not loaded."

            }), 500


        # ====================================================
        # 2. CHECK UPLOADED FILE
        # ====================================================

        print(
            "Received files:",
            list(request.files.keys())
        )


        if "image" not in request.files:

            print(
                "NO IMAGE FIELD RECEIVED"
            )

            return jsonify({

                "success": False,

                "error":
                    "No image field received."

            }), 400


        # ====================================================
        # 3. GET IMAGE FILE
        # ====================================================

        file = request.files["image"]


        # Check filename.

        if file.filename == "":

            print(
                "NO FILE SELECTED"
            )

            return jsonify({

                "success": False,

                "error":
                    "No file selected."

            }), 400


        print(
            "Filename:",
            file.filename
        )


        # ====================================================
        # 4. PREPROCESS IMAGE
        # ====================================================

        print()
        print(
            "Preprocessing image..."
        )


        try:

            data = preprocess_image(
                file
            )

        except UnidentifiedImageError:

            print(
                "INVALID IMAGE"
            )

            return jsonify({

                "success": False,

                "error":
                    "The uploaded file is not a valid image."

            }), 400


        # ====================================================
        # 5. RUN MODEL
        # ====================================================

        print()
        print("**************************************")
        print("CALLING TENSORFLOW MODEL")
        print("**************************************")


        # ----------------------------------------------------
        # Run inference.
        #
        # training=False tells TensorFlow that this is
        # prediction rather than training.
        # ----------------------------------------------------

        prediction = model(
            data,
            training=False
        )


        # ----------------------------------------------------
        # Convert TensorFlow tensor to NumPy.
        # ----------------------------------------------------

        prediction = prediction.numpy()


        print("**************************************")
        print("MODEL FINISHED")
        print("**************************************")


        print(
            "Raw model prediction:",
            prediction
        )


        # ====================================================
        # 6. FLATTEN MODEL OUTPUT
        # ====================================================

        prediction = np.asarray(
            prediction
        ).flatten()


        print(
            "Flattened prediction:",
            prediction
        )


        print(
            "Number of outputs:",
            prediction.size
        )


        # ====================================================
        # 7. CHECK MODEL OUTPUT
        # ====================================================
        #
        # Your model uses:
        #
        #     Dense(2, activation="softmax")
        #
        # Therefore we expect two values.
        #
        # Example:
        #
        #     [0.92, 0.08]
        #
        # ====================================================

        if prediction.size != 2:

            print(
                "INVALID MODEL OUTPUT"
            )

            return jsonify({

                "success": False,

                "error":
                    (
                        "Expected two model outputs "
                        "for Cat and Other Animal, "
                        "but received "
                        + str(prediction.size)
                        + "."
                    )

            }), 500


        # ====================================================
        # 8. GET CLASS INDEX
        # ====================================================
        #
        # np.argmax finds the class with the highest
        # probability.
        #
        # Example:
        #
        #     [0.90, 0.10]
        #
        # argmax = 0
        #
        #     0 = Cat
        #
        # ----------------------------------------------------
        #
        # Example:
        #
        #     [0.05, 0.95]
        #
        # argmax = 1
        #
        #     1 = Other Animal
        #
        # ====================================================

        index = int(
            np.argmax(prediction)
        )


        # ====================================================
        # 9. GET PROBABILITIES
        # ====================================================

        cat_probability = float(
            prediction[0]
        )


        other_probability = float(
            prediction[1]
        )


        # ----------------------------------------------------
        # Make sure values are within 0-1.
        # ----------------------------------------------------

        cat_probability = float(
            np.clip(
                cat_probability,
                0.0,
                1.0
            )
        )


        other_probability = float(
            np.clip(
                other_probability,
                0.0,
                1.0
            )
        )


        # ====================================================
        # 10. DETERMINE PREDICTION
        # ====================================================

        if index == 0:

            label = "Cat"

            confidence = (
                cat_probability * 100
            )


        elif index == 1:

            label = "Other Animal"

            confidence = (
                other_probability * 100
            )


        else:

            label = "Unknown"

            confidence = 0.0


        # ====================================================
        # 11. CONFIDENCE
        # ====================================================

        confidence = float(
            np.clip(
                confidence,
                0.0,
                100.0
            )
        )


        # ====================================================
        # 12. PRINT RESULT
        # ====================================================

        print()
        print("======================================")
        print("PREDICTION RESULT")
        print("======================================")


        print(
            "Cat probability:",
            round(
                cat_probability * 100,
                2
            ),
            "%"
        )


        print(
            "Other Animal probability:",
            round(
                other_probability * 100,
                2
            ),
            "%"
        )


        print(
            "Selected class:",
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
        # 13. CREATE JSON RESPONSE
        # ====================================================

        response = {

            "success": True,

            "prediction": label,

            "confidence": round(
                confidence,
                2
            )

        }


        # ====================================================
        # 14. SEND RESPONSE
        # ====================================================

        print()
        print(
            "Sending JSON:",
            response
        )


        return jsonify(
            response
        )


    # ========================================================
    # INVALID IMAGE ERROR
    # ========================================================

    except UnidentifiedImageError:

        print()
        print(
            "INVALID IMAGE ERROR"
        )


        return jsonify({

            "success": False,

            "error":
                "The uploaded file is not a valid image."

        }), 400


    # ========================================================
    # GENERAL ERROR
    # ========================================================

    except Exception as e:

        print()
        print("######################################")
        print("PREDICTION EXCEPTION")
        print("######################################")


        print(
            repr(e)
        )


        traceback.print_exc()


        return jsonify({

            "success": False,

            "error":
                str(e)

        }), 500


# ============================================================
# LARGE FILE ERROR
# ============================================================

@app.errorhandler(413)
def request_entity_too_large(error):

    return jsonify({

        "success": False,

        "error":
            "File is too large. Maximum size is 10 MB."

    }), 413


# ============================================================
# START APPLICATION
# ============================================================

if __name__ == "__main__":

    # --------------------------------------------------------
    # Use PORT from the hosting environment if available.
    #
    # Otherwise use port 5000.
    # --------------------------------------------------------

    port = int(
        os.environ.get(
            "PORT",
            5000
        )
    )


    print()
    print("======================================")
    print("SERVER STARTING")
    print("======================================")


    print(
        "Port:",
        port
    )


    print(
        "Model:",
        MODEL_PATH
    )


    print(
        "Classes:",
        CLASSES
    )


    print()
    print(
        "Open the application at:"
    )


    print(
        "http://127.0.0.1:"
        + str(port)
    )


    print()


    # --------------------------------------------------------
    # Start Flask server.
    # --------------------------------------------------------

    app.run(
        host="0.0.0.0",
        port=port,
        debug=False
    )