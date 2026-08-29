# ============================================================
# CAT CLASSIFIER - FLASK APPLICATION
# ============================================================
#
# This application:
#
# 1. Loads a trained TensorFlow/Keras model
# 2. Accepts an image from the frontend
# 3. Preprocesses the image
# 4. Sends the image to the AI model
# 5. Determines whether the image is a cat or another animal
# 6. Returns the prediction and confidence as JSON
#
# Expected predictions:
#
#     Cat
#     Other Animal
#
# ============================================================


# ============================================================
# IMPORTS
# ============================================================

import os

# Reduce TensorFlow informational messages.
# This must be set BEFORE importing TensorFlow.
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
# Limit TensorFlow to one CPU thread.
#
# This can help reduce CPU usage, especially when running the
# application on a laptop or a small cloud server.
#
# ============================================================

try:

    tf.config.threading.set_intra_op_parallelism_threads(1)

    tf.config.threading.set_inter_op_parallelism_threads(1)

except Exception:

    # If TensorFlow threading has already been configured,
    # continue running the application.
    pass


# ============================================================
# FLASK APPLICATION
# ============================================================

app = Flask(__name__)


# ============================================================
# SETTINGS
# ============================================================

# Get the folder where app.py is located.
#
# This makes the model path more reliable.
BASE_DIR = os.path.dirname(
    os.path.abspath(__file__)
)


# Path to the trained AI model.
MODEL_PATH = os.path.join(
    BASE_DIR,
    "animal_classifier.keras"
)


# Image size expected by the model.
IMAGE_SIZE = (224, 224)


# Maximum image upload size.
#
# 10 MB should be more than enough for normal image uploads.
app.config["MAX_CONTENT_LENGTH"] = 10 * 1024 * 1024


# ============================================================
# CLASS NAMES
# ============================================================
#
# These names are used when the model has two outputs.
#
# IMPORTANT:
#
# This assumes the model was trained with:
#
#     index 0 = Cat
#     index 1 = Other Animal
#
# If your model was trained with the opposite order,
# these labels must be reversed.
#
# ============================================================

CLASSES = [
    "Cat",
    "Other Animal"
]


# ============================================================
# LOAD MODEL
# ============================================================

print("======================================")
print("CAT CLASSIFIER STARTING")
print("======================================")


# Start with no model.
model = None


try:

    # --------------------------------------------------------
    # Check whether the model file exists
    # --------------------------------------------------------

    if not os.path.exists(MODEL_PATH):

        print("")
        print("MODEL FILE NOT FOUND!")
        print("")
        print("Expected model location:")
        print(MODEL_PATH)

    else:

        print("")
        print("Loading model...")
        print("")

        print("Model path:")
        print(MODEL_PATH)

        # ----------------------------------------------------
        # Load the trained Keras model.
        #
        # compile=False is used because this application
        # only needs the model for prediction.
        # ----------------------------------------------------

        model = tf.keras.models.load_model(
            MODEL_PATH,
            compile=False
        )

        print("")
        print("MODEL LOADED SUCCESSFULLY!")
        print("")

        # Display model input/output information.
        print(
            "INPUT SHAPE:",
            model.input_shape
        )

        print(
            "OUTPUT SHAPE:",
            model.output_shape
        )

        print("")


except Exception as e:

    print("")
    print("======================================")
    print("MODEL LOAD ERROR")
    print("======================================")

    print(
        str(e)
    )

    print("")

    traceback.print_exc()

    model = None


# ============================================================
# HOME ROUTE
# ============================================================
#
# When the user visits:
#
#     http://127.0.0.1:5000/
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
# HEALTH CHECK ROUTE
# ============================================================
#
# This allows you to check whether:
#
# 1. Flask is running
# 2. The AI model loaded successfully
#
# Visit:
#
#     http://127.0.0.1:5000/health
#
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
#
# This function prepares the uploaded image for TensorFlow.
#
# Steps:
#
# 1. Open image
# 2. Convert image to RGB
# 3. Resize to 224 x 224
# 4. Convert to NumPy array
# 5. Normalize pixel values
# 6. Add batch dimension
#
# Final input:
#
#     (1, 224, 224, 3)
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
    # This ensures the image has exactly 3 channels:
    #
    #     Red
    #     Green
    #     Blue
    #
    # --------------------------------------------------------

    image = image.convert("RGB")


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
    # Convert PIL image to NumPy array
    # --------------------------------------------------------

    data = np.asarray(
        image,
        dtype=np.float32
    )

    print(
        "NumPy shape:",
        data.shape
    )


    # --------------------------------------------------------
    # Normalize pixel values
    # --------------------------------------------------------
    #
    # Original pixel values:
    #
    #     0 - 255
    #
    # After normalization:
    #
    #     0 - 1
    #
    # IMPORTANT:
    #
    # This must match the preprocessing used when the
    # model was trained.
    #
    # --------------------------------------------------------

    data = data / 255.0


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
        "Final model input shape:",
        data.shape
    )


    return data


# ============================================================
# PREDICTION ROUTE
# ============================================================
#
# The frontend sends an image using:
#
#     POST /predict
#
# The field name must be:
#
#     image
#
# ============================================================

@app.route(
    "/predict",
    methods=["POST"]
)
def predict():

    print("")
    print("######################################")
    print("PREDICT REQUEST")
    print("######################################")
    print("")


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

                "error": "Model is not loaded"

            }), 500


        # ====================================================
        # 2. CHECK FOR IMAGE
        # ====================================================

        print(
            "Received files:",
            list(request.files.keys())
        )


        # Check whether the frontend sent an "image" field.
        if "image" not in request.files:

            print(
                "NO IMAGE FIELD RECEIVED"
            )

            return jsonify({

                "success": False,

                "error": "No image field received"

            }), 400


        # Get uploaded file.
        file = request.files["image"]


        # ====================================================
        # 3. CHECK FILE NAME
        # ====================================================

        if file.filename == "":

            print(
                "NO FILE SELECTED"
            )

            return jsonify({

                "success": False,

                "error": "No file selected"

            }), 400


        print(
            "Filename:",
            file.filename
        )


        # ====================================================
        # 4. PREPROCESS IMAGE
        # ====================================================

        print("")
        print("Opening and preprocessing image...")
        print("")


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

                "error": "The uploaded file is not a valid image"

            }), 400


        # ====================================================
        # 5. RUN AI MODEL
        # ====================================================

        print("")
        print("**************************************")
        print("CALLING AI MODEL")
        print("**************************************")
        print("")


        # Run the model.
        #
        # training=False means we are performing inference,
        # not training.
        prediction = model(
            data,
            training=False
        )


        # Convert TensorFlow tensor to NumPy.
        prediction = prediction.numpy()


        print("")
        print("**************************************")
        print("MODEL FINISHED")
        print("**************************************")
        print("")


        print(
            "Raw prediction:",
            prediction
        )


        # ====================================================
        # 6. CONVERT PREDICTION
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
        #     [[0.91]]
        #
        # becomes:
        #
        #     [0.91]
        #
        # Or:
        #
        #     [[0.10, 0.90]]
        #
        # becomes:
        #
        #     [0.10, 0.90]
        #

        prediction = prediction.flatten()


        # ====================================================
        # 7. CHECK MODEL OUTPUT
        # ====================================================

        if prediction.size == 0:

            print(
                "MODEL RETURNED EMPTY PREDICTION"
            )

            return jsonify({

                "success": False,

                "error": "Model returned an empty prediction"

            }), 500


        # ====================================================
        # 8. INTERPRET PREDICTION
        # ====================================================
        #
        # There are two possible model formats:
        #
        # FORMAT 1:
        #
        #     Binary output
        #
        #     [0.85]
        #
        # FORMAT 2:
        #
        #     Two-class output
        #
        #     [0.15, 0.85]
        #
        # ====================================================


        # ====================================================
        # BINARY CLASSIFIER
        # ====================================================

        if prediction.size == 1:

            probability = float(
                prediction[0]
            )


            print(
                "Binary probability:",
                probability
            )


            # ------------------------------------------------
            # Make sure the probability is between 0 and 1.
            # ------------------------------------------------

            probability = float(
                np.clip(
                    probability,
                    0.0,
                    1.0
                )
            )


            # ------------------------------------------------
            # CAT / OTHER ANIMAL
            # ------------------------------------------------
            #
            # This assumes the model was trained so that:
            #
            #     probability >= 0.5
            #             = Cat
            #
            #     probability < 0.5
            #             = Other Animal
            #
            # ------------------------------------------------

            if probability >= 0.5:

                label = "Cat"

                confidence = (
                    probability * 100
                )

            else:

                label = "Other Animal"

                confidence = (
                    (1.0 - probability) * 100
                )


        # ====================================================
        # TWO-CLASS CLASSIFIER
        # ====================================================

        else:

            # ------------------------------------------------
            # Make sure at least two outputs exist.
            # ------------------------------------------------

            if prediction.size < 2:

                return jsonify({

                    "success": False,

                    "error": "Invalid model output"

                }), 500


            # ------------------------------------------------
            # Find the class with the highest probability.
            # ------------------------------------------------

            index = int(
                np.argmax(prediction)
            )


            print(
                "Predicted class index:",
                index
            )


            # ------------------------------------------------
            # Make sure the index exists in CLASSES.
            # ------------------------------------------------

            if index >= len(CLASSES):

                label = "Unknown"

                confidence = 0.0

            else:

                # Get class name.
                label = CLASSES[index]


                # Get probability.
                confidence = float(
                    prediction[index]
                )


                # Convert probability to percentage.
                confidence = (
                    confidence * 100
                )


                # Keep confidence between 0 and 100.
                confidence = float(
                    np.clip(
                        confidence,
                        0.0,
                        100.0
                    )
                )


        # ====================================================
        # 9. PRINT FINAL RESULT
        # ====================================================

        print("")
        print("======================================")
        print("PREDICTION RESULT")
        print("======================================")

        print(
            "Prediction:",
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

        print("======================================")
        print("")


        # ====================================================
        # 10. CREATE JSON RESPONSE
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


        # Send result back to frontend.
        return jsonify(
            response
        )


    # ========================================================
    # INVALID IMAGE ERROR
    # ========================================================

    except UnidentifiedImageError:

        print("")
        print("INVALID IMAGE ERROR")
        print("")


        return jsonify({

            "success": False,

            "error": "The uploaded file is not a valid image"

        }), 400


    # ========================================================
    # GENERAL ERROR
    # ========================================================

    except Exception as e:

        print("")
        print("######################################")
        print("PREDICTION EXCEPTION")
        print("######################################")

        print(
            repr(e)
        )

        print("")

        traceback.print_exc()


        return jsonify({

            "success": False,

            "error": str(e)

        }), 500


# ============================================================
# LARGE FILE ERROR HANDLER
# ============================================================
#
# If a user uploads an image larger than 10 MB, Flask returns
# this message instead of producing a generic error.
#
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

    # --------------------------------------------------------
    # Get port from environment.
    #
    # If PORT is not available, use 5000.
    # --------------------------------------------------------

    port = int(
        os.environ.get(
            "PORT",
            5000
        )
    )


    # --------------------------------------------------------
    # Display startup information.
    # --------------------------------------------------------

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

    print("")


    # --------------------------------------------------------
    # Start Flask server.
    #
    # host="0.0.0.0" allows the application to accept
    # connections from other devices and cloud services.
    #
    # debug=False is safer for normal use.
    # --------------------------------------------------------

    app.run(
        host="0.0.0.0",
        port=port,
        debug=False
    )