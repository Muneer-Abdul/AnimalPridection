import os

# Must be before TensorFlow import
os.environ["TF_CPP_MIN_LOG_LEVEL"] = "2"

import traceback
import numpy as np
from PIL import Image
from flask import Flask, render_template, request, jsonify

import tensorflow as tf


# =====================================================
# TensorFlow resource limits
# =====================================================

tf.config.threading.set_intra_op_parallelism_threads(1)
tf.config.threading.set_inter_op_parallelism_threads(1)


# =====================================================
# Flask
# =====================================================

app = Flask(__name__)


# =====================================================
# Settings
# =====================================================

MODEL_PATH = "animal_classifier.keras"
IMAGE_SIZE = (224, 224)


# =====================================================
# Load model
# =====================================================

print("======================================")
print("ANIMAL CLASSIFIER STARTING")
print("======================================")

try:

    print("Loading model...")

    model = tf.keras.models.load_model(
        MODEL_PATH,
        compile=False
    )

    print("MODEL LOADED!")
    print("INPUT:", model.input_shape)
    print("OUTPUT:", model.output_shape)

except Exception as e:

    print("MODEL LOAD ERROR:")
    print(str(e))

    traceback.print_exc()

    model = None


# =====================================================
# Home
# =====================================================

@app.route("/")
def home():

    return render_template("index.html")


# =====================================================
# Health
# =====================================================

@app.route("/health")
def health():

    return jsonify({
        "status": "ok",
        "model_loaded": model is not None
    })


# =====================================================
# Prediction
# =====================================================

@app.route("/predict", methods=["POST"])
def predict():

    print("")
    print("######################################")
    print("PREDICT REQUEST")
    print("######################################")

    try:

        # ---------------------------------------------
        # Model check
        # ---------------------------------------------

        if model is None:

            print("MODEL IS NONE")

            return jsonify({
                "success": False,
                "error": "Model is not loaded"
            }), 500


        # ---------------------------------------------
        # File check
        # ---------------------------------------------

        print(
            "Received files:",
            list(request.files.keys())
        )


        if "image" not in request.files:

            print("NO IMAGE")

            return jsonify({
                "success": False,
                "error": "No image field received"
            }), 400


        file = request.files["image"]


        print(
            "Filename:",
            file.filename
        )


        # ---------------------------------------------
        # Open image
        # ---------------------------------------------

        print("Opening image...")

        image = Image.open(file)

        print(
            "Original:",
            image.size,
            image.mode
        )


        image = image.convert("RGB")


        # ---------------------------------------------
        # Resize
        # ---------------------------------------------

        print("Resizing...")

        image = image.resize(
            IMAGE_SIZE
        )


        # ---------------------------------------------
        # NumPy
        # ---------------------------------------------

        print("Converting to NumPy...")

        data = np.asarray(
            image,
            dtype=np.float32
        )


        print(
            "Array:",
            data.shape
        )


        # ---------------------------------------------
        # Normalize
        # ---------------------------------------------

        data = data / 255.0


        # ---------------------------------------------
        # Batch
        # ---------------------------------------------

        data = np.expand_dims(
            data,
            axis=0
        )


        print(
            "Final input:",
            data.shape
        )


        # ---------------------------------------------
        # Prediction
        # ---------------------------------------------

        print("")
        print("**************************************")
        print("CALLING MODEL.PREDICT")
        print("**************************************")


        prediction = model.predict(
            data,
            batch_size=1,
            verbose=0
        )


        print("**************************************")
        print("MODEL.PREDICT FINISHED")
        print("**************************************")


        print(
            "Raw prediction:",
            prediction
        )


        # ---------------------------------------------
        # Convert prediction
        # ---------------------------------------------

        prediction = np.asarray(
            prediction
        )


        print(
            "Prediction shape:",
            prediction.shape
        )


        # ---------------------------------------------
        # Binary classifier
        # ---------------------------------------------

        if prediction.size == 1:

            probability = float(
                prediction.flatten()[0]
            )


            print(
                "Probability:",
                probability
            )


            if probability >= 0.5:

                label = "Animal"

                confidence = (
                    probability * 100
                )

            else:

                label = "Not Animal"

                confidence = (
                    (1 - probability)
                    * 100
                )


        # ---------------------------------------------
        # Two-class classifier
        # ---------------------------------------------

        else:

            prediction = prediction.flatten()

            index = int(
                np.argmax(prediction)
            )


            classes = [
                "Animal",
                "Not Animal"
            ]


            if index < len(classes):

                label = classes[index]

            else:

                label = "Unknown"


            confidence = (
                float(prediction[index])
                * 100
            )


        print("")
        print("RESULT:", label)
        print(
            "CONFIDENCE:",
            confidence
        )


        # ---------------------------------------------
        # Return
        # ---------------------------------------------

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


        return jsonify(response)


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


# =====================================================
# Start
# =====================================================

if __name__ == "__main__":

    port = int(
        os.environ.get(
            "PORT",
            5000
        )
    )

    app.run(
        host="0.0.0.0",
        port=port
    )