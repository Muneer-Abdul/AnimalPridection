 # 🐱 AnimalPrediction

### AI-Powered Cat vs Other Animal Image Classifier

AnimalPrediction is a machine-learning web application that uses a trained **TensorFlow/Keras deep-learning model** to analyze an uploaded image and determine whether the image contains a **cat** or an **other animal**.

The project combines a trained image-classification model with a **Flask backend** and a simple, responsive **HTML/CSS/JavaScript frontend**.


## 🚀 Project Demo

Upload an image → Let the AI analyze it → Get a prediction and confidence score.

### Example

| Input | Prediction |
|---|---|
| 🐱 Cat image | **Cat** |
| 🐶 Dog image | **Other Animal** |
| 🐃 Buffalo image | **Other Animal** |

The application also provides a confidence percentage for each prediction.


## ✨ Features

- 🐱 Cat image classification
- 🐾 Other-animal classification
- 📷 Image upload and preview
- 🤖 TensorFlow/Keras machine-learning model
- ⚡ Flask-powered backend
- 🌐 Web-based user interface
- 📊 Prediction confidence score
- 🔍 Real-time image prediction
- 📱 Responsive interface for different screen sizes
- 🛡️ Image validation
- 📦 Maximum upload size protection
- ❤️ Simple and easy-to-use interface


## 🧠 How It Works

The application follows this process:

```text
User selects an image
        ↓
Image preview is displayed
        ↓
User clicks "Predict"
        ↓
Frontend sends image to Flask
        ↓
Flask receives the image
        ↓
Image is resized to 224 × 224
        ↓
Image is converted to RGB
        ↓
TensorFlow/Keras model processes image
        ↓
Model predicts the class
        ↓
Cat / Other Animal
        ↓
Confidence score calculated
        ↓
Result displayed on website
