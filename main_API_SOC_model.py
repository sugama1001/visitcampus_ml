from flask import Flask, jsonify, request
import tensorflow as tf
import numpy as np
import tensorflow_text as text
import tensorflow_hub

app = Flask(__name__)

# Load the model outside of the route
model_path = "Scope Of Science Recommendation Model"
try:
    # Use experimental_io_device option
    load_options = tf.saved_model.LoadOptions(experimental_io_device='/job:localhost')
    model = tf.keras.models.load_model(model_path, options=load_options)
    print("Model loaded successfully.")
except Exception as e:
    print("Error loading the model:", str(e))

@app.route("/")
def home():
    return "<h1>The API WORK</h1>"
@app.route('/predict', methods=['POST'])
def predict():
    # Contoh Input:
    # {
    #     "EI_text": "saya suka menyendiri",
    #     "SN_text": "saya selalu mengikuti kata hati",
    #     "TF_text": "saya suka analisis",
    #     "JP_text": "saya tidak suka diatur orang lain"
    # }
    try:
        # Example data for POST request
        data = request.get_json()

        # Extract text inputs
        EI_text = np.array([data['EI_text']])
        SN_text = np.array([data['SN_text']])
        TF_text = np.array([data['TF_text']])
        JP_text = np.array([data['JP_text']])

        # Combine text inputs into an array
        arrays = [EI_text, SN_text, TF_text, JP_text]

        # Make predictions using the loaded model
        predictions = model.predict(arrays)

        json_predictions = [float(prediction[0]) for prediction in predictions]

        return jsonify(json_predictions)

    except Exception as e:
        return jsonify({"error": str(e)})

if __name__ == '__main__':
    app.run(debug=True)
