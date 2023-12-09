from flask import Flask, jsonify, request
import tensorflow as tf
import numpy as np
import tensorflow_text as text
import tensorflow_hub

def gabung_probabilitas(arrays):
    soshum = sum(arrays[0])
    saintek = sum(arrays[1])
    soshum_percentage = round(soshum / (soshum + saintek), 2)

    return [soshum_percentage, 1 - soshum_percentage]

def pembobotan_elemen_MBTI(model_output_value, bobot):
    for index_list, list_data in enumerate(model_output_value):
        for index_data, data in enumerate(list_data):
            for num in range(2):
                bobot[index_list][index_data][num] = round(model_output_value[index_list][index_data]*bobot[index_list][index_data][num], 2)

    output_list = [[], []]

    for bobot_list in bobot:
        output_list[0].append(bobot_list[0])
        output_list[1].append(bobot_list[1])

    return output_list

# Load the model outside of the route
model_path = "Scope Of Science Recommendation Model"
try:
    # Use experimental_io_device option
    load_options = tf.saved_model.LoadOptions(experimental_io_device='/job:localhost')
    model = tf.keras.models.load_model(model_path, options=load_options)
    print("Model loaded successfully.")
except Exception as e:
    print("Error loading the model:", str(e))

app = Flask(__name__)

@app.route("/")
def home():
    return "<h1>The API WORK</h1>"
@app.route('/predict', methods=['POST'])
def predict():
    # {
    #     "EI_text": "saya suka menyendiri",
    #     "SN_text": "saya selalu mengikuti kata hati",
    #     "TF_text": "saya suka analisis",
    #     "JP_text": "nah kamu babi"
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

            

        bobot = [
            [[0.875,0.125],[0.75,0.25]],
            [[0.67,0.33],[0.5,0.5]],
            [[0.75,0.25],[0.71,0.29]],
            [[0.625,0.375],[0.79,0.21]]
        ]
        # json_predictions = [float(prediction[0]) for prediction in predictions]
        json_predictions = [float(prediction[0]) for prediction in predictions]
        json_predictions = [[json_predictions[0],1-json_predictions[0]],[json_predictions[1],1-json_predictions[1]],[json_predictions[2],1-json_predictions[2]],[json_predictions[3],1-json_predictions[3]]]
        json_predictions = pembobotan_elemen_MBTI(json_predictions,bobot)
        json_predictions = gabung_probabilitas(json_predictions)
  
        return jsonify(json_predictions)

    except Exception as e:
        return jsonify({"error": str(e)})

if __name__ == '__main__':
    app.run(debug=True)
