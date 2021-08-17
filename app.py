import os
from flask import Flask, request, jsonify, render_template
from flask_cors import CORS, cross_origin
from com_in_flower_ai_utils.utils import decodeImage
from Predict import Flowers

app = Flask(__name__)
CORS(app)


class Clientapp:
    def __init__(self):
        self.filename = 'InputImage.jpg'
        self.Classifier = Flowers(self.filename)


@app.route("/", methods=['GET'])
@cross_origin()
def home():
    return render_template('index.html')

@app.route("/predict", methods=['POST'])
@cross_origin()
def predictRoute():
    image = request.json['image']
    decodeImage(image, clapp.filename)
    result = clapp.Classifier.prediction_on_flowers()
    return jsonify(result)



if __name__ == "__main__":
    clapp = Clientapp()
    app.run(host='127.0.0.1', port=8000, debug=True)
