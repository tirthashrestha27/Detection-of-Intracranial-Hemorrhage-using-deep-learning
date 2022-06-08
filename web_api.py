import os
import pickle
import flask
from flask import Flask, jsonify, request
import keras
import numpy as np
import json

app = Flask(__name__)
os.environ['CUDA_VISIBLE_DEVICES'] = '-1'

yo = keras.models.load_model(
    'model_checkpoints/weights-improvement-06-0.633729.hdf5')


class NumpyArrayEncoder(json.JSONEncoder):

    def default(self, obj):
        if isinstance(obj, np.ndarray):
            return obj.tolist()
        return json.JSONEncoder.default(self, obj)


@app.route('/predict', methods=['GET'])
def infer_image():
    data = request.json
    print(data)
    y = yo.predict([data['a'], data['b'], data['c']])
    print(y[0])
    encodedNumpyData = json.dumps(y[0], cls=NumpyArrayEncoder)
    return jsonify(encodedNumpyData)


@app.route('/', methods=['GET'])
def index():
    return 'Machine Learning Inference'


if __name__ == '__main__':
    app.run(debug=True)