from flask import Flask, jsonify, request
from src.align.detect_face_custom import detect_face
from src.identify import find_face, calculate_distance, calculate_embeddings
from facenet_pytorch import MTCNN

import numpy as np
from matplotlib import pyplot as plt
import ast
import json

app = Flask(__name__)

class NumpyEncoder(json.JSONEncoder):
    def default(self, obj):
        if isinstance(obj, np.ndarray):
            return obj.tolist()
        return json.JSONEncoder.default(self, obj)


@app.route('/')
def home():
    return 'Hello'

@app.route('/detectFace', methods=['POST'])
def detectFace():
    #img = request.files['file']
    img = request.values['images']
    mtcnn = MTCNN(post_process=False)
    processed_img, left_eye, right_eye, original_img_size = detect_face(img, mtcnn)

    eyes = ' '.join(str(e) for e in left_eye.tolist() + right_eye.tolist())
    # plt.imshow(processed_img)
    # plt.show()

    response = {}
    response["eyes"] = eyes
    response["size"] = original_img_size

    return jsonify(response)

@app.route('/analysis', methods=['POST'])
def analysis():
    file = request.files['file']
    images = eval(request.values['images'].replace("\\", ""))
    results = find_face(file,images)
    return json.dumps(results, cls=NumpyEncoder)

@app.route('/results', methods=['POST'])
def results():
    file = request.files['file']
    images = eval(request.values['images'])
    results = calculate_distance(file, images)
    return json.dumps(results, cls=NumpyEncoder)

@app.route('/extract',methods=['POST'])
def extract():
    images = eval(request.values['images'].replace("\\", ""))
    results = calculate_embeddings(images)
    return json.dumps(results, cls=NumpyEncoder)



if __name__ == "__main__":
    app.run(host="0.0.0.0", port="5000")

application = app