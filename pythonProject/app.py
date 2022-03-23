from flask import Flask, jsonify, request
from model.detect_face import detect_face
from facenet_pytorch import MTCNN

from matplotlib import pyplot as plt

app = Flask(__name__)

@app.route('/')
def home():
    return 'Hello'

@app.route('/model', methods=['POST'])
def model():
    img = request.files['file']
    mtcnn = MTCNN(post_process=False)
    processed_img, left_eye, right_eye = detect_face(img, mtcnn)

    eyes = ' '.join(str(e) for e in left_eye.tolist() + right_eye.tolist())
    print(eyes)
    plt.imshow(processed_img)
    plt.show()

    response = {}
    response["eyes"] = eyes

    return jsonify(response)

if __name__ == "__main__":
    app.run(host="0.0.0.0", port="5000")