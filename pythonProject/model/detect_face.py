from facenet_pytorch.models.utils.detect_face import extract_face
import cv2
import os
from PIL import Image
import numpy as np
import math
import torch
import base64
import requests
from tensorflow.keras.preprocessing import image # tensorflow ver: 2.xx 일때 사용

def loadBase64Img(uri):
	encoded_data = uri.split(',')[1]
	nparr = np.fromstring(base64.b64decode(encoded_data), np.uint8)
	img = cv2.imdecode(nparr, cv2.IMREAD_COLOR)
	return img


#img might be path, base64 or numpy array. Convert it to numpy whatever it is.
def load_image(img):
	exact_image = False; base64_img = False; url_img = False; file_sotrage_img = False;

	if type(img).__name__ == "FileStorage":
		file_sotrage_img = True

	elif type(img).__module__ == np.__name__:
		exact_image = True

	elif len(img) > 11 and img[0:11] == "data:image/":
		base64_img = True

	elif len(img) > 11 and img.startswith("http"):
		url_img = True

	#---------------------------

	if file_sotrage_img == True:
		img = np.fromstring(img.read(), np.uint8)
		img = cv2.imdecode(img, cv2.IMREAD_COLOR)
		img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

	elif base64_img == True:
		img = loadBase64Img(img)

	elif url_img:
		img = np.array(Image.open(requests.get(img, stream=True).raw))

	elif exact_image != True: #image path passed as input
		if os.path.isfile(img) != True:
			raise ValueError("Confirm that ",img," exists")

		# img = cv2.imread(img) # 한글 경로 오류 문제...
		path = np.fromfile(img, np.uint8)
		img = cv2.imdecode(path, cv2.IMREAD_UNCHANGED)

		img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB) ## 얘 때문이었어!!!!!!

	return img

# box 값에 따라 얼굴 이미지를 crop
def crop_image(img, box):
	img = img[int(box[1]):int(box[3]), int(box[0]):int(box[2])]
	return img

def findEuclideanDistance(source_representation, test_representation):
    if type(source_representation) == list:
        source_representation = np.array(source_representation)

    if type(test_representation) == list:
        test_representation = np.array(test_representation)

    euclidean_distance = source_representation - test_representation
    euclidean_distance = np.sum(np.multiply(euclidean_distance, euclidean_distance))
    euclidean_distance = np.sqrt(euclidean_distance)
    return euclidean_distance

#-------------------------------------------------------------------------------------------

def alignment_procedure(img, left_eye, right_eye):

	#this function aligns given face in img based on left and right eye coordinates

	left_eye_x, left_eye_y = left_eye
	right_eye_x, right_eye_y = right_eye

	#-----------------------
	#find rotation direction

	if left_eye_y > right_eye_y:
		point_3rd = (right_eye_x, left_eye_y)
		direction = -1 #rotate same direction to clock
	else:
		point_3rd = (left_eye_x, right_eye_y)
		direction = 1 #rotate inverse direction of clock

	#-----------------------
	#find length of triangle edges

	a = findEuclideanDistance(np.array(left_eye), np.array(point_3rd))
	b = findEuclideanDistance(np.array(right_eye), np.array(point_3rd))
	c = findEuclideanDistance(np.array(right_eye), np.array(left_eye))

	#-----------------------

	#apply cosine rule

	if b != 0 and c != 0: #this multiplication causes division by zero in cos_a calculation

		cos_a = (b*b + c*c - a*a)/(2*b*c)
		angle = np.arccos(cos_a) #angle in radian
		angle = (angle * 180) / math.pi #radian to degree

		#-----------------------
		#rotate base image

		if direction == -1:
			angle = 90 - angle

		img = Image.fromarray(img) # 오류
		img = np.array(img.rotate(direction * angle))

	#-----------------------

	return img #return img anyway

def resize_image(img, target_size=(160, 160)):
	if img.shape[0] > 0 and img.shape[1] > 0:
		factor_0 = target_size[0] / img.shape[0]
		factor_1 = target_size[1] / img.shape[1]
		factor = min(factor_0, factor_1)

		dsize = (int(img.shape[1] * factor), int(img.shape[0] * factor))
		img = cv2.resize(img, dsize)

		# Then pad the other side to the target size by adding black pixels
		diff_0 = target_size[0] - img.shape[0]
		diff_1 = target_size[1] - img.shape[1]
		# Put the base image in the middle of the padded image
		img = np.pad(img, ((diff_0 // 2, diff_0 - diff_0 // 2), (diff_1 // 2, diff_1 - diff_1 // 2), (0, 0)), 'constant')

	#double check: if target image is not still the same size with target.
	if img.shape[0:2] != target_size:
		img = cv2.resize(img, target_size)
    
	#normalizing the image pixels

	img_pixels = image.img_to_array(img) #what this line doing? must?
	img_pixels = np.expand_dims(img_pixels, axis = 0)
	img_pixels /= 255 #normalize input in [0, 1]

	return img_pixels

def get_size(img):
    if isinstance(img, (np.ndarray, torch.Tensor)):
        return img.shape[1::-1]
    else:
        return img.size

## This function applies pre-processing stages of a face recognition pipeline including detection and alignment
# Parameters:
# 	  img_path: exact image path, numpy array or base64 encoded image
#     target_size: 리턴 사이즈
#     mtcnn : mtcnn 객체
# Returns:
# 		deteced and aligned face in numpy format
def detect_face(img_path, mtcnn, margin = 15, target_size = (160, 160)):
	img = load_image(img_path) # 원본 이미지

	boxes, probs, landmarks = mtcnn.detect(img, landmarks=True)

	if (boxes is None): #얼굴을 탐지하지 못한 경우 원본 이미지 반환 -> 예외 발생
		raise Exception('Cannot detect face')
		return img

	box = boxes[0]
	image_size = target_size[0] #160

	margin = [
		margin * (box[2] - box[0]) / (image_size - margin),
		margin * (box[3] - box[1]) / (image_size - margin),
	]
	raw_image_size = get_size(img)

	box = [
		int(max(box[0] - margin[0] / 2, 0)),
		int(max(box[1] - margin[1] / 2, 0)),
		int(min(box[2] + margin[0] / 2, raw_image_size[0])),
		int(min(box[3] + margin[1] / 2, raw_image_size[1])),
	]

	face = crop_image(img, box)

	left_eye = landmarks[0][0]
	right_eye = landmarks[0][1]

	face = alignment_procedure(face, left_eye, right_eye)

	face = resize_image(face, target_size=target_size) #returns (1, 160, 160, 3)

	face_img = face[0]

	return face_img, left_eye, right_eye # 원본 사진에서의 눈 위치 반환.

