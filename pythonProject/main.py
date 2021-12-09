from deepface import DeepFace
import matplotlib.pyplot as plt
import glob
import sys
import os
from sklearn.metrics import accuracy_score
import numpy as np
import imutils

models = ["VGG-Face", "Facenet", "Facenet512", "OpenFace", "DeepFace", "DeepID", "ArcFace", "Dlib"]
metrics = ["cosine", "euclidean", "euclidean_l2"]
backends = ['opencv', 'ssd', 'dlib', 'mtcnn', 'retinaface']

#추가 얼굴 정렬
def alignment(img_name, angle):
    img = plt.imread(img_name)
    img = imutils.rotate(img,angle)
    plt.imsave(img_name,img)

#DB 이미지 리스트 불러오기
db_list = glob.glob("db/*.png")

#이미지 없을 때 예외처리
if not db_list:
    print("이미지가 없습니다.")
    sys.exit()

#DB 이미지 리스트 전처리
for db_img in db_list:
    if "img2_5" in db_img:
        alignment(db_img,135)

    aligned_img = DeepFace.detectFace(db_img, enforce_detection=False)
    plt.imsave(db_img, aligned_img)



#입력 이미지 전처리
test_list = glob.glob("test/*.png")

if not test_list:
    print("이미지가 없습니다.")
    sys.exit()

for test_img in test_list:
    aligned_img = DeepFace.detectFace(test_img, enforce_detection=False)
    plt.imsave(test_img, aligned_img)


file_list = os.listdir('db')

#make ground truth output
def make_output(original_img):
    y_true = []
    for file in file_list:
        # 확장자 제거
        original_file_name = ''.join(original_img.split('.')[:-1])
        file_name = ''.join(file.split('.')[:-1])

        y_true.append(original_file_name in file_name)
    return y_true

#make prediction output
def prediction(df):
    predicted_file_list = df['identity'].values.tolist()
    prediction = []
    y_pred = []
    for predicted_file in predicted_file_list:
        file = ''.join(predicted_file.split('/')[-1])
        file_name = ''.join(file.split('.')[:-1])
        prediction.append(file_name)

    for file in file_list:
        file_name = ''.join(file.split('.')[:-1])
        y_pred.append(file_name in prediction)

    return y_pred


#모든 이미지에 대해 시행한 후 accuracy 누적
test_list = os.listdir('test')
model_accuracy = np.array([[0.0 for j in range(len(models))] for i in range(len(test_list))])

for i in range(len(test_list)):
    original_img = test_list[i]
    y_true = make_output(original_img)

    # 모든 모델에 대해 시행
    for j in range(len(models)):
        df = DeepFace.find(img_path="test/"+original_img, db_path="db", model_name=models[j], distance_metric=metrics[2],
                           enforce_detection=False)
        y_pred = prediction(df)
        model_accuracy[i][j] = accuracy_score(y_true, y_pred)

model_accuracy_mean = np.mean(model_accuracy, axis=0)
accuracy_max = max(model_accuracy_mean)
accuracy_max_index = np.where(model_accuracy_mean == accuracy_max)[0][0]
accuracy_list = model_accuracy[:, accuracy_max_index].reshape(len(test_list),)


for i in range(len(model_accuracy_mean)):
    print(models[i]+" total accuracy : "+str(model_accuracy_mean[i]))

print(models[accuracy_max_index])

for i in range(len(accuracy_list)):
    print("img"+str(i+1)+" accuracy:",accuracy_list[i])

print("total accuracy: ",accuracy_max)