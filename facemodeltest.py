import os
from tensorflow.python.keras.models import load_model
import numpy as np
from sklearn.metrics import precision_score
from sklearn.metrics import f1_score
from sklearn.metrics import accuracy_score
from sklearn.metrics import recall_score
import sklearn
def facemodel_test(moudel_dir,data_dir,label_dir):
    facemodel=tf.keras.models.load_moudel(dir)
    image_data= np.loadtxt(data_dir)
    labels=np.loadtxt(label_dir)
    y_pred=facemodel.predict(image_data)
    print("accuracy_score:", accuracy_score(labels, y_pred) * 100)
    print("recall_score:"  , recall_score(labels, y_pred) * 100)
    print("precision_score:", precision_score(labels, y_pred) * 100)
    print("f1_score:", f1_score(labels, y_pred) * 100)
if __name__ == '__main__':
    moudel_dir="./moudel.h5"
    data_dir="./face_data"
    label_dir="./face_label"
    facemodel_test(moudel_dir,data_dir,label_dir)


