
import tensorflow as tf
import cv2
import numpy as np
import dlib
import sys
import os
from tensorflow.keras.layers import Conv2D, BatchNormalization, Activation, MaxPool2D, Dropout, Flatten, Dense
from tensorflow.keras import Model
from sklearn.model_selection import train_test_split
size = 64
name = os.listdir('./face/')
nl=[]
for name1 in name:
    if(name1=='other'):
        continue
    else:
        nl.append(name1)
print(nl)
class Baseline(Model):
    def __init__(self):
        super(Baseline, self).__init__()
        self.c1 = Conv2D(filters=32, kernel_size=(3, 3), padding='same')  # 卷积层
        self.b1 = BatchNormalization()  # BN层
        self.a1 = Activation('sigmoid')  # 激活层
        self.p1 = MaxPool2D(pool_size=(2, 2), strides=2, padding='same')  # 池化层
        self.d1 = Dropout(0.2)  # dropout层

        self.c2 = Conv2D(filters=64, kernel_size=(3, 3), padding='same')  # 卷积层
        self.b2 = BatchNormalization()  # BN层
        self.a2 = Activation('sigmoid')  # 激活层
        self.p2 = MaxPool2D(pool_size=(2, 2), strides=2, padding='same')  # 池化层
        self.d2 = Dropout(0.2)  # dropout层

        self.c3 = Conv2D(filters=64, kernel_size=(3, 3), padding='same')  # 卷积层
        self.b3 = BatchNormalization()  # BN层
        self.a3 = Activation('sigmoid')  # 激活层
        self.p3 = MaxPool2D(pool_size=(2, 2), strides=2, padding='same')  # 池化层
        self.d3 = Dropout(0.2)  # dropout层

        self.flatten = Flatten()
        self.f1 = Dense(4096, activation='relu')
        self.d2 = Dropout(0.2)
        self.f2 = Dense(512, activation='relu')
        self.d3 = Dropout(0.2)
        self.f3 = Dense(2, activation='softmax')
    def call(self, x):
        x = self.c1(x)
        x = self.b1(x)
        x = self.a1(x)
        x = self.p1(x)
        x = self.d1(x)

        x = self.c2(x)
        x = self.b2(x)
        x = self.a2(x)
        x = self.p2(x)
        x = self.d2(x)

        x = self.c3(x)
        x = self.b3(x)
        x = self.a3(x)
        x = self.p3(x)
        x = self.d3(x)

        x = self.flatten(x)
        x = self.f1(x)
        x = self.d2(x)
        x = self.f2(x)
        x = self.d3(x)
        y = self.f3(x)
        return y
detector = dlib.get_frontal_face_detector()
i = 0
#m = []
model = []
for data in nl:
    model.append(Baseline())
    model[i].compile(optimizer='adam',
               loss=tf.keras.losses.SparseCategoricalCrossentropy(from_logits=False),
               metrics=['sparse_categorical_accuracy'])
    model[i].load_weights('./model/'+data+'/'+data+'_model.ckpt')
    i = i+1
# model=Baseline()
# model.compile(optimizer='adam',
#                 loss=tf.keras.losses.SparseCategoricalCrossentropy(from_logits=False),
#                 metrics=['sparse_categorical_accuracy'])
# model.load_weights('./model/wj/wj_model.ckpt')

cap = cv2.VideoCapture(0)  # 打开摄像头
while True:
    _, img = cap.read()  # 读取
    gray_image = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)  # 灰度化
    dets = detector(gray_image, 1)
    if not len(dets):
        key = cv2.waitKey(30)
        if key == 27:
            sys.exit(0)
    for i, d in enumerate(dets):
        x1 = d.top() if d.top() > 0 else 0
        y1 = d.bottom() if d.bottom() > 0 else 0
        x2 = d.left() if d.left() > 0 else 0
        y2 = d.right() if d.right() > 0 else 0
        """人脸大小64*64"""
        face = img[x1:y1, x2:y2]
        face = cv2.resize(face, (size, size))

        cv2.rectangle(img, (x2, x1), (y2, y1), (255, 0, 0), 3)
        face = np.array(face)
        face = tf.expand_dims(face, axis=0)
        face = tf.cast(face,dtype=tf.float32)
        face = face/255.
        i = 0
        flag = 0
        for n in nl:
            pri = model[i].predict(face)
            if pri[0][0] < 0.5:
                flag = 1
                break
            else:
                i = i+1

        print(pri)
        if flag:
            cv2.putText(img, nl[i], (x2, y2), cv2.FONT_HERSHEY_COMPLEX, 0.5, (0, 255, 0), 2)
        else:
            cv2.putText(img, 'stranger', (x2, y2), cv2.FONT_HERSHEY_COMPLEX, 0.5, (0, 255, 0), 2)
        """通过确定对角线画矩形"""
        # cv2.rectangle(img, (x2, x1), (y2, y1), (255, 0, 0), 3)
    cv2.imshow('image', img)
    key = cv2.waitKey(30)
    if key == 27:
          # #释放摄像头并销毁所有窗口
        cap.release()
        cv2.destroyAllWindows()
        sys.exit(0)

