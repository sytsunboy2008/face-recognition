import tensorflow as tf
import os
import numpy as np
import random
import cv2
from matplotlib import pyplot as plt
from tensorflow.keras.layers import Conv2D, BatchNormalization, Activation, MaxPool2D, Dropout, Flatten, Dense
from tensorflow.keras import Model
from sklearn.model_selection import train_test_split
import tkinter as tk
from tkinter import messagebox

np.set_printoptions(threshold=np.inf)
physical_devices = tf.config.experimental.list_physical_devices('GPU')
tf.config.experimental.set_memory_growth(physical_devices[0], True)



window = tk.Tk()
faces_path = './face/'
faces_other_path = './face/other/'
#batch_size = 50          # 每次取50张图片
#learning_rate = 0.01        # 学习率
size = 64                 # 图片大小64*64*3
imgs = []                 # 存放人脸图片
labs = []                 # 存放人脸图片对应的标签
def readData(path , h = size , w = size):
    for filename in os.listdir(path):
        if filename.endswith('.jpg'):
            filename = path + '/' + filename
            img = cv2.imread(filename)
            top, bottom, left, right = getPaddingSize(img)
            """放大图片扩充图片边缘部分"""
            img = cv2.copyMakeBorder(img, top, bottom, left, right, cv2.BORDER_CONSTANT, value=[0, 0, 0])
            img = cv2.resize(img, (h, w))
            imgs.append(img)                 # 一张张人脸图片加入imgs列表中
            labs.append(path)                # 一张张人脸图片对应的path，即文件夹名faces_my和faces_other，即标签
def getPaddingSize(img):
    height, width, _ = img.shape
    top, bottom, left, right = (0, 0, 0, 0)
    longest = max(height, width)

    if width < longest:
        tmp = longest - width
        left = tmp // 2
        right = tmp - left
    elif height < longest:
        tmp = longest - height
        top = tmp // 2
        bottom = tmp - top
    else:
        pass
    return top, bottom, left, right
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


def xlmx():
    name = e.get()
    #print(name)
    #print(name.encode('UTF-8').isalpha())
    if name.encode('UTF-8').isalpha():
        if os.path.exists('./model/'+name+'/'+name+'_model.ckpt'):
            tk.messagebox.showinfo(title='请重试', message='模型已存在')
            e.delete(0, "end")
            return
        if not os.path.exists('./face/'+name):
            tk.messagebox.showinfo(title='请重试', message='人脸数据不存在')
            e.delete(0, "end")
            return
        """1、读取人脸数据"""
        #for i in range(1,2):
        readData(faces_path+str(name))
        readData(faces_other_path)
        imgss = np.array(imgs)  # 将图片数据与标签转换成数组
        labss = np.array([[1] if lab == faces_path+name else [0] for lab in labs])

        """2、随机划分测试集与训练集"""

        train_x_1, test_x_1, train_y, test_y = train_test_split(imgss, labss, test_size=0.05,
                                                                random_state=random.randint(0, 100))
        train_x_2 = (train_x_1.reshape(train_x_1.shape[0], size, size, 3)) / 255.  # 参数：图片数据的总数，图片的高、宽、通道
        test_x_2 = (test_x_1.reshape(test_x_1.shape[0], size, size, 3)) / 255.

        model = Baseline()
        model.compile(optimizer='adam',
                      loss=tf.keras.losses.SparseCategoricalCrossentropy(from_logits=False),
                      metrics=['sparse_categorical_accuracy'])
        # model.compile(optimizer='adam',
        #                loss=tf.keras.losses.SparseCategoricalCrossentropy(from_logits=False),
        #                metrics=['ce'])
        # model.compile(optimizer='adam',#sgd
        #                loss=tf.keras.losses.SparseCategoricalCrossentropy(from_logits=False),
        #                metrics=['binary_accuracy'])

        checkpoint_save_path = './checkpoint/'+name+'/Baseline.ckpt'

        if os.path.exists(checkpoint_save_path + '.index'):
            print('-------------load the model-----------------')
            model.load_weights(checkpoint_save_path)

        cp_callback = tf.keras.callbacks.ModelCheckpoint(filepath=checkpoint_save_path,
                                                         save_weights_only=True,
                                                         save_best_only=True)
        history = model.fit(train_x_2, train_y, batch_size=32, epochs=2, validation_data=(test_x_1, test_y),
                            validation_freq=1,
                            callbacks=[cp_callback])
        model.save_weights('./model/'+name+'/'+name+'_model.ckpt')
        model.summary()

        ###############################################    show   ###############################################

        # 显示训练集和验证集的acc和loss曲线
        # print(dict[history.history])
        acc = history.history['sparse_categorical_accuracy']
        val_acc = history.history['val_sparse_categorical_accuracy']
        loss = history.history['loss']
        val_loss = history.history['val_loss']
        # acc = history.history['binary_accuracy']
        # val_acc = history.history['val_binary_accuracy']
        # loss = history.history['loss']
        # val_loss = history.history['val_loss']
        plt.subplot(1, 2, 1)
        plt.plot(acc, label='Training Accuracy')
        plt.plot(val_acc, label='Validation Accuracy')
        plt.title('Training and Validation Accuracy')
        plt.legend()

        plt.subplot(1, 2, 2)
        plt.plot(loss, label='Training Loss')
        plt.plot(val_loss, label='Validation Loss')
        plt.title('Training and Validation Loss')
        plt.legend()
        plt.show()

    else:
        tk.messagebox.showinfo(title='请重试', message='请使用英文字符')
        e.delete(0, "end")


window.title('训练模型')
window.geometry('240x200')
l = tk.Label(window, text='输入姓名', font=('Arial', 12), width=15, height=2)
l.pack()
e = tk.Entry(window, show=None)
e.pack()
name = e.get()
b1 = tk.Button(window,text='开始训练', width=15, height=2, command=xlmx)
b1.pack()
b2 = tk.Button(window,text='退出', width=15, height=2, command=window.quit)
b2.pack()
window.mainloop()
