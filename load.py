"""-----------------------------------------
一、采集我的人脸数据集
获取本人的人脸数据集10000张，使用的是dlib来
识别人脸，虽然速度比OpenCV识别慢，但是识别效
果更好。
人脸大小：64*64
-----------------------------------------"""
import cv2
import dlib
import os
import random
import tkinter as tk
from tkinter import messagebox
def img_change(img, light=1, bias=0):
    width = img.shape[1]
    height = img.shape[0]
    for i in range(0, width):
        for j in range(0, height):
            for k in range(3):
                tmp = int(img[j, i, k] * light + bias)
                if tmp > 255:
                    tmp = 255
                elif tmp < 0:
                    tmp = 0
                img[j, i, k] = tmp
    return img


"""特征提取器:dlib自带的frontal_face_detector"""

detector = dlib.get_frontal_face_detector()

def jzrl():
    name = e.get()
    print(name)
    print(name.encode('UTF-8').isalpha())
    if name.encode('UTF-8').isalpha():
        faces_my_path = './face/'
        size = 64
        if not os.path.exists(faces_my_path+name):
            os.makedirs(faces_my_path+name)
            cap = cv2.VideoCapture(0)
            num = 1
            print(str(num))
            #print(cv2.COLOR_BGR2GRAY)
            while True:
                if (num <= 100):  # 10000
                    print('Being processed picture %s' % num)
                    success, img = cap.read()
                    gray_img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
                    """使用特征提取器进行人脸检测"""
                    dets = detector(gray_img, 1)
                    """--------------------------------------------------------------------
                    使用enumerate 函数遍历序列中的元素以及它们的下标,i为人脸序号,d为i对应的元素;
                    left：人脸左边距离图片左边界的距离 ；right：人脸右边距离图片左边界的距离
                    top：人脸上边距离图片上边界的距离 ；bottom：人脸下边距离图片上边界的距离
                    ----------------------------------------------------------------------"""
                    for i, d in enumerate(dets):
                        x1 = d.top() if d.top() > 0 else 0
                        y1 = d.bottom() if d.bottom() > 0 else 0
                        x2 = d.left() if d.left() > 0 else 0
                        y2 = d.right() if d.right() > 0 else 0

                        face = img[x1:y1, x2:y2]
                        """调整图片的对比度与亮度， 对比度与亮度值都取随机数，这样能增加样本的多样性"""
                        face = img_change(face, random.uniform(0.5, 1.5), random.randint(-50, 50))
                        face = cv2.resize(face, (size, size))
                        cv2.imshow('image', face)
                        cv2.imwrite(faces_my_path + name+ '/' + str(num) + '.jpg', face)
                        num += 1
                    key = cv2.waitKey(30)
                    if key == 27:
                        break
                else:
                    print('Finished!')
                    tk.messagebox.showinfo(title='完成', message='人脸加载完成')
                    break

        else:
            tk.messagebox.showinfo(title='请重试', message='人脸已存在')
            print('已存在人脸数据')
    else:
        tk.messagebox.showinfo(title='请重试', message='请使用英文字符')
        e.delete(0, "end")


window = tk.Tk()
window.title('加载人脸')
window.geometry('240x200')
l = tk.Label(window, text='输入姓名', font=('Arial', 12), width=15, height=2)
l.pack()
e = tk.Entry(window, show=None)
e.pack()
b1 = tk.Button(window,text='加载人脸', width=15, height=2, command=jzrl)
b1.pack()
b2 = tk.Button(window,text='退出', width=15, height=2, command=window.quit)
b2.pack()
window.mainloop()

####