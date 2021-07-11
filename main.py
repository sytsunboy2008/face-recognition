# 这是一个示例 Python 脚本。

# 按 Shift+F10 执行或将其替换为您的代码。
# 按 双击 Shift 在所有地方搜索类、文件、工具窗口、操作和设置。


def print_hi(name):
    # 在下面的代码行中使用断点来调试脚本。
    print(f'Hi, {name}')  # 按 Ctrl+F8 切换断点。


# 按间距中的绿色按钮以运行脚本。
if __name__ == '__main__':
    print_hi('PyCharm')

import tkinter as tk
from tkinter import messagebox
import os
window=tk.Tk()
window.title('人脸识别系统')
window.geometry('400x285')
l=tk.Label(window,text='人脸识别系统',font=('Arial',12),width=15,height=2)
l.pack()
def jzrl():
    os.system("python load.py")
def xlmx():
    os.system("python train.py")
def rlsb():
    os.system("python recognition.py")
def scck():
    os.system("python del.py")


b1 = tk.Button(window,text='加载人脸',width=15,height=2,command=jzrl)
b2 = tk.Button(window,text='训练模型',width=15,height=2,command=xlmx)
b3 = tk.Button(window,text='人脸识别',width=15,height=2,command=rlsb)
b4 = tk.Button(window,text='人脸数据管理',width=15,height=2,command=scck)
b1.pack()
b2.pack()
b3.pack()
b4.pack()
b5 = tk.Button(window,text='退出', width=15, height=2, command=window.quit)
b5.pack(anchor = 'se')
window.mainloop()
# 访问 https://www.jetbrains.com/help/pycharm/ 获取 PyCharm 帮助
