import os
import tkinter as tk
from tkinter import messagebox
import shutil
window = tk.Tk()
window.title('人脸数据删除模块')
window.geometry('400x400')

def sc():
    n = str(lb.get(lb.curselection()))
    index = lb.curselection()
    if os.path.exists('./model/' + n):
        shutil.rmtree('./face/' + n)
        # os.removedirs('./face/'+n)
        shutil.rmtree('./model/' + n)
        # os.removedirs('./model/'+n)
        tk.messagebox.showinfo(title='成功', message='成功人脸数据和模型数据')

    else:
        shutil.rmtree('./face/' + n)
        # os.removedirs('./face/' + n)
        tk.messagebox.showinfo(title='成功', message='成功人脸数据')
    lb.delete(index)
name = os.listdir('./face/')

l=tk.Label(window,text='选择人脸数据',font=('Arial',12),width=15,height=2)
l.pack()
var2 = tk.StringVar()
lb = tk.Listbox(window,listvariable=var2)
for name1 in name:
    if(name1=='other'):
        continue
    else:
        lb.insert('end',name1)
lb.pack()
b1 = tk.Button(window, text='删除该数据', width=15, height=2, command=sc)
b1.pack()
b2 = tk.Button(window, text='退出', width=15, height=2, command=window.quit)
b2.pack()
window.mainloop()

