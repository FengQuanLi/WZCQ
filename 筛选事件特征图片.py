import os
# device = torch.device("cuda:0" if (torch.cuda.is_available()) else "cpu")
# print(device)
import json
import cv2
import numpy as np
import time
from PIL import Image, ImageDraw, ImageFont
import shutil

from pynput.keyboard import Controller, Key, Listener
from pynput import keyboard
import threading

态='暂停'
def cv2ImgAddText(img, text, left, top, textColor=(0, 255, 0), textSize=20):
    if (isinstance(img, np.ndarray)):  # 判断是否OpenCV图片类型
        img = Image.fromarray(cv2.cvtColor(img, cv2.COLOR_BGR2RGB))
    # 创建一个可以在给定图像上绘图的对象
    draw = ImageDraw.Draw(img)
    # 字体的格式
    fontStyle = ImageFont.truetype(
        'C:/Windows/Fonts/STHUPO.TTF', textSize, encoding="utf-8")
    #"D:/python/辅助/锐字真言体.ttf"
    # 绘制文本
    draw.text((left, top), text, textColor, font=fontStyle)
    # 转换回OpenCV格式
    return cv2.cvtColor(np.asarray(img), cv2.COLOR_RGB2BGR)
def get_key_name(key):
    if isinstance(key, keyboard.KeyCode):


        return key.char
    else:

        return str(key)
# 监听按压
def on_press(key):
    global 态

    key_name=get_key_name(key)
   # print(key_name)
    # 操作=''
    # if key_name=='w':
    #     W键按下=True
    #
    # elif key_name=='Key.left':
    #     操作='一技能'
    # elif key_name=='Key.down':
    #     操作='二技能'
    # elif key_name=='Key.right':
    #     操作='三技能'
    # elif key_name=='Key.up' :
    #     攻击态=True


# 监听释放
def on_release(key):
    global 态


    key_name=get_key_name(key)


    if key_name=='Key.up' :

        态='弃'
    elif key_name=='Key.left':
        态='普通'
    elif key_name=='Key.down':
        态='过'
    elif key_name=='Key.right':
        态='死亡'
    elif key_name=='a':
        态='击杀敌方英雄'
    elif key_name=='s':
        态='击杀小兵或野怪或推掉塔'
    elif key_name=='d':
        态='被击杀'
    elif key_name == 'w':
        态 = '被击塔攻击'

    #print("已经释放:", key_name)
    if key == Key.esc:
        # 停止监听
        return False

# 开始监听
def start_listen():
    with Listener(on_press=on_press, on_release=on_release) as listener:
        listener.join()

th = threading.Thread(target=start_listen,)
th.start()










#筛选事件特征图片
#1、进入目录打开引索 方法抄
路径json='../判断数据样本test/_判断数据.json'
路径新='../判断数据样本/'
if not os.path.exists(路径新):
    os.makedirs(路径新)
路径新 = 路径新 +'判断新.json'
全部数据={}
with open(路径json, encoding='ansi') as f:
    while True:
        df = f.readline()
        df = df.replace('\'', '\"')

        if df == "":
            break
        单元 = json.loads(df)
        for key in 单元:
            全部数据[key]=单元[key]

#print(全部数据)

for key in 全部数据:
    记录文件 = open(路径新, 'a+')
   # print(key + ':' + 全部数据[key])
    图片路径 = '../判断数据样本test/' + key + '.jpg'
    图片新路径 = '../判断数据样本/'+ key + '.jpg'


    # 截图 = cv2.imread(图片路径)
    截图 = cv2.imdecode(np.fromfile(图片路径, dtype=np.uint8), -1)
    截图 = cv2ImgAddText(截图, 全部数据[key], 0, 0, (000, 222, 111), 25)
    cv2.imshow('AAA', 截图)
    cv2.waitKey()

    while 态 == '暂停':
        time.sleep(0.02)
    新输出={}

    校准输出=全部数据[key]
    if 态=='过':
        校准输出 = 全部数据[key]
    elif 态=='普通':
        校准输出 = '普通'
    elif 态 == '死亡':
        校准输出 = '死亡'
    elif 态 == '被击杀':
        校准输出 = '被击杀'
    elif 态 == '击杀小兵或野怪或推掉塔':
        校准输出 = '击杀小兵或野怪或推掉塔'
    elif 态 == '击杀敌方英雄':
        校准输出 = '击杀敌方英雄'
    elif 态 == '被击塔攻击':
        校准输出 = '被击塔攻击'
    elif 态 == '弃' and key!='162098566208':
        态 = '暂停'
        continue
    else:
        print(1)
    print(key, 校准输出)
    新输出[key]=校准输出
    json.dump(新输出, 记录文件, ensure_ascii=False)
    记录文件.write('\n')
    shutil.copy(图片路径, 图片新路径)

    态 = '暂停'
    记录文件.close()

# def CV信息显示():
#     global 全部数据,态
#
#
#
#
# CV信息= threading.Thread(target=CV信息显示)
# CV信息.start()
# d=666
# for i in range(555):
#
#     while 态=='暂停':
#      time.sleep(1)
#      print(d)
#     态 = '暂停'
#     d=d+1
