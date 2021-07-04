import socket
import json
import sys
import time, threading
import cv2
import torch
import numpy as np
from 辅助功能 import 状态信息综合
import torchvision
from resnet_utils import myResnet
from 模型_策略梯度 import 智能体
from Batch import create_masks
import subprocess
from PyQt5.QtWidgets import QApplication
from PIL import Image, ImageQt
import os
import win32gui, win32ui, win32con
from 取训练数据 import 读出引索
from 运行辅助 import MyMNTDevice,取图
from pynput.keyboard import  Key, Listener
from pynput import keyboard
import random
from  模型_策略梯度 import  Transformer
#window = int(subprocess.check_output(["xdotool", "search" ,"VehiclePhysicsExampleeeveed181"]).decode('ascii').split('\n')[0])
_DEVICE_ID = '68UDU17B14011947'
窗口名称="RNE-AL00"
window = win32gui.FindWindow(0,窗口名称)

device = torch.device("cuda:0" if (torch.cuda.is_available()) else "cpu")
mod = torchvision.models.resnet101(pretrained=True).eval().cuda(device).requires_grad_(False)
resnet101 = myResnet(mod)
model_判断状态=Transformer(6,768,2,12,0.0,6*6*2048)
model_判断状态.load_state_dict(torch.load('weights/model_weights_判断状态L'))
model_判断状态.cuda(device)
N = 15000 # 运行N次后学习
条数 = 100
轮数 = 3
学习率 = 0.0003
智能体 = 智能体(动作数=7, 并行条目数=条数,
          学习率=学习率, 轮数=轮数,
          输入维度=6)


接收反馈=True
总次数=0
阶段分=0
阶段分2=0




步数=0
学习次数=0
分数记录 = []
速度记录=[]
最高分=0
time.sleep(1)
app = QApplication(sys.argv)
screen = app.primaryScreen()

训练数据保存目录='../训练数据2'
计时开始=0

加三技能='d 0 552 1878 100\nc\nu 0\nc\n'
加二技能='d 0 446 1687 100\nc\nu 0\nc\n'
加一技能='d 0 241 1559 100\nc\nu 0\nc\n'
购买='d 0 651 207 100\nc\nu 0\nc\n'
词数词典路径="./json/词_数表.json"
数_词表路径="./json/数_词表.json"
操作查询路径="./json/名称_操作.json"
操作词典={"图片号":"0","移动操作":"无移动","动作操作":"无动作"}

if os.path.isfile(词数词典路径) and os.path.isfile(数_词表路径):
    词_数表, 数_词表 = 读出引索(词数词典路径, 数_词表路径)
with open(词数词典路径, encoding='utf8') as f:
    词数词典 = json.load(f)
with open(操作查询路径, encoding='utf8') as f:
    操作查询词典 = json.load(f)

方向表 = ['上移', '下移', '左移', '右移', '左上移', '左下移', '右上移', '右下移']
设备 = MyMNTDevice(_DEVICE_ID)
旧指令='移动停'
一键按下=False
二键按下=False
三键按下=False
四键按下=False
五键按下=False
六键按下=False
七键按下=False
八键按下=False
继续=True
#需要用一个东西来装关键事件
#需要储存事件以及对应的图片

状态辞典={'击杀小兵或野怪或推掉塔': 1, '击杀敌方英雄': 5, '被击塔攻击': -2, '被击杀': -5,'无状况': 0, '死亡': 0, '其它': -0.03,'普通': 0}
状态辞典A={'击杀小兵或野怪或推掉塔': 0, '击杀敌方英雄': 1, '被击塔攻击': 2,  '被击杀': 3, '无状况': 4, '死亡': 5, '其它': 6,'普通': 7}
状况='无状况'
状态辞典B={'击杀小兵或野怪或推掉塔': 0, '击杀敌方英雄': 1, '被击塔攻击': 2,  '被击杀': 3,  '死亡': 4, '普通': 5}
状态列表=[]
for K in 状态辞典B:
    状态列表.append(K)
def get_key_name(key):
    if isinstance(key, keyboard.KeyCode):


        return key.char
    else:

        return str(key)
def on_release(key):
    global 一键按下,状况
    key_name=get_key_name(key)
    if key_name=='1':
        一键按下=False
    if key_name == '2':
        二键按下 = False

    if key_name == '3':
        三键按下 = False
    if key_name == '4':
        四键按下 = False
    if key_name == '5':
        五键按下 = False
    if key_name == '6':
        六键按下 = False
    if key_name == '7':
        七键按下 = False
    if key_name == '8':
        八键按下 = False
    if key_name == 'Key.page_down':
        状况='无状况'
    print("已经释放:", key_name)
    if key == Key.esc:
        # 停止监听
        return False

def on_press(key):
    global 一键按下,状况,继续

    key_name=get_key_name(key)
    操作=''
    if key_name=='Key.left':
        状况='击杀小兵或野怪或推掉塔'

    if key_name == 'Key.down':
        状况='击杀敌方英雄'
    if key_name == 'Key.right':
        状况='被击塔攻击'
    if key_name == 'Key.up':
        状况='被击杀'
    if key_name == 'Key.page_down':
        状况='其它'
    if key_name == 'q':
        状况='普通'
    if key_name == 'e':
        状况='死亡'
    if key_name == 'i':

        继续 = bool(1 - 继续)

    print(状况)



def start_listen():
    with Listener(on_press=on_press, on_release=on_release) as listener:
        listener.join()
th = threading.Thread(target=start_listen,)
th.start()
判断数据保存目录='../判断数据样本test'
if not os.path.exists(判断数据保存目录):
   os.makedirs(判断数据保存目录)
图片路径=判断数据保存目录+'/'
if not os.path.exists(图片路径):
    os.mkdir(图片路径)

for i in range(6666666):
    图片张量 = torch.Tensor(0)
    操作序列 = np.ones((1, 1))
    操作序列[0] = 128
    计数=0

    while 继续:
        计时开始 = time.time()



        img = screen.grabWindow(window)
        image = ImageQt.fromqimage(img)
        imgA = image.resize((960, 480))
        #imgA = 取图(窗口名称)


        图片数组=np.asarray(imgA)
        截屏 = torch.from_numpy(图片数组).cuda(device).unsqueeze(0).permute(0, 3, 2, 1) / 255
        _, out = resnet101(截屏)
        out = torch.reshape(out, (1,6*6*2048))
        操作序列A=np.ones((1,1))
        操作张量A = torch.from_numpy(操作序列A.astype(np.int64)).cuda(device)
        src_mask, trg_mask = create_masks(操作张量A.unsqueeze(0), 操作张量A.unsqueeze(0), device)
        outA=out.detach()
        实际输出,_=model_判断状态(outA.unsqueeze(0), 操作张量A.unsqueeze(0),trg_mask)
        #实际输出=model_判断状态(out, 操作张量.unsqueeze(0),trg_mask)
        _, 抽样 = torch.topk(实际输出, k=1, dim=-1)
        抽样np = 抽样.cpu().numpy()
        if 图片张量.shape[0] == 0:

            图片张量 = out



        elif 图片张量.shape[0] <120:

            图片张量 = torch.cat((图片张量, out), 0)

            操作序列 = np.append(操作序列, 动作)


        else:

            图片张量 = 图片张量[0:119, :]
            操作序列 = 操作序列[0:119]
            操作序列 = np.append(操作序列, 动作)
            图片张量 = torch.cat((图片张量, out), 0)


        操作张量 = torch.from_numpy(操作序列.astype(np.int64)).cuda(device)
        src_mask, trg_mask = create_masks(操作张量.unsqueeze(0), 操作张量.unsqueeze(0), device)





        状态 = 状态信息综合(图片张量.cpu().numpy(), 操作序列,trg_mask)
        完结=False
        局内计数=0




        动作, 动作可能性, 评价 = 智能体.选择动作(状态,device,0)

        if 计数 % 50 == 0 and 计数 != 0:
            设备.发送(购买)
            设备.发送(加三技能)
            设备.发送(加二技能)
            设备.发送(加一技能)
            设备.发送(操作查询词典['移动停'])
            print(旧指令, '周期')
            time.sleep(0.02)
            设备.发送(操作查询词典[旧指令])

        指令 = 数_词表[str(动作)]
        指令集 = 指令.split('_')
        if 指令集[0]=='无移动':
            指令集[0]='移动停'
        if 指令集[0]==旧指令:

            操作词典['移动操作'] = 指令集[0]
            操作词典['动作操作'] = 指令集[1]
        else:
            操作词典['移动操作'] = 指令集[0]
            操作词典['动作操作'] = 指令集[1]
            旧指令=指令集[0]
            设备.发送(操作查询词典[指令集[0]])
        time.sleep(0.01)
        if 指令集[1] != '无动作' and 指令集[1] != '发起集合' and 指令集[1] != '发起进攻' and 指令集[1] != '发起撤退':
            设备.发送(操作查询词典[指令集[1]])


#状态辞典={'击杀小兵或野怪或推掉塔': 1, '击杀敌方英雄': 5, '被击塔攻击': -2, '被击杀': -5,'无状况': 0, '死亡': 0, '其它': -0.03,'普通': 0}
        if 状况=='其它' or 状况=='无状况':

            状况=状态列表[抽样np[0, 0,0,0]]
        得分=状态辞典[状况]
        #or 状况 == '被击塔攻击' or 状况 == '被击杀'
        #print(状况, '得分', 得分)
        if 状况=='击杀小兵或野怪或推掉塔' or 状况=='击杀敌方英雄'or 状况 == '被击塔攻击' or 状况 == '被击杀':
            print(状况,'得分',得分)
            if 状况 == '击杀小兵或野怪或推掉塔' or 状况 == '击杀敌方英雄' :
                保存事件时间=str(int(time.time()*100))

                imgA.save(图片路径+保存事件时间+'.jpg')
                事件词典={}
                事件词典[保存事件时间]=状况
                记录文件 = open(图片路径 + '_判断数据.json', 'a+')
                json.dump(事件词典, 记录文件, ensure_ascii=False)
                记录文件.write('\n')
                记录文件.close()
        elif 状况=='普通':
            随机筛选=random.randint(0, 5000)
            if 随机筛选%100000==0:

                print(状况,'得分',得分)
                保存事件时间=str(int(time.time()*100))

                imgA.save(图片路径+保存事件时间+'.jpg')
                事件词典={}
                事件词典[保存事件时间]=状况
                记录文件 = open(图片路径 + '_判断数据.json', 'a+')
                json.dump(事件词典, 记录文件, ensure_ascii=False)
                记录文件.write('\n')
                记录文件.close()
        elif 状况=='死亡':
            随机筛选=random.randint(0, 5000)
            if 随机筛选%50000==0:
                print(状况,'得分',得分)
                保存事件时间=str(int(time.time()*100))

                imgA.save(图片路径+保存事件时间+'.jpg')
                事件词典={}
                事件词典[保存事件时间]=状况
                记录文件 = open(图片路径 + '_判断数据.json', 'a+')
                json.dump(事件词典, 记录文件, ensure_ascii=False)
                记录文件.write('\n')
                记录文件.close()
        if 状况!='其它':
            状况='无状况'
        else:
            print('其它得分',得分)



        状态['图片张量']=状态['图片张量'][:, -1:, :]
        状态['操作序列']=状态['操作序列'][-1:]
        状态['trg_mask'] = 0
        #智能体.记录数据(状态, 动作, 动作可能性, 评价, 得分, 完结,计数)

        用时1 = 0.22 - (time.time() - 计时开始)
        if 用时1 > 0:
            time.sleep(用时1)


        计数 = 计数 + 1
        if 计数 % 10 == 0:
            print(用时1)


        if 继续 is False:

         print('学习中。。。。。。。。。。。。。。。。')
         #智能体.学习(device)
         print('分数', 1)
         #智能体.保存模型(学习次数)
         分数记录 = []
         速度记录=[]
         print('学习完毕。。。。。。。。。。。。。。。。')
         #智能体.存硬盘('PPO训练数据/'+str(int(time.time())))
         #智能体.保存模型(学习次数)

    time.sleep(1)
    print('继续',继续)









#     状态=状态_
#     延迟 = 0.22 - (time.time() - 计时开始)
#     if 延迟 > 0:
#         time.sleep(延迟)
#     局内计数 = 局内计数 + 1
#
# 分数记录.append(分数)
#
# 平均分 = np.mean(分数记录[-500:])
# 平均速度 = np.mean(速度记录[-15000:])
# if 平均分 > 最高分:
#     最高分 = 平均分
#
# print('步数', 步数, '平均分', 平均分,'最高分',最高分,'局数',i,'平均速度',平均速度)










    #time.sleep(2)
    # while True:
    #
    #     time.sleep(11)




