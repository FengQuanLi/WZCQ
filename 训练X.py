import torch
import torchvision
from PIL import Image
import numpy as np
import time
import json
from config import GPT2Config, TransformerConfig
from Batch import create_masks

import torch.nn.functional as F
from 取训练数据 import *
from 杂项 import *
import os
import random
from  模型_策略梯度 import  Transformer
from 模型_策略梯度 import 智能体
状态辞典B={'击杀小兵或野怪或推掉塔': 0, '击杀敌方英雄': 1, '被击塔攻击': 2,  '被击杀': 3,  '死亡': 4, '普通': 5}
状态辞典={'击杀小兵或野怪或推掉塔': 2, '击杀敌方英雄': 5, '被击塔攻击': -0.5, '被击杀': -2,'无状况':0.01, '死亡': 0.01, '其它': -0.003,'普通': 0.01}
状态列表=[]
for K in 状态辞典B:
    状态列表.append(K)
训练数据保存目录='../训练数据样本/未用'
if not os.path.exists(训练数据保存目录):
   os.makedirs(训练数据保存目录)
for root, dirs, files in os.walk('../训练数据样本/未用'):
    if len(dirs)>0:
        break

词数词典路径="./json/词_数表.json"
数_词表路径="./json/数_词表.json"
if os.path.isfile(词数词典路径) and os.path.isfile(数_词表路径):
    词_数表, 数_词表 = 读出引索(词数词典路径, 数_词表路径)
with open(词数词典路径, encoding='utf8') as f:
    词数词典=json.load(f)
device = torch.device("cuda:0" if (torch.cuda.is_available()) else "cpu")
#
#
config = TransformerConfig()
模型路径 = 'model_weights_2021-05-7D'

model_判断状态=Transformer(6,768,2,12,0.0,6*6*2048)
model_判断状态.load_state_dict(torch.load('weights/model_weights_判断状态L'))
model_判断状态.cuda(device).requires_grad_(False)
N = 15000 # 运行N次后学习
条数 = 100
轮数 = 3
学习率 = 0.0003
智能体 = 智能体(动作数=7, 并行条目数=条数,
          学习率=学习率, 轮数=轮数,
          输入维度=6)





分块大小=600
游标大小=600
树枝=1

计数=0
time_start=time.time()
for j in range(100):
    #random.shuffle(dirs)
    for 号 in dirs:
        预处理数据 = '../训练数据样本/未用/'+号+'/图片_操作预处理数据2.npz'
        if os.path.isfile(预处理数据):
            npz文件 = np.load(预处理数据, allow_pickle=True)
            图片张量np, 操作序列 = npz文件["图片张量np"], npz文件["操作序列"]
            if 图片张量np.shape[0]<600:
                continue
            循环=True
            游标=0
            操作序列=np.insert(操作序列,0,128)

            操作_分_表 = []
            目标输出_分_表 = []
            图片_分_表 = []

            while 循环:
                if 游标 + 分块大小 < 操作序列.shape[0]:

                    操作_分 = 操作序列[游标:游标 + 分块大小]
                    目标输出_分 = 操作序列[游标 + 1:游标 + 1 + 分块大小]
                    图片_分 = 图片张量np[游标:游标 + 分块大小, :]
                    操作_分_表.append(操作_分)
                    目标输出_分_表.append(目标输出_分)
                    图片_分_表.append(图片_分)
                    游标 = 游标 + 游标大小
                else:
                    操作_分 = 操作序列[-分块大小 - 1:-1]
                    目标输出_分 = 操作序列[-分块大小:]

                    图片_分 = 图片张量np[-分块大小:, :]
                    操作_分_表.append(操作_分)
                    目标输出_分_表.append(目标输出_分)
                    图片_分_表.append(图片_分)
                    循环 = False

            循环=True
            i=0
            while 循环:
                if (i+1)*树枝<len(操作_分_表):

                    操作_分_枝=np.array(操作_分_表[i*树枝:(i+1)*树枝])
                    图片_分_枝 = np.array(图片_分_表[i * 树枝:(i + 1) * 树枝])
                    目标输出_分_枝 = np.array(目标输出_分_表[i * 树枝:(i + 1) * 树枝])



                else:
                    操作_分_枝 = np.array(操作_分_表[i * 树枝:len(操作_分_表)])
                    图片_分_枝 = np.array(图片_分_表[i * 树枝:len(图片_分_表)],dtype=np.float32)
                    目标输出_分_枝 = np.array(目标输出_分_表[i * 树枝:len(目标输出_分_表)])
                    循环 = False

                操作_分_torch=torch.from_numpy(操作_分_枝).cuda(device)
                操作序列A = np.ones_like(操作_分_枝)
                操作序列A_torch = torch.from_numpy(操作序列A).cuda(device)
                图片_分_torch = torch.from_numpy(图片_分_枝).cuda(device)
                目标输出_分_torch = torch.from_numpy(目标输出_分_枝).cuda(device)


                src_mask, trg_mask = create_masks(操作_分_torch, 操作_分_torch, device)
                if 图片_分_torch.shape[0]!=操作_分_torch.shape[0]:
                    continue

                状态={}
                状态['操作序列']=操作_分_枝
                状态['图片张量'] = 图片_分_枝
                状态['trg_mask']=trg_mask

                动作, 动作可能性, 评价 = 智能体.选择动作批量(状态, device, 目标输出_分_torch, True)
                实际输出, _ = model_判断状态(图片_分_torch, 操作序列A_torch, trg_mask)
                _, 抽样 = torch.topk(实际输出, k=1, dim=-1)
                抽样np = 抽样.cpu().numpy()
                回报=np.ones_like(抽样np[0, :, 0])
                回报=回报.astype(np.float32)
                for 计数 in range(抽样np.shape[1]):
                    状况 = 状态列表[抽样np[0, 计数, 0]]

                    得分 = 状态辞典[状况]
                    回报[计数]=得分

                智能体.监督强化学习(device,状态,回报,动作,动作可能性,评价)

                # 输出_实际_A = model(图片_分_torch,操作_分_torch ,trg_mask)
                # lin = 输出_实际_A.view(-1, 输出_实际_A.size(-1))
                # optimizer.zero_grad()
               # loss = F.cross_entropy(lin, 目标输出_分_torch.contiguous().view(-1), ignore_index=-1)
                if 计数 % 1 == 0:
                    #print(loss)





                    time_end = time.time()
                    用时 = time_end - time_start

                    #_, 抽样 = torch.topk(输出_实际_A, k=1, dim=-1)
                    #抽样np = 抽样.cpu().numpy()
                    #打印抽样数据(数_词表, 抽样np[0:1,:,:], 目标输出_分_torch[0,:])
                    print("用时{} 第{}轮 第{}张 号{}".format(用时, j, 计数, 号))
                if 计数 % 45060 == 0:
                    print('888')

                #loss.backward()

                #optimizer.step()
                计数=计数+1
                i=i+1
    智能体.保存模型(j)
    #torch.save(model.state_dict(), 'weights/model_weights_2021-05-7D')
    #torch.save(model.state_dict(), 'weights/model_weights_2021-05-7D{}'.format(str(j)))





