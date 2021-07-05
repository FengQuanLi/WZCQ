# 用强化学习训练AI玩王者荣耀
 ## 说明
一、这个项目由我之前的[AI玩王者荣耀](https://github.com/FengQuanLi/ResnetGPT) 项目发展而来。  
    较之前的项目有进步，也更加复杂。用了强化学习的训练方法，并增加了一个判断回报状态的神经网络。  
 ### 注意! 这个项目处于试验阶段，再加上本人水平有限，所以并不完善，问题很多。
二、运行环境和我之前的项目相似，请参考之前的项目，但还需要添加 PyQt5模块用于截图参考（requirements.txt）。  
环境配置参考视频 1  
链接：https://pan.baidu.com/s/1fJRyX-scxbeOJ2lsddTLiA   
提取码：msr5  
环境配置参考视频 2（minitouch安装）    
链接：https://pan.baidu.com/s/1V8VV24Q1U8ajLKXU9HJT2w     
提取码：rid1  
三、我用的显卡是gtx 1660 。根据以往经验，需要1060或以上算力的显卡。  
四、需要一台打开安卓调试并能玩王者荣耀的手机，虚拟机也可行。  
五、需要下载[scrcpy](https://github.com/Genymobile/scrcpy/blob/master/README.zh-Hans.md)  的windows版本。 把所有文件解压到项目根目录即可（这是我的笨办法） 。  

六、pyminitouch库运行时会自动安装minitouch。如果无法自动安装则需要手动安装[minitouch](https://github.com/openstf/minitouch) ，比较麻烦。  
还有，minitouch不支持Android10及以上系统  
七、本人用的手机分辨率是1080*2160的，本代码并没有针对不同的手机做优化。不同的手机minitouch命令中所描述的位置会有差异，需要对代码做出相应调整，请务必注意。  
八、注意游戏的布局（摇杆和按键移向两边）务必要一样，可参考B站视频[B站视频](https://www.bilibili.com/video/BV13V411W7aW) 。  

  


## 运行游戏AI
  
一、首先下载主模型 你可以从[google云盘](https://drive.google.com/file/d/1oTlAKPwmmfs8BL2GIMqerayeO0dyXfmk/view?usp=sharing) 下载训练过的模型，也可以百度网盘下载  
链接：https://pan.baidu.com/s/1RxkZXD17GCux6ixeCcX2kg   
提取码：ez3d   
后放入weights文件夹下  
注意！要加载不同的模型请 在 模型_策略梯度.py 261行更改。


二、先运行 “启动和结束进程.py” 启动scrcpy

三、启动王者荣耀进入5v5人机对战    运行 “训练数据截取_A.py” 即可。
## 生成训练数据（半自动）
运行 “训练数据截取_A.py” 这时就可以生成训练用的数据。  
按"i"键则结束或则是重新运行  
按键'w' 's ' 'a' 'd'控制方向  左、下、右箭头对应是1、2、3技能，上箭头长按则攻击。其它按键请参考源码。   
注意！！ 如果用按键控制则会记录按键操作数据，否则会记录AI玩游戏的数据。  
根据我的经验，随着模型训练次数增加，手动干预的次数会越来越小。但总体来说训练数据的获取依然需要人为干预，因为游戏结束
后要重新开始需要手动操控（我并没有做自动化脚本）。

# 如何训练主模型
一、下载状态判断模型 你可以从[google云盘](https://drive.google.com/file/d/1eqy-xX29sjEguuQI_1m8qaLEX3g4KAQ7/view?usp=sharing) 下载训练过的模型，也可以百度网盘下载  
链接：https://pan.baidu.com/s/1-UCuPutZQck3Iawot9bGrw 
提取码：545t  
后放入weights文件夹下 
二、数据预处理  
将图片用resnet101预处理后再和对应操作数据一起处理后用numpy数组储存备用。  
具体要做的就是运行 “处理训练数据5.py”   
三、训练  
预处理完成以后运行 “训练X.py”即可。  
注意！模型保存路径 在 模型_策略梯度.py 295和296行更改。  
我的经验是：在原有模型上进行30-40局游戏，训练两轮（会保存两次）即可。
# 如何训练状态判断模型
状态判断模型实际上是一个图像分类的神经网络，用了和主模型基本一样结构，只是参数不同。  
一、获取标注数据  
标注数据是在游戏运行过程中进行标注的，运行 状态标注.py    
Key.left 击杀小兵或野怪或推掉塔
Key.down 击杀敌方英雄
Key.right 被击塔攻击
Key.up 被击杀  
注意这里会加载的标注模型自动参与标注数据以减轻标注工作负担。  
二、校正标注数据  
前一步获取的标注数据是不准确的，需要手动校准。  
运行 筛选事件特征图片.py   
具体操作参考 代码68至81行  注意其中“过”表示认同原始标注。  
三、训练  
运行 训练状态判断模型A.py 
# 其它问题
源码中保留了一些试验过程中的废代码。


