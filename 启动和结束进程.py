
import os

os.system('taskkill /IM scrcpy.exe /F')
os.system('taskkill /IM adb.exe /F')
#os.system('adb connect 127.0.0.1:7555')
os.system("scrcpy --max-size 960")