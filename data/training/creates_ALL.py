import os
import subprocess
from subprocess import PIPE
import samplecreates

#パスの取得&リスト作成　
path = r'C:/work/ImgCap/data/training/pos' #ポジティブ画像があるファイルのフルパス
files = os.listdir(path)

for dir in files:
    samplecreates.main(dir)

subprocess.run("python mergevec.py -v vec -o vec/pos.vec",shell=True) #vecファイルを一つにする