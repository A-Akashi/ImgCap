import subprocess
from subprocess import PIPE

def main(img):

    print(img) #実行中のファイルを確認する

    #opencv_createsamples実行（-numの数字が増やす枚数）
    subprocess.run("opencv_createsamples.exe -img pos/"+img+" -vec vec/positive"+img+".vec -num 1000 -maxidev 40 -maxxangle 0.8 -maxyangle 0.8 -maxzangle 0.5 -w 24 -h 24",shell=True)

subprocess.run("python mergevec.py -v vec -o vec/pos.vec",shell=True) #vecファイルを一つにする
