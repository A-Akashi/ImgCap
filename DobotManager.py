from DobotDriver.DobotWrapper import DobotWrapper
import socket
import threading
import time
import cv2
import numpy as np



class DobotManager:
    def __init__(self):
        self.duringAutoSeq = False
        self.rotation = -35
        self.host = '127.0.0.1'
        self.port = 12345
        self.client_socket = self.create_socket()
        
        # Dobot上の対応点
        #dobot_points = np.array([[x1, y1] , [x2, y2]  , [x3, y3]  , [x4, y4]], dtype=np.float32)
        dobot_points = np.array([[190, -15], [245, -10], [220, 65], [145, 60]], dtype=np.float32)

        # カメラ画像上の対応点
        #camera_points = np.array([[u1, v1] , [u2, v2]  , [u3, v3]  , [u4, v4]], dtype=np.float32)
        camera_points = np.array([[152, 179], [134, 306], [311, 280], [319, 133]], dtype=np.float32)
        
        # 変換行列の作成
        self.transformation_matrix = cv2.getPerspectiveTransform(camera_points, dobot_points)
        return

    def create_socket(self):
        client_socket = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
        client_socket.connect((self.host, self.port))
        return client_socket
    
    
    def move(self):
        
        posX = int(self.GUIManager.destXEntry.get() or 0)
        posY = int(self.GUIManager.destYEntry.get() or 0)
        posZ = int(self.GUIManager.destZEntry.get() or 0)
        rot = int(self.GUIManager.rotEntry.get() or 0)
        
        # 別スレッド化で実行
        threading.Thread(target=self.moveXYZ, args=(posX, posY, posZ, rot)).start()
        return 
    
    def moveXYZ(self, posX, posY, posZ, rot):
        
        # XY駆動
        message = f"move|{posX}|{posY}|stay|{rot}"
        
        self.client_socket.sendall(message.encode())

        data = self.client_socket.recv(1024)
        print(f"Received response: {data.decode()}")

        
        # Z駆動
        message = f"move|{posX}|{posY}|{posZ}|{rot}"
        
        self.client_socket.sendall(message.encode())

        data = self.client_socket.recv(1024)
        print(f"Received response: {data.decode()}")      
        
        return
    
    def grip(self):       
       
        def run_in_Tread_grip() :
            message = "grip"
            self.client_socket.sendall(message.encode())

            data = self.client_socket.recv(1024)
            print(f"Received response: {data.decode()}")
        
        threading.Thread(target=run_in_Tread_grip, args=()).start()
        
        return 
    
    
    def release(self):

        def run_in_Tread_release() :
            message = "release"
            self.client_socket.sendall(message.encode())

            data = self.client_socket.recv(1024)
            print(f"Received response: {data.decode()}")
        
        threading.Thread(target=run_in_Tread_release, args=()).start()
        
        return
    

    def auto_detect(self):
        # AutoによるDobotアーム処理中に設定。
        self.duringAutoSeq = True

        # 検知座標取得
        ImgPosX = self.GUIManager.center_x
        ImgPosY = self.GUIManager.center_y
        
        # Dobot座標系へ変換
        DobotPosX, DobotPosY = self.convert_dobot_coordinate(ImgPosX, ImgPosY) 
        
        threading.Thread(target=self.auto_sequence, args=(DobotPosX, DobotPosY)).start()
        
        return 

    def auto_sequence(self, DobotPosX, DobotPosY) :
        
        # 対象座標へ移動
        self.moveXYZ(DobotPosX, DobotPosY, -24, self.rotation)
        
        time.sleep(0.5)
        
        # 空圧グリッパーを閉じる
        message = "grip"
        self.client_socket.sendall(message.encode())

        data = self.client_socket.recv(1024)
        print(f"Received response: {data.decode()}")
        
        time.sleep(1)
        
        
        # 上に持ち上げる
        self.moveXYZ(DobotPosX, DobotPosY, 20, self.rotation)
        
        # 物体を移動       
        destX, destY = self.get_destination_pos()
        
        message = f"move|{destX}|{destY}|{20}|{-35}"
        
        self.client_socket.sendall(message.encode())

        data = self.client_socket.recv(1024)
        print(f"Received response: {data.decode()}")    
        
        
        time.sleep(1)
        
        # 空圧グリッパーを離す
        message = "release"
        self.client_socket.sendall(message.encode())

        data = self.client_socket.recv(1024)
        print(f"Received response: {data.decode()}")
        
        # AutoによるDobotアーム停止中に設定。
        self.duringAutoSeq = False
        
        return


    # カメラ座標 → Dobot座標に変換
    # (image_x, image_y)がyoloのバウンディングボックスの中心座標
    # (dobot_x, dobot_y)がmove関数に渡す座標
    def convert_dobot_coordinate(self, image_x, image_y):
        image_point = np.array([[image_x, image_y]], dtype=np.float32)
        dobot_point = cv2.perspectiveTransform(image_point.reshape(-1, 1, 2), self.transformation_matrix)
        dobot_x, dobot_y = dobot_point[0][0]
        print(dobot_x, dobot_y)
        
        return int(dobot_x) , int(dobot_y)
    
    # 検知した物体のラベルから格納先の座標を返す
    def get_destination_pos(self) :
        destX = 0
        destY = 0
        
        # 画鋲A
        if self.bboxlabel == 0 :
          destX = 95
          destY = 185
        # 画鋲B
        elif self.bboxlabel == 1 :
          destX = 160
          destY = 190    
        
        return destX, destY


