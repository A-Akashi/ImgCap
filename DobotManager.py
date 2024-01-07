from DobotDriver.DobotWrapper import DobotWrapper
import socket
import threading
import time


class DobotManager:
    def __init__(self):
        self.host = '127.0.0.1'
        self.port = 12345
        self.client_socket = self.create_socket()
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

        # 検知座標取得
        ImgPosX = self.GUIManager.center_x
        ImgPosY = self.GUIManager.center_y
        
        # Dobot座標系へ変換
        DobotPosX, DobotPosY = self.convert_dobot_coordinate(ImgPosX, ImgPosY) 
        
        threading.Thread(target=self.auto_sequence, args=(DobotPosX, DobotPosY)).start()
        
        return 

    def auto_sequence(self, DobotPosX, DobotPosY) :
        
        # 対象座標へ移動
        self.moveXYZ(DobotPosX, DobotPosY, -30, 0)
        
        time.sleep(0.5)
        
        # 空圧グリッパーを閉じる
        message = "grip"
        self.client_socket.sendall(message.encode())

        data = self.client_socket.recv(1024)
        print(f"Received response: {data.decode()}")
        
        time.sleep(1)
        
        # 物体を移動
        moveY = DobotPosY + 100
        
        message = f"move|{DobotPosX}|{moveY}|{20}|{0}"
        
        self.client_socket.sendall(message.encode())

        data = self.client_socket.recv(1024)
        print(f"Received response: {data.decode()}")    
        
        
        time.sleep(1)
        
        # 空圧グリッパーを離す
        message = "release"
        self.client_socket.sendall(message.encode())

        data = self.client_socket.recv(1024)
        print(f"Received response: {data.decode()}")
        
        return



    def convert_dobot_coordinate(self, x, y):
               
        # カメラ座標 → Dobot座標に変換
        # (image_x, image_y)がyoloのバウンディングボックスの中心座標
        # (dobot_x, dobot_y)がmove関数に渡す座標
        image_x, image_y = 122, 122
        image_point = np.array([[image_x, image_y]], dtype=np.float32)
        dobot_point = cv2.perspectiveTransform(image_point.reshape(-1, 1, 2), transformation_matrix)
        dobot_x, dobot_y = dobot_point[0][0]
        print(dobot_x, dobot_y)
        
        return dobot_x , dobot_y


