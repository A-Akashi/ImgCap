import socket
import threading
import time
from DobotDriver.DobotWrapper import DobotWrapper


class DobotServer:
    def __init__(self):
        self.dobot = DobotWrapper("COM3")
        self.isDobotConnected = self.dobot.initiate()


    def handle_client(self, client_socket, address):
        while True:
            data = client_socket.recv(1024)
            if not data:
                break
            
            print(f"Received data from {address}: {data.decode()}")
    
            command = data.decode().split('|')
            
            if (command[0] == "grip"):
                self.grip()                
            elif (command[0] == "release"):
                self.release()

            elif (command[0] == "move"):
                self.move(command)

            client_socket.sendall(b"Command Sequence Finished from the server!")
        
        client_socket.close()
                  


    def start_server(self, host = '127.0.0.1', port = 12345):
        server_socket = socket.socket(socket.AF_INET, socket.SOCK_STREAM)

        server_socket.bind((host, port))

        server_socket.listen()

        print(f"Server listening on {host}:{port}")
        
        while True:
            client_socket, client_address = server_socket.accept()

            print(f"Connection established from {client_address}")

            client_handler = threading.Thread(target=self.handle_client, args=(client_socket, client_address))
            client_handler.start()


    def move(self, command):
        if self.isDobotConnected == False :
            print("Dobot is not Connected.")
            return
        
        # 現在位置取得
        curPos = self.dobot.get_position()
        print(f"curPosX[0] {curPos[0]} curPosY[1] {curPos[1]} curPosZ[2] {curPos[2]} curPosR[3] {curPos[3]}")
               
        posX = int(command[1])
        posY = int(command[2])
        
        
        if command[3] == "stay":
            posZ = int(curPos[2])
        else:
            posZ = int(command[3])
        
        Rot = int(command[4])
        
        self.dobot.move_arm(posX, posY, posZ, Rot) 
        
        return

    def grip(self):
        if self.isDobotConnected == False :
            print("Dobot is not Connected.")
            return
        
        self.dobot.grip(True)
        return 
    
    
    def release(self):
        if self.isDobotConnected == False :
            print("Dobot is not Connected.")
            return
        
        self.dobot.grip(False)
        return 
        
        

if __name__ == "__main__":
    server = DobotServer()
    server.start_server()