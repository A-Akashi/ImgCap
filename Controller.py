import cv2

class CameraController:
    def __init__(self):
        self.video_source = 0
        self.vid = cv2.VideoCapture(self.video_source)
        self.is_camera_open = False
    
    def release(self):
        if self.vid.isOpened():
           self.vid.release()

    def connect(self):
        self.is_camera_open = True

    def disconnect(self):
        self.is_camera_open = False

    def isCameraOpen(self):
        return self.is_camera_open

    def get_Frame(self):
        ret, frame = self.vid.read()
        return ret, frame

    def snapshot(self):
        ret, frame = self.vid.read()
        if ret:
            cv2.imwrite("snapshot.png", frame)